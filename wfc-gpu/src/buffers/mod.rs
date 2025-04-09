// wfc-gpu/src/buffers/mod.rs

//! Module organizing different GPU buffer types used in WFC-GPU.

// Imports
use crate::{error_recovery::RecoverableGpuOp, GpuError};
use bytemuck::{Pod, Zeroable};
use futures::future::{try_join_all, FutureExt, TryFutureExt};
use log::{debug, error, info, warn};
use std::{
    any::Any,
    collections::HashMap,
    future::Future,
    marker::Unpin,
    ops::Deref,
    pin::Pin,
    sync::Arc,
    time::{Duration, Instant},
};
use wfc_core::{
    grid::{GridDefinition, PossibilityGrid},
    BoundaryCondition,
};
use wfc_rules::AdjacencyRules;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BufferAddress, CommandEncoder, Device, MapMode, Queue,
};

// Declare submodules
pub mod entropy_buffers;
pub mod grid_buffers;
pub mod rule_buffers;
pub mod worklist_buffers;

// Re-export key structs from submodules
pub use entropy_buffers::EntropyBuffers;
pub use grid_buffers::GridBuffers;
pub use rule_buffers::RuleBuffers;
pub use worklist_buffers::WorklistBuffers;

// --- Struct Definitions --- //

/// Uniform buffer structure holding parameters accessible by GPU compute shaders.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
pub struct GpuParamsUniform {
    pub grid_width: u32,
    pub grid_height: u32,
    pub grid_depth: u32,
    pub num_tiles: u32,
    pub num_axes: u32,
    pub boundary_mode: u32,
    pub heuristic_type: u32,
    pub tie_breaking: u32,
    pub max_propagation_steps: u32,
    pub contradiction_check_frequency: u32,
    pub worklist_size: u32,
    pub grid_element_count: u32,
    pub _padding: u32,
}

/// Uniform buffer structure for entropy shader parameters.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuEntropyShaderParams {
    pub grid_dims: [u32; 3],
    pub heuristic_type: u32,
    pub num_tiles: u32,
    pub u32s_per_cell: u32,
    pub _padding1: u32,
    pub _padding2: u32,
}

/// DynamicBufferConfig contains settings for how buffers are resized
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicBufferConfig {
    pub growth_factor: f32,
    pub min_buffer_size: u64,
    pub max_buffer_size: u64,
    pub auto_resize: bool,
}

impl Default for DynamicBufferConfig {
    fn default() -> Self {
        Self {
            growth_factor: 1.5,
            min_buffer_size: 1024,
            max_buffer_size: 1024 * 1024 * 1024,
            auto_resize: true,
        }
    }
}

/// Stores all the GPU buffers needed for the WFC algorithm.
#[derive(Debug, Clone)]
pub struct GpuBuffers {
    pub grid_buffers: GridBuffers,
    pub rule_buffers: RuleBuffers,
    pub entropy_buffers: EntropyBuffers,
    pub contradiction_flag_buf: Arc<wgpu::Buffer>,
    pub staging_contradiction_flag_buf: Arc<wgpu::Buffer>,
    pub contradiction_location_buf: Arc<wgpu::Buffer>,
    pub staging_contradiction_location_buf: Arc<wgpu::Buffer>,
    pub params_uniform_buf: Arc<wgpu::Buffer>,
    pub pass_statistics_buf: Arc<wgpu::Buffer>,
    pub staging_pass_statistics_buf: Arc<wgpu::Buffer>,
    pub worklist_buffers: WorklistBuffers,
    pub num_cells: usize,
    pub original_grid_dims: Option<(usize, usize, usize)>,
    pub grid_dims: (usize, usize, usize),
    pub num_tiles: usize,
    pub num_axes: usize,
    pub boundary_mode: wfc_core::BoundaryCondition,
    pub entropy_params_buffer: Arc<wgpu::Buffer>,
    pub dynamic_buffer_config: Option<DynamicBufferConfig>,
}

/// Holds the results downloaded from the GPU after relevant compute passes.
#[derive(Debug, Clone, Default)]
pub struct GpuDownloadResults {
    pub entropy: Option<Vec<f32>>,
    pub min_entropy_info: Option<(f32, u32)>,
    pub contradiction_flag: Option<bool>,
    pub contradiction_location: Option<u32>,
    pub grid_possibilities: Option<Vec<u32>>,
}

/// Specifies which data to download from the GPU.
pub struct DownloadRequest {
    pub download_entropy: bool,
    pub download_min_entropy_info: bool,
    pub download_grid_possibilities: bool,
    pub download_contradiction_location: bool,
}

impl Default for DownloadRequest {
    /// Creates a new `DownloadRequest` with default values (all false).
    fn default() -> Self {
        Self {
            download_entropy: false,
            download_min_entropy_info: false,
            download_grid_possibilities: false,
            download_contradiction_location: false,
        }
    }
}

// --- Implementations --- //

impl GpuBuffers {
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        initial_grid: &PossibilityGrid,
        rules: &AdjacencyRules,
        boundary_mode: BoundaryCondition,
    ) -> Result<Self, GpuError> {
        info!(
            "Creating GPU buffers with boundary mode: {:?}...",
            boundary_mode
        );
        let width = initial_grid.width;
        let height = initial_grid.height;
        let depth = initial_grid.depth;
        let num_cells = width * height * depth;
        let num_tiles = rules.num_tiles();
        let num_axes = rules.num_axes();

        let default_dynamic_config = DynamicBufferConfig::default();
        let grid_buffers = GridBuffers::new(device, initial_grid, &default_dynamic_config)?;
        let worklist_buffers = WorklistBuffers::new(device, num_cells, &default_dynamic_config)?;
        let entropy_buffers = EntropyBuffers::new(device, num_cells, &default_dynamic_config)?;
        let rule_buffers = RuleBuffers::new(device, rules, &default_dynamic_config)?;

        let params = GpuParamsUniform {
            grid_width: width as u32,
            grid_height: height as u32,
            grid_depth: depth as u32,
            num_tiles: num_tiles as u32,
            num_axes: num_axes as u32,
            boundary_mode: match boundary_mode {
                BoundaryCondition::Finite => 0,
                BoundaryCondition::Periodic => 1,
            },
            heuristic_type: 0,
            tie_breaking: 0,
            max_propagation_steps: 1000,
            contradiction_check_frequency: 10,
            worklist_size: 0,
            grid_element_count: (num_cells * grid_buffers.u32s_per_cell) as u32,
            _padding: 0,
        };

        let contradiction_buffer_size = std::mem::size_of::<u32>() as u64;
        let contradiction_location_buffer_size = std::mem::size_of::<u32>() as u64;

        let contradiction_flag_buf = Self::create_buffer(
            device,
            contradiction_buffer_size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            Some("Contradiction Flag"),
        );
        let staging_contradiction_flag_buf = Self::create_buffer(
            device,
            contradiction_buffer_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            Some("Staging Contradiction Flag"),
        );
        let contradiction_location_buf = Self::create_buffer(
            device,
            contradiction_location_buffer_size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            Some("Contradiction Location"),
        );
        let staging_contradiction_location_buf = Self::create_buffer(
            device,
            contradiction_location_buffer_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            Some("Staging Contradiction Location"),
        );
        let params_uniform_buf = Arc::new(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Params Uniform"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            },
        ));
        let pass_statistics_buf = Self::create_buffer(
            device,
            16,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            Some("Pass Statistics"),
        );
        let staging_pass_statistics_buf = Self::create_buffer(
            device,
            16,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            Some("Staging Pass Statistics"),
        );
        let entropy_params_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Entropy Params Uniform"),
            size: std::mem::size_of::<GpuEntropyShaderParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        info!("GPU buffers created successfully.");
        Ok(Self {
            grid_buffers,
            rule_buffers,
            entropy_buffers,
            contradiction_flag_buf,
            staging_contradiction_flag_buf,
            contradiction_location_buf,
            staging_contradiction_location_buf,
            params_uniform_buf,
            pass_statistics_buf,
            staging_pass_statistics_buf,
            worklist_buffers,
            num_cells,
            original_grid_dims: Some((width, height, depth)),
            grid_dims: (width, height, depth),
            num_tiles,
            num_axes,
            boundary_mode,
            entropy_params_buffer,
            dynamic_buffer_config: Some(default_dynamic_config),
        })
    }

    pub fn create_buffer(
        device: &wgpu::Device,
        size: u64,
        usage: wgpu::BufferUsages,
        label: Option<&str>,
    ) -> Arc<wgpu::Buffer> {
        let padded_size = size.max(4); // Ensure minimum size of 4 bytes for alignment
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size: padded_size,
            usage,
            mapped_at_creation: false,
        });
        Arc::new(buffer)
    }

    pub fn resize_buffer(
        device: &wgpu::Device,
        old_buffer: &Arc<wgpu::Buffer>,
        new_size: u64,
        usage: wgpu::BufferUsages,
        label: Option<&str>,
        config: &DynamicBufferConfig,
    ) -> Arc<wgpu::Buffer> {
        let old_size = old_buffer.size();
        let mut actual_new_size = (new_size as f64 * config.growth_factor as f64).ceil() as u64;
        actual_new_size = actual_new_size.max(config.min_buffer_size);
        actual_new_size = actual_new_size.min(config.max_buffer_size);
        log::debug!(
            "Resizing buffer '{}' from {} to {} bytes (required: {})",
            label.unwrap_or("Unnamed"),
            old_size,
            actual_new_size,
            new_size
        );
        Self::create_buffer(device, actual_new_size, usage, label)
    }

    pub fn is_buffer_sufficient(buffer: &Arc<wgpu::Buffer>, required_size: u64) -> bool {
        buffer.size() >= required_size
    }

    pub fn resize_for_grid(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth: u32,
        num_tiles: u32,
        config: &DynamicBufferConfig,
    ) -> Result<(), String> {
        self.grid_buffers
            .ensure_grid_possibilities_buffer(device, width, height, depth, num_tiles, config)?;
        let num_cells = (width * height * depth) as usize;
        self.entropy_buffers
            .ensure_buffers(device, num_cells, config)?;
        self.worklist_buffers
            .ensure_worklist_buffers(device, width, height, depth, config)?;
        self.num_cells = num_cells;
        self.grid_dims = (width as usize, height as usize, depth as usize);
        Ok(())
    }

    pub async fn download_results(
        &self,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        request: DownloadRequest,
    ) -> Result<GpuDownloadResults, GpuError> {
        let recoverable_op = RecoverableGpuOp::default();

        // Clone necessary Arcs OUTSIDE the closure
        let device_clone = device.clone();
        let queue_clone = queue.clone();
        let buffers_clone = self.clone(); // Clone self (GpuBuffers needs Clone)
        let staging_entropy_buffer = self.entropy_buffers.staging_entropy_buf.clone();
        let staging_min_info_buffer = self.entropy_buffers.staging_min_entropy_info_buf.clone();
        let staging_grid_buffer = self.grid_buffers.staging_grid_possibilities_buf.clone();
        let staging_flag_buffer = self.staging_contradiction_flag_buf.clone();
        let staging_loc_buffer = self.staging_contradiction_location_buf.clone();

        recoverable_op
            .try_with_recovery(move || {
                // Now, move the cloned Arcs INTO the inner async move block
                // where they are actually used by download_buffer_data

                async move {
                    let mut futures: Vec<
                        Pin<
                            Box<
                                dyn Future<Output = Result<(String, Box<dyn Any>), GpuError>>
                                    + Send,
                            >,
                        >,
                    > = Vec::new();

                    // --- Entropy ---
                    if request.download_entropy {
                        let device_for_future = device_clone.clone();
                        let queue_for_future = queue_clone.clone();
                        let buffer_to_download = buffers_clone.entropy_buffers.entropy_buf.clone();
                        let staging_buffer_for_future = staging_entropy_buffer.clone();
                        let num_cells = buffers_clone.num_cells;

                        let entropy_future = async move {
                            download_buffer_data::<f32>(
                                device_for_future,                                    // Owned Arc
                                queue_for_future,                                     // Owned Arc
                                buffer_to_download,                                   // Owned Arc
                                staging_buffer_for_future,                            // Owned Arc
                                num_cells as u64 * std::mem::size_of::<f32>() as u64, // Correct size calc
                                Some("Entropy".to_string()),
                            )
                            .await
                        }
                        .map(|res| {
                            res.map(|data| ("entropy".to_string(), Box::new(data) as Box<dyn Any>))
                        })
                        .boxed();
                        futures.push(entropy_future);
                    }

                    // --- Min Entropy Info ---
                    if request.download_min_entropy_info {
                        let device_for_future = device_clone.clone();
                        let queue_for_future = queue_clone.clone();
                        let buffer_to_download =
                            buffers_clone.entropy_buffers.min_entropy_info_buf.clone();
                        let staging_buffer_for_future = staging_min_info_buffer.clone();
                        let num_cells = buffers_clone.num_cells;
                        let u32s_per_cell = buffers_clone.grid_buffers.u32s_per_cell; // Need this

                        let min_info_future = async move {
                            download_buffer_data::<u32>(
                                device_for_future,
                                queue_for_future,
                                buffer_to_download,
                                staging_buffer_for_future,
                                num_cells as u64
                                    * u32s_per_cell as u64
                                    * std::mem::size_of::<u32>() as u64, // Correct size
                                Some("Min Entropy Info".to_string()),
                            )
                            .await
                        }
                        .map(|res| {
                            res.map(|data| {
                                (
                                    "min_entropy_info".to_string(),
                                    Box::new(data) as Box<dyn Any>,
                                )
                            })
                        })
                        .boxed();
                        futures.push(min_info_future);
                    }

                    // --- Grid Possibilities ---
                    if request.download_grid_possibilities {
                        let device_for_future = device_clone.clone();
                        let queue_for_future = queue_clone.clone();
                        let buffer_to_download =
                            buffers_clone.grid_buffers.grid_possibilities_buf.clone();
                        let staging_buffer_for_future = staging_grid_buffer.clone();
                        let num_cells = buffers_clone.num_cells;
                        let u32s_per_cell = buffers_clone.grid_buffers.u32s_per_cell;

                        let grid_future = async move {
                            download_buffer_data::<u32>(
                                device_for_future,
                                queue_for_future,
                                buffer_to_download,
                                staging_buffer_for_future,
                                num_cells as u64
                                    * u32s_per_cell as u64
                                    * std::mem::size_of::<u32>() as u64, // Correct size
                                Some("Grid Possibilities".to_string()),
                            )
                            .await
                        }
                        .map(|res| {
                            res.map(|data| {
                                (
                                    "grid_possibilities".to_string(),
                                    Box::new(data) as Box<dyn Any>,
                                )
                            })
                        })
                        .boxed();
                        futures.push(grid_future);
                    }

                    // --- Contradiction Flag ---
                    if request.download_contradiction_location {
                        // Flag download tied to location download
                        let device_for_future = device_clone.clone();
                        let queue_for_future = queue_clone.clone();
                        let buffer_to_download = buffers_clone.contradiction_flag_buf.clone();
                        let staging_buffer_for_future = staging_flag_buffer.clone();
                        let num_cells = buffers_clone.num_cells;

                        let flag_future = async move {
                            download_buffer_data::<u32>(
                                device_for_future,
                                queue_for_future,
                                buffer_to_download,
                                staging_buffer_for_future,
                                num_cells as u64 * std::mem::size_of::<u32>() as u64, // Correct size
                                Some("Contradiction Flag".to_string()),
                            )
                            .await
                        }
                        .map(|res| {
                            res.map(|data| {
                                (
                                    "contradiction_flag".to_string(),
                                    Box::new(data) as Box<dyn Any>,
                                )
                            })
                        })
                        .boxed();
                        futures.push(flag_future);
                    }

                    // --- Contradiction Location ---
                    if request.download_contradiction_location {
                        let device_for_future = device_clone.clone();
                        let queue_for_future = queue_clone.clone();
                        let buffer_to_download = buffers_clone.contradiction_location_buf.clone();
                        let staging_buffer_for_future = staging_loc_buffer.clone();
                        let num_cells = buffers_clone.num_cells;

                        let loc_future = async move {
                            download_buffer_data::<u32>(
                                device_for_future,
                                queue_for_future,
                                buffer_to_download,
                                staging_buffer_for_future,
                                num_cells as u64 * std::mem::size_of::<u32>() as u64, // Correct size
                                Some("Contradiction Location".to_string()),
                            )
                            .await
                        }
                        .map(|res| {
                            res.map(|data| {
                                (
                                    "contradiction_location".to_string(),
                                    Box::new(data) as Box<dyn Any>,
                                )
                            })
                        })
                        .boxed();
                        futures.push(loc_future);
                    }

                    // --- Wait for all download futures ---
                    // Call map_downloaded_data which now needs to be static or take Arcs
                    // For simplicity, let's just process results here directly
                    let results = try_join_all(futures).await?;
                    let mut final_results = GpuDownloadResults::default();
                    let downloaded_data: HashMap<String, Box<dyn Any>> =
                        results.into_iter().collect();

                    for (key, data_box) in downloaded_data {
                        match key.as_str() {
                            "entropy" => {
                                if let Ok(b) = data_box.downcast::<Vec<f32>>() {
                                    final_results.entropy = Some(*b);
                                } else {
                                    warn!("Failed to downcast entropy data");
                                }
                            }
                            "min_entropy_info" => {
                                if let Ok(b) = data_box.downcast::<Vec<u32>>() {
                                    // Assuming min entropy info is (value: f32, index: u32)
                                    if b.len() >= 2 {
                                        // Check length before accessing
                                        final_results.min_entropy_info =
                                            Some((f32::from_bits(b[0]), b[1]));
                                    } else {
                                        warn!("min_entropy_info data too short");
                                    }
                                } else {
                                    warn!("Failed to downcast min_entropy_info data");
                                }
                            }
                            "grid_possibilities" => {
                                if let Ok(d) = data_box.downcast::<Vec<u32>>() {
                                    final_results.grid_possibilities = Some(*d);
                                } else {
                                    warn!("Failed to downcast grid_possibilities data");
                                }
                            }
                            "contradiction_flag" => {
                                if let Ok(b) = data_box.downcast::<Vec<u32>>() {
                                    if !b.is_empty() {
                                        // Check length before accessing
                                        final_results.contradiction_flag = Some(b[0] != 0);
                                    } else {
                                        warn!("contradiction_flag data is empty");
                                    }
                                } else {
                                    warn!("Failed to downcast contradiction_flag data");
                                }
                            }
                            "contradiction_location" => {
                                if let Ok(b) = data_box.downcast::<Vec<u32>>() {
                                    if !b.is_empty() {
                                        // Check length before accessing
                                        final_results.contradiction_location = Some(b[0]);
                                    } else {
                                        warn!("contradiction_location data is empty");
                                    }
                                } else {
                                    warn!("Failed to downcast contradiction_location data");
                                }
                            }
                            _ => warn!("Unknown key in downloaded data: {}", key),
                        }
                    }
                    Ok(final_results)
                }
            })
            .await
    }

    // Removed map_downloaded_data as logic is moved into download_results
}

pub async fn download_buffer_data<T>(
    // Change signature to take owned Arcs
    device: Arc<Device>,
    queue: Arc<Queue>,
    buffer_to_download: Arc<wgpu::Buffer>,
    staging_buffer: Arc<wgpu::Buffer>,
    buffer_size: u64, // Use buffer_size directly, already calculated in bytes
    label: Option<String>,
) -> Result<Vec<T>, GpuError>
where
    T: Pod + Send + 'static,
{
    let start_time = Instant::now();
    let label = label.unwrap_or_else(|| "Unnamed Buffer Download".to_string());
    debug!(
        "Starting download for '{}' ({} bytes)...",
        label, buffer_size
    );

    // Ensure staging buffer is large enough
    if staging_buffer.size() < buffer_size {
        // This should ideally trigger a resize, but for now, error out.
        // Or, the staging buffers should be created with sufficient size initially.
        return Err(GpuError::BufferError(format!(
            "Staging buffer for '{}' ({}) is smaller than required download size ({})",
            label,
            staging_buffer.size(),
            buffer_size
        )));
    }

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(&format!("Download Encoder for {}", label)),
    });

    encoder.copy_buffer_to_buffer(&buffer_to_download, 0, &staging_buffer, 0, buffer_size);

    queue.submit(Some(encoder.finish()));
    debug!("Copy command submitted for '{}'", label);

    // Map the staging buffer
    let buffer_slice = staging_buffer.slice(..buffer_size);
    let (sender, receiver) = futures::channel::oneshot::channel();
    buffer_slice.map_async(MapMode::Read, move |result| {
        if let Err(e) = sender.send(result) {
            error!("Failed to send map_async result for '{}': {:?}", label, e);
        }
    });

    // Poll the device while waiting for the map operation
    // Use a timeout to prevent indefinite waiting
    let map_timeout = Duration::from_secs(10); // 10 second timeout
    let map_start = Instant::now();
    loop {
        match receiver.try_recv() {
            Ok(Some(Ok(()))) => {
                debug!(
                    "Buffer '{}' mapped successfully after {:?}.",
                    label,
                    map_start.elapsed()
                );
                break; // Success!
            }
            Ok(Some(Err(e))) => {
                error!("Failed to map buffer '{}': {:?}", label, e);
                staging_buffer.unmap(); // Unmap on error
                return Err(GpuError::BufferMapFailed(e.to_string()));
            }
            Ok(None) => {
                // Channel closed unexpectedly
                error!("Mapping channel closed unexpectedly for '{}'", label);
                return Err(GpuError::InternalError("Mapping channel closed".into()));
            }
            Err(futures::channel::oneshot::Canceled) => {
                // Try_recv error (usually means not ready yet)
                device.poll(wgpu::Maintain::Wait); // Wait for GPU to finish
                                                   // Add a small sleep to avoid busy-waiting intensely
                tokio::time::sleep(Duration::from_millis(1)).await;
                if map_start.elapsed() > map_timeout {
                    error!("Timeout waiting for buffer map for '{}'", label);
                    staging_buffer.unmap(); // Attempt to unmap on timeout
                    return Err(GpuError::BufferMapTimeout(label));
                }
            }
        }
    }

    // Read data from the mapped buffer
    let data = {
        let mapped_range = staging_buffer.slice(..buffer_size).get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&mapped_range).to_vec();
        result
    };

    staging_buffer.unmap();
    debug!(
        "Download for '{}' completed in {:?}",
        label,
        start_time.elapsed()
    );
    Ok(data)
}

impl GpuDownloadResults {
    /// Converts the downloaded grid possibilities data into a `PossibilityGrid`.
    ///
    /// # Arguments
    ///
    /// * `grid_dims` - The dimensions (width, height, depth) of the grid.
    /// * `num_tiles` - The total number of tile types.
    ///
    /// # Returns
    ///
    /// `Ok(PossibilityGrid)` if conversion is successful, or `Err(GpuError)` if data is missing or invalid.
    pub fn to_possibility_grid(
        &self,
        grid_dims: (usize, usize, usize),
        num_tiles: usize,
    ) -> Result<PossibilityGrid, GpuError> {
        let (width, height, depth) = grid_dims;
        let num_cells = width * height * depth;
        let u32s_per_cell = (num_tiles + 31) / 32;

        let grid_possibilities = self.grid_possibilities.as_ref().ok_or_else(|| {
            GpuError::BufferOperationError("Grid possibilities data not downloaded".to_string())
        })?;

        if grid_possibilities.len() < num_cells * u32s_per_cell {
            return Err(GpuError::BufferSizeMismatch(format!(
                "Downloaded grid data size mismatch: expected {} u32s, got {}",
                num_cells * u32s_per_cell,
                grid_possibilities.len()
            )));
        }

        let mut grid = PossibilityGrid::new(width, height, depth, num_tiles);
        let mut cell_index = 0;
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    if let Some(cell) = grid.get_mut(x, y, z) {
                        cell.fill(false); // Clear initial state
                        let base_index = cell_index * u32s_per_cell;
                        for i in 0..u32s_per_cell {
                            if base_index + i < grid_possibilities.len() {
                                let bits = grid_possibilities[base_index + i];
                                for bit_pos in 0..32 {
                                    let tile_idx = i * 32 + bit_pos;
                                    if tile_idx < num_tiles {
                                        if ((bits >> bit_pos) & 1) == 1 {
                                            cell.set(tile_idx, true);
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        warn!(
                            "Attempted to access invalid grid cell ({}, {}, {}) during conversion",
                            x, y, z
                        );
                    }
                    cell_index += 1;
                }
            }
        }

        Ok(grid)
    }
}

// Test module remains in the respective files (grid_buffers.rs, worklist_buffers.rs)
