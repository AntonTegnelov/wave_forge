// wfc-gpu/src/buffers/mod.rs

//! Module organizing different GPU buffer types used in WFC-GPU.

// Imports
use crate::{error_recovery::RecoverableGpuOp, GpuError};
use bytemuck::{Pod, Zeroable};
use futures::future::{try_join_all, FutureExt};
use log::{error, info, trace, warn};
use std::{any::Any, collections::HashMap, future::Future, pin::Pin, sync::Arc};
use wfc_core::{grid::PossibilityGrid, BoundaryCondition};
use wfc_rules::AdjacencyRules;
use wgpu::{self, util::DeviceExt};
use wgpu::{BufferAsyncError, BufferSlice, Device, MapMode, Queue};

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
        // Clone Arcs needed by the closure outside of it
        let device_clone = device.clone();
        let queue_clone = queue.clone();
        let buffers_clone = self.clone(); // Clone self (GpuBuffers) to access buffers inside closure

        recoverable_op
            .try_with_recovery(move || {
                // Use the cloned Arcs inside the closure
                let device = device_clone;
                let queue = queue_clone;
                let buffers = buffers_clone; // Use the cloned GpuBuffers

                // Move staging buffer clones here, inside the closure but outside the ifs
                let staging_entropy_buffer = buffers.entropy_buffers.staging_entropy_buf.clone();
                let staging_min_info_buffer =
                    buffers.entropy_buffers.staging_min_entropy_info_buf.clone();
                let staging_grid_buffer =
                    buffers.grid_buffers.staging_grid_possibilities_buf.clone();
                let staging_flag_buffer = buffers.staging_contradiction_flag_buf.clone();
                let staging_loc_buffer = buffers.staging_contradiction_location_buf.clone();

                async move {
                    let mut futures: Vec<
                        Pin<
                            Box<
                                dyn Future<Output = Result<(String, Box<dyn Any>), GpuError>>
                                    + Send,
                            >,
                        >,
                    > = Vec::new();

                    if request.download_entropy {
                        // Use buffers reference from closure
                        let entropy_buffer = &buffers.entropy_buffers.entropy_buf;
                        let _expected_size = self.num_cells;
                        let entropy_future = download_buffer_data::<f32>(
                            &device,
                            &queue,
                            entropy_buffer,
                            &staging_entropy_buffer, // Use cloned staging buffer
                            _expected_size as u64,
                            Some("Entropy".to_string()),
                        )
                        .map(|res| {
                            res.map(|data| ("entropy".to_string(), Box::new(data) as Box<dyn Any>))
                        })
                        .boxed();
                        futures.push(entropy_future);
                    }

                    if request.download_min_entropy_info {
                        let min_info_buffer = &buffers.entropy_buffers.min_entropy_info_buf;
                        let _expected_size = self.num_cells * self.grid_buffers.u32s_per_cell;
                        let min_info_future = download_buffer_data::<u32>(
                            &device,
                            &queue,
                            min_info_buffer,
                            &staging_min_info_buffer, // Use cloned staging buffer
                            _expected_size as u64,
                            Some("Min Entropy Info".to_string()),
                        )
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

                    if request.download_grid_possibilities {
                        let grid_buffer = &buffers.grid_buffers.grid_possibilities_buf;
                        let _expected_size = self.num_cells * self.grid_buffers.u32s_per_cell;
                        let grid_future = download_buffer_data::<u32>(
                            &device,
                            &queue,
                            grid_buffer,
                            &staging_grid_buffer, // Use cloned staging buffer
                            _expected_size as u64,
                            Some("Grid Possibilities".to_string()),
                        )
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

                    if request.download_contradiction_location {
                        let flag_buffer = &buffers.contradiction_flag_buf;
                        let _expected_size = self.num_cells;
                        let flag_future = download_buffer_data::<u32>(
                            &device,
                            &queue,
                            flag_buffer,
                            &staging_flag_buffer, // Use cloned staging buffer
                            _expected_size as u64,
                            Some("Contradiction Flag".to_string()),
                        )
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

                        let loc_buffer = &buffers.contradiction_location_buf;
                        let _expected_size = self.num_cells;
                        let loc_future = download_buffer_data::<u32>(
                            &device,
                            &queue,
                            loc_buffer,
                            &staging_loc_buffer, // Use cloned staging buffer
                            _expected_size as u64,
                            Some("Contradiction Location".to_string()),
                        )
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
                    self.map_downloaded_data(futures).await
                }
            })
            .await
    }

    async fn map_downloaded_data(
        &self,
        futures: Vec<
            Pin<Box<dyn Future<Output = Result<(String, Box<dyn Any>), GpuError>> + Send>>,
        >,
    ) -> Result<GpuDownloadResults, GpuError> {
        let results = try_join_all(futures).await?;
        let mut final_results = GpuDownloadResults::default();

        let downloaded_data: HashMap<String, Box<dyn Any>> = results.into_iter().collect();

        for (key, data_box) in downloaded_data {
            match key.as_str() {
                "entropy" => {
                    if let Ok(b) = data_box.downcast::<Vec<f32>>() {
                        final_results.entropy = Some(*b);
                    } else {
                        warn!("Failed to downcast entropy data Box<dyn Any> to Box<Vec<f32>>");
                    }
                }
                "min_entropy_info" => {
                    if let Ok(b) = data_box.downcast::<Vec<u32>>() {
                        if b.len() >= 2 {
                            final_results.min_entropy_info = Some((f32::from_bits(b[0]), b[1]));
                        } else {
                            warn!("min_entropy_info data has len {}, expected >= 2", b.len());
                        }
                    } else {
                        warn!("Failed to downcast min_entropy_info data Box<dyn Any> to Box<Vec<u32>>");
                    }
                }
                "grid_possibilities" => {
                    if let Ok(d) = data_box.downcast::<Vec<u32>>() {
                        final_results.grid_possibilities = Some(*d);
                    } else {
                        warn!("Failed to downcast grid_possibilities data Box<dyn Any> to Box<Vec<u32>>");
                    }
                }
                "contradiction_flag" => {
                    if let Ok(b) = data_box.downcast::<Vec<u32>>() {
                        if !b.is_empty() {
                            final_results.contradiction_flag = Some(b[0] != 0);
                        } else {
                            warn!("contradiction_flag data is empty");
                        }
                    } else {
                        warn!("Failed to downcast contradiction_flag data Box<dyn Any> to Box<Vec<u32>>");
                    }
                }
                "contradiction_location" => {
                    if let Ok(b) = data_box.downcast::<Vec<u32>>() {
                        if !b.is_empty() {
                            final_results.contradiction_location = Some(b[0]);
                        } else {
                            warn!("contradiction_location data is empty");
                        }
                    } else {
                        warn!("Failed to downcast contradiction_location data Box<dyn Any> to Box<Vec<u32>>");
                    }
                }
                _ => warn!("Unknown key found during data mapping: {}", key),
            }
        }
        Ok(final_results)
    }
}

/// Downloads data from a GPU buffer into a CPU-accessible structure.
///
/// This function handles the asynchronous mapping, polling, and reading
/// required to safely get data back from the GPU.
///
/// # Type Parameters
///
/// * `T`: The type of data expected in the buffer (must be `Pod` and `Send`).
///
/// # Arguments
///
/// * `device` - Reference to the WGPU device.
/// * `queue` - Reference to the WGPU queue.
/// * `buffer_to_download` - Arc reference to the source GPU buffer.
/// * `staging_buffer` - Arc reference to the staging buffer (must have MAP_READ usage).
/// * `buffer_size` - The number of bytes to copy and read.
/// * `label` - An optional label for logging/debugging.
///
/// # Returns
///
/// A `Result` containing a `Vec<T>` with the downloaded data, or a `GpuError`.
pub async fn download_buffer_data<T>(
    device: &Device,
    queue: &Queue,
    buffer_to_download: &Arc<wgpu::Buffer>,
    staging_buffer: &Arc<wgpu::Buffer>,
    buffer_size: u64,
    label: Option<String>,
) -> Result<Vec<T>, GpuError>
where
    T: Pod + Send + 'static,
{
    if staging_buffer.size() < buffer_size {
        return Err(GpuError::BufferSizeMismatch(format!(
            "Staging buffer '{}' too small ({} bytes) for download ({} bytes)",
            label.as_deref().unwrap_or("Unnamed"),
            staging_buffer.size(),
            buffer_size
        )));
    }

    trace!(
        "Downloading {} bytes from buffer '{}'",
        buffer_size,
        label.as_deref().unwrap_or("Unnamed")
    );

    // 1. Copy data from GPU buffer to staging buffer
    let label_clone = label.clone();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(&format!(
            "Download Encoder for {}",
            label.as_deref().unwrap_or("buffer")
        )),
    });
    encoder.copy_buffer_to_buffer(buffer_to_download, 0, staging_buffer, 0, buffer_size);
    queue.submit(Some(encoder.finish()));
    trace!(
        "Copy command submitted for buffer '{}'",
        label.as_deref().unwrap_or("Unnamed")
    );

    // 2. Map the staging buffer for reading
    let buffer_slice: BufferSlice = staging_buffer.slice(..buffer_size);
    let (sender, receiver) = futures::channel::oneshot::channel();
    buffer_slice.map_async(MapMode::Read, move |result| {
        let closure_label = label_clone;
        if let Err(e) = sender.send(result) {
            error!(
                "Failed to send map_async result for buffer '{}': {:?}",
                closure_label.as_deref().unwrap_or("Unnamed"),
                e
            );
        }
    });
    trace!(
        "map_async called for buffer '{}'",
        label.as_deref().unwrap_or("Unnamed")
    );

    // 3. Poll the device until the mapping is complete (important!)
    // Use Maintain::Wait for simplicity, or Maintain::Poll in a loop for non-blocking.
    device.poll(wgpu::Maintain::Wait);
    trace!(
        "Device polled, waiting for map_async callback for '{}'",
        label.as_deref().unwrap_or("Unnamed")
    );

    // 4. Receive the mapping result
    let map_result: Result<(), BufferAsyncError> = match receiver.await {
        Ok(result) => result,
        Err(e) => {
            error!(
                "Failed to receive map_async result for buffer '{}': {}",
                label.as_deref().unwrap_or("Unnamed"),
                e
            );
            return Err(GpuError::TransferError(format!(
                "Channel error receiving map result for {}: {}",
                label.as_deref().unwrap_or("buffer"),
                e
            )));
        }
    };

    if let Err(e) = map_result {
        error!(
            "Failed to map buffer '{}' for reading: {}",
            label.as_deref().unwrap_or("Unnamed"),
            e
        );
        return Err(GpuError::TransferError(format!(
            "Failed to map buffer '{}': {}",
            label.as_deref().unwrap_or("Unnamed"),
            e
        )));
    }
    trace!(
        "Buffer '{}' mapped successfully",
        label.as_deref().unwrap_or("Unnamed")
    );

    // 5. Get the mapped buffer view and copy data
    let mapped_data = {
        let view = staging_buffer.slice(..buffer_size).get_mapped_range();
        trace!(
            "Got mapped range for buffer '{}'",
            label.as_deref().unwrap_or("Unnamed")
        );
        // Copy data immediately while the view is valid
        let data_vec = bytemuck::cast_slice::<u8, T>(&view).to_vec();
        // Drop the view implicitly here by ending the scope
        trace!(
            "Copied data from mapped buffer '{}'",
            label.as_deref().unwrap_or("Unnamed")
        );
        data_vec
    };

    // 6. Unmap the buffer
    staging_buffer.unmap();
    trace!(
        "Unmapped buffer '{}'",
        label.as_deref().unwrap_or("Unnamed")
    );

    Ok(mapped_data)
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
