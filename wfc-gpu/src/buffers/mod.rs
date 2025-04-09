// wfc-gpu/src/buffers/mod.rs

//! Module organizing different GPU buffer types used in WFC-GPU.

// Imports
use crate::error_recovery::GpuError;
use bytemuck::{Pod, Zeroable};
use log::{debug, error, info, warn};
use std::mem;
use std::sync::Arc;
use std::time::Instant;
use wfc_core::{grid::PossibilityGrid, BoundaryCondition};
use wfc_rules::AdjacencyRules;
use wgpu::{self, Device, MapMode, Queue};

// Re-export buffer modules
pub use entropy_buffers::EntropyBuffers;
pub use grid_buffers::GridBuffers;
pub use rule_buffers::RuleBuffers;
pub use worklist_buffers::WorklistBuffers;

// Declare submodules
pub mod entropy_buffers;
pub mod grid_buffers;
pub mod rule_buffers;
pub mod worklist_buffers;

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

/// Request for downloading data from GPU buffers.
#[derive(Default, Debug)]
pub struct DownloadRequest {
    pub download_entropy: bool,
    pub download_min_entropy_info: bool,
    pub download_grid_possibilities: bool,
    pub download_contradiction_flag: bool,
    pub download_contradiction_location: bool,
}

impl Default for DownloadRequest {
    /// Creates a new `DownloadRequest` with default values (all false).
    fn default() -> Self {
        Self {
            download_entropy: false,
            download_min_entropy_info: false,
            download_grid_possibilities: false,
            download_contradiction_flag: false,
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
        request: DownloadRequest,
    ) -> Result<GpuDownloadResults, GpuError> {
        let download_start = Instant::now();
        debug!("Starting download with request: {:?}", request);

        let mut final_results = GpuDownloadResults::default();

        // First collect all async operations into a Vec
        if request.download_entropy {
            let num_cells = self.grid_dims.0 * self.grid_dims.1 * self.grid_dims.2;
            let device = Arc::new(Device::default()); // We'll get these from GpuSynchronizer instead
            let queue = Arc::new(Queue::default());

            debug!("Entropy download requested for {} cells", num_cells);

            // We'll refactor this to use a different approach that doesn't require different async block types
            let download_size = num_cells * mem::size_of::<f32>();
            let buffer = self.entropy_buffers.entropy_buf.slice(..);

            let data = download_buffer_data::<f32>(
                device.clone(),
                queue.clone(),
                buffer,
                self.entropy_buffers.entropy_buf.slice(..),
                download_size,
                Some("Entropy Data".to_string()),
            )
            .await?;

            final_results.entropy = Some(data);
        }

        if request.download_min_entropy_info {
            debug!("Min entropy index download requested");

            let device = Arc::new(Device::default()); // These are placeholders
            let queue = Arc::new(Queue::default());

            let download_size = 5 * mem::size_of::<u32>();
            let buffer = self.entropy_buffers.min_entropy_info_buf.slice(..);

            let data = download_buffer_data::<u32>(
                device.clone(),
                queue.clone(),
                buffer,
                self.entropy_buffers.min_entropy_info_buf.slice(..),
                download_size,
                Some("Min Entropy Info".to_string()),
            )
            .await?;

            if data.len() >= 2 {
                // First value is min value, second is index
                let min_value = f32::from_bits(data[0]);
                let min_index = data[1];
                final_results.min_entropy_info = Some((min_value, min_index));
            }
        }

        if request.download_grid_possibilities {
            debug!("Grid state download requested");

            let device = Arc::new(Device::default()); // These are placeholders
            let queue = Arc::new(Queue::default());

            let num_cells = self.grid_dims.0 * self.grid_dims.1 * self.grid_dims.2;
            let u32s_per_cell = (self.num_tiles + 31) / 32; // Ceiling division by 32
            let download_size = num_cells * u32s_per_cell * mem::size_of::<u32>();
            let buffer = self.grid_buffers.grid_possibilities_buf.slice(..);

            let data = download_buffer_data::<u32>(
                device.clone(),
                queue.clone(),
                buffer,
                self.grid_buffers.grid_possibilities_buf.slice(..),
                download_size,
                Some("Grid Data".to_string()),
            )
            .await?;

            final_results.grid_possibilities = Some(data);
        }

        if request.download_contradiction_flag {
            debug!("Contradiction flag download requested");

            let device = Arc::new(Device::default()); // These are placeholders
            let queue = Arc::new(Queue::default());

            let download_size = mem::size_of::<u32>();
            let buffer = self.contradiction_flag_buf.slice(..);

            let data = download_buffer_data::<u32>(
                device.clone(),
                queue.clone(),
                buffer,
                self.contradiction_flag_buf.slice(..),
                download_size,
                Some("Contradiction Flag".to_string()),
            )
            .await?;

            let flag_value = if data.is_empty() { 0 } else { data[0] };
            final_results.contradiction_flag = Some(flag_value > 0);
        }

        if request.download_contradiction_location {
            debug!("Contradiction location download requested");

            let device = Arc::new(Device::default()); // These are placeholders
            let queue = Arc::new(Queue::default());

            let download_size = 3 * mem::size_of::<u32>();
            let buffer = self.contradiction_location_buf.slice(..);

            let data = download_buffer_data::<u32>(
                device.clone(),
                queue.clone(),
                buffer,
                self.contradiction_location_buf.slice(..),
                download_size,
                Some("Contradiction Location".to_string()),
            )
            .await?;

            if data.len() >= 3 {
                let x = data[0] as usize;
                let y = data[1] as usize;
                let z = data[2] as usize;
                let coords = (x, y, z);
                final_results.contradiction_location = Some(coords);
            }
        }

        debug!("Download completed in {:.2?}", download_start.elapsed());
        Ok(final_results)
    }

    /// Converts raw buffer data to a PossibilityGrid.
    ///
    /// # Arguments
    ///
    /// * `data` - Raw buffer data as a vector of u32
    /// * `grid_dims` - Grid dimensions (width, height, depth)
    /// * `num_tiles` - Number of tile types
    ///
    /// # Returns
    ///
    /// `Ok(PossibilityGrid)` if conversion is successful, or `Err(GpuError)` if data is invalid.
    pub fn to_possibility_grid_from_data(
        &self,
        data: &[u32],
        grid_dims: (usize, usize, usize),
        num_tiles: usize,
    ) -> Result<PossibilityGrid, GpuError> {
        let (width, height, depth) = grid_dims;
        let num_cells = width * height * depth;
        let u32s_per_cell = (num_tiles + 31) / 32;
        let expected_data_size = num_cells * u32s_per_cell;

        if data.len() < expected_data_size {
            return Err(GpuError::BufferSizeMismatch(format!(
                "Grid possibilities data is too small: expected at least {} elements, got {}",
                expected_data_size,
                data.len()
            )));
        }

        // Create a new grid with the specified dimensions
        let mut grid = PossibilityGrid::new(width, height, depth, num_tiles);

        // Fill the grid with the downloaded data
        let mut cell_index = 0;
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    if let Some(cell) = grid.get_mut(x, y, z) {
                        let data_start = cell_index * u32s_per_cell;
                        let data_end = data_start + u32s_per_cell;

                        if data_end <= data.len() {
                            // Copy the raw data to the cell's possibilities
                            for (i, &value) in data[data_start..data_end].iter().enumerate() {
                                cell.set_raw_word(i, value);
                            }
                        } else {
                            warn!(
                                "Invalid grid cell access during conversion: ({}, {}, {})",
                                x, y, z
                            );
                        }
                    }
                    cell_index += 1;
                }
            }
        }

        Ok(grid)
    }
}

/// Downloads data from a GPU buffer to CPU memory using a staging buffer.
///
/// # Arguments
///
/// * `device` - Arc-wrapped WGPU device
/// * `queue` - Arc-wrapped WGPU queue
/// * `source_buffer` - The source buffer to download from
/// * `staging_buffer` - The staging buffer to use for the download
/// * `buffer_size` - Size of the data to download in bytes
/// * `label` - Optional label for debugging
///
/// # Returns
///
/// * `Ok(Vec<T>)` - The downloaded data as a vector
/// * `Err(GpuError)` - If an error occurred during download
pub async fn download_buffer_data<T: bytemuck::Pod + bytemuck::Zeroable>(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    source_buffer: &wgpu::Buffer,
    staging_buffer: &wgpu::Buffer,
    buffer_size: u64,
    label: Option<String>,
) -> Result<Vec<T>, GpuError> {
    // Create buffer slice
    let label_str = label.as_deref().unwrap_or("unnamed buffer");
    debug!(
        "Starting download of '{}' ({} bytes)",
        label_str, buffer_size
    );

    // Ensure buffers are large enough
    if source_buffer.size() < buffer_size {
        return Err(GpuError::BufferSizeMismatch(format!(
            "Source buffer for '{}' is smaller than required size ({} < {})",
            label_str,
            source_buffer.size(),
            buffer_size
        )));
    }

    if staging_buffer.size() < buffer_size {
        return Err(GpuError::BufferSizeMismatch(format!(
            "Staging buffer for '{}' is smaller than required size ({} < {})",
            label_str,
            staging_buffer.size(),
            buffer_size
        )));
    }

    // Create command encoder
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(&format!("Download encoder for '{}'", label_str)),
    });

    // Copy source buffer to staging buffer
    encoder.copy_buffer_to_buffer(source_buffer, 0, staging_buffer, 0, buffer_size);

    queue.submit(Some(encoder.finish()));
    debug!("Copy command submitted for '{}'", label_str);

    // Map the staging buffer
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures::channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });

    // Wait for the mapping operation to complete
    device.poll(wgpu::Maintain::Wait);
    debug!("Device polled for '{}'", label_str);

    // Handle the mapping result
    let map_start = std::time::Instant::now();
    let map_timeout = std::time::Duration::from_secs(2);

    // Use a mutable variable for receiver handling
    let mut recv = receiver;
    loop {
        match recv.try_recv() {
            Ok(Some(Ok(()))) => {
                break;
            }
            Ok(Some(Err(e))) => {
                error!("Failed to map buffer '{}': {:?}", label_str, e);
                staging_buffer.unmap(); // Unmap on error
                return Err(GpuError::BufferMapError(format!(
                    "Failed to map buffer '{}': {}",
                    label_str,
                    e.to_string()
                )));
            }
            Ok(None) | Err(_) => {
                // We're still waiting
                if map_start.elapsed() > map_timeout {
                    error!("Timeout waiting for buffer map for '{}'", label_str);
                    staging_buffer.unmap(); // Attempt to unmap on timeout
                    return Err(GpuError::BufferMapTimeout(label_str.to_string()));
                }
                tokio::task::yield_now().await;
            }
        }
    }

    // Read data from the mapped buffer
    let data = {
        let mapped_range = buffer_slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&mapped_range).to_vec();
        result
    };

    staging_buffer.unmap();
    debug!(
        "Download for '{}' completed in {:?}",
        label_str,
        map_start.elapsed()
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
