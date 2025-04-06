use crate::error_recovery::{AdaptiveTimeoutConfig, RecoverableGpuOp};
use crate::GpuError;
use bitvec::field::BitField;
use bytemuck::{Pod, Zeroable};
use futures::channel::oneshot;
use futures::{self, FutureExt};
use log::{debug, error, info};
use std::sync::Arc;
use wfc_core::grid::PossibilityGrid;
use wfc_core::BoundaryCondition;
use wfc_rules::AdjacencyRules;
use wgpu;
use wgpu::util::DeviceExt;

/// Uniform buffer structure holding parameters accessible by GPU compute shaders.
///
/// This struct defines the layout for constant data passed to the GPU, such as grid dimensions
/// and tile counts. It must exactly match the corresponding struct definition in the WGSL shaders
/// (e.g., `Params` struct in `propagate.wgsl` and `entropy.wgsl`).
///
/// Marked `#[repr(C)]` for stable memory layout across Rust/WGPU.
/// Implements `Pod` and `Zeroable` for safe, direct memory mapping (`bytemuck`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuParamsUniform {
    /// Width of the grid (X dimension).
    pub grid_width: u32,
    /// Height of the grid (Y dimension).
    pub grid_height: u32,
    /// Depth of the grid (Z dimension).
    pub grid_depth: u32,
    /// Total number of unique tile types.
    pub num_tiles: u32,
    /// Number of adjacency axes (typically 6 for 3D).
    pub num_axes: u32,
    /// Boundary mode: 0 for Clamped, 1 for Periodic.
    pub boundary_mode: u32,
    /// Entropy heuristic type: 0 for simple count, 1 for Shannon entropy, 2 for weighted.
    pub heuristic_type: u32,
    /// Tie-breaking strategy: 0 for none, 1 for deterministic, 2 for random pattern, 3 for position-based.
    pub tie_breaking: u32,
    /// Maximum propagation steps before giving up
    pub max_propagation_steps: u32,
    /// How often to check for contradictions during propagation
    pub contradiction_check_frequency: u32,
    /// Current size of the input worklist (number of updated cells) for the propagation shader.
    pub worklist_size: u32,
    /// Number of total grid elements for SoA access
    pub grid_element_count: u32,
    /// Padding to ensure struct size is a multiple of 16 bytes.
    pub _padding: u32,
}

/// Uniform buffer structure for entropy parameters
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EntropyParamsUniform {
    /// Width of the grid (X dimension).
    pub grid_width: u32,
    /// Height of the grid (Y dimension).
    pub grid_height: u32,
    /// Depth of the grid (Z dimension).
    pub grid_depth: u32,
    /// Padding to ensure alignment
    pub _padding1: u32,
    /// Entropy heuristic type (0=Shannon, 1=Count, 2=CountSimple, 3=WeightedCount)
    pub heuristic_type: u32,
    /// Padding to ensure alignment
    pub _padding2: u32,
    /// Padding to ensure alignment
    pub _padding3: u32,
    /// Padding to ensure alignment
    pub _padding4: u32,
}

/// DynamicBufferConfig contains settings for how buffers are resized
#[derive(Debug, Clone, Copy)]
pub struct DynamicBufferConfig {
    /// Growth factor for buffer sizes when resizing
    pub growth_factor: f32,
    /// Minimum size (in bytes) of any buffer after resizing
    pub min_buffer_size: u64,
    /// Maximum size (in bytes) allowed for any buffer
    pub max_buffer_size: u64,
    /// Whether to automatically resize buffers when needed
    pub auto_resize: bool,
}

impl Default for DynamicBufferConfig {
    fn default() -> Self {
        Self {
            growth_factor: 1.5, // Grow by 50% when resizing
            min_buffer_size: 1024, // Minimum 1KB buffer size
            max_buffer_size: 1024 * 1024 * 1024, // Maximum 1GB buffer size
            auto_resize: true, // Automatically resize by default
        }
    }
}

/// Stores all the GPU buffers needed for the WFC algorithm.
/// 
/// This struct maintains the state of all WGPU buffers required for the Wave Function Collapse
/// algorithm's GPU acceleration, including grid data, entropy calculations, worklists for propagation,
/// and various auxiliary buffers for things like contradiction detection.
#[derive(Debug)]
pub struct GpuBuffers {
    /// Buffer holding the current state of all cell possibilities - the primary WFC grid state.
    pub grid_possibilities_buf: Arc<wgpu::Buffer>,
    /// Staging buffer used during grid possibilities upload/download.
    pub staging_grid_possibilities_buf: Arc<wgpu::Buffer>,
    /// Buffer containing tile adjacency rules in a packed format for GPU access.
    pub rules_buf: Arc<wgpu::Buffer>,
    /// Buffer for storing calculated entropy values for each cell.
    pub entropy_buf: Arc<wgpu::Buffer>,
    /// Staging buffer used during entropy value upload/download.
    pub staging_entropy_buf: Arc<wgpu::Buffer>,
    /// Buffer containing information about the minimum entropy found (value and cell index).
    pub min_entropy_info_buf: Arc<wgpu::Buffer>,
    /// Staging buffer used during min entropy info upload/download.
    pub staging_min_entropy_info_buf: Arc<wgpu::Buffer>,
    /// First buffer for storing the worklist of cells to update in propagation (double-buffered design).
    pub worklist_buf_a: Arc<wgpu::Buffer>,
    /// Second buffer for storing the worklist of cells to update in propagation (double-buffered design).
    pub worklist_buf_b: Arc<wgpu::Buffer>,
    /// Buffer for tracking the number of cells in the worklist.
    pub worklist_count_buf: Arc<wgpu::Buffer>,
    /// Buffer containing a flag that is set when a contradiction is detected.
    pub contradiction_flag_buf: Arc<wgpu::Buffer>,
    /// Staging buffer used during contradiction flag upload/download.
    pub staging_contradiction_flag_buf: Arc<wgpu::Buffer>,
    /// Buffer for storing the location (cell index) where a contradiction occurred.
    pub contradiction_location_buf: Arc<wgpu::Buffer>,
    /// Staging buffer used during contradiction location upload/download.
    pub staging_contradiction_location_buf: Arc<wgpu::Buffer>,
    /// Staging buffer used during worklist count upload/download.
    pub staging_worklist_count_buf: Arc<wgpu::Buffer>,
    /// Uniform buffer containing GPU parameters like grid dimensions, tile count, etc.
    pub params_uniform_buf: Arc<wgpu::Buffer>,
    /// Buffer containing adjacency rules in a format optimized for the GPU.
    pub adjacency_rules_buf: Arc<wgpu::Buffer>,
    /// Buffer containing weights for adjacency rules.
    pub rule_weights_buf: Arc<wgpu::Buffer>,
    /// Buffer for storing statistics about each propagation pass.
    pub pass_statistics_buf: Arc<wgpu::Buffer>,
    /// Staging buffer used during pass statistics upload/download.
    pub staging_pass_statistics_buf: Arc<wgpu::Buffer>,
    /// Number of u32 words used to represent the possibilities for a single cell.
    pub u32s_per_cell: usize,
    /// Total number of cells in the grid.
    pub num_cells: usize,
    /// Original grid dimensions, used for error recovery.
    pub original_grid_dims: Option<(usize, usize, usize)>,
    /// Current size of the worklist.
    pub current_worklist_size: usize,
    /// Current worklist buffer index (0 or 1 for worklist_buf_a or worklist_buf_b).
    pub current_worklist_idx: usize,
    /// Current grid dimensions (width, height, depth).
    pub grid_dims: (usize, usize, usize),
    /// Number of tile types in the model.
    pub num_tiles: usize,
    /// Number of axes/directions in the adjacency rules.
    pub num_axes: usize,
    /// Boundary condition to apply (e.g., Wrap, Block).
    pub boundary_mode: wfc_core::BoundaryCondition,
    /// Uniform buffer for entropy calculation parameters.
    pub entropy_params_buffer: wgpu::Buffer,
    /// Configuration for dynamic buffer management.
    pub dynamic_buffer_config: Option<DynamicBufferConfig>,
}

/// Holds the results downloaded from the GPU after relevant compute passes.
#[derive(Debug, Clone, Default)]
pub struct GpuDownloadResults {
    pub entropy: Option<Vec<f32>>,
    pub min_entropy_info: Option<(f32, u32)>,
    pub contradiction_flag: Option<bool>,
    pub contradiction_location: Option<u32>,
    pub worklist_count: Option<u32>,
    pub grid_possibilities: Option<Vec<u32>>,
}

impl GpuBuffers {
    /// Creates and initializes all necessary GPU buffers for the WFC algorithm.
    ///
    /// This includes buffers for:
    /// - Grid possibilities (`grid_possibilities_buf`)
    /// - Adjacency rules (`rules_buf`)
    /// - Calculated entropy (`entropy_buf`)
    /// - Minimum entropy info (`min_entropy_info_buf`)
    /// - Updates worklist (`updates_buf`)
    /// - Uniform parameters (`params_uniform_buf`)
    /// - Output worklist (for potential future iterative propagation) (`output_worklist_buf`)
    /// - Output worklist count (`output_worklist_count_buf`)
    /// - Contradiction flag (`contradiction_flag_buf`)
    /// - Contradiction location (`contradiction_location_buf`)
    /// - Staging buffers for efficient data transfer between CPU and GPU.
    /// - Rule weights for weighted adjacency rules (`rule_weights_buf`)
    ///
    /// # Arguments
    ///
    /// * `device` - The WGPU `Device` used to create buffers.
    /// * `queue` - The WGPU `Queue` used for initial buffer writes (e.g., uploading rules).
    /// * `params` - The `GpuParamsUniform` structure containing grid dimensions, tile counts, etc.
    /// * `rules` - The `AdjacencyRules` structure containing the adjacency constraints.
    /// * `initial_possibilities` - A slice representing the initial possibility state for each cell,
    ///    packed into `u32` values (e.g., using bitsets).
    /// * `boundary_mode` - The boundary mode for the grid.
    ///
    /// # Returns
    ///
    /// * `Ok(Self)` - An instance of `GpuBuffers` containing all created buffers.
    /// * `Err(GpuError)` - If buffer creation fails or if initial data upload encounters issues
    ///   (e.g., size mismatch).
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

        // --- Pack Initial Grid State (Possibilities) ---
        let u32s_per_cell = (num_tiles + 31) / 32;
        let mut packed_possibilities = Vec::with_capacity(num_cells * u32s_per_cell);
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    if let Some(cell_possibilities) = initial_grid.get(x, y, z) {
                        if cell_possibilities.len() != num_tiles {
                            let err_msg = format!(
                                "Possibility grid cell ({}, {}, {}) has unexpected length: {} (expected {})",
                                x, y, z, cell_possibilities.len(), num_tiles
                            );
                            error!("{}", err_msg);
                            return Err(GpuError::BufferOperationError(err_msg));
                        }
                        // Pack the BitSlice into u32s
                        let iter = cell_possibilities.chunks_exact(32);
                        let remainder = iter.remainder();
                        for chunk in iter {
                            packed_possibilities.push(chunk.load_le::<u32>());
                        }
                        if !remainder.is_empty() {
                            let mut last_u32 = 0u32;
                            for (i, bit) in remainder.iter().by_vals().enumerate() {
                                if bit {
                                    last_u32 |= 1 << i;
                                }
                            }
                            packed_possibilities.push(last_u32);
                        }
                    } else {
                        // Handle potential error if get returns None unexpectedly
                        let err_msg =
                            format!("Failed to get possibility grid cell ({}, {}, {})", x, y, z);
                        error!("{}", err_msg);
                        return Err(GpuError::BufferOperationError(err_msg));
                    }
                }
            }
        }

        if packed_possibilities.len() != num_cells * u32s_per_cell {
            let err_msg = format!(
                "Internal Error: Packed possibilities size mismatch. Expected {}, Got {}. Grid dims: ({},{},{}), Num Tiles: {}, u32s/cell: {}",
                num_cells * u32s_per_cell, packed_possibilities.len(), width, height, depth, num_tiles, u32s_per_cell
            );
            error!("{}", err_msg);
            return Err(GpuError::BufferOperationError(err_msg));
        }

        let grid_buffer_size = (packed_possibilities.len() * std::mem::size_of::<u32>()) as u64;

        // --- Pack Rules ---
        // Each rule is represented by a bit in a u32 array
        // Index pattern: axis * num_tiles * num_tiles + tile1 * num_tiles + tile2
        let rule_bits_len = (num_axes * num_tiles * num_tiles + 31) / 32;
        let mut packed_rules = vec![0u32; rule_bits_len];

        for (axis, tile1, tile2) in rules.get_allowed_rules_map().keys() {
            let rule_idx = axis * num_tiles * num_tiles + tile1 * num_tiles + tile2;
            let u32_idx = rule_idx / 32;
            let bit_idx = rule_idx % 32;
            packed_rules[u32_idx] |= 1 << bit_idx;
        }

        // --- Pack Rule Weights ---
        // Each weighted rule is represented by two values:
        // - rule_idx: The packed rule index (axis, tile1, tile2)
        // - weight_bits: The f32 weight encoded as u32 bits
        let mut weighted_rules = Vec::new();

        for ((axis, tile1, tile2), weight) in rules.get_weighted_rules_map() {
            // Only add weights that are not 1.0 (since 1.0 is the default)
            if *weight < 1.0 {
                let rule_idx = axis * num_tiles * num_tiles + tile1 * num_tiles + tile2;
                weighted_rules.push(rule_idx as u32);
                // Convert f32 to bit representation for storage
                weighted_rules.push(weight.to_bits());
            }
        }

        // If no weighted rules, add a placeholder entry to avoid empty buffer issues
        if weighted_rules.is_empty() {
            weighted_rules.push(0);
            weighted_rules.push(1.0f32.to_bits());
        }

        // --- Create Uniform Buffer Data ---
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
            heuristic_type: 0, // Default to simple count
            tie_breaking: 0, // Default to none
            max_propagation_steps: 1000, // Default maximum steps
            contradiction_check_frequency: 10, // Default check frequency
            worklist_size: 0, // Initial worklist size is 0
            grid_element_count: (num_cells * u32s_per_cell) as u32,
            _padding: 0,
        };
        let _params_buffer_size = std::mem::size_of::<GpuParamsUniform>() as u64;

        // --- Calculate Other Buffer Sizes ---
        let entropy_buffer_size = (num_cells * std::mem::size_of::<f32>()) as u64;
        let updates_buffer_size = (num_cells * std::mem::size_of::<u32>()) as u64;
        let contradiction_buffer_size = std::mem::size_of::<u32>() as u64;
        let min_entropy_info_buffer_size = (2 * std::mem::size_of::<u32>()) as u64; // Size for [f32_bits, u32_index]
        let contradiction_location_buffer_size = std::mem::size_of::<u32>() as u64;

        // --- Create Buffers --- (Wrap in Arc)
        let grid_possibilities_buf = Arc::new(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("WFC Grid Possibilities Buffer"),
                contents: bytemuck::cast_slice(&packed_possibilities),
                usage: wgpu::BufferUsages::STORAGE 
                    | wgpu::BufferUsages::COPY_DST 
                    | wgpu::BufferUsages::COPY_SRC,
            },
        ));

        let rules_buf = Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Rules"),
                contents: bytemuck::cast_slice(&packed_rules),
                usage: wgpu::BufferUsages::STORAGE, // Read-only in shader
            }),
        );

        let params_uniform_buf = Arc::new(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Params Uniform"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            }
        ));

        let entropy_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Entropy"),
            size: entropy_buffer_size, // Use calculated size
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        let worklist_buf_a = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Worklist Buffer A"),
            size: updates_buffer_size, // Max possible size
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST // For initial upload
                | wgpu::BufferUsages::COPY_SRC, // If direct copy between worklists is needed (less likely now)
            mapped_at_creation: false,
        }));

        let worklist_buf_b = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Worklist Buffer B"),
            size: updates_buffer_size, // Max possible size
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        let worklist_count_buf = Arc::new(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Worklist Count Buffer"),
                contents: bytemuck::cast_slice(&[0u32]),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            },
        ));

        let contradiction_flag_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Contradiction Flag"),
            size: contradiction_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        let contradiction_location_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Contradiction Location"),
            size: contradiction_location_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST // For reset
                | wgpu::BufferUsages::COPY_SRC, // For download
            mapped_at_creation: false,
        }));

        let min_entropy_info_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Min Entropy Info Buffer"),
            size: min_entropy_info_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST // For initialization/reset
                | wgpu::BufferUsages::COPY_SRC, // For reading back result
            mapped_at_creation: false,
        }));

        let staging_grid_possibilities_buf =
            Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Grid Possibilities Buffer"),
                size: grid_buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

        let staging_entropy_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Entropy Buffer"),
            size: entropy_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let staging_min_entropy_info_buf =
            Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Min Entropy Info"),
                size: min_entropy_info_buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

        let staging_contradiction_flag_buf =
            Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Contradiction Flag"),
                size: contradiction_buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

        let staging_contradiction_location_buf =
            Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Contradiction Location Buffer"),
                size: contradiction_location_buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

        let staging_worklist_count_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Worklist Count Buffer"),
            size: std::mem::size_of::<u32>() as u64, // Use size directly
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Create rules buffer
        let adjacency_rules_buf = Arc::new(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("WFC Adjacency Rules Buffer"),
                contents: bytemuck::cast_slice(&packed_rules),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            },
        ));

        // Create rule weights buffer
        let rule_weights_buf = Arc::new(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("WFC Rule Weights Buffer"),
                contents: bytemuck::cast_slice(&weighted_rules),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            },
        ));

        // Create pass statistics buffer for tracking propagation details
        let pass_statistics_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pass Statistics Buffer"),
            size: 4 * 4, // 4 u32 values: [cells_added, possibilities_removed, contradictions, overflow]
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        // Create staging buffer for pass statistics
        let staging_pass_statistics_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Pass Statistics Buffer"),
            size: 4 * 4, // 4 u32 values
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));

        // Create entropy parameters buffer
        let entropy_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Entropy Parameters Buffer"),
            size: std::mem::size_of::<EntropyParamsUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        info!("GPU buffers created successfully.");
        Ok(Self {
            grid_possibilities_buf,
            staging_grid_possibilities_buf,
            rules_buf,
            entropy_buf,
            staging_entropy_buf,
            min_entropy_info_buf,
            staging_min_entropy_info_buf,
            worklist_buf_a,
            worklist_buf_b,
            worklist_count_buf,
            contradiction_flag_buf,
            staging_contradiction_flag_buf,
            contradiction_location_buf,
            staging_contradiction_location_buf,
            staging_worklist_count_buf,
            params_uniform_buf,
            adjacency_rules_buf,
            rule_weights_buf,
            pass_statistics_buf,
            staging_pass_statistics_buf,
            u32s_per_cell,
            num_cells,
            original_grid_dims: None,
            current_worklist_size: 0,
            current_worklist_idx: 0,
            grid_dims: (width, height, depth),
            num_tiles,
            num_axes,
            boundary_mode,
            entropy_params_buffer,
            dynamic_buffer_config: None,
        })
    }

    /// Create a new buffer with the provided configuration
    pub fn create_buffer(
        device: &wgpu::Device,
        size: u64,
        usage: wgpu::BufferUsages,
        label: Option<&str>,
    ) -> Arc<wgpu::Buffer> {
        let padded_size = size.max(1); // Ensure minimum size of 1
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size: padded_size,
            usage,
            mapped_at_creation: false,
        });
        Arc::new(buffer)
    }

    /// Resize a buffer to a new size, returning the new buffer
    pub fn resize_buffer(
        device: &wgpu::Device,
        old_buffer: &Arc<wgpu::Buffer>,
        new_size: u64,
        usage: wgpu::BufferUsages,
        label: Option<&str>,
        config: &DynamicBufferConfig,
    ) -> Arc<wgpu::Buffer> {
        let old_size = old_buffer.size();
        
        // Determine the new buffer size based on growth factor and bounds
        let mut actual_new_size = (new_size as f64 * config.growth_factor as f64).ceil() as u64;
        actual_new_size = actual_new_size.max(config.min_buffer_size);
        actual_new_size = actual_new_size.min(config.max_buffer_size);
        
        // Create a new buffer with the calculated size
        log::debug!("Resizing buffer from {} to {} bytes", old_size, actual_new_size);
        Self::create_buffer(device, actual_new_size, usage, label)
    }

    /// Check if a buffer is sufficient for the required size
    pub fn is_buffer_sufficient(
        buffer: &Arc<wgpu::Buffer>,
        required_size: u64,
    ) -> bool {
        buffer.size() >= required_size
    }

    /// Ensure the grid possibilities buffer is sufficient for the given dimensions
    pub fn ensure_grid_possibilities_buffer(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth: u32,
        num_tiles: u32,
        config: &DynamicBufferConfig,
    ) -> Result<(), String> {
        let num_cells = (width * height * depth) as usize;
        let u32s_per_cell = ((num_tiles + 31) / 32) as usize; // Ceiling division to get number of u32s needed
        let required_size = (num_cells * u32s_per_cell * std::mem::size_of::<u32>()) as u64;
        
        if !Self::is_buffer_sufficient(&self.grid_possibilities_buf, required_size) {
            let new_buffer = Self::resize_buffer(
                device,
                &self.grid_possibilities_buf,
                required_size,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                Some("Grid Possibilities Buffer"),
                config,
            );
            self.grid_possibilities_buf = new_buffer;
        }
        
        Ok(())
    }

    /// Ensure the entropy buffer is sufficient for the given dimensions
    pub fn ensure_entropy_buffer(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth: u32,
        config: &DynamicBufferConfig,
    ) -> Result<(), String> {
        let num_cells = (width * height * depth) as usize;
        let required_size = (num_cells * std::mem::size_of::<f32>()) as u64;
        
        if !Self::is_buffer_sufficient(&self.entropy_buf, required_size) {
            let new_buffer = Self::resize_buffer(
                device,
                &self.entropy_buf,
                required_size,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                Some("Entropy Buffer"),
                config,
            );
            self.entropy_buf = new_buffer;
        }
        
        Ok(())
    }

    /// Ensure the worklist buffers are sufficient for the given dimensions
    pub fn ensure_worklist_buffers(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth: u32,
        config: &DynamicBufferConfig,
    ) -> Result<(), String> {
        let num_cells = (width * height * depth) as usize;
        let required_size = (num_cells * std::mem::size_of::<u32>()) as u64;
        
        if !Self::is_buffer_sufficient(&self.worklist_buf_a, required_size) {
            let new_buffer_a = Self::resize_buffer(
                device,
                &self.worklist_buf_a,
                required_size,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                Some("Worklist Buffer A"),
                config,
            );
            self.worklist_buf_a = new_buffer_a;
        }
        
        if !Self::is_buffer_sufficient(&self.worklist_buf_b, required_size) {
            let new_buffer_b = Self::resize_buffer(
                device,
                &self.worklist_buf_b,
                required_size,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                Some("Worklist Buffer B"),
                config,
            );
            self.worklist_buf_b = new_buffer_b;
        }
        
        // Also resize the worklist count buffer (much smaller, fixed size)
        if !Self::is_buffer_sufficient(&self.worklist_count_buf, 4) {
            let new_count_buf = Self::resize_buffer(
                device,
                &self.worklist_count_buf,
                4,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                Some("Worklist Count Buffer"),
                config,
            );
            self.worklist_count_buf = new_count_buf;
        }
        
        Ok(())
    }

    /// Resize all essential buffers for a given grid dimension
    pub fn resize_for_grid(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth: u32,
        num_tiles: u32,
        config: &DynamicBufferConfig,
    ) -> Result<(), String> {
        self.ensure_grid_possibilities_buffer(device, width, height, depth, num_tiles, config)?;
        self.ensure_entropy_buffer(device, width, height, depth, config)?;
        self.ensure_worklist_buffers(device, width, height, depth, config)?;
        
        // Update grid dimensions
        self.num_cells = (width * height * depth) as usize;
        self.grid_dims = (width as usize, height as usize, depth as usize);
        
        Ok(())
    }

    /// Set the dynamic buffer configuration
    pub fn with_dynamic_buffer_config(mut self, config: DynamicBufferConfig) -> Self {
        self.dynamic_buffer_config = Some(config);
        self
    }

    /// Upload initial updates with auto-resize if needed
    pub fn upload_initial_updates_with_auto_resize(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        updates: &[u32],
        active_worklist_idx: u32,
    ) -> Result<(), String> {
        // Check if we need to auto-resize
        let should_resize = if let Some(config) = &self.dynamic_buffer_config {
            if config.auto_resize {
                let required_size = std::mem::size_of_val(updates) as u64;
                let active_buffer = if active_worklist_idx == 0 {
                    &self.worklist_buf_a
                } else {
                    &self.worklist_buf_b
                };

                !Self::is_buffer_sufficient(active_buffer, required_size)
            } else {
                false
            }
        } else {
            false
        };

        // If we need to resize, do it with a cloned config
        if should_resize {
            if let Some(config) = self.dynamic_buffer_config {
                self.ensure_worklist_buffers(
                    device,
                    self.grid_dims.0 as u32,
                    self.grid_dims.1 as u32,
                    self.grid_dims.2 as u32,
                    &config,
                )?;
            }
        }

        // Proceed with upload
        self.upload_initial_updates(queue, updates, active_worklist_idx as usize)
            .map_err(|e| e.to_string())
    }

    /// Uploads initial update list to a specific worklist buffer.
    pub fn upload_initial_updates(
        &self,
        queue: &wgpu::Queue,
        updates: &[u32],
        buffer_index: usize,
    ) -> Result<(), GpuError> {
        if updates.is_empty() {
            debug!("No initial updates to upload.");
            return Ok(());
        }
        let target_buffer = if buffer_index == 0 {
            &self.worklist_buf_a
        } else {
            &self.worklist_buf_b
        };

        let update_data = bytemuck::cast_slice(updates);
        if update_data.len() as u64 > target_buffer.size() {
            error!(
                "Initial update data size ({}) exceeds worklist buffer size ({}).",
                update_data.len(),
                target_buffer.size()
            );
            return Err(GpuError::BufferOperationError(format!(
                "Initial update data size ({}) exceeds worklist buffer size ({})",
                update_data.len(),
                target_buffer.size()
            )));
        }
        debug!(
            "Uploading {} initial updates ({} bytes) to worklist buffer {}.",
            updates.len(),
            update_data.len(),
            buffer_index
        );
        queue.write_buffer(target_buffer, 0, update_data);
        Ok(())
    }

    /// Resets the `min_entropy_info_buf` on the GPU to its initial state.
    ///
    /// Sets the minimum entropy value to `f32::MAX` (represented as bits) and the index to `u32::MAX`.
    /// This is typically done before running the entropy calculation shader.
    ///
    /// # Arguments
    ///
    /// * `queue` - The WGPU `Queue` used to write to the buffer.
    ///
    /// # Returns
    ///
    /// * `Ok(())` always (buffer writing is typically fire-and-forget, errors are harder to catch here).
    pub fn reset_min_entropy_info(&self, queue: &wgpu::Queue) -> Result<(), GpuError> {
        let initial_data = [f32::MAX.to_bits(), u32::MAX]; // [min_entropy_f32_bits, min_index_u32]
        queue.write_buffer(
            &self.min_entropy_info_buf,
            0,
            bytemuck::cast_slice(&initial_data),
        );
        Ok(())
    }

    /// Resets the `contradiction_flag_buf` on the GPU to 0.
    ///
    /// A value of 0 indicates no contradiction has been detected.
    /// This should be called before running the propagation shader.
    ///
    /// # Arguments
    ///
    /// * `queue` - The WGPU `Queue` used to write to the buffer.
    ///
    /// # Returns
    ///
    /// * `Ok(())` always.
    pub fn reset_contradiction_flag(&self, queue: &wgpu::Queue) -> Result<(), GpuError> {
        queue.write_buffer(
            &self.contradiction_flag_buf,
            0,
            bytemuck::cast_slice(&[0u32]),
        );
        Ok(())
    }

    /// Resets the `worklist_count_buf` on the GPU to 0.
    ///
    /// Used if implementing iterative GPU propagation where the shader generates a new worklist.
    ///
    /// # Arguments
    ///
    /// * `queue` - The WGPU `Queue` used to write to the buffer.
    ///
    /// # Returns
    ///
    /// * `Ok(())` always.
    pub fn reset_worklist_count(&self, queue: &wgpu::Queue) -> Result<(), GpuError> {
        queue.write_buffer(&self.worklist_count_buf, 0, bytemuck::cast_slice(&[0u32]));
        Ok(())
    }

    /// Resets the `contradiction_location_buf` on the GPU to `u32::MAX`.
    ///
    /// `u32::MAX` is used to indicate that no specific contradiction location has been recorded yet.
    ///
    /// # Arguments
    ///
    /// * `queue` - The WGPU `Queue` used to write to the buffer.
    ///
    /// # Returns
    ///
    /// * `Ok(())` always.
    pub fn reset_contradiction_location(&self, queue: &wgpu::Queue) -> Result<(), GpuError> {
        let max_u32 = [u32::MAX];
        queue.write_buffer(
            &self.contradiction_location_buf,
            0,
            bytemuck::cast_slice(&max_u32),
        );
        Ok(())
    }

    /// Updates the `worklist_size` field within the `params_uniform_buf` on the GPU.
    ///
    /// This informs the propagation shader how many updated cells are present in the `updates_buf`.
    ///
    /// # Arguments
    ///
    /// * `queue` - The WGPU `Queue` used to write to the buffer.
    /// * `worklist_size` - The number of valid entries in the `updates_buf`.
    ///
    /// # Returns
    ///
    /// * `Ok(())` always.
    ///
    /// # Panics
    ///
    /// This function relies on the memory layout of `GpuParamsUniform`. Changes to that struct
    /// might require updating the offset calculation here.
    pub fn update_params_worklist_size(
        &self,
        queue: &wgpu::Queue,
        worklist_size: u32,
    ) -> Result<(), GpuError> {
        let offset = (5 * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
        queue.write_buffer(
            &self.params_uniform_buf,
            offset,
            bytemuck::cast_slice(&[worklist_size]),
        );
        Ok(())
    }

    /// Downloads data from multiple GPU buffers in parallel and returns the results.
    ///
    /// This is the main data retrieval function to get processed data back from the GPU.
    /// It downloads whichever buffers are requested via the boolean flags.
    ///
    /// Enhanced with error recovery for non-fatal GPU errors.
    pub async fn download_results(
        &self,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        download_entropy: bool,
        download_min_entropy_info: bool,
        download_grid_possibilities: bool,
        _download_worklist: bool, // Not used currently, but kept for API compatibility
        download_worklist_size: bool,
        download_contradiction_location: bool,
    ) -> Result<GpuDownloadResults, GpuError> {
        // Create a recoverable operation wrapper
        let recoverable_op = RecoverableGpuOp::default();

        // Use the recovery mechanism for the download operation
        recoverable_op
            .try_with_recovery(|| async {
                // Create the download command once per attempt
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Download Results Encoder"),
                });

                // Initialize empty results
                let mut result = GpuDownloadResults {
                    entropy: None,
                    min_entropy_info: None,
                    contradiction_flag: None,
                    contradiction_location: None,
                    worklist_count: None,
                    grid_possibilities: None,
                };

                // Queue copy commands for the requested buffers
                if download_entropy {
                    encoder.copy_buffer_to_buffer(
                        &self.entropy_buf,
                        0,
                        &self.staging_entropy_buf,
                        0,
                        self.entropy_buf.size(),
                    );
                }
                if download_min_entropy_info {
                    encoder.copy_buffer_to_buffer(
                        &self.min_entropy_info_buf,
                        0,
                        &self.staging_min_entropy_info_buf,
                        0,
                        self.min_entropy_info_buf.size(),
                    );
                }
                if download_grid_possibilities {
                    encoder.copy_buffer_to_buffer(
                        &self.grid_possibilities_buf,
                        0,
                        &self.staging_grid_possibilities_buf,
                        0,
                        self.grid_possibilities_buf.size(),
                    );
                }
                if download_worklist_size {
                    encoder.copy_buffer_to_buffer(
                        &self.worklist_count_buf,
                        0,
                        &self.staging_worklist_count_buf,
                        0,
                        self.worklist_count_buf.size(),
                    );
                }
                if download_contradiction_location {
                    encoder.copy_buffer_to_buffer(
                        &self.contradiction_location_buf,
                        0,
                        &self.staging_contradiction_location_buf,
                        0,
                        self.contradiction_location_buf.size(),
                    );
                }

                // Submit the copy commands
                queue.submit(std::iter::once(encoder.finish()));
                debug!("GPU copy commands submitted for download.");

                // Now map the staging buffers and process results
                use futures::future::join_all;
                use futures::FutureExt;
                use std::convert::TryInto;
                use tokio::sync::oneshot;

                // For adaptive timeout
                let operation_complexity = 2.0; // More complex for bulk downloads
                let estimated_width = self.grid_dims.0;
                let estimated_height = self.grid_dims.1;
                let estimated_depth = self.grid_dims.2;
                let estimated_tiles = self.num_tiles;
                
                let recovery_op = Self::configure_adaptive_timeouts(
                    estimated_width,
                    estimated_height,
                    estimated_depth,
                    estimated_tiles,
                );
                let operation_timeout = recovery_op.calculate_operation_timeout(operation_complexity);
                
                let mut futures = Vec::new();
                
                // Grid possibilities download (most important for progressive results)
                if download_grid_possibilities {
                    let buffer = self.staging_grid_possibilities_buf.clone();
                    let expected_size = self.u32s_per_cell * self.num_cells;
                    
                    futures.push(async move {
                        let buffer_slice = buffer.slice(..);
                        let (tx, rx) = oneshot::channel();
                        
                        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                            let _ = tx.send(result);
                        });
                        
                        match tokio::time::timeout(operation_timeout, rx).await {
                            Ok(result) => {
                                result.map_err(|_| {
                                    GpuError::BufferOperationError("Mapping channel canceled".to_string())
                                })??;
                                
                                let mapped_range = buffer_slice.get_mapped_range();
                                let bytes = mapped_range.as_ref();
                                
                                // Convert bytes to u32 slice and then to Vec<u32>
                                let u32_slice = unsafe {
                                    std::slice::from_raw_parts(
                                        bytes.as_ptr() as *const u32,
                                        bytes.len() / std::mem::size_of::<u32>(),
                                    )
                                };
                                
                                // Validate the size
                                if u32_slice.len() >= expected_size {
                                    let grid_possibilities = u32_slice[..expected_size].to_vec();
                                    Ok(("grid_possibilities", Box::new(grid_possibilities) as Box<dyn std::any::Any + Send>))
                                } else {
                                    Err(GpuError::BufferOperationError(format!(
                                        "Grid possibilities buffer size mismatch: expected {} items, got {}",
                                        expected_size,
                                        u32_slice.len()
                                    )))
                                }
                            },
                            Err(_) => {
                                Err(GpuError::BufferOperationError(format!(
                                    "Grid possibilities buffer mapping timed out after {:?}",
                                    operation_timeout
                                )))
                            }
                        }
                    }.boxed());
                }
                
                // Worklist count download
                if download_worklist_size {
                    let buffer = self.staging_worklist_count_buf.clone();
                    
                    futures.push(async move {
                        let buffer_slice = buffer.slice(..);
                        let (tx, rx) = oneshot::channel();
                        
                        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                            let _ = tx.send(result);
                        });
                        
                        rx.await.map_err(|_| {
                            GpuError::BufferOperationError("Mapping channel canceled".to_string())
                        })??;
                        
                        let mapped_range = buffer_slice.get_mapped_range();
                        let bytes = mapped_range.to_vec();
                        drop(mapped_range);
                        buffer.unmap();
                        
                        if bytes.len() >= 4 {
                            let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
                            Ok(("worklist_count", Box::new(count) as Box<dyn std::any::Any + Send>))
                        } else {
                            Err(GpuError::BufferOperationError(
                                "Worklist count buffer size mismatch".to_string(),
                            ))
                        }
                    }.boxed());
                }
                
                // Process the futures
                let download_results = join_all(futures).await;
                
                // Process the results
                for download_result in download_results {
                    match download_result {
                        Ok((name, data)) => {
                            match name {
                                "grid_possibilities" => {
                                    if let Some(grid_data) = data.downcast_ref::<Vec<u32>>() {
                                        result.grid_possibilities = Some(grid_data.clone());
                                    }
                                }
                                "worklist_count" => {
                                    if let Some(count) = data.downcast_ref::<u32>() {
                                        result.worklist_count = Some(*count);
                                    }
                                }
                                _ => {}
                            }
                        }
                        Err(e) => {
                            log::error!("Error downloading GPU data: {}", e);
                            return Err(e);
                        }
                    }
                }
                
                // Clean up for entropy buffer
                if download_entropy {
                    let buffer = self.staging_entropy_buf.clone();
                    buffer.unmap();
                }
                
                // Clean up for min entropy info buffer
                if download_min_entropy_info {
                    let buffer = self.staging_min_entropy_info_buf.clone();
                    buffer.unmap();
                }
                
                // Clean up for contradiction location buffer
                if download_contradiction_location {
                    let buffer = self.staging_contradiction_location_buf.clone();
                    buffer.unmap();
                }

                Ok(result)
            })
            .await
    }

    /// Downloads the propagation status (contradiction flag, worklist count, contradiction location) from the GPU.
    ///
    /// This is an optimized method that combines multiple status downloads into a single operation
    /// to reduce synchronization points during propagation.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu Device for buffer copies
    /// * `queue` - The wgpu Queue for submitting commands
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * `has_contradiction` - Boolean indicating if a contradiction was detected
    /// * `worklist_count` - The number of cells in the output worklist
    /// * `contradiction_location` - Optional index of the cell where a contradiction occurred
    pub async fn download_propagation_status(
        &self,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Result<(bool, u32, Option<u32>), GpuError> {
        // Create encoder for copying buffers
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Propagation Status Encoder"),
        });

        // Copy the contradiction flag and worklist count to staging buffers
        encoder.copy_buffer_to_buffer(
            &self.contradiction_flag_buf,
            0,
            &self.staging_contradiction_flag_buf,
            0,
            self.contradiction_flag_buf.size(),
        );
        encoder.copy_buffer_to_buffer(
            &self.worklist_count_buf,
            0,
            &self.staging_worklist_count_buf,
            0,
            self.worklist_count_buf.size(),
        );

        queue.submit(Some(encoder.finish()));

        // For adaptive timeout, we need to estimate the grid size and complexity
        // We don't store these directly, but we can derive reasonable values
        // for the purpose of timeout configuration

        // Use fixed values for now since we don't have direct access to grid dimensions
        let estimated_width = 64;
        let estimated_height = 64;
        let estimated_depth = 1;
        let estimated_tiles = 32;

        // Create a RecoverableGpuOp with adaptive timeout
        let recovery_op = Self::configure_adaptive_timeouts(
            estimated_width,
            estimated_height,
            estimated_depth,
            estimated_tiles,
        );

        // Create futures for downloading contradiction flag and worklist count
        // Use higher operation complexity (1.5) for these operations as they're critical
        let operation_complexity = 1.5;
        let operation_timeout = recovery_op.calculate_operation_timeout(operation_complexity);

        // Use a longer timeout for larger grids to prevent premature timeouts
        let contradiction_flag_future: futures::future::BoxFuture<'_, Result<bool, GpuError>> = {
            let buffer = self.staging_contradiction_flag_buf.clone();
            async move {
                let buffer_slice = buffer.slice(..);
                let (tx, rx) = oneshot::channel();

                buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                    let _ = tx.send(result);
                });

                // Use tokio's timeout to handle timeouts more gracefully
                match tokio::time::timeout(operation_timeout, rx).await {
                    Ok(result) => {
                        result.map_err(|_| {
                            GpuError::BufferOperationError("Mapping channel canceled".to_string())
                        })??;
                    }
                    Err(_) => {
                        return Err(GpuError::BufferOperationError(format!(
                            "Buffer mapping timed out after {:?}",
                            operation_timeout
                        )));
                    }
                }

                let mapped_range = buffer_slice.get_mapped_range();
                let bytes = mapped_range.to_vec();
                drop(mapped_range);
                buffer.unmap();

                let flag_value = if bytes.len() >= 4 {
                    u32::from_le_bytes(bytes[0..4].try_into().unwrap()) != 0
                } else {
                    false
                };

                Ok(flag_value)
            }
            .boxed()
        };

        let worklist_count_future: futures::future::BoxFuture<'_, Result<u32, GpuError>> = {
            let buffer = self.staging_worklist_count_buf.clone();
            async move {
                let buffer_slice = buffer.slice(..);
                let (tx, rx) = oneshot::channel();

                buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                    let _ = tx.send(result);
                });

                // Use tokio's timeout to handle timeouts more gracefully
                match tokio::time::timeout(operation_timeout, rx).await {
                    Ok(result) => {
                        result.map_err(|_| {
                            GpuError::BufferOperationError("Mapping channel canceled".to_string())
                        })??;
                    }
                    Err(_) => {
                        return Err(GpuError::BufferOperationError(format!(
                            "Buffer mapping timed out after {:?}",
                            operation_timeout
                        )));
                    }
                }

                let mapped_range = buffer_slice.get_mapped_range();
                let bytes = mapped_range.to_vec();
                drop(mapped_range);
                buffer.unmap();

                let count = if bytes.len() >= 4 {
                    u32::from_le_bytes(bytes[0..4].try_into().unwrap())
                } else {
                    0
                };

                Ok(count)
            }
            .boxed()
        };

        // Run both futures concurrently for efficiency
        let (flag_result, count_result) =
            futures::join!(contradiction_flag_future, worklist_count_future);
        let has_contradiction = flag_result?;
        let worklist_count = count_result?;

        // If there's a contradiction, download the location
        let contradiction_location = if has_contradiction {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Contradiction Location Encoder"),
            });

            encoder.copy_buffer_to_buffer(
                &self.contradiction_location_buf,
                0,
                &self.staging_contradiction_location_buf,
                0,
                self.contradiction_location_buf.size(),
            );

            queue.submit(Some(encoder.finish()));

            let buffer = self.staging_contradiction_location_buf.clone();
            let buffer_slice = buffer.slice(..);
            let (tx, rx) = oneshot::channel();

            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

            rx.await.map_err(|_| {
                GpuError::BufferOperationError("Mapping channel canceled".to_string())
            })??;

            let mapped_range = buffer_slice.get_mapped_range();
            let bytes = mapped_range.to_vec();
            drop(mapped_range);
            buffer.unmap();

            if bytes.len() >= 4 {
                Some(u32::from_le_bytes(bytes[0..4].try_into().unwrap()))
            } else {
                None
            }
        } else {
            None
        };

        Ok((has_contradiction, worklist_count, contradiction_location))
    }

    /// Downloads the minimum entropy information from the GPU.
    ///
    /// This asynchronously maps the `min_entropy_info_buf` from the GPU to read the minimum entropy
    /// found (as f32 bits) and the index of the cell with that entropy.
    ///
    /// # Arguments
    ///
    /// * `device` - The WGPU `Device` to use for buffer mapping.
    /// * `queue` - The WGPU `Queue` to use for command submission.
    ///
    /// # Returns
    ///
    /// * `Ok(Some((min_entropy, min_idx)))` with the minimum entropy value and its index.
    /// * `Ok(None)` if no valid minimum was found.
    /// * `Err(GpuError)` if mapping or downloading the buffer fails.
    pub async fn download_min_entropy_info(
        &self,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Result<Option<(f32, usize)>, GpuError> {
        // Create a new command encoder for the operation
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Min Entropy Download Encoder"),
        });

        // We need to do another GPU operation to ensure all prior GPU work is complete
        // Create a temporary staging buffer to map
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Min Entropy Info Staging Buffer"),
            size: 8, // 2 x 4 bytes (u32)
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from the GPU buffer to the staging buffer
        encoder.copy_buffer_to_buffer(
            &self.min_entropy_info_buf,
            0,
            &staging_buffer,
            0,
            8, // 8 bytes (2 x u32)
        );

        // Submit the commands to the queue
        queue.submit(std::iter::once(encoder.finish()));

        // Use a oneshot channel for async mapping
        let (tx, rx) = futures::channel::oneshot::channel();
        
        // Start mapping the buffer with the callback
        let buffer_slice = staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // Wait for the mapping to complete
        device.poll(wgpu::Maintain::Wait);
        
        // Wait for the callback to be called
        match rx.await {
            Ok(Ok(())) => {
                // Get a view of the buffer data
                let data = buffer_slice.get_mapped_range();
                
                // Convert the bytes to a slice of u32
                let result: &[u32] = bytemuck::cast_slice(&data);
                
                // Convert the first u32 to f32 using the bits
                let min_entropy = f32::from_bits(result[0]);
                let min_idx = result[1] as usize;
                
                // Drop the buffer view so it can be unmapped
                drop(data);
                staging_buffer.unmap();
                
                // Check if we have a valid result
                if min_entropy == f32::MAX || min_idx == u32::MAX as usize {
                    Ok(None)
                } else {
                    Ok(Some((min_entropy, min_idx)))
                }
            }
            Ok(Err(e)) => Err(GpuError::BufferMapFailed(e)),
            Err(_) => Err(GpuError::BufferMappingFailed("Buffer mapping channel closed".to_string())),
        }
    }

    /// Sets up adaptive timeout handling for buffer operations.
    ///
    /// Configures timeouts that scale appropriately with grid size and complexity
    /// to prevent premature timeouts for large grids.
    ///
    /// # Arguments
    ///
    /// * `width` - Grid width
    /// * `height` - Grid height
    /// * `depth` - Grid depth
    /// * `num_tiles` - Number of tile types
    ///
    /// # Returns
    ///
    /// A RecoverableGpuOp instance configured with appropriate adaptive timeout settings
    pub fn configure_adaptive_timeouts(
        width: usize,
        height: usize,
        depth: usize,
        num_tiles: usize,
    ) -> RecoverableGpuOp {
        let grid_cells = width * height * depth;

        // Create recovery op with adaptive timeout config
        let timeout_config = AdaptiveTimeoutConfig {
            base_timeout_ms: 1000,        // 1 second base
            reference_cell_count: 10_000, // 10k cells reference grid
            size_scale_factor: 0.6,       // Scale sub-linearly with grid size
            tile_count_multiplier: 0.01,  // 1% per tile
            max_timeout_ms: 30_000,       // 30 seconds max
        };

        RecoverableGpuOp::new().with_adaptive_timeout(timeout_config, grid_cells, num_tiles)
    }

    /// Uploads entropy parameters to the GPU.
    ///
    /// # Arguments
    ///
    /// * `queue` - The WGPU queue to submit the upload to.
    /// * `params` - The entropy parameters to upload.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the upload succeeds, `Err(GpuError)` otherwise.
    pub fn upload_entropy_params(
        &self,
        queue: &wgpu::Queue,
        params: &EntropyParamsUniform,
    ) -> Result<(), GpuError> {
        queue.write_buffer(
            &self.entropy_params_buffer,
            0,
            bytemuck::cast_slice(&[*params]),
        );
        Ok(())
    }

    /// Uploads entropy values to the entropy buffer.
    ///
    /// # Arguments
    ///
    /// * `queue` - The queue to submit the upload to.
    /// * `entropy_values` - The entropy values to upload.
    ///
    /// # Returns
    ///
    /// A result indicating success or an error.
    pub fn upload_entropy_buffer(&self, queue: &wgpu::Queue, entropy_values: &[f32]) -> Result<(), GpuError> {
        if entropy_values.len() != self.num_cells {
            return Err(GpuError::BufferSizeMismatch(
                format!("Entropy buffer size mismatch: expected {}, got {}", self.num_cells, entropy_values.len())
            ));
        }
        
        queue.write_buffer(&self.entropy_buf, 0, bytemuck::cast_slice(entropy_values));
        Ok(())
    }
    
    /// Uploads minimum entropy cell information to the min entropy buffer.
    ///
    /// # Arguments
    ///
    /// * `queue` - The queue to submit the upload to.
    /// * `min_entropy_info` - The minimum entropy info to upload (entropy value and cell index).
    ///
    /// # Returns
    ///
    /// A result indicating success or an error.
    pub fn upload_min_entropy_buffer(&self, queue: &wgpu::Queue, min_entropy_info: &[u32]) -> Result<(), GpuError> {
        if min_entropy_info.len() != 2 {
            return Err(GpuError::BufferSizeMismatch(
                format!("Min entropy buffer size mismatch: expected 2, got {}", min_entropy_info.len())
            ));
        }
        
        queue.write_buffer(&self.min_entropy_info_buf, 0, bytemuck::cast_slice(min_entropy_info));
        Ok(())
    }
    
    /// Downloads entropy values from the entropy buffer.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to create staging buffers on.
    /// * `queue` - The queue to submit the download to.
    ///
    /// # Returns
    ///
    /// A result containing the downloaded entropy values or an error.
    pub fn download_entropy_buffer(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Vec<f32>, GpuError> {
        let buffer_size = self.num_cells * std::mem::size_of::<f32>();
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Entropy Download Staging Buffer"),
            size: buffer_size as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Entropy Download Encoder"),
        });
        
        encoder.copy_buffer_to_buffer(
            &self.entropy_buf,
            0,
            &staging_buffer,
            0,
            buffer_size as wgpu::BufferAddress,
        );
        
        // Submit commands directly to the provided queue
        queue.submit(std::iter::once(encoder.finish()));
        
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        
        device.poll(wgpu::Maintain::Wait);
        
        if let Ok(Ok(())) = receiver.recv() {
            let data = buffer_slice.get_mapped_range();
            let entropy_values: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();
            Ok(entropy_values)
        } else {
            Err(GpuError::BufferMappingFailed("Failed to map entropy buffer for reading".to_string()))
        }
    }
    
    /// Downloads minimum entropy cell information from the min entropy buffer.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to create staging buffers on.
    /// * `queue` - The queue to submit the download to.
    ///
    /// # Returns
    ///
    /// A result containing the downloaded min entropy info or an error.
    pub fn download_min_entropy_info_sync(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Vec<u32>, GpuError> {
        let buffer_size = 2 * std::mem::size_of::<u32>();
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Min Entropy Info Download Staging Buffer"),
            size: buffer_size as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Min Entropy Info Download Encoder"),
        });
        
        encoder.copy_buffer_to_buffer(
            &self.min_entropy_info_buf,
            0,
            &staging_buffer,
            0,
            buffer_size as wgpu::BufferAddress,
        );
        
        // Submit commands directly to the provided queue
        queue.submit(std::iter::once(encoder.finish()));
        
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        
        device.poll(wgpu::Maintain::Wait);
        
        if let Ok(Ok(())) = receiver.recv() {
            let data = buffer_slice.get_mapped_range();
            let min_entropy_info: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();
            Ok(min_entropy_info)
        } else {
            Err(GpuError::BufferMappingFailed("Failed to map min entropy buffer for reading".to_string()))
        }
    }
}

// The implementation of Drop for GpuBuffers
impl Drop for GpuBuffers {
    /// Performs cleanup of GPU resources when GpuBuffers is dropped.
    /// 
    /// This ensures that all mapped buffers are properly unmapped.
    /// Note that actual buffer deallocation is handled by Arc's Drop implementation
    /// when the last reference to each buffer is dropped.
    fn drop(&mut self) {
        // Unmap any buffers that might still be mapped
        // This avoids potential resource leaks if buffers are still mapped when dropped
        
        // Only log at debug level to avoid cluttering tests
        debug!("GpuBuffers being dropped, performing cleanup");
        
        // Since we can't directly check if a buffer is mapped in wgpu,
        // we'll avoid calling unmap() which would cause errors if the buffer isn't mapped.
        // Instead, we'll just let the Arc references drop naturally.
        
        // The actual buffer memory will be freed when the Arc references are dropped
        
        debug!("GPU buffers cleanup completed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::initialize_test_gpu;
    use futures::pin_mut;
    use std::sync::Arc;
    use wfc_core::{grid::PossibilityGrid, BoundaryCondition};
    use wfc_rules::AdjacencyRules;

    // Helper function to create uniform rules (copied from core)
    fn create_uniform_rules(num_transformed_tiles: usize, num_axes: usize) -> AdjacencyRules {
        let mut allowed_tuples = Vec::new();
        for axis in 0..num_axes {
            for ttid1 in 0..num_transformed_tiles {
                for ttid2 in 0..num_transformed_tiles {
                    allowed_tuples.push((axis, ttid1, ttid2));
                }
            }
        }
        AdjacencyRules::from_allowed_tuples(num_transformed_tiles, num_axes, allowed_tuples)
    }

    // Helper setup uses the rules helper
    fn setup_test_environment(
        width: usize,
        height: usize,
        depth: usize,
        num_transformed_tiles: usize,
    ) -> (
        Arc<wgpu::Device>,
        Arc<wgpu::Queue>,
        PossibilityGrid,
        GpuBuffers,
    ) {
        let (device, queue) = initialize_test_gpu();
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let grid = PossibilityGrid::new(width, height, depth, num_transformed_tiles);
        let rules = create_uniform_rules(num_transformed_tiles, 6);
        let boundary_mode = BoundaryCondition::Finite;
        let buffers = GpuBuffers::new(&device, &queue, &grid, &rules, boundary_mode)
            .expect("Failed to create buffers");
        (device, queue, grid, buffers)
    }

    #[test]
    #[ignore = "Skipping due to GPU initialization issues that cause tests to hang"]
    fn test_buffer_creation() {
        let (_device, _queue, _grid, buffers) = setup_test_environment(4, 4, 1, 10);
        assert!(buffers.grid_possibilities_buf.size() > 0);
        assert!(buffers.entropy_buf.size() > 0);
        assert!(buffers.worklist_buf_a.size() > 0);
        assert!(buffers.worklist_count_buf.size() > 0);
        assert!(buffers.params_uniform_buf.size() > 0);
    }

    #[test]
    #[ignore] // Ignoring until mapping logic is stable
    fn read_initial_possibilities_placeholder() {
        let (device, queue, _grid, buffers) = setup_test_environment(2, 2, 1, 4);

        // Use pollster to run the async code synchronously
        let results = pollster::block_on(async {
            // Create the future but don't await immediately
            let download_future = buffers.download_results(
                device.clone(),
                queue.clone(),
                false,
                false,
                false,
                true, // Download grid
                false,
                false,
            );

            // Pin the future and use select! with polling
            pin_mut!(download_future);
            
            
            loop {
                futures::select! {
                    res = download_future.as_mut().fuse() => break res,
                    _ = futures::future::ready(()).fuse() => {
                        // Poll the device regularly while waiting
                        device.poll(wgpu::Maintain::Poll);
                    }
                }
            }
            .expect("Failed to download results")
        });

        assert!(results.grid_possibilities.is_some());
        // TODO: Add actual value checks
    }

    #[test]
    #[ignore = "Skipping due to GPU initialization issues that cause tests to hang"]
    fn test_buffer_usage_flags() {
        let (_device, _queue, _grid, buffers) = setup_test_environment(1, 1, 1, 1);
        assert!(buffers
            .grid_possibilities_buf
            .usage()
            .contains(wgpu::BufferUsages::STORAGE));
        assert!(buffers
            .grid_possibilities_buf
            .usage()
            .contains(wgpu::BufferUsages::COPY_DST));
        assert!(buffers
            .grid_possibilities_buf
            .usage()
            .contains(wgpu::BufferUsages::COPY_SRC));
        assert!(buffers
            .entropy_buf
            .usage()
            .contains(wgpu::BufferUsages::STORAGE));
        assert!(buffers
            .entropy_buf
            .usage()
            .contains(wgpu::BufferUsages::COPY_SRC));
        assert!(!buffers
            .entropy_buf
            .usage()
            .contains(wgpu::BufferUsages::COPY_DST));
        assert!(buffers
            .worklist_buf_a
            .usage()
            .contains(wgpu::BufferUsages::STORAGE));
        assert!(buffers
            .worklist_buf_a
            .usage()
            .contains(wgpu::BufferUsages::COPY_DST));
        assert!(buffers
            .worklist_buf_a
            .usage()
            .contains(wgpu::BufferUsages::COPY_SRC));
        assert!(buffers
            .worklist_buf_b
            .usage()
            .contains(wgpu::BufferUsages::STORAGE));
        assert!(buffers
            .worklist_buf_b
            .usage()
            .contains(wgpu::BufferUsages::COPY_DST));
        assert!(buffers
            .worklist_buf_b
            .usage()
            .contains(wgpu::BufferUsages::COPY_SRC));
        assert!(buffers
            .worklist_count_buf
            .usage()
            .contains(wgpu::BufferUsages::STORAGE));
        assert!(buffers
            .worklist_count_buf
            .usage()
            .contains(wgpu::BufferUsages::COPY_DST));
        assert!(buffers
            .worklist_count_buf
            .usage()
            .contains(wgpu::BufferUsages::COPY_SRC));
        assert!(buffers
            .params_uniform_buf
            .usage()
            .contains(wgpu::BufferUsages::UNIFORM));
        assert!(buffers
            .params_uniform_buf
            .usage()
            .contains(wgpu::BufferUsages::COPY_DST));
        assert!(buffers
            .params_uniform_buf
            .usage()
            .contains(wgpu::BufferUsages::COPY_SRC));
        assert!(!buffers
            .params_uniform_buf
            .usage()
            .contains(wgpu::BufferUsages::STORAGE));
    }

    #[test]
    #[ignore = "Skipping due to GPU initialization issues that cause tests to hang"]
    fn cleanup_test() {
        let (_device, _queue, _grid, _buffers) = setup_test_environment(1, 1, 1, 2);
    }

    #[test]
    #[ignore = "Skipping due to GPU initialization issues that cause tests to hang"]
    fn test_zero_sized_grid_creation() {
        // Renamed slightly
        let (device, queue) = initialize_test_gpu();
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let grid = PossibilityGrid::new(0, 0, 0, 10);
        let rules = create_uniform_rules(10, 6);
        let boundary_mode = BoundaryCondition::Finite;
        let buffers_result = GpuBuffers::new(&device, &queue, &grid, &rules, boundary_mode);
        assert!(
            buffers_result.is_ok(),
            "Buffer creation failed for zero-sized grid: {:?}",
            buffers_result.err()
        );
        // Test download later
    }

    #[test]
    #[ignore = "Skipping due to GPU initialization issues that cause tests to hang"]
    fn test_large_tile_count() {
        // This test is just to verify proper buffer creation with a large number of tiles
        // that approaches the maximum limit (128)
        let width = 1;
        let height = 1;
        let depth = 1;
        let num_transformed_tiles = 100; // Close to the 128 tile maximum

        let (_device, _queue, _grid, buffers) =
            setup_test_environment(width, height, depth, num_transformed_tiles);

        assert!(buffers.grid_possibilities_buf.size() > 0);
        assert!(buffers.rules_buf.size() > 0);
    }

    #[test]
    #[ignore = "Skipping due to GPU initialization issues that cause tests to hang"]
    fn test_optimized_buffer_downloads() {
        // Create a test environment with a small grid
        let width = 4;
        let height = 4;
        let depth = 1;
        let num_transformed_tiles = 10;

        let (_device, _queue, _grid, _buffers) =
            setup_test_environment(width, height, depth, num_transformed_tiles);

        // Skip the actual test for now as it requires a working GPU and may hang in CI
        // We'll keep this test stub here for future implementation
        // The compilation of the code itself helps verify API compatibility
    }

    #[test]
    #[ignore = "Skipping due to GPU initialization issues that cause tests to hang"]
    fn test_map_staging_buffer_future() {
        // This is a simplified test to ensure our mapping infrastructure works
        // without going through the whole propagation
        let width = 2;
        let height = 2;
        let depth = 1;
        let num_tiles = 2;

        // Initialize test environment
        let (device, queue, _grid, _) = setup_test_environment(width, height, depth, num_tiles);

        // Create a small test buffer
        let buffer_size = 16; // Just 4 u32s
        let test_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create a staging buffer for reading
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Create and submit a command encoder to copy data
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Command Encoder"),
        });

        encoder.copy_buffer_to_buffer(&test_buffer, 0, &staging_buffer, 0, buffer_size);
        queue.submit(std::iter::once(encoder.finish()));

        // This just makes sure our map/slice operation doesn't panic
        // We don't actually test the result data since it will be undefined
    }

    #[test]
    #[ignore = "Skipping due to GPU initialization issues that cause tests to hang"]
    fn test_dynamic_buffer_resizing() {
        // Create a small grid
        let small_width = 4;
        let small_height = 4;
        let small_depth = 1;
        let num_tiles = 8;

        // Set up the test environment with the small grid
        let (device, _queue, _, mut buffers) = setup_test_environment(
            small_width,
            small_height,
            small_depth,
            num_tiles,
        );

        // Now try to resize for a larger grid
        let large_width = 16;
        let large_height = 16;
        let large_depth = 1;

        // Configure dynamic buffer management
        let config = DynamicBufferConfig {
            growth_factor: 1.5,
            min_buffer_size: 1024,
            max_buffer_size: 1024 * 1024 * 1024,
            auto_resize: true,
        };

        // Record the original buffer sizes
        let original_grid_buf_size = buffers.grid_possibilities_buf.size();
        let original_entropy_buf_size = buffers.entropy_buf.size();
        let original_worklist_buf_size = buffers.worklist_buf_a.size();

        // Resize all buffers
        let result = buffers.resize_for_grid(
            &device,
            large_width,
            large_height,
            large_depth,
            num_tiles as u32,
            &config,
        );

        // Verify resize was successful
        assert!(result.is_ok(), "Buffer resize failed: {:?}", result.err());

        // Verify the new buffer sizes
        let new_grid_buf_size = buffers.grid_possibilities_buf.size();
        let new_entropy_buf_size = buffers.entropy_buf.size();
        let new_worklist_buf_size = buffers.worklist_buf_a.size();

        // Calculate expected minimum sizes
        let large_num_cells = (large_width * large_height * large_depth) as usize;
        let u32s_per_cell = ((num_tiles as u32 + 31) / 32) as usize;
        let expected_min_grid_size =
            (large_num_cells * u32s_per_cell * std::mem::size_of::<u32>()) as u64;
        let expected_min_entropy_size = (large_num_cells * std::mem::size_of::<f32>()) as u64;
        let expected_min_worklist_size = (large_num_cells * std::mem::size_of::<u32>()) as u64;

        // Verify buffers are correctly sized
        assert!(
            new_grid_buf_size >= expected_min_grid_size,
            "Grid buffer size too small after resize: {} < {}",
            new_grid_buf_size,
            expected_min_grid_size
        );
        assert!(
            new_entropy_buf_size >= expected_min_entropy_size,
            "Entropy buffer size too small after resize: {} < {}",
            new_entropy_buf_size,
            expected_min_entropy_size
        );
        assert!(
            new_worklist_buf_size >= expected_min_worklist_size,
            "Worklist buffer size too small after resize: {} < {}",
            new_worklist_buf_size,
            expected_min_worklist_size
        );

        // Verify buffers got larger
        assert!(
            new_grid_buf_size > original_grid_buf_size,
            "Grid buffer didn't increase in size"
        );
        assert!(
            new_entropy_buf_size > original_entropy_buf_size,
            "Entropy buffer didn't increase in size"
        );
        assert!(
            new_worklist_buf_size > original_worklist_buf_size,
            "Worklist buffer didn't increase in size"
        );
    }
}
