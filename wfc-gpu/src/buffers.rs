//! Manages the creation, storage, and access of GPU buffers for the WFC algorithm.

// use crate::error_recovery::{AdaptiveTimeoutConfig, RecoverableGpuOp}; // Moved/Unused
use crate::{error_recovery::RecoverableGpuOp, GpuError};
// use futures::channel::oneshot; // Moved/Unused
// use futures::{self, FutureExt}; // Moved/Unused
// use log::{debug, error, info, warn}; // Use specific imports where needed or remove if unused
use bitvec::field::BitField;
use bytemuck::{Pod, Zeroable};
use log::{error, info};
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
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
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

/// Uniform buffer structure for entropy shader parameters.
/// Must match the `Params` struct in `entropy.wgsl`.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuEntropyShaderParams {
    pub grid_dims: [u32; 3], // width, height, depth
    pub heuristic_type: u32, // 0=Shannon, 1=Count, etc.
    pub num_tiles: u32,
    pub u32s_per_cell: u32,
    // Add padding if necessary to meet alignment rules (e.g., vec3 needs 16-byte alignment)
    pub _padding1: u32,
    pub _padding2: u32,
}

/// DynamicBufferConfig contains settings for how buffers are resized
#[derive(Debug, Clone, Copy, PartialEq)]
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
            growth_factor: 1.5,                  // Grow by 50% when resizing
            min_buffer_size: 1024,               // Minimum 1KB buffer size
            max_buffer_size: 1024 * 1024 * 1024, // Maximum 1GB buffer size
            auto_resize: true,                   // Automatically resize by default
        }
    }
}

/// Stores all the GPU buffers needed for the WFC algorithm.
///
/// This struct maintains the state of all WGPU buffers required for the Wave Function Collapse
/// algorithm's GPU acceleration, including grid data, entropy calculations, worklists for propagation,
/// and various auxiliary buffers for things like contradiction detection.
#[derive(Debug, Clone)]
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

/// Specifies which data to download from the GPU.
pub struct DownloadRequest {
    pub download_entropy: bool,
    pub download_min_entropy_info: bool,
    pub download_grid_possibilities: bool,
    // pub download_worklist: bool, // Not used currently
    pub download_worklist_size: bool,
    pub download_contradiction_location: bool,
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
            heuristic_type: 0,                 // Default to simple count
            tie_breaking: 0,                   // Default to none
            max_propagation_steps: 1000,       // Default maximum steps
            contradiction_check_frequency: 10, // Default check frequency
            worklist_size: 0,                  // Initial worklist size is 0
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
            },
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
            size: std::mem::size_of::<GpuEntropyShaderParams>() as wgpu::BufferAddress,
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
        log::debug!(
            "Resizing buffer from {} to {} bytes",
            old_size,
            actual_new_size
        );
        Self::create_buffer(device, actual_new_size, usage, label)
    }

    /// Check if a buffer is sufficient for the required size
    pub fn is_buffer_sufficient(buffer: &Arc<wgpu::Buffer>, required_size: u64) -> bool {
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
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
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
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
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
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
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
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
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
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
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

    /// Convenience method to call upload_initial_updates_with_auto_resize.
    /// This wrapper keeps the original call signature used elsewhere for now.
    pub fn upload_initial_updates_wrapper(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        updates: &[u32],
        active_worklist_idx: usize, // Keep usize to match callers
    ) -> Result<(), String> {
        // Renamed call to the correct function
        self.upload_initial_updates_with_auto_resize(
            device,
            queue,
            updates,
            active_worklist_idx as u32, // Convert usize to u32 here
        )
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
        request: DownloadRequest,
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
                let mut download_data = GpuDownloadResults::default();

                // --- Queue copy commands ---
                if request.download_entropy {
                    encoder.copy_buffer_to_buffer(
                        &self.entropy_buf,
                        0,
                        &self.staging_entropy_buf,
                        0,
                        self.entropy_buf.size(),
                    );
                }
                if request.download_min_entropy_info {
                    encoder.copy_buffer_to_buffer(
                        &self.min_entropy_info_buf,
                        0,
                        &self.staging_min_entropy_info_buf,
                        0,
                        self.min_entropy_info_buf.size(),
                    );
                }
                if request.download_grid_possibilities {
                    encoder.copy_buffer_to_buffer(
                        &self.grid_possibilities_buf,
                        0,
                        &self.staging_grid_possibilities_buf,
                        0,
                        self.grid_possibilities_buf.size(),
                    );
                }
                if request.download_worklist_size {
                    encoder.copy_buffer_to_buffer(
                        &self.worklist_count_buf,
                        0,
                        &self.staging_worklist_count_buf,
                        0,
                        self.worklist_count_buf.size(),
                    );
                }
                // Assuming contradiction flag is needed if location is
                if request.download_contradiction_location {
                    encoder.copy_buffer_to_buffer(
                        &self.contradiction_flag_buf,
                        0,
                        &self.staging_contradiction_flag_buf,
                        0,
                        self.contradiction_flag_buf.size(),
                    );
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
                // info!("GPU copy commands submitted for download."); // Use info! if needed

                // --- Map staging buffers and process results ---
                use futures::future::join_all;
                use futures::FutureExt;
                use tokio::sync::oneshot;

                // Configure timeout (simplified example, enhance if needed)
                let operation_timeout = std::time::Duration::from_secs(10);

                let mut futures = Vec::new();

                // Helper closure to map buffer and process result
                async fn map_and_process<T: Pod + Send + 'static>(
                    buffer: Arc<wgpu::Buffer>,
                    timeout: std::time::Duration,
                    expected_items: Option<usize>, // None means single value
                ) -> Result<Box<dyn std::any::Any + Send>, GpuError> {
                    let buffer_slice = buffer.slice(..);
                    let (tx, rx) = oneshot::channel();
                    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                        let _ = tx.send(result);
                    });

                    match tokio::time::timeout(timeout, rx).await {
                        Ok(channel_result) => {
                            channel_result.map_err(|_| {
                                GpuError::BufferOperationError(
                                    "Mapping channel canceled".to_string(),
                                )
                            })??;
                            let mapped_range = buffer_slice.get_mapped_range();
                            let bytes = mapped_range.as_ref();
                            let result_data = bytemuck::cast_slice::<u8, T>(bytes).to_vec();
                            drop(mapped_range); // Important: drop before unmap
                            buffer.unmap();

                            if let Some(expected) = expected_items {
                                if result_data.len() >= expected {
                                    Ok(Box::new(result_data[..expected].to_vec())
                                        as Box<dyn std::any::Any + Send>)
                                } else {
                                    Err(GpuError::BufferSizeMismatch(format!(
                                        "Expected {} items, got {}",
                                        expected,
                                        result_data.len()
                                    )))
                                }
                            } else if !result_data.is_empty() {
                                Ok(Box::new(result_data[0]) as Box<dyn std::any::Any + Send>)
                            } else {
                                Err(GpuError::BufferSizeMismatch(
                                    "Expected single value, got empty".to_string(),
                                ))
                            }
                        }
                        Err(_) => Err(GpuError::BufferOperationError(format!(
                            "Buffer mapping timed out after {:?}",
                            timeout
                        ))),
                    }
                }

                // --- Create futures for requested downloads ---
                if request.download_grid_possibilities {
                    let buffer = self.staging_grid_possibilities_buf.clone();
                    let expected_size = self.u32s_per_cell * self.num_cells;
                    futures.push(
                        map_and_process::<u32>(buffer, operation_timeout, Some(expected_size))
                            .map(|r| r.map(|d| ("grid_possibilities", d)))
                            .boxed(),
                    );
                }
                if request.download_entropy {
                    let buffer = self.staging_entropy_buf.clone();
                    let expected_size = self.num_cells;
                    futures.push(
                        map_and_process::<f32>(buffer, operation_timeout, Some(expected_size))
                            .map(|r| r.map(|d| ("entropy", d)))
                            .boxed(),
                    );
                }
                if request.download_min_entropy_info {
                    let buffer = self.staging_min_entropy_info_buf.clone();
                    futures.push(
                        map_and_process::<u32>(buffer, operation_timeout, Some(2))
                            .map(|r| r.map(|d| ("min_entropy_info", d)))
                            .boxed(),
                    ); // Expect [f32_bits, u32_idx]
                }
                if request.download_worklist_size {
                    let buffer = self.staging_worklist_count_buf.clone();
                    futures.push(
                        map_and_process::<u32>(buffer, operation_timeout, None)
                            .map(|r| r.map(|d| ("worklist_count", d)))
                            .boxed(),
                    );
                }
                if request.download_contradiction_location {
                    let flag_buffer = self.staging_contradiction_flag_buf.clone();
                    futures.push(
                        map_and_process::<u32>(flag_buffer, operation_timeout, None)
                            .map(|r| r.map(|d| ("contradiction_flag", d)))
                            .boxed(),
                    );
                    let loc_buffer = self.staging_contradiction_location_buf.clone();
                    futures.push(
                        map_and_process::<u32>(loc_buffer, operation_timeout, None)
                            .map(|r| r.map(|d| ("contradiction_location", d)))
                            .boxed(),
                    );
                }

                // --- Process results ---
                let download_results_vec = join_all(futures).await;
                for res in download_results_vec {
                    match res {
                        Ok((name, data_box)) => match name {
                            "grid_possibilities" => {
                                download_data.grid_possibilities =
                                    data_box.downcast::<Vec<u32>>().ok().map(|b| *b)
                            }
                            "entropy" => {
                                download_data.entropy =
                                    data_box.downcast::<Vec<f32>>().ok().map(|b| *b)
                            }
                            "min_entropy_info" => {
                                if let Some(d) = data_box.downcast::<Vec<u32>>().ok() {
                                    if d.len() >= 2 {
                                        download_data.min_entropy_info =
                                            Some((f32::from_bits(d[0]), d[1]));
                                    }
                                }
                            }
                            "worklist_count" => {
                                download_data.worklist_count =
                                    data_box.downcast::<u32>().ok().map(|b| *b)
                            }
                            "contradiction_flag" => {
                                download_data.contradiction_flag =
                                    data_box.downcast::<u32>().ok().map(|b| *b != 0)
                            }
                            "contradiction_location" => {
                                download_data.contradiction_location =
                                    data_box.downcast::<u32>().ok().map(|b| *b)
                            }
                            _ => {}
                        },
                        Err(e) => return Err(e), // Propagate the first error
                    }
                }

                Ok(download_data)
            })
            .await
    }

    // /// Downloads the propagation status ... // Keep this commented or remove if not needed
}
