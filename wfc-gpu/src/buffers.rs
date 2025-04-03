use crate::GpuError;
use bitvec::field::BitField;
use bytemuck::{Pod, Zeroable};
use log::{debug, error, info, warn};
use std::num::NonZeroU64;
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
/// Marked `#[repr(C)]` for stable memory layout across Rust/WGSL.
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
    /// Current size of the input worklist (number of updated cells) for the propagation shader.
    pub worklist_size: u32,
    /// Boundary mode: 0 for Clamped, 1 for Periodic.
    pub boundary_mode: u32,
    /// Padding to ensure struct size is a multiple of 16 bytes.
    pub _padding1: u32, // Ensure alignment to 16 bytes
}

/// Manages the collection of WGPU buffers required for GPU-accelerated WFC.
///
/// This struct encapsulates all GPU memory allocation and management for the algorithm,
/// including buffers for:
/// - Storing the possibility state of the grid (`grid_possibilities_buf`).
/// - Storing the adjacency rules (`rules_buf`).
/// - Storing calculated entropy values (`entropy_buf`).
/// - Holding shader parameters (`params_uniform_buf`).
/// - Managing worklists for propagation (`updates_buf`, `output_worklist_buf`, `output_worklist_count_buf`).
/// - Communicating results and status flags (e.g., `contradiction_flag_buf`, `min_entropy_info_buf`).
///
/// It also includes corresponding staging buffers (prefixed `staging_`) used for efficiently
/// transferring data between the CPU and GPU, particularly for downloading results.
#[derive(Debug)] // Add Debug derive
#[allow(dead_code)] // Allow unused fields/methods during development
#[derive(Clone)] // Derive Clone
pub struct GpuBuffers {
    /// **GPU Buffer**: Stores the possibility bitvector for each grid cell.
    /// Each cell's possibilities are packed into `num_tiles_u32` elements.
    /// Usage: `STORAGE | COPY_DST | COPY_SRC`
    pub grid_possibilities_buf: Arc<wgpu::Buffer>,
    /// **Staging Buffer**: Used for downloading the final grid possibilities state to the CPU.
    /// Usage: `MAP_READ | COPY_DST`
    staging_grid_possibilities_buf: Arc<wgpu::Buffer>,
    /// **GPU Buffer**: Stores the flattened adjacency rules, packed into u32s.
    /// Read-only by shaders.
    /// Usage: `STORAGE`
    pub rules_buf: Arc<wgpu::Buffer>,
    /// **GPU Buffer**: Stores the calculated entropy value (f32) for each grid cell.
    /// Written to by the entropy shader, read back to CPU.
    /// Usage: `STORAGE | COPY_SRC`
    pub entropy_buf: Arc<wgpu::Buffer>,
    /// **Staging Buffer**: Used for downloading the entropy grid to the CPU.
    /// Usage: `MAP_READ | COPY_DST`
    staging_entropy_buf: Arc<wgpu::Buffer>,
    /// **GPU Buffer**: Stores the minimum positive entropy found and its index.
    /// Layout: `[f32_entropy_bits: u32, flat_index: u32]`.
    /// Written atomically by the entropy shader.
    /// Usage: `STORAGE | COPY_DST | COPY_SRC`
    pub min_entropy_info_buf: Arc<wgpu::Buffer>,
    /// **Staging Buffer**: Used for downloading the minimum entropy info.
    /// Usage: `MAP_READ | COPY_DST`
    staging_min_entropy_info_buf: Arc<wgpu::Buffer>,
    /// **GPU Buffer A**: Worklist buffer for propagation (ping-pong A).
    /// Stores flat indices of updated cells. Usage depends on iteration.
    /// Usage: `STORAGE | COPY_DST | COPY_SRC`
    pub worklist_buf_a: Arc<wgpu::Buffer>,
    /// **GPU Buffer B**: Worklist buffer for propagation (ping-pong B).
    /// Stores flat indices of updated cells. Usage depends on iteration.
    /// Usage: `STORAGE | COPY_DST | COPY_SRC`
    pub worklist_buf_b: Arc<wgpu::Buffer>,
    /// **GPU Buffer**: Atomic counter for the *output* worklist size in the current propagation step.
    /// Reset before each step, read after to know size for next step.
    /// Usage: `STORAGE | COPY_DST | COPY_SRC`
    pub worklist_count_buf: Arc<wgpu::Buffer>,
    /// **GPU Buffer**: Flag (u32) set by the propagation shader if a contradiction is detected.
    /// 0 = no contradiction, 1 = contradiction.
    /// Usage: `STORAGE | COPY_DST | COPY_SRC`
    pub contradiction_flag_buf: Arc<wgpu::Buffer>,
    /// **GPU Buffer**: Uniform buffer holding `GpuParamsUniform`.
    /// Usage: `UNIFORM | COPY_DST | COPY_SRC`
    pub params_uniform_buf: Arc<wgpu::Buffer>,
    /// **Staging Buffer**: Used for downloading the contradiction flag.
    /// Usage: `MAP_READ | COPY_DST`
    staging_contradiction_flag_buf: Arc<wgpu::Buffer>,
    /// **GPU Buffer**: Stores the flat index of the first cell where a contradiction was detected.
    /// Written atomically by the propagation shader.
    /// Usage: `STORAGE | COPY_DST | COPY_SRC`
    pub contradiction_location_buf: Arc<wgpu::Buffer>,
    /// **Staging Buffer**: Used for downloading the contradiction location index.
    /// Usage: `MAP_READ | COPY_DST`
    staging_contradiction_location_buf: Arc<wgpu::Buffer>,
    /// Staging buffers (used for CPU <-> GPU data transfer)
    staging_params_buf: Arc<wgpu::Buffer>,
    staging_updates_buf: Arc<wgpu::Buffer>,
    staging_entropy_buf: Arc<wgpu::Buffer>,
    staging_min_entropy_info_buf: Arc<wgpu::Buffer>,
    staging_contradiction_flag_buf: Arc<wgpu::Buffer>,
    staging_contradiction_location_buf: Arc<wgpu::Buffer>,
    staging_grid_possibilities_buf: Arc<wgpu::Buffer>,
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
        queue: &wgpu::Queue,
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
        // Iterate through all combinations (axis, ttid1, ttid2) and check if allowed
        let num_rules_total = num_axes * num_tiles * num_tiles;
        let u32s_for_rules = (num_rules_total + 31) / 32; // ceil(num_rules_total / 32)
        let mut packed_rules = vec![0u32; u32s_for_rules];

        let mut rule_idx = 0;
        for axis in 0..num_axes {
            for ttid1 in 0..num_tiles {
                for ttid2 in 0..num_tiles {
                    if rules.check(ttid1, ttid2, axis) {
                        let u32_idx = rule_idx / 32;
                        let bit_idx = rule_idx % 32;
                        if u32_idx < packed_rules.len() {
                            packed_rules[u32_idx] |= 1 << bit_idx;
                        }
                    }
                    rule_idx += 1;
                }
            }
        }

        // Optional sanity check (should always pass if loops are correct)
        if rule_idx != num_rules_total {
            warn!(
                "Rule packing index mismatch: iterated {} rules, expected {}",
                rule_idx, num_rules_total
            );
        }

        let _rules_buffer_size = (packed_rules.len() * std::mem::size_of::<u32>()) as u64;

        // --- Create Uniform Buffer Data ---
        let params = GpuParamsUniform {
            grid_width: width as u32,
            grid_height: height as u32,
            grid_depth: depth as u32,
            num_tiles: num_tiles as u32,
            num_axes: num_axes as u32,
            worklist_size: 0, // Initial worklist size is 0
            boundary_mode: match boundary_mode {
                BoundaryCondition::Finite => 0,
                BoundaryCondition::Periodic => 1,
            },
            _padding1: 0,
        };
        let _params_buffer_size = std::mem::size_of::<GpuParamsUniform>() as u64;

        // --- Calculate Other Buffer Sizes ---
        let entropy_buffer_size = (num_cells * std::mem::size_of::<f32>()) as u64;
        // Input worklist (updates) can contain up to num_cells indices
        let updates_buffer_size = (num_cells * std::mem::size_of::<u32>()) as u64;
        // Output worklist can also contain up to num_cells indices
        let worklist_count_buffer_size = std::mem::size_of::<u32>() as u64;
        let contradiction_buffer_size = std::mem::size_of::<u32>() as u64;
        let min_entropy_info_buffer_size = (2 * std::mem::size_of::<u32>()) as u64; // Size for [f32_bits, u32_index]
        let contradiction_location_buffer_size = std::mem::size_of::<u32>() as u64;

        // --- Create Buffers --- (Wrap in Arc)
        let grid_possibilities_buf = Arc::new(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Grid Possibilities"),
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

        let worklist_count_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Worklist Count Buffer"),
            size: worklist_count_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST // For reset
                | wgpu::BufferUsages::COPY_SRC, // For download
            mapped_at_creation: false,
        }));

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
            params_uniform_buf,
            staging_params_buf: Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Params Buffer"),
                size: std::mem::size_of::<GpuParamsUniform>() as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            staging_updates_buf: Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Updates Buffer"),
                size: updates_buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
        })
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
        debug!("Resetting min entropy info buffer on GPU.");
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
        debug!("Resetting contradiction flag buffer on GPU.");
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
        debug!("Resetting worklist count buffer on GPU.");
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
        debug!(
            "Updating params uniform buffer worklist_size to {} on GPU.",
            worklist_size
        );
        // Calculate the offset of the worklist_size field within the GpuParamsUniform struct.
        // WARNING: This assumes the layout defined in GpuParamsUniform.
        // If the struct changes, this offset needs to be updated.
        // grid_width, grid_height, grid_depth, num_tiles, num_axes are all u32.
        let offset = (5 * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
        queue.write_buffer(
            &self.params_uniform_buf,
            offset,
            bytemuck::cast_slice(&[worklist_size]),
        );
        Ok(())
    }

    /// Downloads multiple results from GPU buffers using a single command submission.
    /// Returns a future that resolves to a `GpuDownloadResults` struct.
    ///
    /// NOTE: This implementation still uses `map_async` + `device.poll`, which might block
    /// if not used within a compatible async runtime or if `Maintain::Wait` is used.
    /// True non-blocking behavior requires integration with an executor.
    pub async fn download_results(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        // Add parameters to specify *which* results to download, e.g.:
        download_entropy: bool,
        download_min_entropy: bool,
        download_contradiction: bool,
        download_possibilities: bool,
        num_tiles: u32, // Needed only if downloading possibilities
    ) -> Result<GpuDownloadResults, GpuError> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Download Encoder"),
        });

        // Use a temporary struct to hold results within the async blocks
        // We need Arc<Mutex<..>> because multiple async blocks might mutate it concurrently
        // If awaiting sequentially, this isn't strictly needed, but it's safer for future parallel awaits.
        let results = Arc::new(tokio::sync::Mutex::new(GpuDownloadResults::default()));

        // Collect futures for mapping operations
        let mut map_futures = Vec::new();

        // --- Entropy Download --- (If requested)
        if download_entropy {
            let buffer_size = self.entropy_buf.size();
            let staging_buffer = Arc::clone(&self.staging_entropy_buf);
            encoder.copy_buffer_to_buffer(&self.entropy_buf, 0, &staging_buffer, 0, buffer_size);

            let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
            let buffer_slice = staging_buffer.slice(..);
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).expect("Receiver dropped for entropy map");
            });

            let results_clone = Arc::clone(&results);
            map_futures.push(async move {
                match rx.receive().await {
                    Some(Ok(())) => {
                        let data = buffer_slice.get_mapped_range();
                        let entropy_vec: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
                        let mut results_guard = results_clone.lock().await;
                        results_guard.entropy = Some(entropy_vec);
                        drop(data); // Unmap buffer
                        staging_buffer.unmap();
                        Ok(())
                    }
                    Some(Err(e)) => Err(GpuError::BufferMapError(e)),
                    None => Err(GpuError::InternalError(
                        "Entropy map channel closed unexpectedly".to_string(),
                    )),
                }
            });
        }

        // --- Min Entropy Info Download --- (If requested)
        if download_min_entropy {
            let buffer_size = self.min_entropy_info_buf.size();
            let staging_buffer = Arc::clone(&self.staging_min_entropy_info_buf);
            encoder.copy_buffer_to_buffer(
                &self.min_entropy_info_buf,
                0,
                &staging_buffer,
                0,
                buffer_size,
            );

            let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
            let buffer_slice = staging_buffer.slice(..);
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result)
                    .expect("Receiver dropped for min_entropy map");
            });

            let results_clone = Arc::clone(&results);
            map_futures.push(async move {
                match rx.receive().await {
                    Some(Ok(())) => {
                        let data = buffer_slice.get_mapped_range();
                        let result_raw: &[u32] = bytemuck::cast_slice(&data);
                        if result_raw.len() >= 2 {
                            let entropy_bits = result_raw[0];
                            let index = result_raw[1];
                            let entropy = f32::from_bits(entropy_bits);
                            let mut results_guard = results_clone.lock().await;
                            results_guard.min_entropy_info = Some((entropy, index));
                        } else {
                            error!(
                                "Min entropy buffer size mismatch on download. Len: {}",
                                result_raw.len()
                            );
                            // Don't set the result, let Option stay None
                        }
                        drop(data);
                        staging_buffer.unmap();
                        Ok(()) // Even if size mismatch, mapping itself succeeded
                    }
                    Some(Err(e)) => Err(GpuError::BufferMapError(e)),
                    None => Err(GpuError::InternalError(
                        "Min entropy map channel closed unexpectedly".to_string(),
                    )),
                }
            });
        }

        // --- Contradiction Flag & Location Download --- (If requested)
        if download_contradiction {
            // Flag
            let buffer_size_flag = self.contradiction_flag_buf.size();
            let staging_buffer_flag = Arc::clone(&self.staging_contradiction_flag_buf);
            encoder.copy_buffer_to_buffer(
                &self.contradiction_flag_buf,
                0,
                &staging_buffer_flag,
                0,
                buffer_size_flag,
            );

            let (tx_flag, rx_flag) = futures_intrusive::channel::shared::oneshot_channel();
            let buffer_slice_flag = staging_buffer_flag.slice(..);
            buffer_slice_flag.map_async(wgpu::MapMode::Read, move |result| {
                tx_flag
                    .send(result)
                    .expect("Receiver dropped for contradiction_flag map");
            });

            let results_clone_flag = Arc::clone(&results);
            map_futures.push(async move {
                match rx_flag.receive().await {
                    Some(Ok(())) => {
                        let data = buffer_slice_flag.get_mapped_range();
                        let result_raw: &[u32] = bytemuck::cast_slice(&data);
                        let flag = result_raw.first().copied().unwrap_or(0) != 0;
                        let mut results_guard = results_clone_flag.lock().await;
                        results_guard.contradiction_flag = Some(flag);
                        drop(data);
                        staging_buffer_flag.unmap();
                        Ok(())
                    }
                    Some(Err(e)) => Err(GpuError::BufferMapError(e)),
                    None => Err(GpuError::InternalError(
                        "Contradiction flag map channel closed unexpectedly".to_string(),
                    )),
                }
            });

            // Location
            let buffer_size_loc = self.contradiction_location_buf.size();
            let staging_buffer_loc = Arc::clone(&self.staging_contradiction_location_buf);
            encoder.copy_buffer_to_buffer(
                &self.contradiction_location_buf,
                0,
                &staging_buffer_loc,
                0,
                buffer_size_loc,
            );

            let (tx_loc, rx_loc) = futures_intrusive::channel::shared::oneshot_channel();
            let buffer_slice_loc = staging_buffer_loc.slice(..);
            buffer_slice_loc.map_async(wgpu::MapMode::Read, move |result| {
                tx_loc
                    .send(result)
                    .expect("Receiver dropped for contradiction_location map");
            });

            let results_clone_loc = Arc::clone(&results);
            map_futures.push(async move {
                match rx_loc.receive().await {
                    Some(Ok(())) => {
                        let data = buffer_slice_loc.get_mapped_range();
                        let result_raw: &[u32] = bytemuck::cast_slice(&data);
                        // Use MAX as sentinel for "no specific location found" or initial state
                        let loc = result_raw.first().copied().unwrap_or(u32::MAX);
                        let mut results_guard = results_clone_loc.lock().await;
                        results_guard.contradiction_location = Some(loc);
                        drop(data);
                        staging_buffer_loc.unmap();
                        Ok(())
                    }
                    Some(Err(e)) => Err(GpuError::BufferMapError(e)),
                    None => Err(GpuError::InternalError(
                        "Contradiction location map channel closed unexpectedly".to_string(),
                    )),
                }
            });
        }

        // --- Possibilities Download --- (If requested)
        if download_possibilities {
            // Size depends on num_tiles and possibilities per tile (u32)
            let possibilities_per_tile =
                self.grid_possibilities_buf.size() / self.params_uniform_buf.num_tiles as u64;
            let buffer_size = possibilities_per_tile * num_tiles as u64;

            if buffer_size > self.staging_grid_possibilities_buf.size() {
                error!(
                    "Calculated possibilities download size ({}) exceeds staging buffer size ({})",
                    buffer_size,
                    self.staging_grid_possibilities_buf.size()
                );
                return Err(GpuError::BufferOperationError(format!(
                    "Possibilities staging buffer too small. Required: {}, Available: {}",
                    buffer_size,
                    self.staging_grid_possibilities_buf.size()
                )));
            } else if buffer_size == 0 {
                warn!("Possibilities download requested but calculated size is 0.");
                // Set result to empty vec? or None? Let's go with None.
                let mut results_guard = results.lock().await;
                results_guard.grid_possibilities = None;
            } else {
                let staging_buffer = Arc::clone(&self.staging_grid_possibilities_buf);
                encoder.copy_buffer_to_buffer(
                    &self.grid_possibilities_buf,
                    0,
                    &staging_buffer,
                    0,
                    buffer_size,
                );

                let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
                // Slice the staging buffer to the expected size
                let buffer_slice = staging_buffer.slice(..buffer_size);
                buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                    tx.send(result)
                        .expect("Receiver dropped for grid_possibilities map");
                });

                let results_clone = Arc::clone(&results);
                map_futures.push(async move {
                    match rx.receive().await {
                        Some(Ok(())) => {
                            let data = buffer_slice.get_mapped_range();
                            let possibilities_vec: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
                            let mut results_guard = results_clone.lock().await;
                            results_guard.grid_possibilities = Some(possibilities_vec);
                            drop(data);
                            staging_buffer.unmap();
                            Ok(())
                        }
                        Some(Err(e)) => Err(GpuError::BufferMapError(e)),
                        None => Err(GpuError::InternalError(
                            "Possibilities map channel closed unexpectedly".to_string(),
                        )),
                    }
                });
            }
        }

        // --- Submit and Poll --- (Still uses polling, but submits once)
        queue.submit(Some(encoder.finish()));
        debug!("Submitted batched download commands. Polling device...");
        device.poll(wgpu::Maintain::Wait); // Block until queue is empty
        debug!("Device polling complete.");

        // Wait for all map_async callbacks to complete
        debug!("Awaiting {} map futures...", map_futures.len());
        // Use try_join_all to run map callbacks concurrently (if runtime supports it)
        // and fail early if any map operation fails.
        futures::future::try_join_all(map_futures).await?;
        debug!("All map futures completed.");

        // Extract the results from the Mutex
        // Use Arc::try_unwrap if this is the only strong reference left, otherwise clone.
        let final_results = Arc::try_unwrap(results)
            .map_err(|_| GpuError::InternalError("Failed to unwrap Arc for results".to_string()))?
            .into_inner();

        Ok(final_results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::setup_wgpu;
    use crate::BoundaryCondition; // Make sure this is imported
    use wfc_core::graph::graph_from_rules;
    use wfc_core::grid::{Direction, Grid, PossibilityGrid}; // Import PossibilityGrid
    use wfc_core::rules::AdjacencyRules;

    // Helper function to create a simple grid and rules for testing
    fn setup_test_environment() -> (Grid, AdjacencyRules, Arc<wgpu::Device>, Arc<wgpu::Queue>) {
        let (device, queue) = setup_wgpu();
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Simple 2x2 grid, 2 tiles (0 and 1)
        let width = 2;
        let height = 2;
        let num_tiles = 2;
        let grid = Grid::new(width, height, num_tiles);

        // Simple rules: tile 0 can be adjacent to 0, tile 1 can be adjacent to 1
        let mut rules = AdjacencyRules::new(num_tiles);
        rules.allow(0, 0, Direction::Up);
        rules.allow(0, 0, Direction::Down);
        rules.allow(0, 0, Direction::Left);
        rules.allow(0, 0, Direction::Right);
        rules.allow(1, 1, Direction::Up);
        rules.allow(1, 1, Direction::Down);
        rules.allow(1, 1, Direction::Left);
        rules.allow(1, 1, Direction::Right);

        (grid, rules, device, queue)
    }

    // ... other tests: test_buffer_creation, test_initial_state, etc. ...

    #[test]
    fn test_download_buffers() {
        let (grid, rules, device, queue) = setup_test_environment();
        let buffers =
            GpuBuffers::new(&device, &queue, &grid, &rules, BoundaryCondition::Finite).unwrap();

        // --- Pre-populate GPU buffers with known data for testing downloads ---
        // Example: Set entropy, min_entropy, contradiction, possibilities
        let initial_entropy = vec![1.0f32; grid.num_cells()]; // Example entropy
        let initial_min_entropy = [f32::to_bits(0.5f32), 1u32]; // Example: entropy 0.5 at index 1
        let initial_contradiction_flag = [1u32]; // Example: contradiction detected
        let initial_contradiction_loc = [2u32]; // Example: contradiction at index 2
                                                // Example: Allow only tile 0 everywhere (represented by bitmask 1)
        let initial_possibilities = vec![1u32; grid.num_tiles() as usize];

        queue.write_buffer(
            &buffers.entropy_buf,
            0,
            bytemuck::cast_slice(&initial_entropy),
        );
        queue.write_buffer(
            &buffers.min_entropy_info_buf,
            0,
            bytemuck::cast_slice(&initial_min_entropy),
        );
        queue.write_buffer(
            &buffers.contradiction_flag_buf,
            0,
            bytemuck::cast_slice(&initial_contradiction_flag),
        );
        queue.write_buffer(
            &buffers.contradiction_location_buf,
            0,
            bytemuck::cast_slice(&initial_contradiction_loc),
        );
        queue.write_buffer(
            &buffers.grid_possibilities_buf,
            0,
            bytemuck::cast_slice(&initial_possibilities),
        );
        // Ensure writes are submitted before download
        device.poll(wgpu::Maintain::Wait);

        // --- Perform Batched Download ---
        let num_tiles = grid.num_tiles(); // Get actual number of tiles

        let results = pollster::block_on(buffers.download_results(
            &device, &queue, true,      // download_entropy
            true,      // download_min_entropy
            true,      // download_contradiction
            true,      // download_possibilities
            num_tiles, // Pass num_tiles
        ));

        // --- Assertions ---
        assert!(
            results.is_ok(),
            "Batched download failed: {:?}",
            results.err()
        );
        let downloaded_data = results.unwrap();

        // Assert that requested data was downloaded
        assert!(
            downloaded_data.entropy.is_some(),
            "Entropy was not downloaded"
        );
        assert!(
            downloaded_data.min_entropy_info.is_some(),
            "Min entropy info was not downloaded"
        );
        assert!(
            downloaded_data.contradiction_flag.is_some(),
            "Contradiction flag was not downloaded"
        );
        assert!(
            downloaded_data.contradiction_location.is_some(),
            "Contradiction location was not downloaded"
        );
        assert!(
            downloaded_data.grid_possibilities.is_some(),
            "Grid possibilities were not downloaded"
        );

        // Assert the content of the downloaded data matches the initial values
        if let Some(entropy) = downloaded_data.entropy {
            assert_eq!(
                entropy.len(),
                grid.num_cells(),
                "Downloaded entropy length mismatch"
            );
            assert!(
                entropy.iter().all(|&x| (x - 1.0).abs() < f32::EPSILON),
                "Downloaded entropy content mismatch"
            );
        }
        if let Some((min_entropy, min_index)) = downloaded_data.min_entropy_info {
            assert!(
                (min_entropy - 0.5).abs() < f32::EPSILON,
                "Downloaded min entropy value mismatch"
            );
            assert_eq!(min_index, 1, "Downloaded min entropy index mismatch");
        }
        if let Some(flag) = downloaded_data.contradiction_flag {
            assert_eq!(flag, true, "Downloaded contradiction flag mismatch");
        }
        if let Some(loc) = downloaded_data.contradiction_location {
            assert_eq!(loc, 2, "Downloaded contradiction location mismatch");
        }
        if let Some(possibilities) = downloaded_data.grid_possibilities {
            assert_eq!(
                possibilities.len(),
                grid.num_tiles() as usize,
                "Downloaded possibilities length mismatch"
            );
            assert!(
                possibilities.iter().all(|&x| x == 1u32),
                "Downloaded possibilities content mismatch"
            );
        }
    }
}
