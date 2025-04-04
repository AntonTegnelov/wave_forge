use crate::GpuError;
use bitvec::field::BitField;
use bytemuck::{Pod, Zeroable};
use futures::channel::oneshot;
use futures::pin_mut;
use futures::{self, FutureExt};
use log::{debug, error, info, warn};
use std::sync::Arc;
use std::time::Duration;
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
    staging_worklist_count_buf: Arc<wgpu::Buffer>,
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
        let updates_buffer_size = (num_cells * std::mem::size_of::<u32>()) as u64;
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

    /// Creates a future that maps the staging buffer and returns the raw bytes.
    /// Note: This function should be static but needs access to buffer mapping futures.
    async fn map_staging_buffer_to_vec(buffer: Arc<wgpu::Buffer>) -> Result<Vec<u8>, GpuError> {
        let (sender, receiver) = oneshot::channel::<Result<(), GpuError>>();
        let buffer_slice = buffer.slice(..);
        // Ensure map_async is called correctly
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result.map_err(|e| GpuError::BufferMapFailed(e)));
        });
        // Polling happens in the caller (download_results)
        let map_result = receiver
            .await
            .map_err(|_| GpuError::BufferOperationError("Mapping channel closed".to_string()))?;
        map_result?; // Propagate mapping error
        let data = {
            let view = buffer_slice.get_mapped_range();
            view.to_vec()
        };
        buffer.unmap();
        Ok(data)
    }

    /// Downloads requested results from GPU buffers to the CPU.
    pub async fn download_results(
        &self,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        download_entropy: bool,
        download_min_entropy: bool,
        download_contradiction_flag: bool,
        download_grid: bool,
        download_worklist_count: bool,
        download_contradiction_location: bool,
    ) -> Result<GpuDownloadResults, GpuError> {
        let mut results = GpuDownloadResults::default();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Download Results Encoder"),
        });

        // Queue copy commands (Unchanged)
        if download_entropy {
            encoder.copy_buffer_to_buffer(
                &self.entropy_buf,
                0,
                &self.staging_entropy_buf,
                0,
                self.entropy_buf.size(),
            );
        }
        if download_min_entropy {
            encoder.copy_buffer_to_buffer(
                &self.min_entropy_info_buf,
                0,
                &self.staging_min_entropy_info_buf,
                0,
                self.min_entropy_info_buf.size(),
            );
        }
        if download_contradiction_flag {
            encoder.copy_buffer_to_buffer(
                &self.contradiction_flag_buf,
                0,
                &self.staging_contradiction_flag_buf,
                0,
                self.contradiction_flag_buf.size(),
            );
        }
        if download_grid {
            encoder.copy_buffer_to_buffer(
                &self.grid_possibilities_buf,
                0,
                &self.staging_grid_possibilities_buf,
                0,
                self.grid_possibilities_buf.size(),
            );
        }
        if download_worklist_count {
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

        queue.submit(Some(encoder.finish()));
        info!("GPU copy commands submitted for download.");

        // Ensure GPU is idle before attempting to map buffers
        device.poll(wgpu::Maintain::Wait);
        info!("GPU polled, proceeding with buffer mapping.");

        // Vector to hold futures for mapping operations
        let mut mapping_futures = Vec::new();

        // Helper closure to create mapping future
        let create_mapping_future = |buffer: Arc<wgpu::Buffer>| -> futures::future::BoxFuture<
            'static,
            Result<wgpu::BufferView, GpuError>,
        > {
            async move {
                let (tx, rx) = oneshot::channel();
                buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |result| {
                        let _ = tx.send(result);
                    });
                // Polling might still be needed here occasionally for certain drivers/setups
                // device.poll(wgpu::Maintain::Poll); // Keep polling minimal if Wait is used above
                rx.await
                    .map_err(|_| {
                        GpuError::BufferOperationError("Mapping channel canceled".to_string())
                    })?
                    .map_err(|e| {
                        GpuError::BufferOperationError(format!("Buffer mapping failed: {:?}", e))
                    })
            }
            .boxed()
        };

        // --- Create Mapping Futures ---
        if download_entropy {
            mapping_futures.push(create_mapping_future(self.staging_entropy_buf.clone()));
        }
        if download_min_entropy {
            mapping_futures.push(create_mapping_future(
                self.staging_min_entropy_info_buf.clone(),
            ));
        }
        if download_contradiction_flag {
            mapping_futures.push(create_mapping_future(
                self.staging_contradiction_flag_buf.clone(),
            ));
        }
        if download_grid {
            mapping_futures.push(create_mapping_future(
                self.staging_grid_possibilities_buf.clone(),
            ));
        }
        if download_worklist_count {
            mapping_futures.push(create_mapping_future(
                self.staging_worklist_count_buf.clone(),
            ));
        }
        if download_contradiction_location {
            mapping_futures.push(create_mapping_future(
                self.staging_contradiction_location_buf.clone(),
            ));
        }

        // Wait for all mapping operations concurrently
        let mapped_results = futures::future::join_all(mapping_futures).await;

        // Process results
        let mut result_idx = 0;
        if download_entropy {
            let view = mapped_results[result_idx].as_ref().map_err(|e| e.clone())?;
            let data: &[f32] = bytemuck::cast_slice(&view);
            results.entropy = Some(data.to_vec());
            result_idx += 1;
        }
        if download_min_entropy {
            let view = mapped_results[result_idx].as_ref().map_err(|e| e.clone())?;
            if view.len() >= 8 {
                let entropy_bits = u32::from_le_bytes(view[0..4].try_into().unwrap());
                let index = u32::from_le_bytes(view[4..8].try_into().unwrap());
                results.min_entropy_info = Some((f32::from_bits(entropy_bits), index));
            } else {
                warn!("Min entropy buffer size mismatch during download");
            }
            result_idx += 1;
        }
        if download_contradiction_flag {
            let view = mapped_results[result_idx].as_ref().map_err(|e| e.clone())?;
            if view.len() >= 4 {
                let flag_value: u32 = u32::from_le_bytes(view[0..4].try_into().unwrap());
                results.contradiction_flag = Some(flag_value != 0);
            } else {
                warn!("Contradiction flag buffer size mismatch during download");
            }
            result_idx += 1;
        }
        if download_grid {
            let view = mapped_results[result_idx].as_ref().map_err(|e| e.clone())?;
            let data: &[u32] = bytemuck::cast_slice(&view);
            results.grid_possibilities = Some(data.to_vec());
            result_idx += 1;
        }
        if download_worklist_count {
            let view = mapped_results[result_idx].as_ref().map_err(|e| e.clone())?;
            if view.len() >= 4 {
                let count: u32 = u32::from_le_bytes(view[0..4].try_into().unwrap());
                results.worklist_count = Some(count);
            } else {
                warn!("Worklist count buffer size mismatch during download");
            }
            result_idx += 1;
        }
        if download_contradiction_location {
            let view = mapped_results[result_idx].as_ref().map_err(|e| e.clone())?;
            if view.len() >= 4 {
                let location: u32 = u32::from_le_bytes(view[0..4].try_into().unwrap());
                results.contradiction_location = Some(location);
            } else {
                warn!("Contradiction location buffer size mismatch during download");
            }
            result_idx += 1;
        }

        debug!("All requested data downloaded from GPU.");
        Ok(results)
    }

    /// Helper function to calculate the expected size of the possibility grid data in bytes.
    fn get_possibility_data_size(&self, num_tiles: u32) -> u64 {
        // Calculate size based on buffer size and number of tiles per u32
        // This assumes the buffer holds data for all cells.
        let _u32s_per_cell = (num_tiles + 31) / 32;
        // The size should just be the buffer's actual size.
        // Let's rely on the existing buffer size directly.
        self.grid_possibilities_buf.size()
    }
}

// Enum to help process results from try_join_all
enum DownloadedData {
    Entropy(Vec<f32>),
    MinEntropyInfo((f32, u32)),
    ContradictionFlag(bool),
    GridPossibilities(Vec<u32>),
    WorklistCount(u32),
    ContradictionLocation(u32),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::initialize_test_gpu;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio;
    use wfc_core::{grid::PossibilityGrid, BoundaryCondition};
    use wfc_rules::AdjacencyRules;
    use wgpu::util::DeviceExt;

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

    #[tokio::test]
    async fn test_buffer_creation() {
        let (_device, _queue, _grid, buffers) = setup_test_environment(4, 4, 1, 10);
        assert!(buffers.grid_possibilities_buf.size() > 0);
        assert!(buffers.entropy_buf.size() > 0);
        assert!(buffers.worklist_buf_a.size() > 0);
        assert!(buffers.worklist_count_buf.size() > 0);
        assert!(buffers.params_uniform_buf.size() > 0);
    }

    #[tokio::test]
    #[ignore] // Ignoring until mapping logic is stable
    async fn read_initial_possibilities_placeholder() {
        let (device, queue, _grid, buffers) = setup_test_environment(2, 2, 1, 4);

        let results = buffers
            .download_results(
                device.clone(),
                queue.clone(),
                false,
                false,
                false,
                true, // Download grid
                false,
                false,
            )
            .await
            .expect("Failed to download results");

        assert!(results.grid_possibilities.is_some());
        // TODO: Add actual value checks
    }

    #[tokio::test]
    async fn test_buffer_usage_flags() {
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

    #[tokio::test]
    async fn cleanup_test() {
        let (_device, _queue, _grid, _buffers) = setup_test_environment(1, 1, 1, 2);
    }

    #[tokio::test]
    async fn test_zero_sized_grid_creation() {
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

    #[tokio::test]
    async fn test_large_tile_count() {
        let num_tiles = 1000;
        let (_device, _queue, grid, buffers) = setup_test_environment(2, 2, 1, num_tiles);
        let u32s_per_cell = (num_tiles + 31) / 32;
        let expected_possibilities_size =
            (grid.width * grid.height * grid.depth * u32s_per_cell * std::mem::size_of::<u32>())
                as u64;
        assert_eq!(
            buffers.grid_possibilities_buf.size(),
            expected_possibilities_size
        );
    }

    #[tokio::test]
    async fn test_map_staging_buffer_future() {
        let (device, queue) = initialize_test_gpu();
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let data_to_write: Vec<u32> = vec![1, 2, 3, 4, 5];
        let buffer_size = (data_to_write.len() * std::mem::size_of::<u32>()) as u64;

        let gpu_buffer = Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Test GPU Buffer"),
                contents: bytemuck::cast_slice(&data_to_write),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            }),
        );

        let staging_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Copy Encoder"),
        });
        encoder.copy_buffer_to_buffer(&gpu_buffer, 0, &staging_buffer, 0, buffer_size);
        queue.submit(Some(encoder.finish()));

        // Call the helper function that returns the future
        let map_future = GpuBuffers::map_staging_buffer_to_vec(staging_buffer.clone());
        pin_mut!(map_future); // Pin the future so it can be used with select!

        // Poll the device while waiting for the future
        let result = loop {
            futures::select! {
                res = map_future.as_mut().fuse() => break res,
                _ = tokio::time::sleep(Duration::from_millis(1)).fuse() => {
                    device.poll(wgpu::Maintain::Poll);
                }
            }
        };

        let downloaded_bytes = result.expect("Mapping future failed");
        let downloaded_data: Vec<u32> = bytemuck::cast_slice(&downloaded_bytes).to_vec();

        assert_eq!(downloaded_data, data_to_write);
    }
}
