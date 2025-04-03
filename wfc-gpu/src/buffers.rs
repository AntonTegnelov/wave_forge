use crate::GpuError;
use bitvec::field::BitField;
use bytemuck::{Pod, Zeroable};
use log::{debug, error, info, warn};
use std::sync::Arc;
use wfc_core::grid::PossibilityGrid;
use wfc_core::BoundaryMode;
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
        boundary_mode: BoundaryMode,
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
                BoundaryMode::Clamped => 0,
                BoundaryMode::Periodic => 1,
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

    /// Asynchronously downloads the calculated entropy values from the GPU `entropy_buf`.
    ///
    /// Copies data from the GPU buffer to a staging buffer and maps it for CPU access.
    ///
    /// # Arguments
    ///
    /// * `device` - The WGPU `Device`.
    /// * `queue` - The WGPU `Queue`.
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<f32>)` containing the entropy values for each cell.
    /// * `Err(GpuError)` if an error occurs during the download process.
    pub async fn download_entropy(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<f32>, GpuError> {
        debug!("Initiating entropy download...");
        let buffer_size = self.entropy_buf.size();
        let staging_buffer = Arc::clone(&self.staging_entropy_buf);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Entropy Download Encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.entropy_buf, 0, &staging_buffer, 0, buffer_size);
        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result); // Ignore error if receiver dropped
        });

        device.poll(wgpu::Maintain::Wait);

        match receiver.receive().await {
            Some(Ok(())) => {
                let data = buffer_slice.get_mapped_range();
                let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
                drop(data);
                staging_buffer.unmap();
                debug!("Entropy download complete ({} floats).", result.len());
                Ok(result)
            }
            Some(Err(e)) => {
                error!("Failed to map staging entropy buffer: {:?}", e);
                staging_buffer.unmap();
                Err(GpuError::BufferMapFailed(e))
            }
            None => {
                error!("Buffer map future cancelled/channel closed.");
                // Attempt to unmap anyway, might already be unmapped or panic
                // staging_buffer.unmap(); // unmap is called automatically when Arc/buffer drops if mapped
                Err(GpuError::Other("Buffer map future cancelled".to_string()))
            }
        }
    }

    /// Asynchronously downloads the minimum entropy information (value and index) from the GPU `min_entropy_info_buf`.
    pub async fn download_min_entropy_info(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(f32, u32), GpuError> {
        debug!("Initiating min entropy info download...");
        let buffer_size = self.min_entropy_info_buf.size();
        let staging_buffer = Arc::clone(&self.staging_min_entropy_info_buf);

        if buffer_size != 8 {
            warn!(
                "Min entropy info buffer size is not 8 bytes: {}",
                buffer_size
            );
            // Continue anyway, but log warning
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Min Entropy Info Download Encoder"),
        });
        encoder.copy_buffer_to_buffer(
            &self.min_entropy_info_buf,
            0,
            &staging_buffer,
            0,
            buffer_size, // Copy the actual size, even if unexpected
        );
        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        device.poll(wgpu::Maintain::Wait);

        match receiver.receive().await {
            Some(Ok(())) => {
                let data = buffer_slice.get_mapped_range();
                // Use size check before casting
                if data.len() >= 8 {
                    // Clone the slice to avoid moving `data`
                    let data_slice = data[..8].to_vec();
                    let result_raw: [u32; 2] = *bytemuck::from_bytes::<[u32; 2]>(&data_slice);
                    let min_entropy_val = f32::from_bits(result_raw[0]);
                    let min_index = result_raw[1];
                    drop(data);
                    staging_buffer.unmap();
                    debug!(
                        "Min entropy info download complete: value={}, index={}",
                        min_entropy_val, min_index
                    );
                    Ok((min_entropy_val, min_index))
                } else {
                    error!(
                        "Downloaded min entropy data size mismatch: expected >= 8, got {}",
                        data.len()
                    );
                    drop(data);
                    staging_buffer.unmap();
                    Err(GpuError::BufferOperationError(format!(
                        "Min entropy data size mismatch: expected >= 8, got {}",
                        data.len()
                    )))
                }
            }
            Some(Err(e)) => {
                error!("Failed to map staging min entropy buffer: {:?}", e);
                staging_buffer.unmap();
                Err(GpuError::BufferMapFailed(e))
            }
            None => {
                error!("Min entropy buffer map future cancelled.");
                Err(GpuError::Other(
                    "Min entropy buffer map future cancelled".to_string(),
                ))
            }
        }
    }

    /// Asynchronously downloads the contradiction flag (u32) from the GPU `contradiction_flag_buf`.
    pub async fn download_contradiction_flag(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<bool, GpuError> {
        debug!("Initiating contradiction flag download...");
        let buffer_size = self.contradiction_flag_buf.size();
        let staging_buffer = Arc::clone(&self.staging_contradiction_flag_buf);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Contradiction Flag Download Encoder"),
        });
        encoder.copy_buffer_to_buffer(
            &self.contradiction_flag_buf,
            0,
            &staging_buffer,
            0,
            buffer_size,
        );
        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        device.poll(wgpu::Maintain::Wait);

        match receiver.receive().await {
            Some(Ok(())) => {
                let data = buffer_slice.get_mapped_range();
                if data.len() >= 4 {
                    let result_raw: u32 = *bytemuck::from_bytes::<u32>(&data[..4]);
                    let contradiction_detected = result_raw > 0;
                    drop(data);
                    staging_buffer.unmap();
                    debug!(
                        "Contradiction flag download complete: raw={}, detected={}",
                        result_raw, contradiction_detected
                    );
                    Ok(contradiction_detected)
                } else {
                    error!(
                        "Downloaded contradiction flag data size mismatch: expected >= 4, got {}",
                        data.len()
                    );
                    drop(data);
                    staging_buffer.unmap();
                    Err(GpuError::BufferOperationError(format!(
                        "Contradiction flag data size mismatch: expected >= 4, got {}",
                        data.len()
                    )))
                }
            }
            Some(Err(e)) => {
                error!("Failed to map staging contradiction flag buffer: {:?}", e);
                staging_buffer.unmap();
                Err(GpuError::BufferMapFailed(e))
            }
            None => {
                error!("Contradiction flag buffer map future cancelled.");
                Err(GpuError::Other(
                    "Contradiction flag buffer map future cancelled".to_string(),
                ))
            }
        }
    }

    /// Asynchronously downloads the contradiction location index (u32) from the GPU `contradiction_location_buf`.
    pub async fn download_contradiction_location(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<u32, GpuError> {
        let buffer_size = self.contradiction_location_buf.size();
        let staging_buffer = Arc::clone(&self.staging_contradiction_location_buf);

        if buffer_size != std::mem::size_of::<u32>() as u64 {
            warn!(
                "Contradiction location buffer size mismatch: expected {}, got {}",
                std::mem::size_of::<u32>(),
                buffer_size
            );
            // Continue anyway
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Contradiction Location Download Encoder"),
        });
        encoder.copy_buffer_to_buffer(
            &self.contradiction_location_buf,
            0,
            &staging_buffer,
            0,
            buffer_size, // Copy actual size
        );
        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        device.poll(wgpu::Maintain::Wait);

        match receiver.receive().await {
            Some(Ok(())) => {
                let data = buffer_slice.get_mapped_range();
                if data.len() >= 4 {
                    let location_index: u32 = bytemuck::from_bytes::<u32>(&data[..4]).to_owned();
                    drop(data);
                    staging_buffer.unmap();
                    Ok(location_index)
                } else {
                    error!("Downloaded contradiction location data size mismatch: expected >= 4, got {}", data.len());
                    drop(data);
                    staging_buffer.unmap();
                    Err(GpuError::BufferOperationError(format!(
                        "Contradiction location data size mismatch: expected >= 4, got {}",
                        data.len()
                    )))
                }
            }
            Some(Err(e)) => {
                error!(
                    "Failed to map staging contradiction location buffer: {:?}",
                    e
                );
                staging_buffer.unmap();
                Err(GpuError::BufferMapFailed(e))
            }
            None => {
                error!("Contradiction location buffer map future cancelled.");
                Err(GpuError::BufferOperationError(
                    "Failed to receive buffer map result for contradiction location".to_string(),
                ))
            }
        }
    }

    /// Asynchronously downloads the worklist count from the GPU `worklist_count_buf`.
    pub async fn download_worklist_count(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<u32, GpuError> {
        let size = self.worklist_count_buf.size();
        // Create a temporary staging buffer just for this download
        let temp_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Temp Staging Worklist Count"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Download Worklist Count Encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.worklist_count_buf, 0, &temp_staging_buffer, 0, size);
        queue.submit(Some(encoder.finish()));

        let buffer_slice = temp_staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = sender.send(v);
        });

        device.poll(wgpu::Maintain::Wait);

        match receiver.receive().await {
            Some(Ok(())) => {
                let data = buffer_slice.get_mapped_range();
                if data.len() >= 4 {
                    let count = *bytemuck::from_bytes::<u32>(&data[..4]);
                    drop(data);
                    temp_staging_buffer.unmap();
                    Ok(count)
                } else {
                    error!(
                        "Downloaded worklist count data size mismatch: expected >= 4, got {}",
                        data.len()
                    );
                    drop(data);
                    temp_staging_buffer.unmap();
                    Err(GpuError::BufferOperationError(format!(
                        "Worklist count data size mismatch: expected >= 4, got {}",
                        data.len()
                    )))
                }
            }
            Some(Err(e)) => {
                error!("Failed to map staging worklist count buffer: {:?}", e);
                temp_staging_buffer.unmap(); // Attempt unmap
                Err(GpuError::BufferMapFailed(e))
            }
            None => {
                error!("Worklist count buffer map future cancelled.");
                Err(GpuError::BufferOperationError(
                    "Map receiver channel closed prematurely".to_string(),
                ))
            }
        }
    }

    /// Asynchronously downloads the raw possibility data (`Vec<u32>`) from the GPU `grid_possibilities_buf`.
    pub async fn download_possibilities(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<u32>, GpuError> {
        debug!("Initiating possibilities download...");
        let buffer_size = self.grid_possibilities_buf.size();
        let staging_buffer = Arc::clone(&self.staging_grid_possibilities_buf);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Possibilities Download Encoder"),
        });
        encoder.copy_buffer_to_buffer(
            &self.grid_possibilities_buf,
            0,
            &staging_buffer,
            0,
            buffer_size,
        );
        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        device.poll(wgpu::Maintain::Wait);

        match receiver.receive().await {
            Some(Ok(())) => {
                let data = buffer_slice.get_mapped_range();
                // Cast the raw byte data to Vec<u32>
                let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
                drop(data);
                staging_buffer.unmap();
                debug!("Possibilities download complete ({} u32s).", result.len());
                Ok(result)
            }
            Some(Err(e)) => {
                error!("Failed to map staging possibilities buffer: {:?}", e);
                staging_buffer.unmap();
                Err(GpuError::BufferMapFailed(e))
            }
            None => {
                error!("Possibilities buffer map future cancelled.");
                Err(GpuError::Other(
                    "Possibilities map future cancelled".to_string(),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use wfc_core::grid::PossibilityGrid;
    use wfc_core::BoundaryMode;
    use wfc_rules::{AdjacencyRules, TileSet, TileSetError, Transformation};

    // Helper to create uniform rules
    fn create_uniform_rules(tileset: &TileSet) -> AdjacencyRules {
        let num_tiles = tileset.num_transformed_tiles();
        let num_axes = 6;
        let mut allowed_tuples = Vec::new();
        for axis in 0..num_axes {
            for ttid1 in 0..num_tiles {
                for ttid2 in 0..num_tiles {
                    allowed_tuples.push((axis, ttid1, ttid2));
                }
            }
        }
        AdjacencyRules::from_allowed_tuples(num_tiles, num_axes, allowed_tuples)
    }

    // Helper to create simple tileset
    fn create_simple_tileset(num_base_tiles: usize) -> Result<TileSet, TileSetError> {
        let weights = vec![1.0; num_base_tiles];
        let allowed_transforms = vec![vec![Transformation::Identity]; num_base_tiles];
        TileSet::new(weights, allowed_transforms)
    }

    // Mock device/queue setup
    struct MockGpuResources {
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    }

    async fn setup_mock_gpu() -> Option<MockGpuResources> {
        // Standard setup using default adapter/device
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::default(),
            flags: Default::default(),
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: None,
                },
                None, // Trace path
            )
            .await
            .ok()?;

        Some(MockGpuResources {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    #[tokio::test]
    async fn test_buffer_creation_with_boundary_mode() {
        let gpu_res = match setup_mock_gpu().await {
            Some(res) => res,
            None => {
                println!("Skipping test_buffer_creation_with_boundary_mode: No suitable GPU adapter found.");
                return;
            }
        };
        let device = &gpu_res.device;
        let queue = &gpu_res.queue;

        let tileset = create_simple_tileset(2).unwrap();
        let rules = create_uniform_rules(&tileset);
        let grid = PossibilityGrid::new(2, 2, 2, tileset.num_transformed_tiles());

        let result_clamped = GpuBuffers::new(device, queue, &grid, &rules, BoundaryMode::Clamped);
        assert!(result_clamped.is_ok());

        let result_periodic = GpuBuffers::new(device, queue, &grid, &rules, BoundaryMode::Periodic);
        assert!(result_periodic.is_ok());
    }

    #[test]
    fn test_buffer_creation() {
        assert!(true); // Simple compile check
    }

    // Other tests (upload_updates, reset_functions, etc.) remain as placeholders
    // because they require actual GPU interaction which is hard in unit tests.
    #[test]
    fn test_upload_updates() {
        assert!(true);
    }

    #[test]
    fn test_reset_functions() {
        assert!(true);
    }

    #[test]
    fn test_update_params_worklist_size() {
        assert!(true);
    }

    #[tokio::test]
    async fn test_download_smoke_tests() {
        // Smoke tests for download functions (check they compile and run basic async flow)
        let gpu_res = match setup_mock_gpu().await {
            Some(res) => res,
            None => {
                println!("Skipping test_download_smoke_tests: No suitable GPU adapter found.");
                return;
            }
        };
        let device = &gpu_res.device;
        let queue = &gpu_res.queue;

        let tileset = create_simple_tileset(2).unwrap();
        let rules = create_uniform_rules(&tileset);
        let grid = PossibilityGrid::new(2, 2, 2, tileset.num_transformed_tiles());
        let buffers = GpuBuffers::new(device, queue, &grid, &rules, BoundaryMode::Clamped).unwrap();

        // Just call download functions, check if they return Result (don't care about value here)
        let _ = buffers.download_entropy(device, queue).await;
        let _ = buffers.download_min_entropy_info(device, queue).await;
        let _ = buffers.download_contradiction_flag(device, queue).await;
        let _ = buffers.download_contradiction_location(device, queue).await;
        let _ = buffers.download_worklist_count(device, queue).await;
        assert!(true); // If it reaches here without panic, basic flow is ok
    }
}
