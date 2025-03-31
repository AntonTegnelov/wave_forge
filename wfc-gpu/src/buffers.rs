use crate::GpuError;
use bytemuck::{Pod, Zeroable};
use wfc_core::{grid::PossibilityGrid, rules::AdjacencyRules};
use wgpu;
use wgpu::util::DeviceExt; // Import for create_buffer_init

/// Uniform buffer structure for passing parameters to GPU compute shaders.
///
/// This structure must match the layout of the equivalent struct in WGSL shaders.
/// It contains all grid dimensions, tile counts, and runtime values needed by shaders.
///
/// # Memory Layout Considerations
///
/// The struct is marked with `repr(C)` to ensure consistent memory layout between
/// Rust and shader code. It also implements Pod and Zeroable for safe casting.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuParamsUniform {
    pub grid_width: u32,
    pub grid_height: u32,
    pub grid_depth: u32,
    pub num_tiles: u32,
    pub num_tiles_u32: u32, // Number of u32s needed per cell for possibilities
    pub num_axes: u32,
    pub worklist_size: u32, // Add worklist size field
    pub _padding1: u32,     // Adjust padding if needed
}

/// Manages GPU buffers for the Wave Function Collapse algorithm.
///
/// This struct handles all GPU memory management, including:
/// - Grid possibility data (bitvectors packed into u32 arrays)
/// - Adjacency rules in packed format
/// - Entropy calculation buffers
/// - Propagation worklists and counters
/// - Flags for detecting contradictions
///
/// # Synchronization and Hang Prevention
///
/// The buffer operations include several measures to prevent GPU hangs:
/// - Explicit polling when waiting for GPU operations
/// - Proper buffer size checking before operations
/// - Staging buffers for safe memory transfers
/// - Explicit unmapping of GPU buffers
/// - Careful handling of asynchronous buffer operations
pub struct GpuBuffers {
    // Grid state (possibilities) - likely atomic u32 for bitvec representation
    pub grid_possibilities_buf: wgpu::Buffer,
    // Adjacency rules (flattened)
    pub rules_buf: wgpu::Buffer,
    // Entropy output buffer
    pub entropy_buf: wgpu::Buffer,
    // Buffer for updated coordinates (input to propagation worklist)
    pub updates_buf: wgpu::Buffer,
    // Buffer for the next propagation worklist (output from shader)
    pub output_worklist_buf: wgpu::Buffer,
    // Buffer to hold the count for the output worklist (atomic u32)
    pub output_worklist_count_buf: wgpu::Buffer,
    // Staging buffers for reading results back to CPU (e.g., entropy, contradiction)
    pub contradiction_flag_buf: wgpu::Buffer,
    pub params_uniform_buf: wgpu::Buffer,
}

impl GpuBuffers {
    /// Creates a new set of GPU buffers for WFC computation.
    ///
    /// Initializes all necessary buffers with appropriate sizes and content based on
    /// the initial grid and rules. The buffers are allocated on the GPU and filled
    /// with initial data.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to create buffers on
    /// * `initial_grid` - The initial grid with possibility data
    /// * `rules` - The adjacency rules defining valid tile arrangements
    ///
    /// # Returns
    ///
    /// A Result containing either the initialized buffers or a GPU error
    ///
    /// # Implementation Details
    ///
    /// 1. Packs grid possibilities into bit vectors (u32 arrays)
    /// 2. Packs adjacency rules into bit vectors
    /// 3. Creates uniform buffer with grid parameters
    /// 4. Allocates working buffers for computation
    pub fn new(
        device: &wgpu::Device,
        initial_grid: &PossibilityGrid,
        rules: &AdjacencyRules,
    ) -> Result<Self, GpuError> {
        let width = initial_grid.width;
        let height = initial_grid.height;
        let depth = initial_grid.depth;
        let num_cells = width * height * depth;
        let num_tiles = rules.num_tiles();
        let num_axes = rules.num_axes();

        // --- Pack Possibilities (Manual Bit Packing) ---
        let bits_per_cell = num_tiles;
        let u32s_per_cell = (bits_per_cell + 31) / 32; // Ceiling division
        let mut packed_possibilities: Vec<u32> = Vec::with_capacity(num_cells * u32s_per_cell);

        for cell_bitvec in initial_grid.get_cell_data() {
            let mut cell_data_u32 = vec![0u32; u32s_per_cell];
            for (i, bit) in cell_bitvec.iter().by_vals().enumerate() {
                if bit {
                    let u32_idx = i / 32;
                    let bit_idx = i % 32;
                    if u32_idx < cell_data_u32.len() {
                        // Ensure index is in bounds
                        cell_data_u32[u32_idx] |= 1 << bit_idx;
                    }
                }
            }
            packed_possibilities.extend_from_slice(&cell_data_u32);
        }
        let _grid_buffer_size = (packed_possibilities.len() * std::mem::size_of::<u32>()) as u64;

        // --- Pack Rules ---
        let num_rules = num_axes * num_tiles * num_tiles;
        let u32s_for_rules = (num_rules + 31) / 32;
        let mut packed_rules = vec![0u32; u32s_for_rules];
        for (i, &allowed) in rules.get_allowed_rules().iter().enumerate() {
            if allowed {
                let u32_idx = i / 32;
                let bit_idx = i % 32;
                packed_rules[u32_idx] |= 1 << bit_idx;
            }
        }
        let _rules_buffer_size = (packed_rules.len() * std::mem::size_of::<u32>()) as u64;

        // --- Create Uniform Buffer Data ---
        let params = GpuParamsUniform {
            grid_width: width as u32,
            grid_height: height as u32,
            grid_depth: depth as u32,
            num_tiles: num_tiles as u32,
            num_tiles_u32: u32s_per_cell as u32,
            num_axes: num_axes as u32,
            worklist_size: 0, // Initial worklist size is 0
            _padding1: 0,     // Ensure padding is correct if struct changes
        };
        let _params_buffer_size = std::mem::size_of::<GpuParamsUniform>() as u64;

        // --- Calculate Other Buffer Sizes ---
        let entropy_buffer_size = (num_cells * std::mem::size_of::<f32>()) as u64;
        // Input worklist (updates) can contain up to num_cells indices
        let updates_buffer_size = (num_cells * std::mem::size_of::<u32>()) as u64;
        // Output worklist can also contain up to num_cells indices
        let output_worklist_buffer_size = updates_buffer_size;
        let output_worklist_count_buffer_size = std::mem::size_of::<u32>() as u64;
        let contradiction_buffer_size = std::mem::size_of::<u32>() as u64;

        // --- Create Buffers --- (Use calculated sizes)
        let grid_possibilities_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Possibilities"),
            contents: bytemuck::cast_slice(&packed_possibilities),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let rules_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Rules"),
            contents: bytemuck::cast_slice(&packed_rules),
            usage: wgpu::BufferUsages::STORAGE, // Read-only in shader
        });

        let params_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Uniform"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let entropy_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Entropy"),
            size: entropy_buffer_size, // Use calculated size
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let updates_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Updates Worklist"),
            size: updates_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_worklist_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Worklist"),
            size: output_worklist_buffer_size,
            // Needs STORAGE for shader write, COPY_SRC if read back needed
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let output_worklist_count_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Worklist Count"),
            size: output_worklist_count_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let contradiction_flag_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Contradiction Flag"),
            size: contradiction_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Ok(Self {
            grid_possibilities_buf,
            rules_buf,
            params_uniform_buf,
            entropy_buf,
            updates_buf,
            output_worklist_buf,
            output_worklist_count_buf,
            contradiction_flag_buf,
        })
    }

    /// Uploads a list of updated cell coordinates to the GPU.
    ///
    /// This is used to tell the propagation shader which cells have been updated
    /// and need their constraints propagated to neighbors.
    ///
    /// # Arguments
    ///
    /// * `queue` - The GPU command queue to submit the upload
    /// * `updates` - Array of cell indices (packed as u32) to process
    ///
    /// # Returns
    ///
    /// Result indicating success or a buffer operation error
    ///
    /// # Safety
    ///
    /// Checks that the update data doesn't exceed buffer capacity before uploading
    pub fn upload_updates(&self, queue: &wgpu::Queue, updates: &[u32]) -> Result<(), GpuError> {
        let updates_byte_size = std::mem::size_of_val(updates) as u64;
        if updates_byte_size > self.updates_buf.size() {
            return Err(GpuError::BufferOperationError(format!(
                "Update data size ({} bytes) exceeds updates buffer capacity ({} bytes)",
                updates_byte_size,
                self.updates_buf.size()
            )));
        }
        queue.write_buffer(&self.updates_buf, 0, bytemuck::cast_slice(updates));
        Ok(())
    }

    /// Downloads the entropy grid results from the GPU.
    ///
    /// This async function copies entropy data from the GPU to a staging buffer,
    /// then maps that buffer for CPU access to read back the results.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `queue` - The GPU command queue
    ///
    /// # Returns
    ///
    /// Result containing either the entropy values or a GPU error
    ///
    /// # Asynchronous Operation
    ///
    /// This function is async because it needs to wait for the GPU
    /// to complete the copy operation and mapping before reading data.
    pub async fn download_entropy(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<f32>, GpuError> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Entropy Download Encoder"),
        });

        // Create temporary staging buffer
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Temp Entropy Staging"),
            size: self.entropy_buf.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &self.entropy_buf,
            0,
            &staging_buffer, // Use temp buffer
            0,
            staging_buffer.size(),
        );
        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..); // Use temp buffer
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        // Wait for the mapping to complete
        receiver.receive().await.ok_or_else(|| {
            GpuError::Other(
                "Channel closed before receiving map result for entropy buffer".to_string(),
            )
        })??;

        let entropy_data: Vec<f32> = {
            let data = buffer_slice.get_mapped_range();
            let result = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            result
        };
        Ok(entropy_data)
    }

    /// Resets the contradiction flag buffer to 0.
    ///
    /// This needs to be done before each propagation run to ensure
    /// we start with a clean state for contradiction detection.
    ///
    /// # Arguments
    ///
    /// * `queue` - The GPU command queue
    ///
    /// # Returns
    ///
    /// Result indicating success or a buffer operation error
    pub fn reset_contradiction_flag(&self, queue: &wgpu::Queue) -> Result<(), GpuError> {
        queue.write_buffer(&self.contradiction_flag_buf, 0, bytemuck::bytes_of(&0u32));
        Ok(())
    }

    /// Resets the output worklist count buffer to 0.
    ///
    /// This prepares the output worklist for a new propagation run
    /// by resetting its atomic counter to zero.
    ///
    /// # Arguments
    ///
    /// * `queue` - The GPU command queue
    ///
    /// # Returns
    ///
    /// Result indicating success or a buffer operation error
    pub fn reset_output_worklist_count(&self, queue: &wgpu::Queue) -> Result<(), GpuError> {
        queue.write_buffer(
            &self.output_worklist_count_buf,
            0,
            bytemuck::bytes_of(&0u32),
        );
        Ok(())
    }

    /// Updates the worklist_size field in the params uniform buffer.
    ///
    /// This tells the GPU shader how many items are in the worklist
    /// to process during propagation.
    ///
    /// # Arguments
    ///
    /// * `queue` - The GPU command queue
    /// * `worklist_size` - Number of cells in the current worklist
    ///
    /// # Returns
    ///
    /// Result indicating success or a buffer operation error
    ///
    /// # Implementation Details
    ///
    /// Uses a calculated offset to update just the worklist_size field
    /// within the larger params uniform structure.
    pub fn update_params_worklist_size(
        &self,
        queue: &wgpu::Queue,
        worklist_size: u32,
    ) -> Result<(), GpuError> {
        // Calculate the offset of the worklist_size field within the GpuParamsUniform struct.
        // This is fragile if the struct layout changes. Consider using `offset_of!` macro from a crate
        // or defining offsets explicitly. For now, assume it's after 6 * u32.
        let offset = (6 * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
        if offset + std::mem::size_of::<u32>() as u64 > self.params_uniform_buf.size() {
            return Err(GpuError::BufferOperationError(
                "Calculated offset for worklist_size exceeds params buffer size.".to_string(),
            ));
        }
        queue.write_buffer(
            &self.params_uniform_buf,
            offset,
            bytemuck::bytes_of(&worklist_size),
        );
        Ok(())
    }

    /// Downloads the contradiction flag from the GPU.
    ///
    /// This async function checks if a contradiction was detected during
    /// propagation by reading back the contradiction flag.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `queue` - The GPU command queue
    ///
    /// # Returns
    ///
    /// Result containing a boolean (true if contradiction detected)
    /// or a GPU error
    ///
    /// # Asynchronous Operation
    ///
    /// This function uses explicit GPU polling and synchronization to
    /// ensure all GPU operations complete before reading results.
    /// It prevents hangs by explicitly waiting for GPU completion.
    pub async fn download_contradiction_flag(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<bool, GpuError> {
        log::debug!("Creating staging buffer for contradiction flag download...");
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Contradiction Download Encoder"),
        });

        // Create temporary staging buffer
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Temp Contradiction Staging"),
            size: self.contradiction_flag_buf.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        log::debug!("Copying contradiction flag to staging buffer...");
        encoder.copy_buffer_to_buffer(
            &self.contradiction_flag_buf,
            0,
            &staging_buffer, // Use temp buffer
            0,
            staging_buffer.size(),
        );
        let submission_index = queue.submit(std::iter::once(encoder.finish()));
        log::debug!(
            "Contradiction flag copy submitted with index: {:?}",
            submission_index
        );

        // Explicitly wait for the GPU to finish the copy
        device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission_index));
        log::debug!("GPU signaled completion of contradiction flag copy.");

        log::debug!("Mapping staging buffer...");
        let buffer_slice = staging_buffer.slice(..); // Use temp buffer
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        // Poll device to ensure mapping has a chance to complete
        device.poll(wgpu::Maintain::Wait);

        // Wait for the mapping to complete
        log::debug!("Awaiting buffer mapping...");
        receiver.receive().await.ok_or_else(|| {
            GpuError::Other(
                "Channel closed before receiving map result for contradiction buffer".to_string(),
            )
        })??;
        log::debug!("Buffer mapping completed.");

        let flag_value: u32 = {
            let data = buffer_slice.get_mapped_range();
            let result = if data.len() >= std::mem::size_of::<u32>() {
                log::debug!("Reading contradiction flag value: {} bytes", data.len());
                Ok(*bytemuck::from_bytes::<u32>(&data))
            } else {
                log::error!("Contradiction buffer too small: {} bytes", data.len());
                Err(GpuError::BufferOperationError(format!(
                    "Contradiction staging buffer too small ({} bytes), expected {}",
                    data.len(),
                    std::mem::size_of::<u32>()
                )))
            };
            drop(data); // Explicitly drop guard (unmap)
            result? // Propagate error
        };

        log::debug!("Contradiction flag value: {}", flag_value);
        staging_buffer.unmap(); // Explicitly unmap the buffer
        Ok(flag_value != 0)
    }

    /// Downloads the full params uniform struct from the GPU.
    ///
    /// This is primarily used for debugging to verify that parameters
    /// were uploaded correctly.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `queue` - The GPU command queue
    ///
    /// # Returns
    ///
    /// Result containing the params struct or a GPU error
    pub async fn download_params(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<GpuParamsUniform, GpuError> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Params Download Encoder"),
        });

        // Create temporary staging buffer
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Temp Params Staging"),
            size: self.params_uniform_buf.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &self.params_uniform_buf,
            0,
            &staging_buffer, // Use temp buffer
            0,
            staging_buffer.size(),
        );
        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..); // Use temp buffer
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        // Wait for the mapping to complete
        receiver.receive().await.ok_or_else(|| {
            GpuError::Other(
                "Channel closed before receiving map result for params buffer".to_string(),
            )
        })??;

        let params_data = {
            let data = buffer_slice.get_mapped_range();
            let result = if data.len() >= std::mem::size_of::<GpuParamsUniform>() {
                Ok(*bytemuck::from_bytes::<GpuParamsUniform>(
                    &data[..std::mem::size_of::<GpuParamsUniform>()],
                ))
            } else {
                Err(GpuError::BufferOperationError(format!(
                    "Params staging buffer too small ({} bytes), expected {}",
                    data.len(),
                    std::mem::size_of::<GpuParamsUniform>()
                )))
            };
            // Explicitly drop the guard (which unmaps) before returning the result
            drop(data);
            result? // Propagate error if buffer was too small
        };
        // Buffer is now unmapped

        Ok(params_data)
    }
}
