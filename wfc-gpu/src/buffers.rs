use crate::GpuError;
use bytemuck::{Pod, Zeroable};
use wfc_core::{grid::PossibilityGrid, rules::AdjacencyRules};
use wgpu;
use wgpu::util::DeviceExt; // Import for create_buffer_init

// Uniform buffer structure - MUST match shader layout
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

// Placeholder struct for managing GPU buffers
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

    // Uploads a list of updated cell coordinates (packed as u32 indices) to the updates buffer.
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

    // Downloads the entropy grid results from the staging buffer.
    // This is an async operation as it involves mapping the staging buffer.
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
    pub fn reset_contradiction_flag(&self, queue: &wgpu::Queue) -> Result<(), GpuError> {
        queue.write_buffer(&self.contradiction_flag_buf, 0, bytemuck::bytes_of(&0u32));
        Ok(())
    }

    /// Resets the output worklist count buffer to 0.
    pub fn reset_output_worklist_count(&self, queue: &wgpu::Queue) -> Result<(), GpuError> {
        queue.write_buffer(
            &self.output_worklist_count_buf,
            0,
            bytemuck::bytes_of(&0u32),
        );
        Ok(())
    }

    /// Updates the worklist_size field within the params uniform buffer.
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

    /// Downloads the contradiction flag result from the staging buffer.
    /// Returns true if a contradiction was detected (flag is non-zero), false otherwise.
    pub async fn download_contradiction_flag(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<bool, GpuError> {
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

        encoder.copy_buffer_to_buffer(
            &self.contradiction_flag_buf,
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
                "Channel closed before receiving map result for contradiction buffer".to_string(),
            )
        })??;

        let flag_value: u32 = {
            let data = buffer_slice.get_mapped_range();
            let result = if data.len() >= std::mem::size_of::<u32>() {
                Ok(*bytemuck::from_bytes::<u32>(&data))
            } else {
                Err(GpuError::BufferOperationError(format!(
                    "Contradiction staging buffer too small ({} bytes), expected {}",
                    data.len(),
                    std::mem::size_of::<u32>()
                )))
            };
            drop(data); // Explicitly drop guard (unmap)
            result? // Propagate error
        };
        Ok(flag_value != 0)
    }

    /// Downloads the GpuParamsUniform struct from the params staging buffer.
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
