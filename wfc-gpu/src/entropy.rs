use crate::{
    buffers::{GpuBuffers, GpuEntropyShaderParams},
    pipeline::ComputePipelines,
    sync::GpuSynchronizer,
    GpuError,
};
use log::{debug, error, warn};
use pollster;
use std::sync::Arc;
use wfc_core::entropy::EntropyError as CoreEntropyError;
use wfc_core::{
    entropy::{EntropyCalculator, EntropyError, EntropyHeuristicType},
    grid::{EntropyGrid, PossibilityGrid},
};
use wgpu;

/// GPU-accelerated entropy calculator for use in WFC algorithm.
///
/// This component computes the entropy of each cell in the grid and identifies
/// the cell with the minimum positive entropy, which is a key step in the WFC algorithm.
#[derive(Debug, Clone)]
pub struct GpuEntropyCalculator {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipelines: Arc<ComputePipelines>,
    buffers: Arc<GpuBuffers>,
    synchronizer: Arc<GpuSynchronizer>,
    grid_dims: (usize, usize, usize),
    heuristic_type: EntropyHeuristicType,
}

impl GpuEntropyCalculator {
    /// Creates a new GPU entropy calculator with the default Shannon entropy heuristic.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        pipelines: Arc<ComputePipelines>,
        buffers: Arc<GpuBuffers>,
        grid_dims: (usize, usize, usize),
    ) -> Self {
        let synchronizer = Arc::new(GpuSynchronizer::new(
            device.clone(),
            queue.clone(),
            buffers.clone(),
        ));
        Self {
            device,
            queue,
            pipelines,
            buffers,
            synchronizer,
            grid_dims,
            heuristic_type: EntropyHeuristicType::default(),
        }
    }

    /// Creates a new GPU entropy calculator with a specific entropy heuristic.
    pub fn with_heuristic(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        pipelines: Arc<ComputePipelines>,
        buffers: Arc<GpuBuffers>,
        grid_dims: (usize, usize, usize),
        heuristic_type: EntropyHeuristicType,
    ) -> Self {
        let synchronizer = Arc::new(GpuSynchronizer::new(
            device.clone(),
            queue.clone(),
            buffers.clone(),
        ));
        Self {
            device,
            queue,
            pipelines,
            buffers,
            synchronizer,
            grid_dims,
            heuristic_type,
        }
    }

    /// Asynchronous version of calculate_entropy that allows proper async/await patterns
    pub async fn calculate_entropy_async(
        &self,
        grid: &PossibilityGrid,
    ) -> Result<EntropyGrid, EntropyError> {
        debug!("Entering calculate_entropy_async");
        let (width, height, depth) = (grid.width, grid.height, grid.depth);
        let num_cells = width * height * depth;

        // TODO: Re-evaluate buffer resizing logic. Should it be here or handled
        // by a higher-level coordinator? For now, assume buffers are correctly sized.

        // Reset min entropy buffer (Access via entropy_buffers)
        self.synchronizer.reset_min_entropy_buffer()?;

        // Convert grid possibilities to u32 arrays and upload them
        let u32s_per_cell = self.buffers.grid_buffers.u32s_per_cell;
        let mut packed_data = Vec::with_capacity(num_cells * u32s_per_cell);
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    if let Some(cell) = grid.get(x, y, z) {
                        let mut cell_data = vec![0u32; u32s_per_cell];
                        for (i, bit) in cell.iter().enumerate() {
                            if *bit {
                                let u32_idx = i / 32;
                                let bit_idx = i % 32;
                                if u32_idx < cell_data.len() {
                                    cell_data[u32_idx] |= 1 << bit_idx;
                                }
                            }
                        }
                        packed_data.extend_from_slice(&cell_data);
                    } else {
                        // Handle case where grid.get might return None if dimensions are mismatched
                        // Fill with zeros or handle as error depending on expected behavior
                        packed_data.extend(vec![0u32; u32s_per_cell]);
                    }
                }
            }
        }

        // Upload the data to the GPU buffer (Access via grid_buffers)
        self.queue.write_buffer(
            &self.buffers.grid_buffers.grid_possibilities_buf,
            0,
            bytemuck::cast_slice(&packed_data),
        );

        // --- Create and configure entropy parameters ---
        let entropy_shader_params = GpuEntropyShaderParams {
            grid_dims: [width as u32, height as u32, depth as u32],
            heuristic_type: match self.heuristic_type {
                EntropyHeuristicType::Shannon => 0,
                EntropyHeuristicType::Count => 1,
                EntropyHeuristicType::CountSimple => 2,
                EntropyHeuristicType::WeightedCount => 3,
            },
            num_tiles: self.buffers.num_tiles as u32,
            u32s_per_cell: u32s_per_cell as u32,
            _padding1: 0,
            _padding2: 0,
        };

        // --- Write entropy parameters to buffer ---
        self.synchronizer
            .upload_entropy_params(&entropy_shader_params)?;

        // --- Create bind groups for the entropy shader ---
        // (Layouts might be better managed within ComputePipelines)
        let entropy_params_bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Entropy Params Bind Group Layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let entropy_params_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Entropy Params Bind Group"),
            layout: &entropy_params_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.buffers.entropy_params_buffer.as_entire_binding(), // Direct access ok
            }],
        });

        // --- Dispatch Compute Shader ---
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Entropy Encoder"),
            });

        // Create main bind group (group 0)
        let grid_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Grid Possibilities Bind Group"),
            layout: &self.pipelines.entropy_bind_group_layout_0,
            entries: &[
                wgpu::BindGroupEntry {
                    // Grid Possibilities
                    binding: 0,
                    resource: self
                        .buffers
                        .grid_buffers // Access via grid_buffers
                        .grid_possibilities_buf
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    // Entropy Output
                    binding: 1,
                    resource: self.buffers.entropy_buffers.entropy_buf.as_entire_binding(), // Corrected access
                },
                wgpu::BindGroupEntry {
                    // Params (General)
                    binding: 2,
                    resource: self.buffers.params_uniform_buf.as_entire_binding(), // Direct access ok
                },
                wgpu::BindGroupEntry {
                    // Min Entropy Output
                    binding: 3,
                    resource: self
                        .buffers
                        .entropy_buffers
                        .min_entropy_info_buf
                        .as_entire_binding(), // Corrected access
                },
            ],
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Entropy Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipelines.entropy_pipeline);
            compute_pass.set_bind_group(0, &grid_bind_group, &[]); // Main data buffers
            compute_pass.set_bind_group(1, &entropy_params_bind_group, &[]); // Entropy specific params

            // Dispatch - Calculate workgroup counts
            let workgroup_size = 8; // Must match shader's workgroup_size
            let workgroup_x = width.div_ceil(workgroup_size);
            let workgroup_y = height.div_ceil(workgroup_size);
            let workgroup_z = depth;
            compute_pass.dispatch_workgroups(
                workgroup_x as u32,
                workgroup_y as u32,
                workgroup_z as u32,
            );
        } // End compute pass scope

        // Submit to Queue
        self.queue.submit(std::iter::once(encoder.finish()));

        // --- Download Results ---
        // Create encoder for copy
        let mut copy_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Copy Entropy Results Encoder"),
                });
        copy_encoder.copy_buffer_to_buffer(
            &self.buffers.entropy_buffers.entropy_buf, // Source: Use entropy_buffers
            0,
            &self.buffers.entropy_buffers.staging_entropy_buf, // Destination: Use entropy_buffers
            0,
            self.buffers.entropy_buffers.entropy_buf.size(), // Use entropy_buffers
        );
        self.queue.submit(Some(copy_encoder.finish()));

        // Map and read entropy data (BLOCKING - Needs Async Version)
        let entropy_data_result = crate::buffers::map_and_process::<f32>(
            self.buffers.entropy_buffers.staging_entropy_buf.clone(), // Use entropy_buffers
            std::time::Duration::from_secs(5),
            Some(num_cells),
        )
        .await;

        let entropy_data = match entropy_data_result {
            Ok(data_box) => *data_box.downcast::<Vec<f32>>().map_err(|_| {
                EntropyError::InternalError("Failed to downcast entropy data".into())
            })?,
            Err(e) => return Err(EntropyError::GpuError(e.to_string())),
        };

        debug!("Exiting calculate_entropy_async");
        let mut entropy_grid =
            EntropyGrid::new(self.grid_dims.0, self.grid_dims.1, self.grid_dims.2);
        if entropy_grid.data.len() == entropy_data.len() {
            entropy_grid.data = entropy_data;
            Ok(entropy_grid)
        } else {
            error!(
                "Entropy data size mismatch: expected {}, got {}",
                entropy_grid.data.len(),
                entropy_data.len()
            );
            Err(CoreEntropyError::Other(
                "Downloaded entropy data size mismatch".into(),
            ))
        }
    }

    /// Asynchronously selects the cell with the lowest positive entropy.
    /// This reads the result buffer populated by the entropy compute shader.
    pub async fn select_lowest_entropy_cell_async(
        &self,
        _entropy_grid: &EntropyGrid, // Grid itself is not needed as data is on GPU
    ) -> Option<(usize, usize, usize)> {
        debug!("Entering select_lowest_entropy_cell_async");

        // --- Download Min Entropy Info ---
        // Create encoder for copy
        let mut copy_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Copy Min Entropy Info Encoder"),
                });
        copy_encoder.copy_buffer_to_buffer(
            &self.buffers.entropy_buffers.min_entropy_info_buf, // Source: Use entropy_buffers
            0,
            &self.buffers.entropy_buffers.staging_min_entropy_info_buf, // Destination: Use entropy_buffers
            0,
            self.buffers.entropy_buffers.min_entropy_info_buf.size(), // Use entropy_buffers
        );
        self.queue.submit(Some(copy_encoder.finish()));

        // Map and read min entropy info (BLOCKING - Needs Async Version)
        let min_info_result = crate::buffers::map_and_process::<u32>(
            self.buffers
                .entropy_buffers
                .staging_min_entropy_info_buf
                .clone(), // Use entropy_buffers
            std::time::Duration::from_secs(5),
            Some(2), // Expect [f32_bits, u32_index]
        )
        .await;

        match min_info_result {
            Ok(data_box) => {
                let data = data_box.downcast::<Vec<u32>>().ok()?;
                if data.len() < 2 {
                    warn!(
                        "Downloaded min_entropy_info data has insufficient length ({})",
                        data.len()
                    );
                    return None;
                }
                let _min_entropy_bits = data[0];
                let min_index = data[1];

                // Check if a valid minimum was found (index != u32::MAX)
                if min_index != u32::MAX {
                    let (width, height, _depth) = self.grid_dims;
                    let idx = min_index as usize;
                    let z = idx / (width * height);
                    let y = (idx % (width * height)) / width;
                    let x = idx % width;
                    debug!("Selected lowest entropy cell: ({}, {}, {})", x, y, z);
                    Some((x, y, z))
                } else {
                    debug!("No cell with positive entropy found (or grid fully collapsed/contradiction).");
                    None // Grid might be fully collapsed or in a contradiction state
                }
            }
            Err(e) => {
                error!("Failed to download min entropy info: {}", e);
                None
            }
        }
    }
}

impl EntropyCalculator for GpuEntropyCalculator {
    fn calculate_entropy(&self, grid: &PossibilityGrid) -> Result<EntropyGrid, EntropyError> {
        // Use pollster instead of futures::executor::block_on
        pollster::block_on(self.calculate_entropy_async(grid))
    }

    fn select_lowest_entropy_cell(
        &self,
        entropy_grid: &EntropyGrid,
    ) -> Option<(usize, usize, usize)> {
        // Use pollster instead of futures::executor::block_on
        pollster::block_on(self.select_lowest_entropy_cell_async(entropy_grid))
    }

    fn set_entropy_heuristic(&mut self, heuristic_type: EntropyHeuristicType) -> bool {
        self.heuristic_type = heuristic_type;
        true // GPU implementation supports all heuristic types
    }

    fn get_entropy_heuristic(&self) -> EntropyHeuristicType {
        self.heuristic_type
    }
}

impl From<GpuError> for EntropyError {
    fn from(gpu_error: GpuError) -> Self {
        match gpu_error {
            GpuError::BufferOperationError(s)
            | GpuError::CommandExecutionError(s)
            | GpuError::ShaderError(s)
            | GpuError::TransferError(s)
            | GpuError::BufferSizeMismatch(s) => {
                EntropyError::Other(format!("GPU Communication Error: {}", s))
            }
            GpuError::ValidationError(e) => EntropyError::Other(format!("GPU Setup Error: {}", e)),
            GpuError::Other(s) => EntropyError::Other(s),
            _ => EntropyError::Other(format!("Unhandled GpuError: {:?}", gpu_error)),
        }
    }
}

pub trait GpuEntropyCalculatorExt {
    fn select_lowest_entropy_cell_sync(
        &self,
        entropy_grid: &EntropyGrid,
    ) -> Option<(usize, usize, usize)>;
}

impl GpuEntropyCalculatorExt for GpuEntropyCalculator {
    fn select_lowest_entropy_cell_sync(
        &self,
        entropy_grid: &EntropyGrid,
    ) -> Option<(usize, usize, usize)> {
        self.select_lowest_entropy_cell(entropy_grid)
    }
}

/// Maps a GPU error encountered during entropy calculation to a core WFC EntropyError.
fn map_gpu_error_to_entropy_error(gpu_error: GpuError) -> CoreEntropyError {
    CoreEntropyError::Other(format!("GPU Error: {}", gpu_error))
}

#[cfg(test)]
mod tests {
    // ... existing code ...
}
