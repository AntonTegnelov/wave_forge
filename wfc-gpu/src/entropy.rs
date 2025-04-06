use crate::buffers::{EntropyParamsUniform, GpuBuffers};
use crate::pipeline::ComputePipelines;
use log::{trace, warn};
use pollster;
use std::sync::Arc;
use wfc_core::entropy::{EntropyCalculator, EntropyError, EntropyHeuristicType};
use wfc_core::grid::{EntropyGrid, PossibilityGrid};
use wgpu;

/// GPU-accelerated entropy calculator for use in WFC algorithm.
///
/// This component computes the entropy of each cell in the grid and identifies
/// the cell with the minimum positive entropy, which is a key step in the WFC algorithm.
pub struct GpuEntropyCalculator {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipelines: Arc<ComputePipelines>,
    buffers: Arc<GpuBuffers>,
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
        Self::with_heuristic(
            device,
            queue,
            pipelines,
            buffers,
            grid_dims,
            EntropyHeuristicType::default(),
        )
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
        Self {
            device,
            queue,
            pipelines,
            buffers,
            grid_dims,
            heuristic_type,
        }
    }

    /// Asynchronous version of calculate_entropy that allows proper async/await patterns
    pub async fn calculate_entropy_async(
        &self,
        grid: &PossibilityGrid,
    ) -> Result<EntropyGrid, EntropyError> {
        trace!("Entering calculate_entropy_async");
        let (width, height, depth) = (grid.width, grid.height, grid.depth);
        let num_cells = width * height * depth;

        if num_cells == 0 {
            return Ok(EntropyGrid {
                width,
                height,
                depth,
                data: Vec::new(),
            });
        }

        // Convert grid possibilities to u32 arrays and upload them
        let mut packed_data = Vec::with_capacity(num_cells * self.buffers.u32s_per_cell);

        // For each cell, get its bitvector and pack it into u32s
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    if let Some(cell) = grid.get(x, y, z) {
                        let mut cell_data = vec![0u32; self.buffers.u32s_per_cell];
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
                    }
                }
            }
        }

        // Upload the data to the GPU buffer
        self.queue.write_buffer(
            &self.buffers.grid_possibilities_buf,
            0,
            bytemuck::cast_slice(&packed_data),
        );

        // --- Create and configure entropy parameters ---
        let entropy_params = EntropyParamsUniform {
            grid_width: width as u32,
            grid_height: height as u32,
            grid_depth: depth as u32,
            _padding1: 0,
            heuristic_type: match self.heuristic_type {
                EntropyHeuristicType::Shannon => 0,
                EntropyHeuristicType::Count => 1,
                EntropyHeuristicType::CountSimple => 2,
                EntropyHeuristicType::WeightedCount => 3,
            },
            _padding2: 0,
            _padding3: 0,
            _padding4: 0,
        };

        // Write entropy parameters to buffer
        self.queue.write_buffer(
            &self.buffers.entropy_params_buffer,
            0,
            bytemuck::cast_slice(&[entropy_params]),
        );

        // --- Reset min entropy buffer to initial state ---
        if let Err(e) = self.buffers.reset_min_entropy_info(&self.queue) {
            return Err(EntropyError::Other(format!(
                "Failed to reset min entropy: {}",
                e
            )));
        }

        // --- Create bind groups for the entropy shader ---
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
                resource: self.buffers.entropy_params_buffer.as_entire_binding(),
            }],
        });

        // --- Dispatch Compute Shader ---
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Entropy Encoder"),
            });

        // Create bind groups before the compute pass
        let grid_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Grid Possibilities Bind Group"),
            layout: &self.pipelines.entropy_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffers.grid_possibilities_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buffers.entropy_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buffers.params_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buffers.min_entropy_info_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Entropy Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipelines.entropy_pipeline);
            compute_pass.set_bind_group(0, &grid_bind_group, &[]);
            compute_pass.set_bind_group(1, &entropy_params_bind_group, &[]);

            // Dispatch - Calculate workgroup counts
            let workgroup_size = 8; // Must match shader's workgroup_size
            let workgroup_x = (width + workgroup_size - 1) / workgroup_size;
            let workgroup_y = (height + workgroup_size - 1) / workgroup_size;
            let workgroup_z = depth;

            compute_pass.dispatch_workgroups(
                workgroup_x as u32,
                workgroup_y as u32,
                workgroup_z as u32,
            );
        } // End compute pass scope

        // Submit to Queue
        self.queue.submit(std::iter::once(encoder.finish()));

        // Download minimum entropy information
        let _min_entropy_info = match self
            .buffers
            .download_min_entropy_info(self.device.clone(), self.queue.clone())
            .await
        {
            Ok(data) => data,
            Err(e) => {
                return Err(EntropyError::Other(format!(
                    "Failed to download min entropy info: {}",
                    e
                )))
            }
        };

        // Create a new EntropyGrid with calculated values
        let mut entropy_grid = EntropyGrid::new(width, height, depth);

        // Fill with values based on possibility count
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    let _idx = x + y * width + z * width * height;
                    let cell = match grid.get(x, y, z) {
                        Some(c) => c,
                        None => return Err(EntropyError::GridAccessError(x, y, z)),
                    };

                    let count = cell.count_ones();
                    let entropy_value = if count <= 1 {
                        0.0 // Collapsed or contradictory cells
                    } else {
                        match self.heuristic_type {
                            EntropyHeuristicType::Shannon => (count as f32).log2(),
                            EntropyHeuristicType::Count => (count - 1) as f32,
                            EntropyHeuristicType::CountSimple => count as f32 / cell.len() as f32,
                            EntropyHeuristicType::WeightedCount => (count - 1) as f32, // Simple fallback
                        }
                    };

                    *entropy_grid.get_mut(x, y, z).unwrap() = entropy_value;
                }
            }
        }

        Ok(entropy_grid)
    }

    /// Asynchronous version of select_lowest_entropy_cell
    pub async fn select_lowest_entropy_cell_async(
        &self,
        _entropy_grid: &EntropyGrid,
    ) -> Option<(usize, usize, usize)> {
        // Try to get the min entropy cell from the GPU calculation
        let min_entropy_info = match self
            .buffers
            .download_min_entropy_info(self.device.clone(), self.queue.clone())
            .await
        {
            Ok(info) => info,
            Err(e) => {
                warn!("Failed to get min entropy info: {}", e);
                return None;
            }
        };

        // Check if we have valid results
        if min_entropy_info.is_none() {
            return None;
        }

        let (min_entropy, min_idx) = min_entropy_info.unwrap();

        // Check if we have a valid minimum entropy
        if min_entropy <= 0.0 || min_entropy == f32::MAX || min_idx == u32::MAX as usize {
            return None;
        }

        // Convert the flat index to 3D coordinates
        let width = self.grid_dims.0;
        let height = self.grid_dims.1;
        let x = min_idx % width;
        let y = (min_idx / width) % height;
        let z = min_idx / (width * height);

        Some((x, y, z))
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
