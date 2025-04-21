use crate::shader::pipeline::ComputePipelines;
use crate::{buffers::GpuBuffers, gpu::sync::GpuSynchronizer, utils::error_recovery::GridCoord};
use async_trait;
use std::default::Default;
use std::sync::Arc;
use wfc_core::{grid::PossibilityGrid, propagator::PropagationError};

/// Direct propagation strategy - propagates constraints directly across
/// the entire grid without any partitioning or optimization.
#[derive(Debug)]
pub struct DirectPropagationStrategy {
    name: String,
    #[allow(dead_code)]
    max_iterations: u32,
    pipelines: Arc<ComputePipelines>,
}

impl DirectPropagationStrategy {
    /// Create a new direct propagation strategy
    pub fn new(max_iterations: u32, pipelines: Arc<ComputePipelines>) -> Self {
        Self {
            name: "Direct Propagation".to_string(),
            max_iterations,
            pipelines,
        }
    }

    /// Helper method to create a bind group for a propagation pass
    fn create_propagation_bind_group_for_pass(
        &self,
        device: &wgpu::Device,
        buffers: &GpuBuffers,
        current_worklist_idx: usize,
    ) -> wgpu::BindGroup {
        // Get the appropriate worklist buffers based on current index
        let (input_worklist, output_worklist) = if current_worklist_idx == 0 {
            (
                &buffers.worklist_buffers.worklist_buf_a,
                &buffers.worklist_buffers.worklist_buf_b,
            )
        } else {
            (
                &buffers.worklist_buffers.worklist_buf_b,
                &buffers.worklist_buffers.worklist_buf_a,
            )
        };

        // Create bind group
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Propagation Pass Bind Group"),
            layout: &self.pipelines.propagation_bind_group_layout,
            entries: &[
                // Bind the params uniform buffer
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.params_uniform_buf.as_entire_binding(),
                },
                // Bind the grid possibilities buffer
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers
                        .grid_buffers
                        .grid_possibilities_buf
                        .as_entire_binding(),
                },
                // Bind the input worklist buffer
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: input_worklist.as_entire_binding(),
                },
                // Bind the output worklist buffer
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_worklist.as_entire_binding(),
                },
                // Bind the worklist count buffer
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers
                        .worklist_buffers
                        .worklist_count_buf
                        .as_entire_binding(),
                },
                // Bind the contradiction flag buffer
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.contradiction_flag_buf.as_entire_binding(),
                },
                // Bind the adjacency rules buffer
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.rule_buffers.rules_buf.as_entire_binding(),
                },
            ],
        })
    }
}

/// Implement Default trait for DirectPropagationStrategy
impl Default for DirectPropagationStrategy {
    fn default() -> Self {
        unimplemented!(
            "DirectPropagationStrategy requires pipelines, cannot be created with default()"
        )
    }
}

impl crate::propagator::PropagationStrategy for DirectPropagationStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn prepare(&self, _synchronizer: &GpuSynchronizer) -> Result<(), PropagationError> {
        // Direct propagation doesn't need special preparation
        Ok(())
    }

    fn cleanup(&self, _synchronizer: &GpuSynchronizer) -> Result<(), PropagationError> {
        // Direct propagation doesn't need special cleanup
        Ok(())
    }
}

#[async_trait::async_trait]
impl crate::propagator::AsyncPropagationStrategy for DirectPropagationStrategy {
    async fn propagate(
        &self,
        grid: &mut PossibilityGrid,
        updated_cells: &[GridCoord],
        buffers: &Arc<GpuBuffers>,
        synchronizer: &GpuSynchronizer,
    ) -> Result<(), PropagationError> {
        // Skip if no cells to process
        if updated_cells.is_empty() {
            return Ok(());
        }

        let device = synchronizer.device();
        let queue = synchronizer.queue();

        // Reset contradiction flag
        queue.write_buffer(
            &buffers.contradiction_flag_buf,
            0,
            bytemuck::cast_slice(&[0u32]),
        );

        // Convert updated cells to indices and write to worklist
        let updated_indices: Vec<u32> = updated_cells
            .iter()
            .map(|coord| {
                (coord.x + coord.y * grid.width + coord.z * grid.width * grid.height) as u32
            })
            .collect();

        // Write initial worklist
        queue.write_buffer(
            &buffers.worklist_buffers.worklist_buf_a,
            0,
            bytemuck::cast_slice(&updated_indices),
        );
        queue.write_buffer(
            &buffers.worklist_buffers.worklist_count_buf,
            0,
            bytemuck::cast_slice(&[updated_indices.len() as u32]),
        );

        let mut current_worklist_idx = 0;
        let mut iteration = 0;

        // Main propagation loop
        loop {
            iteration += 1;
            if iteration > self.max_iterations {
                return Err(PropagationError::InternalError(
                    "Maximum propagation iterations reached".to_string(),
                ));
            }

            // Create bind group for this pass
            let bind_group =
                self.create_propagation_bind_group_for_pass(device, buffers, current_worklist_idx);

            // Create command encoder
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Propagation Pass Encoder"),
            });

            // Begin compute pass
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Propagation Pass"),
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(&self.pipelines.propagation_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                // Dispatch workgroups
                let num_workgroups = (updated_indices.len() as u32 + 63) / 64;
                compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
            }

            // Submit commands
            queue.submit(Some(encoder.finish()));

            // Check for contradictions
            let contradiction_flag: Vec<u32> = synchronizer
                .download_buffer(
                    &buffers.contradiction_flag_buf,
                    0,
                    std::mem::size_of::<u32>(),
                )
                .map_err(|e| PropagationError::GpuCommunicationError(e.to_string()))?;

            if contradiction_flag[0] != 0 {
                return Err(PropagationError::Contradiction(0, 0, 0));
            }

            // Check output worklist size
            let output_worklist_count: Vec<u32> = synchronizer
                .download_buffer(
                    &buffers.worklist_buffers.worklist_count_buf,
                    0,
                    std::mem::size_of::<u32>(),
                )
                .map_err(|e| PropagationError::GpuCommunicationError(e.to_string()))?;

            // If no more cells to process, we're done
            if output_worklist_count[0] == 0 {
                break;
            }

            // Swap worklists for next iteration
            current_worklist_idx = 1 - current_worklist_idx;
        }

        Ok(())
    }
}
