use crate::shader::pipeline::ComputePipelines;
use crate::{
    buffers::{GpuBuffers, GpuParamsUniform},
    gpu::sync::GpuSynchronizer,
    utils::error_recovery::GridCoord,
};
use async_trait;
use std::sync::Arc;
use wfc_core::{grid::PossibilityGrid, propagator::PropagationError};

/// Workgroup size (X) declared by `propagate.wgsl`.
const PROPAGATION_WORKGROUP_SIZE: u32 = 64;

#[derive(Debug)]
pub struct DirectPropagationStrategy {
    name: String,
    max_iterations: u32,
    pipelines: Arc<ComputePipelines>,
    /// Dispatches the given number of passes without reading anything back, for measuring how much
    /// of a pass is GPU work and how much is waiting for the two round-trips it normally performs.
    /// Only a benchmark sets this; it produces no useful grid.
    blind_passes: Option<u32>,
    /// With `blind_passes`, records every dispatch into one encoder and submits once, to separate
    /// per-submit cost from per-dispatch encoding cost.
    blind_single_submit: bool,
}

impl DirectPropagationStrategy {
    pub fn new(max_iterations: u32, pipelines: Arc<ComputePipelines>) -> Self {
        Self {
            name: "Direct Propagation".to_string(),
            max_iterations,
            pipelines,
            blind_passes: None,
            blind_single_submit: false,
        }
    }

    /// Runs `passes` dispatches over the whole grid without the per-pass readbacks. Benchmark only:
    /// the result is not a fixpoint and contradictions go unnoticed. See docs/performance.md.
    #[doc(hidden)]
    pub fn benchmark_blind_passes(
        max_iterations: u32,
        pipelines: Arc<ComputePipelines>,
        passes: u32,
    ) -> Self {
        Self {
            blind_passes: Some(passes),
            ..Self::new(max_iterations, pipelines)
        }
    }

    /// Like [`Self::benchmark_blind_passes`], but all dispatches go into one command buffer.
    #[doc(hidden)]
    pub fn benchmark_blind_batched(
        max_iterations: u32,
        pipelines: Arc<ComputePipelines>,
        passes: u32,
    ) -> Self {
        Self {
            blind_passes: Some(passes),
            blind_single_submit: true,
            ..Self::new(max_iterations, pipelines)
        }
    }

    /// Records `passes` dispatches into one encoder and submits them together.
    fn run_passes_batched(
        &self,
        buffers: &GpuBuffers,
        synchronizer: &GpuSynchronizer,
        input_count: u32,
        passes: u32,
    ) {
        let _span = tracing::info_span!("propagation_passes_batched", passes).entered();
        let device = synchronizer.device();
        let queue = synchronizer.queue();
        queue.write_buffer(
            &buffers.params_uniform_buf,
            std::mem::offset_of!(GpuParamsUniform, worklist_size) as u64,
            bytemuck::bytes_of(&input_count),
        );
        let bind_groups = [
            self.create_propagation_bind_group_for_pass(device, buffers, 0),
            self.create_propagation_bind_group_for_pass(device, buffers, 1),
        ];
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Propagation Passes Encoder (batched)"),
        });
        for pass in 0..passes {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Propagation Pass (batched)"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipelines.propagation_pipeline);
            compute_pass.set_bind_group(0, &bind_groups[(pass % 2) as usize], &[]);
            compute_pass.dispatch_workgroups(
                input_count.div_ceil(PROPAGATION_WORKGROUP_SIZE),
                1,
                1,
            );
        }
        queue.submit(Some(encoder.finish()));
    }

    /// Submits one pass without reading the contradiction flag or the worklist count.
    fn run_pass_blind(
        &self,
        buffers: &GpuBuffers,
        synchronizer: &GpuSynchronizer,
        worklist_idx: usize,
        input_count: u32,
    ) {
        let _span = tracing::info_span!("propagation_pass_blind", input_count).entered();
        let device = synchronizer.device();
        let queue = synchronizer.queue();
        queue.write_buffer(
            &buffers.params_uniform_buf,
            std::mem::offset_of!(GpuParamsUniform, worklist_size) as u64,
            bytemuck::bytes_of(&input_count),
        );
        queue.write_buffer(
            &buffers.worklist_buffers.worklist_count_buf,
            0,
            bytemuck::bytes_of(&0u32),
        );
        let bind_group = self.create_propagation_bind_group_for_pass(device, buffers, worklist_idx);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Propagation Pass Encoder (blind)"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Propagation Pass (blind)"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipelines.propagation_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(
                input_count.div_ceil(PROPAGATION_WORKGROUP_SIZE),
                1,
                1,
            );
        }
        queue.submit(Some(encoder.finish()));
    }

    fn worklist_buffer(buffers: &GpuBuffers, worklist_idx: usize) -> &wgpu::Buffer {
        if worklist_idx == 0 {
            &buffers.worklist_buffers.worklist_buf_a
        } else {
            &buffers.worklist_buffers.worklist_buf_b
        }
    }

    fn create_propagation_bind_group_for_pass(
        &self,
        device: &wgpu::Device,
        buffers: &GpuBuffers,
        current_worklist_idx: usize,
    ) -> wgpu::BindGroup {
        let input_worklist = Self::worklist_buffer(buffers, current_worklist_idx);
        let output_worklist = Self::worklist_buffer(buffers, 1 - current_worklist_idx);

        // Binding order must match propagate.wgsl
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Propagation Pass Bind Group"),
            layout: &self.pipelines.propagation_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers
                        .grid_buffers
                        .grid_possibilities_buf
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.rule_buffers.rules_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.rule_buffers.rule_weights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: input_worklist.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_worklist.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.params_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers
                        .worklist_buffers
                        .worklist_count_buf
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffers.contradiction_flag_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: buffers.contradiction_location_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: buffers.pass_statistics_buf.as_entire_binding(),
                },
            ],
        })
    }

    fn download_u32(
        synchronizer: &GpuSynchronizer,
        buffer: &wgpu::Buffer,
    ) -> Result<u32, PropagationError> {
        let data: Vec<u32> = synchronizer
            .download_buffer(buffer, 0, std::mem::size_of::<u32>())
            .map_err(|e| PropagationError::GpuCommunicationError(e.to_string()))?;
        data.first().copied().ok_or_else(|| {
            PropagationError::GpuCommunicationError("Empty buffer download".to_string())
        })
    }

    /// Runs one propagation pass over the first `input_count` cells of the active worklist.
    /// Returns how many neighbour updates the pass queued into the other worklist.
    fn run_pass(
        &self,
        buffers: &GpuBuffers,
        synchronizer: &GpuSynchronizer,
        worklist_idx: usize,
        input_count: u32,
    ) -> Result<u32, PropagationError> {
        let _span = tracing::info_span!("propagation_pass", input_count).entered();
        let device = synchronizer.device();
        let queue = synchronizer.queue();

        queue.write_buffer(
            &buffers.params_uniform_buf,
            std::mem::offset_of!(GpuParamsUniform, worklist_size) as u64,
            bytemuck::bytes_of(&input_count),
        );
        queue.write_buffer(
            &buffers.worklist_buffers.worklist_count_buf,
            0,
            bytemuck::bytes_of(&0u32),
        );

        let bind_group = self.create_propagation_bind_group_for_pass(device, buffers, worklist_idx);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Propagation Pass Encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Propagation Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipelines.propagation_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(
                input_count.div_ceil(PROPAGATION_WORKGROUP_SIZE),
                1,
                1,
            );
        }
        queue.submit(Some(encoder.finish()));

        if Self::download_u32(synchronizer, &buffers.contradiction_flag_buf)? != 0 {
            let flat_index =
                Self::download_u32(synchronizer, &buffers.contradiction_location_buf)? as usize;
            let (width, height, _) = buffers.grid_dims;
            return Err(PropagationError::Contradiction(
                flat_index % width,
                (flat_index / width) % height,
                flat_index / (width * height),
            ));
        }

        Self::download_u32(synchronizer, &buffers.worklist_buffers.worklist_count_buf)
    }
}

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
        Ok(())
    }

    fn cleanup(&self, _synchronizer: &GpuSynchronizer) -> Result<(), PropagationError> {
        Ok(())
    }
}

#[async_trait::async_trait]
impl crate::propagator::AsyncPropagationStrategy for DirectPropagationStrategy {
    async fn propagate(
        &self,
        _grid: &mut PossibilityGrid,
        updated_cells: &[GridCoord],
        buffers: &Arc<GpuBuffers>,
        synchronizer: &GpuSynchronizer,
    ) -> Result<(), PropagationError> {
        if updated_cells.is_empty() {
            return Ok(());
        }

        let queue = synchronizer.queue();
        let (width, height, depth) = buffers.grid_dims;
        let num_cells = (width * height * depth) as u32;
        let capacity = (buffers.worklist_buffers.worklist_buf_a.size()
            / std::mem::size_of::<u32>() as u64) as u32;
        let all_cells: Vec<u32> = (0..num_cells).collect();

        queue.write_buffer(
            &buffers.contradiction_flag_buf,
            0,
            bytemuck::bytes_of(&0u32),
        );

        let initial: Vec<u32> = updated_cells
            .iter()
            .map(|c| (c.x + c.y * width + c.z * width * height) as u32)
            .collect();
        let mut worklist_idx = 0;
        queue.write_buffer(
            Self::worklist_buffer(buffers, worklist_idx),
            0,
            bytemuck::cast_slice(&initial),
        );
        let mut input_count = initial.len() as u32;

        if let Some(passes) = self.blind_passes {
            if self.blind_single_submit {
                self.run_passes_batched(buffers, synchronizer, input_count, passes);
                let _ = synchronizer
                    .device()
                    .poll(wgpu::PollType::wait_indefinitely());
                return Ok(());
            }
            for _ in 0..passes {
                self.run_pass_blind(buffers, synchronizer, worklist_idx, input_count);
                worklist_idx = 1 - worklist_idx;
            }
            // One wait at the end, so the timing covers work the GPU actually finished.
            let _ = synchronizer
                .device()
                .poll(wgpu::PollType::wait_indefinitely());
            return Ok(());
        }

        for _ in 0..self.max_iterations {
            let queued = self.run_pass(buffers, synchronizer, worklist_idx, input_count)?;

            if queued == 0 {
                // The shader restricts neighbours with atomicAnd, so no update can be lost and an
                // empty worklist is a real fixpoint. This used to need a confirming sweep over every
                // cell, which was half of all propagation passes (docs/solver-fit.md).
                return Ok(());
            }

            worklist_idx = 1 - worklist_idx;
            if queued > capacity {
                // The output worklist overflowed and dropped entries, so re-check every cell.
                queue.write_buffer(
                    Self::worklist_buffer(buffers, worklist_idx),
                    0,
                    bytemuck::cast_slice(&all_cells),
                );
                input_count = num_cells;
            } else {
                input_count = queued;
            }
        }

        Err(PropagationError::InternalError(format!(
            "Propagation did not converge within {} passes",
            self.max_iterations
        )))
    }
}
