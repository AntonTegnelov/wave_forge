use crate::{
    buffers::{GpuBuffers, GpuParamsUniform},
    pipeline::ComputePipelines,
};
use async_trait::async_trait;
use log::{debug, info, warn};
use std::sync::Arc;
use wfc_core::{
    grid::PossibilityGrid,
    propagator::propagator::{ConstraintPropagator, PropagationError},
};
use wfc_rules::AdjacencyRules;

/// GPU implementation of the ConstraintPropagator trait.
#[derive(Debug, Clone)]
pub struct GpuConstraintPropagator {
    // References to shared GPU resources (Reverted to hold individual components)
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) pipelines: Arc<ComputePipelines>,
    pub(crate) buffers: Arc<GpuBuffers>,
    // State for ping-pong buffer index
    current_worklist_idx: usize,
    pub(crate) params: GpuParamsUniform,
}

impl GpuConstraintPropagator {
    /// Creates a new `GpuConstraintPropagator`.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        pipelines: Arc<ComputePipelines>,
        buffers: Arc<GpuBuffers>,
        _grid_dims: (usize, usize, usize),
        _boundary_mode: wfc_core::BoundaryCondition,
        params: GpuParamsUniform,
    ) -> Self {
        Self {
            device,
            queue,
            pipelines,
            buffers,
            current_worklist_idx: 0,
            params,
        }
    }

    /// Gets the binding resource for the current input worklist buffer.
    fn input_worklist_binding(&self) -> wgpu::BindingResource {
        if self.current_worklist_idx == 0 {
            self.buffers.worklist_buf_a.as_entire_binding()
        } else {
            self.buffers.worklist_buf_b.as_entire_binding()
        }
    }

    /// Gets the binding resource for the current output worklist buffer.
    fn output_worklist_binding(&self) -> wgpu::BindingResource {
        if self.current_worklist_idx == 0 {
            // Input is A, Output is B
            self.buffers.worklist_buf_b.as_entire_binding()
        } else {
            // Input is B, Output is A
            self.buffers.worklist_buf_a.as_entire_binding()
        }
    }
}

#[async_trait]
impl ConstraintPropagator for GpuConstraintPropagator {
    /// Performs constraint propagation on the GPU using compute shaders.
    ///
    /// This method handles the propagation of changes across the grid based on
    /// the `updated_coords` provided. It leverages the GPU for parallel processing.
    ///
    /// # Workflow:
    ///
    /// 1.  **Upload Updates**: Writes the `updated_coords` list to the dedicated `updated_cells_buf` on the GPU.
    /// 2.  **Reset Contradiction Flag**: Clears the `contradiction_flag_buf` before running the shader.
    /// 3.  **Shader Dispatch**: Executes the `propagate.wgsl` compute shader.
    ///     -   The shader reads the updated cells.
    ///     -   For each updated cell, it examines neighbors based on `rules_buf`.
    ///     -   It calculates the intersection of allowed tiles and updates the main `grid_possibilities_buf`.
    ///     -   If a contradiction (empty possibility set) is detected, it sets the `contradiction_flag_buf`.
    ///     -   (Note: The current shader might perform a fixed number of iterations or a simplified propagation step).
    /// 4.  **Synchronization**: Submits the command buffer and waits for the GPU to finish (implicitly via buffer mapping or explicit polling if needed).
    /// 5.  **Check Contradiction**: Reads the `contradiction_flag_buf` back to the CPU.
    /// 6.  **Return Result**: Returns `Ok(())` if the contradiction flag is not set, or `Err(PropagationError::Contradiction(x, y, z))`
    ///     if a contradiction was detected (coordinates might be approximate or the first detected one).
    ///
    /// # Arguments
    ///
    /// * `_grid`: Mutable reference to the CPU-side `PossibilityGrid`. **Note:** This implementation primarily operates
    ///            on the GPU buffers (`self.buffers`). The CPU grid is *not* directly modified by this function.
    ///            The caller is responsible for potentially reading back the updated grid state from the GPU
    ///            using separate buffer reading methods if needed after the run completes.
    /// * `updated_coords`: A vector of `(x, y, z)` coordinates indicating cells whose possibilities have recently changed
    ///                     (e.g., due to a collapse) and need to be propagated from.
    /// * `_rules`: Reference to the `AdjacencyRules`. **Note:** This is not directly used, as the rules are assumed
    ///             to be already present in `self.buffers.rules_buf` on the GPU.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the propagation shader executed successfully without detecting a contradiction.
    /// * `Err(PropagationError::Contradiction(x, y, z))` if the shader detected and flagged a contradiction.
    ///   The coordinates might indicate the first cell where the contradiction was found.
    /// * `Err(PropagationError::Gpu*Error)` if a GPU-specific error occurs during buffer updates or shader execution,
    ///   wrapped within the `PropagationError` enum.
    async fn propagate(
        &mut self,
        _grid: &mut PossibilityGrid,
        updated_coords: Vec<(usize, usize, usize)>,
        _rules: &AdjacencyRules,
    ) -> Result<(), PropagationError> {
        info!(
            "Starting GPU propagation with {} initial update(s).",
            updated_coords.len()
        );

        // 1. Upload initial updates to the worklist buffer (e.g., worklist_buf_a)
        let initial_updates: Vec<u32> = updated_coords
            .into_iter()
            .map(|(x, y, z)| {
                (z * self.params.grid_height as usize * self.params.grid_width as usize
                    + y * self.params.grid_width as usize
                    + x) as u32
            })
            .collect();

        // Decide initial buffer index (e.g., 0 for buf_a)
        self.current_worklist_idx = 0; // Start with buffer A
        let mut current_worklist_size = initial_updates.len() as u32;

        self.buffers
            .upload_initial_updates(&self.queue, &initial_updates, self.current_worklist_idx)
            .map_err(|e| PropagationError::GpuSetupError(e.to_string()))?;

        // Set initial worklist size in params uniform
        self.buffers
            .update_params_worklist_size(&self.queue, current_worklist_size)
            .map_err(|e| PropagationError::GpuSetupError(e.to_string()))?;

        let mut propagation_pass = 0;
        const MAX_PROPAGATION_PASSES: u32 = 100; // Safeguard against infinite loops

        while current_worklist_size > 0 && propagation_pass < MAX_PROPAGATION_PASSES {
            propagation_pass += 1;
            debug!(
                "Propagation Pass {}: Worklist size = {}",
                propagation_pass, current_worklist_size
            );

            // Create bind group for current pass using our worklist binding methods
            let input_worklist_resource = self.input_worklist_binding();
            let output_worklist_resource = self.output_worklist_binding();

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Propagation Bind Group Pass {}", propagation_pass)),
                layout: &self.pipelines.propagation_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffers.grid_possibilities_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.buffers.rules_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: input_worklist_resource,
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_worklist_resource,
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.buffers.params_uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.buffers.contradiction_flag_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.buffers.worklist_count_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: self.buffers.contradiction_location_buf.as_entire_binding(),
                    },
                ],
            });

            // Reset output worklist count and contradiction flag/location
            self.buffers
                .reset_worklist_count(&self.queue)
                .map_err(|e| PropagationError::GpuSetupError(e.to_string()))?;
            self.buffers
                .reset_contradiction_flag(&self.queue)
                .map_err(|e| PropagationError::GpuSetupError(e.to_string()))?;
            self.buffers
                .reset_contradiction_location(&self.queue)
                .map_err(|e| PropagationError::GpuSetupError(e.to_string()))?;

            // --- Dispatch Propagation Compute Shader ---
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("Propagation Encoder Pass {}", propagation_pass)),
                });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Propagation Compute Pass {}", propagation_pass)),
                    timestamp_writes: None, // Add timestamps if needed
                });
                compute_pass.set_pipeline(&self.pipelines.propagation_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                // Calculate dispatch size based on worklist size
                let workgroup_size = 64; // Should match shader
                let dispatch_x = (current_worklist_size + workgroup_size - 1) / workgroup_size;
                compute_pass.dispatch_workgroups(dispatch_x, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));

            // --- Download results (Contradiction Flag, New Worklist Size) ---
            // Consider making download_results async if pollster is removed later.
            let results = self
                .buffers
                .download_results(
                    self.device.clone(),
                    self.queue.clone(),
                    false, // entropy
                    false, // min_entropy
                    true,  // contradiction_flag
                    false, // grid
                    true,  // worklist_count
                    true,  // download_contradiction_location
                )
                .await
                .map_err(|e| PropagationError::GpuCommunicationError(e.to_string()))?;

            // Check for contradiction
            if let Some(true) = results.contradiction_flag {
                // If contradiction found, try to get location
                let location_index = results.contradiction_location.unwrap_or(u32::MAX);
                let (x, y, z) = if location_index != u32::MAX {
                    let width = self.params.grid_width as usize;
                    let height = self.params.grid_height as usize;
                    let z = location_index as usize / (width * height);
                    let y = (location_index as usize % (width * height)) / width;
                    let x = location_index as usize % width;
                    (x, y, z)
                } else {
                    (usize::MAX, usize::MAX, usize::MAX) // Indicate unknown location
                };
                warn!(
                    "GPU propagation detected contradiction at index {} ({},{},{}).",
                    location_index, x, y, z
                );
                return Err(PropagationError::Contradiction(x, y, z));
            }

            // Update worklist size for the next iteration
            current_worklist_size = results.worklist_count.unwrap_or(0);
            log::trace!(
                "GPU Propagation Iteration {}: New worklist size = {}",
                propagation_pass,
                current_worklist_size
            );

            // Ping-pong buffers for next iteration
            self.current_worklist_idx = 1 - self.current_worklist_idx;
        }

        // Safety break check
        if propagation_pass >= MAX_PROPAGATION_PASSES && current_worklist_size > 0 {
            warn!("GPU Propagation reached max passes ({}) with remaining worklist size {}. Potential infinite loop or slow convergence.", MAX_PROPAGATION_PASSES, current_worklist_size);
            // Decide whether to error or continue
            // return Err(PropagationError::InternalError("Max propagation passes reached".to_string()));
        }

        info!(
            "GPU propagation finished after {} passes.",
            propagation_pass
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // Empty test module - will be implemented later
}
