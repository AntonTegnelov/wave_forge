use crate::{buffers::GpuBuffers, pipeline::ComputePipelines};
use log;
use std::sync::Arc;
use wfc_core::{
    grid::PossibilityGrid,
    propagator::{ConstraintPropagator, PropagationError},
};
use wfc_rules::AdjacencyRules;

const MAX_PROPAGATION_ITERATIONS: u32 = 100; // Safeguard against infinite loops
const PROPAGATION_BATCH_SIZE: usize = 4096; // Number of updates to process per GPU dispatch

/// GPU implementation of the ConstraintPropagator trait.
/// Holds references to GPU resources needed for constraint propagation.
#[derive(Clone)] // Make it cloneable if needed
pub struct GpuConstraintPropagator {
    // References to shared GPU resources
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) pipelines: Arc<ComputePipelines>, // Assume ComputePipelines is shareable/Clone
    pub(crate) buffers: Arc<GpuBuffers>,         // Assume GpuBuffers is shareable/Clone
    pub(crate) grid_dims: (usize, usize, usize),
}

impl GpuConstraintPropagator {
    /// Creates a new `GpuConstraintPropagator`.
    ///
    /// This typically takes resources initialized elsewhere (e.g., by a main GPU manager).
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        pipelines: Arc<ComputePipelines>,
        buffers: Arc<GpuBuffers>,
        grid_dims: (usize, usize, usize),
    ) -> Self {
        Self {
            device,
            queue,
            pipelines,
            buffers,
            grid_dims,
        }
    }
}

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
    fn propagate(
        &mut self,
        _grid: &mut PossibilityGrid,
        updated_coords: Vec<(usize, usize, usize)>,
        _rules: &AdjacencyRules,
    ) -> Result<(), PropagationError> {
        log::debug!("Starting GPU iterative propagation...");
        let propagate_start = std::time::Instant::now();

        let (width, height, _depth) = self.grid_dims;
        let _max_worklist_size = (width * height * _depth) as u64; // Max possible work items

        // --- 1. Prepare Initial Worklist ---
        let mut current_worklist: Vec<u32> = updated_coords
            .iter()
            .map(|&(x, y, z)| (z * width * height + y * width + x) as u32)
            .collect();

        if current_worklist.is_empty() {
            log::debug!("GPU propagate: No initial updates to process.");
            return Ok(());
        }

        let mut iteration = 0;
        let mut total_dispatches = 0;

        // --- 2. Iterative Propagation Loop ---
        while !current_worklist.is_empty() {
            iteration += 1;
            log::debug!(
                "GPU Propagation Iteration: {}, Worklist Size: {}",
                iteration,
                current_worklist.len()
            );
            if iteration > MAX_PROPAGATION_ITERATIONS {
                log::error!(
                    "GPU propagation exceeded max iterations ({}), assuming divergence.",
                    MAX_PROPAGATION_ITERATIONS
                );
                return Err(PropagationError::GpuCommunicationError(
                    "Propagation exceeded max iterations".to_string(),
                ));
            }

            let mut next_worklist_indices = Vec::new(); // Collect indices for the *next* iteration
            let mut contradiction_found_in_iteration = false;
            let mut contradiction_location = u32::MAX;

            // Process the current worklist in batches
            for batch in current_worklist.chunks(PROPAGATION_BATCH_SIZE) {
                total_dispatches += 1;
                let batch_size = batch.len() as u32;
                log::trace!(
                    "  Dispatching Batch ({} updates), Total Dispatches: {}",
                    batch_size,
                    total_dispatches
                );

                // --- 2a. Upload Current Batch & Reset Buffers ---
                self.buffers
                    .upload_updates(&self.queue, batch)
                    .map_err(|e| {
                        PropagationError::GpuCommunicationError(format!(
                            "Failed to upload update batch: {}",
                            e
                        ))
                    })?;

                self.buffers
                    .reset_contradiction_flag(&self.queue)
                    .map_err(|e| {
                        PropagationError::GpuCommunicationError(format!(
                            "Failed to reset contradiction flag: {}",
                            e
                        ))
                    })?;

                self.buffers
                    .reset_contradiction_location(&self.queue)
                    .map_err(|e| {
                        PropagationError::GpuCommunicationError(format!(
                            "Failed to reset contradiction location: {}",
                            e
                        ))
                    })?;

                self.buffers
                    .reset_output_worklist_count(&self.queue)
                    .map_err(|e| {
                        PropagationError::GpuCommunicationError(format!(
                            "Failed to reset output worklist count: {}",
                            e
                        ))
                    })?;

                // --- 2b. Update Uniforms ---
                self.buffers
                    .update_params_worklist_size(&self.queue, batch_size)
                    .map_err(|e| {
                        PropagationError::GpuCommunicationError(format!(
                            "Failed to update worklist size uniform: {}",
                            e
                        ))
                    })?;

                // --- 2c. Create Command Encoder & Bind Group ---
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some(&format!("Propagation Encoder Iter {} Batch", iteration)),
                        });

                let propagation_bind_group =
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(&format!("Propagation Bind Group Iter {} Batch", iteration)),
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
                                resource: self.buffers.updates_buf.as_entire_binding(),
                            }, // Input worklist (current batch)
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: self.buffers.output_worklist_buf.as_entire_binding(),
                            }, // Output worklist (for *next* iteration)
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: self.buffers.params_uniform_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: self
                                    .buffers
                                    .output_worklist_count_buf
                                    .as_entire_binding(),
                            }, // Output count
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: self.buffers.contradiction_flag_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 7,
                                resource: self
                                    .buffers
                                    .contradiction_location_buf
                                    .as_entire_binding(),
                            },
                        ],
                    });

                // --- 2d. Dispatch Compute ---
                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some(&format!(
                                "Propagation Compute Pass Iter {} Batch",
                                iteration
                            )),
                            timestamp_writes: None,
                        });
                    compute_pass.set_pipeline(&self.pipelines.propagation_pipeline);
                    compute_pass.set_bind_group(0, &propagation_bind_group, &[]);

                    let workgroup_size = 64u32;
                    let workgroups_needed = std::cmp::max(1, batch_size.div_ceil(workgroup_size));
                    compute_pass.dispatch_workgroups(workgroups_needed, 1, 1);
                } // End compute pass

                // --- 2e. Submit and Wait ---
                self.queue.submit(std::iter::once(encoder.finish()));
                self.device.poll(wgpu::Maintain::Wait);

                // --- 2f. Check for Contradiction (after this batch) ---
                let contradiction_in_batch = pollster::block_on(
                    self.buffers
                        .download_contradiction_flag(&self.device, &self.queue),
                )
                .map_err(|e| {
                    PropagationError::GpuCommunicationError(format!(
                        "Failed to download contradiction flag: {}",
                        e
                    ))
                })?;

                if contradiction_in_batch {
                    log::warn!(
                        "Contradiction detected during GPU propagation (Iteration {}, Batch Dispatch {})!",
                        iteration, total_dispatches
                    );
                    contradiction_found_in_iteration = true;
                    // Download location
                    contradiction_location = pollster::block_on(
                        self.buffers
                            .download_contradiction_location(&self.device, &self.queue),
                    )
                    .map_err(|e| {
                        PropagationError::GpuCommunicationError(format!(
                            "Failed to download contradiction location: {}",
                            e
                        ))
                    })?;
                    break; // Stop processing batches for this iteration if contradiction found
                }

                // --- 2g. Download Output Worklist Count (for this batch) ---
                let output_count = pollster::block_on(
                    self.buffers
                        .download_output_worklist_count(&self.device, &self.queue),
                )
                .map_err(|e| {
                    PropagationError::GpuCommunicationError(format!(
                        "Failed to download output worklist count: {}",
                        e
                    ))
                })?;

                // --- 2h. Download Output Worklist Indices (for this batch) ---
                if output_count > 0 {
                    let output_indices = self
                        .download_output_worklist(output_count as usize)
                        .map_err(|e| {
                            PropagationError::GpuCommunicationError(format!(
                                "Failed to download output worklist indices: {}",
                                e
                            ))
                        })?;
                    next_worklist_indices.extend(output_indices);
                }
            } // End batch loop

            // --- Check Contradiction after all batches for the iteration ---
            if contradiction_found_in_iteration {
                // Convert flat index to 3D coordinates
                let (x, y, z) = if contradiction_location != u32::MAX {
                    let (grid_w, grid_h, _) = self.grid_dims;
                    let z_coord = contradiction_location / (grid_w * grid_h) as u32;
                    let rem = contradiction_location % (grid_w * grid_h) as u32;
                    let y_coord = rem / grid_w as u32;
                    let x_coord = rem % grid_w as u32;
                    (x_coord as usize, y_coord as usize, z_coord as usize)
                } else {
                    (0, 0, 0) // Default/fallback location if specific one wasn't captured
                };
                log::error!(
                    "GPU Propagation failed due to contradiction at ({}, {}, {}).",
                    x,
                    y,
                    z
                );
                return Err(PropagationError::Contradiction(x, y, z));
            }

            // --- Prepare for Next Iteration ---
            // Deduplicate indices collected from all batches
            next_worklist_indices.sort_unstable();
            next_worklist_indices.dedup();
            current_worklist = next_worklist_indices;
        } // End while loop

        let duration = propagate_start.elapsed();
        log::debug!(
            "GPU iterative propagation finished in {:?} ({} iterations, {} dispatches).",
            duration,
            iteration,
            total_dispatches
        );
        Ok(())
    }
} // <<< END of impl ConstraintPropagator for GpuConstraintPropagator

// Helper needed to download the output worklist
impl GpuConstraintPropagator {
    async fn download_output_worklist_async(
        &self,
        count: usize,
    ) -> Result<Vec<u32>, PropagationError> {
        if count == 0 {
            return Ok(Vec::new());
        }

        let buffer_size = (count * std::mem::size_of::<u32>()) as u64;
        if buffer_size > self.buffers.output_worklist_buf.size() {
            return Err(PropagationError::GpuCommunicationError(format!(
                "Requested output worklist download size ({}) exceeds buffer size ({})",
                buffer_size,
                self.buffers.output_worklist_buf.size()
            )));
        }

        // Create a temporary staging buffer for download
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Output Worklist Download"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Output Worklist Download Encoder"),
            });

        encoder.copy_buffer_to_buffer(
            &self.buffers.output_worklist_buf, // Source: GPU output worklist
            0,
            &staging_buffer, // Destination: Staging buffer
            0,
            buffer_size,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map the staging buffer
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender
                .send(result)
                .expect("Failed to send map result for output worklist");
        });

        self.device.poll(wgpu::Maintain::Wait); // Wait for GPU

        match receiver.receive().await {
            Some(Ok(())) => {
                let data = buffer_slice.get_mapped_range();
                let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
                drop(data); // Unmap before buffer is potentially dropped
                staging_buffer.unmap();
                Ok(result)
            }
            Some(Err(e)) => {
                staging_buffer.unmap(); // Attempt unmap on error
                Err(PropagationError::GpuCommunicationError(format!(
                    "Failed to map output worklist staging buffer: {}",
                    e
                )))
            }
            None => {
                staging_buffer.unmap(); // Attempt unmap on error
                Err(PropagationError::GpuCommunicationError(
                    "Output worklist map future cancelled".to_string(),
                ))
            }
        }
    }

    fn download_output_worklist(&self, count: usize) -> Result<Vec<u32>, PropagationError> {
        pollster::block_on(self.download_output_worklist_async(count))
    }
}

// Ensure necessary types (ComputePipelines, GpuBuffers) are cloneable and fields are accessible
// No code here, just a comment
