use crate::{buffers::GpuBuffers, pipeline::ComputePipelines};
use log;
use std::sync::Arc;
use wfc_core::{
    grid::PossibilityGrid,
    propagator::{ConstraintPropagator, PropagationError},
    BoundaryMode,
};
use wfc_rules::AdjacencyRules;

const MAX_PROPAGATION_ITERATIONS: u32 = 100; // Safeguard against infinite loops
const PROPAGATION_BATCH_SIZE: usize = 4096; // Number of updates to process per GPU dispatch
const PROPAGATION_WORKGROUP_SIZE: u32 = 64; // Match the value in propagate.wgsl

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
    pub(crate) boundary_mode: BoundaryMode,
    // Add state for ping-pong buffer index
    current_worklist_idx: usize,
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
        boundary_mode: BoundaryMode,
    ) -> Self {
        Self {
            device,
            queue,
            pipelines,
            buffers,
            grid_dims,
            boundary_mode,
            current_worklist_idx: 0, // Start with buffer A
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

        // --- 1. Upload Initial Worklist ---
        let initial_worklist: Vec<u32> = updated_coords
            .iter()
            .map(|&(x, y, z)| (z * width * height + y * width + x) as u32)
            .collect();

        if initial_worklist.is_empty() {
            log::debug!("GPU propagate: No initial updates to process.");
            return Ok(());
        }

        // Upload to the starting buffer (e.g., buffer A)
        self.buffers
            .upload_initial_updates(&self.queue, &initial_worklist, self.current_worklist_idx)
            .map_err(|e| {
                PropagationError::GpuCommunicationError(format!(
                    "Failed to upload initial updates: {}",
                    e
                ))
            })?;

        let mut current_worklist_size = initial_worklist.len() as u32;
        let mut iteration = 0;
        let mut total_dispatches = 0;

        // --- 2. Iterative Propagation Loop ---
        while current_worklist_size > 0 {
            iteration += 1;
            log::debug!(
                "GPU Propagation Iteration: {}, Current Worklist Size: {}",
                iteration,
                current_worklist_size
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

            let mut contradiction_found_in_iteration = false;
            let mut contradiction_location = u32::MAX;
            let mut total_output_count_this_iteration: u32 = 0;

            // Calculate number of batches needed for the current worklist size
            let num_batches = current_worklist_size.div_ceil(PROPAGATION_BATCH_SIZE as u32);

            for batch_index in 0..num_batches {
                total_dispatches += 1;
                // Calculate the size of this specific batch
                let batch_offset = batch_index * PROPAGATION_BATCH_SIZE as u32;
                let batch_size = std::cmp::min(
                    PROPAGATION_BATCH_SIZE as u32,
                    current_worklist_size - batch_offset,
                );

                log::trace!(
                    "  Dispatching Batch {}/{} ({} updates), Total Dispatches: {}",
                    batch_index + 1,
                    num_batches,
                    batch_size,
                    total_dispatches
                );

                // --- 2a. Reset Output Count & Contradiction Buffers ---
                // Reset count buffer BEFORE dispatching the batch that writes to it
                self.buffers
                    .reset_worklist_count(&self.queue)
                    .map_err(|e| {
                        PropagationError::GpuCommunicationError(format!(
                            "Failed to reset worklist count: {}",
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

                // --- 2b. Update Uniforms (Worklist size for the *current* batch) ---
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
                            label: Some(&format!(
                                "Propagation Encoder Iter {} Batch {}",
                                iteration, batch_index
                            )),
                        });

                let propagation_bind_group =
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(&format!(
                            "Propagation Bind Group Iter {} Batch {}",
                            iteration, batch_index
                        )),
                        layout: &self.pipelines.propagation_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                // 0: Grid Possibilities
                                binding: 0,
                                resource: self.buffers.grid_possibilities_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                // 1: Rules
                                binding: 1,
                                resource: self.buffers.rules_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                // 2: Input Worklist (Current Ping-Pong Buffer)
                                binding: 2,
                                resource: self.input_worklist_binding(), // Use helper
                            },
                            wgpu::BindGroupEntry {
                                // 3: Output Worklist (Other Ping-Pong Buffer)
                                binding: 3,
                                resource: self.output_worklist_binding(), // Use helper
                            },
                            wgpu::BindGroupEntry {
                                // 4: Params Uniform
                                binding: 4,
                                resource: self.buffers.params_uniform_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                // 5: Output Worklist Count
                                binding: 5,
                                resource: self.buffers.worklist_count_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                // 6: Contradiction Flag
                                binding: 6,
                                resource: self.buffers.contradiction_flag_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                // 7: Contradiction Location
                                binding: 7,
                                resource: self
                                    .buffers
                                    .contradiction_location_buf
                                    .as_entire_binding(),
                            },
                        ],
                    });

                // --- 2d. Dispatch Compute ---
                // Note: The shader now reads from the *input* buffer based on thread_idx,
                // effectively processing only the part of the buffer relevant to this batch.
                // We dispatch enough workgroups to cover the *batch size*.
                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some(&format!(
                                "Propagation Compute Pass Iter {} Batch {}",
                                iteration, batch_index
                            )),
                            timestamp_writes: None,
                        });
                    compute_pass.set_pipeline(&self.pipelines.propagation_pipeline);
                    compute_pass.set_bind_group(0, &propagation_bind_group, &[]);

                    // Use the constant for workgroup size
                    let workgroups_needed =
                        std::cmp::max(1, batch_size.div_ceil(PROPAGATION_WORKGROUP_SIZE));
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
                        "Contradiction detected during GPU propagation (Iteration {}, Batch {})!",
                        iteration,
                        batch_index + 1
                    );
                    contradiction_found_in_iteration = true;
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
                    break; // Stop processing batches for this iteration
                }

                // --- 2g. Download Output Worklist Count (generated by this batch) ---
                // This count tells us how many items were WRITTEN to the output buffer BY THIS BATCH.
                let output_count_this_batch = pollster::block_on(
                    self.buffers
                        .download_worklist_count(&self.device, &self.queue),
                )
                .map_err(|e| {
                    PropagationError::GpuCommunicationError(format!(
                        "Failed to download worklist count: {}",
                        e
                    ))
                })?;

                // Accumulate the total count for the next iteration
                // Note: This assumes the shader correctly uses atomicAdd on the count buffer.
                // A potential issue: If batches run concurrently, atomicAdd might be wrong.
                // However, we wait after each batch, so it *should* be correct, but less efficient.
                // A fully parallel approach would reset the counter once per iteration and download once.
                // For now, this sequential batch approach with per-batch count download is simpler.
                total_output_count_this_iteration += output_count_this_batch;

                // --- 2h. No need to download indices anymore ---
                // Indices are now directly in the other GPU buffer.
            } // End batch loop

            // --- Check Contradiction after all batches ---
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
            // Update the worklist size for the next iteration
            current_worklist_size = total_output_count_this_iteration;

            // Swap the input/output buffers for the next iteration
            self.current_worklist_idx = 1 - self.current_worklist_idx;
        } // End while loop (current_worklist_size > 0)

        let duration = propagate_start.elapsed();
        log::debug!(
            "GPU iterative propagation finished in {:?} ({} iterations, {} dispatches).",
            duration,
            iteration,
            total_dispatches
        );
        Ok(())
    }
}

// Ensure necessary types (ComputePipelines, GpuBuffers) are cloneable and fields are accessible
// No code here, just a comment

#[cfg(test)]
mod tests {
    use super::*; // Import items from parent module
    use crate::accelerator::GpuAccelerator;
    use crate::GpuError;
    use bitvec::prelude::bitvec;
    use std::sync::Arc;
    use wfc_core::{grid::PossibilityGrid, BoundaryMode};
    use wfc_rules::{AdjacencyRules, TileId, TileSet, TileSetError, Transformation};

    // --- Test Setup Helpers ---

    // Helper to create a simple TileSet for testing
    fn create_test_tileset(num_base_tiles: usize) -> Result<TileSet, TileSetError> {
        let weights = vec![1.0; num_base_tiles];
        let allowed_transforms = vec![vec![Transformation::Identity]; num_base_tiles];
        TileSet::new(weights, allowed_transforms)
    }

    // Helper to create specific adjacency rules for boundary tests
    // Rule: Tile 0 can only be adjacent to Tile 0, Tile 1 only to Tile 1
    fn create_boundary_test_rules(tileset: &TileSet) -> AdjacencyRules {
        let num_tiles = tileset.num_transformed_tiles();
        assert_eq!(num_tiles, 2, "This rule set assumes exactly 2 tiles");
        let num_axes = 6;
        let mut allowed_tuples = Vec::new();
        for axis in 0..num_axes {
            // Tile 0 -> Tile 0
            allowed_tuples.push((axis, 0, 0)); // Use usize directly
                                               // Tile 1 -> Tile 1
            allowed_tuples.push((axis, 1, 1)); // Use usize directly
        }
        AdjacencyRules::from_allowed_tuples(num_tiles, num_axes, allowed_tuples)
    }

    // Async helper to initialize GpuAccelerator for tests
    async fn setup_test_accelerator(
        grid: &PossibilityGrid,
        rules: &AdjacencyRules,
        boundary_mode: BoundaryMode,
    ) -> Result<GpuAccelerator, GpuError> {
        // Only run if a GPU is available
        match GpuAccelerator::new(grid, rules, boundary_mode).await {
            Ok(acc) => Ok(acc),
            Err(e) => {
                eprintln!(
                    "GPU Accelerator setup failed, likely no suitable adapter: {}. Skipping test.",
                    e
                );
                Err(e) // Propagate error to indicate skip
            }
        }
    }

    // Helper to download the grid state from GPU buffers
    async fn download_grid_state(
        accelerator: &GpuAccelerator,
        grid_dims: (usize, usize, usize),
        num_tiles: usize,
    ) -> Result<PossibilityGrid, GpuError> {
        let (width, height, depth) = grid_dims;
        let packed_data = accelerator
            .buffers()
            .download_possibilities(&accelerator.device(), &accelerator.queue())
            .await?;

        let mut new_grid = PossibilityGrid::new(width, height, depth, num_tiles);
        let u32s_per_cell = (num_tiles + 31) / 32;
        let mut current_idx = 0;

        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    if let Some(cell_bv) = new_grid.get_mut(x, y, z) {
                        cell_bv.fill(false); // Clear default values
                        let mut tile_idx = 0;
                        for u32_chunk_idx in 0..u32s_per_cell {
                            let packed_value = packed_data[current_idx + u32_chunk_idx];
                            for bit_idx in 0..32 {
                                if tile_idx < num_tiles {
                                    if (packed_value >> bit_idx) & 1 == 1 {
                                        cell_bv.set(tile_idx, true);
                                    }
                                    tile_idx += 1;
                                }
                            }
                        }
                    }
                    current_idx += u32s_per_cell;
                }
            }
        }
        Ok(new_grid)
    }

    // --- Boundary Condition Tests ---

    #[tokio::test]
    async fn test_propagate_clamped_boundary() {
        let width = 3;
        let height = 1;
        let depth = 1;
        let num_tiles = 2;

        let tileset = create_test_tileset(num_tiles).unwrap();
        let rules = create_boundary_test_rules(&tileset);
        let mut initial_grid = PossibilityGrid::new(width, height, depth, num_tiles);

        // Collapse center cell (1, 0, 0) to only allow Tile 0
        let center_cell = initial_grid.get_mut(1, 0, 0).unwrap();
        center_cell.set(1, false); // Disallow Tile 1

        let boundary_mode = BoundaryMode::Clamped;
        let mut accelerator =
            match setup_test_accelerator(&initial_grid, &rules, boundary_mode).await {
                Ok(acc) => acc,
                Err(_) => return, // Skip test if GPU setup fails
            };

        // Propagate the change from the center cell
        let updated_coords = vec![(1, 0, 0)];
        let prop_result = accelerator.propagate(&mut initial_grid, updated_coords, &rules);
        assert!(prop_result.is_ok());

        // Download the grid state after propagation
        let final_grid = download_grid_state(&accelerator, (width, height, depth), num_tiles)
            .await
            .expect("Failed to download grid state");

        // Assertions for Clamped mode:
        // - Center cell (1,0,0) should only have Tile 0 possible.
        let center_final = final_grid.get(1, 0, 0).unwrap();
        assert!(center_final[0]);
        assert!(!center_final[1]);

        // - Left neighbor (0,0,0) should only have Tile 0 possible (constrained by center).
        let left_final = final_grid.get(0, 0, 0).unwrap();
        assert!(left_final[0]);
        assert!(!left_final[1]);

        // - Right neighbor (2,0,0) should only have Tile 0 possible (constrained by center).
        let right_final = final_grid.get(2, 0, 0).unwrap();
        assert!(right_final[0]);
        assert!(!right_final[1]);
    }

    #[tokio::test]
    async fn test_propagate_periodic_boundary() {
        let width = 3;
        let height = 1;
        let depth = 1;
        let num_tiles = 2;

        let tileset = create_test_tileset(num_tiles).unwrap();
        let rules = create_boundary_test_rules(&tileset);
        let mut initial_grid = PossibilityGrid::new(width, height, depth, num_tiles);

        // Collapse cell (0, 0, 0) to only allow Tile 0
        let cell_0 = initial_grid.get_mut(0, 0, 0).unwrap();
        cell_0.set(1, false); // Disallow Tile 1

        let boundary_mode = BoundaryMode::Periodic;
        let mut accelerator =
            match setup_test_accelerator(&initial_grid, &rules, boundary_mode).await {
                Ok(acc) => acc,
                Err(_) => return, // Skip test if GPU setup fails
            };

        // Propagate the change from cell (0, 0, 0)
        let updated_coords = vec![(0, 0, 0)];
        let prop_result = accelerator.propagate(&mut initial_grid, updated_coords, &rules);
        assert!(prop_result.is_ok());

        // Download the grid state after propagation
        let final_grid = download_grid_state(&accelerator, (width, height, depth), num_tiles)
            .await
            .expect("Failed to download grid state");

        // Assertions for Periodic mode:
        // - Cell (0,0,0) should only have Tile 0 possible.
        let cell_0_final = final_grid.get(0, 0, 0).unwrap();
        assert!(cell_0_final[0]);
        assert!(!cell_0_final[1]);

        // - Right neighbor (1,0,0) should only have Tile 0 possible.
        let cell_1_final = final_grid.get(1, 0, 0).unwrap();
        assert!(cell_1_final[0]);
        assert!(!cell_1_final[1]);

        // - Cell (2,0,0) (neighbor via wrap-around) should only have Tile 0 possible.
        let cell_2_final = final_grid.get(2, 0, 0).unwrap();
        assert!(cell_2_final[0]);
        assert!(!cell_2_final[1]);
    }
}
