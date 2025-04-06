use crate::{
    buffers::{GpuBuffers, GpuParamsUniform},
    debug_viz::{DebugVisualizer, GpuBuffersDebugExt},
    pipeline::ComputePipelines,
    subgrid::{
        divide_into_subgrids, extract_subgrid, merge_subgrids, SubgridConfig, SubgridRegion,
    },
};
use async_trait::async_trait;
use log::{debug, info};
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
    // Early termination settings
    early_termination_threshold: u32,
    early_termination_consecutive_passes: u32,
    // Subgrid processing configuration
    subgrid_config: Option<SubgridConfig>,
    // Debug visualization
    debug_visualizer: Option<Arc<std::sync::Mutex<DebugVisualizer>>>,
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
            early_termination_threshold: 10, // Default: if fewer than 10 cells are affected, consider early termination
            early_termination_consecutive_passes: 3, // Default: require 3 consecutive passes below threshold
            subgrid_config: None,                    // Disabled by default
            debug_visualizer: None,                  // Debug visualization disabled by default
        }
    }

    /// Sets the early termination parameters.
    ///
    /// Early termination stops the propagation process when the worklist size remains
    /// below a certain threshold for a number of consecutive passes.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The maximum number of cells that can be updated before considering termination.
    /// * `consecutive_passes` - The number of consecutive passes that must be below the threshold.
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining.
    pub fn with_early_termination(mut self, threshold: u32, consecutive_passes: u32) -> Self {
        self.early_termination_threshold = threshold;
        self.early_termination_consecutive_passes = consecutive_passes;
        self
    }

    /// Enables parallel subgrid processing for large grids.
    ///
    /// When enabled, large grids will be divided into smaller subgrids
    /// that can be processed independently, improving performance.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration for subgrid division and processing.
    ///
    /// # Returns
    ///
    /// `Self` for method chaining.
    pub fn with_parallel_subgrid_processing(mut self, config: SubgridConfig) -> Self {
        self.subgrid_config = Some(config);
        self
    }

    /// Disables parallel subgrid processing.
    ///
    /// # Returns
    ///
    /// `Self` for method chaining.
    pub fn without_parallel_subgrid_processing(mut self) -> Self {
        self.subgrid_config = None;
        self
    }

    /// Sets the debug visualizer.
    ///
    /// # Arguments
    ///
    /// * `visualizer` - The debug visualizer to use for capturing algorithm state.
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining.
    pub fn with_debug_visualizer(mut self, visualizer: DebugVisualizer) -> Self {
        self.debug_visualizer = Some(Arc::new(std::sync::Mutex::new(visualizer)));
        self
    }

    /// Disables debug visualization.
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining.
    pub fn without_debug_visualization(mut self) -> Self {
        self.debug_visualizer = None;
        self
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

    /// Performs constraint propagation on a subgrid.
    ///
    /// This method applies constraint propagation to a specified subgrid region.
    ///
    /// # Arguments
    ///
    /// * `grid` - The full grid (used for context, not directly modified).
    /// * `subgrid` - The extracted subgrid to process.
    /// * `region` - The region information for the subgrid.
    /// * `updated_coords` - Coordinates within the subgrid that have been updated.
    /// * `rules` - The adjacency rules for constraint propagation.
    ///
    /// # Returns
    ///
    /// * `Ok(updated_subgrid)` - The processed subgrid if successful.
    /// * `Err(PropagationError)` - If a contradiction was detected or other error occurred.
    async fn propagate_subgrid(
        &mut self,
        _grid: &PossibilityGrid,
        subgrid: PossibilityGrid,
        region: &SubgridRegion,
        updated_coords: Vec<(usize, usize, usize)>,
        rules: &AdjacencyRules,
    ) -> Result<PossibilityGrid, PropagationError> {
        // Create a temporary GpuBuffers for this subgrid
        let temp_params = GpuParamsUniform {
            grid_width: subgrid.width as u32,
            grid_height: subgrid.height as u32,
            grid_depth: subgrid.depth as u32,
            num_tiles: self.params.num_tiles,
            num_axes: self.params.num_axes,
            worklist_size: 0, // Will be set during propagation
            boundary_mode: self.params.boundary_mode,
            _padding1: 0,
        };

        // Convert global coordinates to local subgrid coordinates
        let local_updated_coords: Vec<(usize, usize, usize)> = updated_coords
            .into_iter()
            .filter_map(|(x, y, z)| region.to_local_coords(x, y, z))
            .collect();

        // Early exit if no cells need updating in this subgrid
        if local_updated_coords.is_empty() {
            return Ok(subgrid);
        }

        debug!(
            "Processing subgrid region ({},{},{}) to ({},{},{}) with {} updates",
            region.start_x,
            region.start_y,
            region.start_z,
            region.end_x,
            region.end_y,
            region.end_z,
            local_updated_coords.len()
        );

        // Create temporary buffers for this subgrid
        let temp_buffers = Arc::new(
            GpuBuffers::new(
                &self.device,
                &self.queue,
                &subgrid,
                rules,
                match self.params.boundary_mode {
                    0 => wfc_core::BoundaryCondition::Finite,
                    _ => wfc_core::BoundaryCondition::Periodic,
                },
            )
            .map_err(|e| PropagationError::GpuSetupError(e.to_string()))?,
        );

        // Modify the constraint propagator to use these buffers temporarily
        let original_buffers = self.buffers.clone();
        let original_params = self.params;
        self.buffers = temp_buffers;
        self.params = temp_params;

        // Convert coordinates to indices for propagate_internal
        let (width, height, _) = (subgrid.width, subgrid.height, subgrid.depth);
        let local_updated_indices: Vec<u32> = local_updated_coords
            .into_iter()
            .map(|(x, y, z)| (x + y * width + z * width * height) as u32)
            .collect();

        // Use the standard propagation logic on the subgrid
        let mut temp_subgrid = subgrid;
        match self
            .propagate_internal(&mut temp_subgrid, local_updated_indices, rules)
            .await
        {
            Ok(()) => {
                // Restore the original buffers and params
                self.buffers = original_buffers;
                self.params = original_params;
                Ok(temp_subgrid)
            }
            Err(e) => {
                // Restore the original buffers and params before returning error
                self.buffers = original_buffers;
                self.params = original_params;
                Err(e)
            }
        }
    }

    /// Performs the actual constraint propagation on the GPU.
    async fn propagate_internal(
        &mut self,
        _grid: &mut PossibilityGrid,
        updated_indices: Vec<u32>,
        _rules: &AdjacencyRules,
    ) -> Result<(), PropagationError> {
        let (total_width, total_height, total_depth) = (
            self.params.grid_width,
            self.params.grid_height,
            self.params.grid_depth,
        );
        let total_cells = (total_width * total_height * total_depth) as usize;

        // Prepare initial worklist
        let worklist_size = updated_indices.len() as u32;
        if worklist_size == 0 {
            debug!("No cells to update, skipping propagation");
            return Ok(());
        }

        debug!("Initial worklist size: {}", worklist_size);

        // Reset buffers for this propagation step
        self.buffers
            .reset_contradiction_flag(&self.queue)
            .map_err(|e| PropagationError::GpuCommunicationError(e.to_string()))?;
        self.buffers
            .reset_contradiction_location(&self.queue)
            .map_err(|e| PropagationError::GpuCommunicationError(e.to_string()))?;
        self.buffers
            .reset_worklist_count(&self.queue)
            .map_err(|e| PropagationError::GpuCommunicationError(e.to_string()))?;

        // Upload initial worklist indices to the active buffer
        self.buffers
            .upload_initial_updates(&self.queue, &updated_indices, self.current_worklist_idx)
            .map_err(|e| PropagationError::GpuCommunicationError(e.to_string()))?;

        // Update params with the current worklist size
        self.buffers
            .update_params_worklist_size(&self.queue, worklist_size)
            .map_err(|e| PropagationError::GpuCommunicationError(e.to_string()))?;

        // Determine dispatch size (ceiling division of total_cells by workgroup size)
        let mut num_cells_to_process = worklist_size;
        let workgroup_size = 64;
        let mut dispatch_size = (num_cells_to_process + workgroup_size - 1) / workgroup_size;

        // If dispatch_size is zero (e.g., empty worklist), use at least one workgroup
        if dispatch_size == 0 {
            dispatch_size = 1;
        }

        // Prepare bind group parameters for the propagation compute pass
        let bind_group_layout = self
            .pipelines
            .get_propagation_bind_group_layout()
            .map_err(|e| PropagationError::GpuSetupError(e.to_string()))?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Propagation Command Encoder"),
            });

        // Run the iterative propagation algorithm until no more updates or we hit a limit
        let mut iteration = 0;
        let max_iterations = 100; // Safety limit to prevent infinite loops
        let mut contradiction = false;
        let mut contradiction_location = None;
        let mut early_termination_count = 0;

        while num_cells_to_process > 0 && iteration < max_iterations && !contradiction {
            debug!(
                "Propagation iteration {}: Processing {} cells",
                iteration, num_cells_to_process
            );

            // Create a new bind group for this iteration
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Propagation Bind Group {}", iteration)),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffers.params_uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.buffers.grid_possibilities_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.buffers.adjacency_rules_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.input_worklist_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.output_worklist_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.buffers.worklist_count_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.buffers.contradiction_flag_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: self.buffers.contradiction_location_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: self.buffers.rule_weights_buf.as_entire_binding(),
                    },
                ],
            });

            // Execute propagation pass
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Propagation Pass {}", iteration)),
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(
                    self.pipelines
                        .get_propagation_pipeline(
                            self.device.features().contains(wgpu::Features::SHADER_I16),
                        )
                        .map_err(|e| PropagationError::GpuSetupError(e.to_string()))?,
                );
                compute_pass.set_bind_group(0, &bind_group, &[]);
                compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
            }

            // Submit commands
            self.queue.submit(std::iter::once(encoder.finish()));

            // Create a new encoder for the next iteration
            encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("Propagation Command Encoder {}", iteration + 1)),
                });

            // Check for contradictions and get next worklist size
            let (has_contradiction, next_worklist_size, contradiction_idx) = self
                .buffers
                .download_propagation_status(Arc::clone(&self.device), Arc::clone(&self.queue))
                .await
                .map_err(|e| PropagationError::GpuCommunicationError(e.to_string()))?;

            // Take a debug snapshot after this propagation step
            if let Some(visualizer) = &self.debug_visualizer {
                if let Ok(mut vis) = visualizer.lock() {
                    if vis.is_enabled() {
                        let _ = self.buffers.take_debug_snapshot(
                            Arc::clone(&self.device),
                            Arc::clone(&self.queue),
                            &mut vis,
                        );
                    }
                }
            }

            if has_contradiction {
                contradiction = true;
                contradiction_location = contradiction_idx;
                debug!(
                    "Contradiction detected at cell {}",
                    contradiction_idx.unwrap_or(total_cells as u32)
                );
                break;
            }

            // Update worklist size and ping-pong buffer index
            num_cells_to_process = next_worklist_size;
            // Toggle the current index between 0 and 1
            self.current_worklist_idx = 1 - self.current_worklist_idx;

            debug!(
                "Iteration {} completed: Next worklist size: {}",
                iteration, num_cells_to_process
            );

            // Check for early termination condition (small worklist for multiple consecutive passes)
            if num_cells_to_process <= self.early_termination_threshold {
                early_termination_count += 1;
                if early_termination_count >= self.early_termination_consecutive_passes {
                    debug!(
                        "Early termination after {} iterations: {} cells affected, below threshold {} for {} consecutive passes",
                        iteration + 1,
                        num_cells_to_process,
                        self.early_termination_threshold,
                        self.early_termination_consecutive_passes
                    );
                    break;
                }
            } else {
                // Reset the counter if we have a larger worklist again
                early_termination_count = 0;
            }

            // Prepare dispatch size for next iteration
            dispatch_size = (num_cells_to_process + workgroup_size - 1) / workgroup_size;
            if dispatch_size == 0 {
                dispatch_size = 1;
            }

            // Update params with new worklist size
            self.buffers
                .update_params_worklist_size(&self.queue, num_cells_to_process)
                .map_err(|e| PropagationError::GpuCommunicationError(e.to_string()))?;

            iteration += 1;
        }

        debug!(
            "Propagation completed after {} iterations{}",
            iteration,
            if contradiction {
                format!(
                    " with contradiction at cell {:?}",
                    contradiction_location.map(|idx| {
                        let _z = idx / (total_width * total_height);
                        let _y = (idx % (total_width * total_height)) / total_width;
                        let x = idx % total_width;
                        x as usize
                    })
                )
            } else {
                String::new()
            }
        );

        // If a contradiction was detected, propagate that information
        if contradiction {
            return Err(PropagationError::Contradiction(
                contradiction_location
                    .map(|idx| {
                        let _z = idx / (total_width * total_height);
                        let _y = (idx % (total_width * total_height)) / total_width;
                        let x = idx % total_width;
                        x as usize
                    })
                    .unwrap_or(0),
                contradiction_location
                    .map(|idx| {
                        let _z = idx / (total_width * total_height);
                        let y = (idx % (total_width * total_height)) / total_width;
                        let _x = idx % total_width;
                        y as usize
                    })
                    .unwrap_or(0),
                contradiction_location
                    .map(|idx| {
                        let z = idx / (total_width * total_height);
                        let _y = (idx % (total_width * total_height)) / total_width;
                        let _x = idx % total_width;
                        z as usize
                    })
                    .unwrap_or(0),
            ));
        }

        Ok(())
    }

    /// Performs propagation using parallel subgrid processing.
    ///
    /// This method divides the grid into smaller subgrids, processes each subgrid separately,
    /// and then merges the results back into the main grid.
    ///
    /// # Arguments
    ///
    /// * `grid` - The grid to propagate.
    /// * `updated_indices` - Indices of cells that have been updated.
    /// * `rules` - The adjacency rules for propagation.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if propagation succeeded
    /// * `Err(PropagationError)` if propagation failed (e.g., contradiction)
    async fn propagate_with_subgrids(
        &mut self,
        grid: &mut PossibilityGrid,
        updated_indices: Vec<u32>,
        rules: &AdjacencyRules,
    ) -> Result<(), PropagationError> {
        let config = self.subgrid_config.clone().unwrap();

        // Take initial snapshot if debug visualization is enabled
        if let Some(visualizer) = &self.debug_visualizer {
            if let Ok(mut vis) = visualizer.lock() {
                if vis.is_enabled() {
                    let _ = self.buffers.take_debug_snapshot(
                        Arc::clone(&self.device),
                        Arc::clone(&self.queue),
                        &mut vis,
                    );
                }
            }
        }

        // Convert indices back to coordinates for subgrid processing
        let (width, height, depth) = (grid.width, grid.height, grid.depth);
        let updated_coords: Vec<(usize, usize, usize)> = updated_indices
            .iter()
            .map(|&idx| {
                let idx = idx as usize;
                let z = idx / (width * height);
                let remainder = idx % (width * height);
                let y = remainder / width;
                let x = remainder % width;
                (x, y, z)
            })
            .collect();

        // Divide the grid into subgrids
        let subgrid_regions = divide_into_subgrids(grid.width, grid.height, grid.depth, &config)
            .map_err(|e| PropagationError::GpuSetupError(e.to_string()))?;

        info!(
            "Parallel subgrid processing: Divided {}x{}x{} grid into {} subgrids",
            grid.width,
            grid.height,
            grid.depth,
            subgrid_regions.len()
        );

        // Extract and process each subgrid
        let mut processed_subgrids = Vec::new();
        for region in &subgrid_regions {
            // Extract the subgrid
            let subgrid = extract_subgrid(grid, region)
                .map_err(|e| PropagationError::GpuSetupError(e.to_string()))?;

            // Process the subgrid
            let processed = self
                .propagate_subgrid(grid, subgrid, region, updated_coords.clone(), rules)
                .await?;

            processed_subgrids.push((region.clone(), processed));

            // Take a snapshot after each subgrid if debug visualization is enabled
            if let Some(visualizer) = &self.debug_visualizer {
                if let Ok(mut vis) = visualizer.lock() {
                    if vis.is_enabled() {
                        let _ = self.buffers.take_debug_snapshot(
                            Arc::clone(&self.device),
                            Arc::clone(&self.queue),
                            &mut vis,
                        );
                    }
                }
            }
        }

        // Merge the processed subgrids back into the main grid
        let merged_updates = merge_subgrids(grid, &processed_subgrids)
            .map_err(|e| PropagationError::GpuSetupError(e.to_string()))?;

        info!(
            "Parallel subgrid processing: Merged {} subgrids with {} updated cells",
            processed_subgrids.len(),
            merged_updates.len()
        );

        // Take a snapshot after merging if debug visualization is enabled
        if let Some(visualizer) = &self.debug_visualizer {
            if let Ok(mut vis) = visualizer.lock() {
                if vis.is_enabled() {
                    let _ = self.buffers.take_debug_snapshot(
                        Arc::clone(&self.device),
                        Arc::clone(&self.queue),
                        &mut vis,
                    );
                }
            }
        }

        // If we have any updates from the merge, we need to propagate them again
        if !merged_updates.is_empty() {
            // For the final pass, disable subgrid processing to avoid recursion
            let original_config = self.subgrid_config.take();
            let result = self.propagate(grid, merged_updates, rules).await;
            self.subgrid_config = original_config;
            return result;
        }

        Ok(())
    }

    /// Initialize a GpuConstraintPropagator with most features enabled
    pub fn init_default() -> Self {
        // Reasonable defaults for most use cases
        let _subgrid_config = SubgridConfig {
            max_subgrid_size: 32,
            overlap_size: 2,
            min_size: 64,
        };

        // Other initialization code...
        // ...
        // (This is just a placeholder method - the real implementation would
        // need actual device and other resources)
        unimplemented!("This is just a stub method, not intended for actual use")
    }
}

impl Drop for GpuConstraintPropagator {
    /// Performs cleanup of GPU resources when GpuConstraintPropagator is dropped.
    ///
    /// This ensures proper cleanup following RAII principles.
    fn drop(&mut self) {
        debug!("GpuConstraintPropagator being dropped, releasing references to GPU resources");

        // The actual cleanup happens automatically through Arc's reference counting
        // when the device, queue, pipelines, and buffers references are dropped.

        // If there's a debug visualizer, ensure it's properly cleaned up
        if self.debug_visualizer.is_some() {
            debug!("Cleaning up debug visualizer resources");
            // The mutex will be cleaned up when the Arc is dropped
            // We don't need to do anything special here
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
    /// # Arguments
    ///
    /// * `grid` - The grid on which to propagate constraints.
    /// * `updated_coords` - Cell coordinates that have been updated.
    /// * `rules` - The adjacency rules that define valid cell states.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if propagation completes successfully
    /// * `Err(PropagationError)` if a contradiction is found or a GPU error occurs
    async fn propagate(
        &mut self,
        grid: &mut PossibilityGrid,
        updated_coords: Vec<(usize, usize, usize)>,
        rules: &AdjacencyRules,
    ) -> Result<(), PropagationError> {
        // Convert coordinates to indices
        let (width, height, _) = (grid.width, grid.height, grid.depth);
        let updated_indices: Vec<u32> = updated_coords
            .into_iter()
            .map(|(x, y, z)| (x + y * width + z * width * height) as u32)
            .collect();

        // Attempt parallel subgrid processing if configured
        if self.subgrid_config.is_some() {
            return self
                .propagate_with_subgrids(grid, updated_indices, rules)
                .await;
        }

        // Take initial snapshot if debug visualization is enabled
        if let Some(visualizer) = &self.debug_visualizer {
            if let Ok(mut vis) = visualizer.lock() {
                if vis.is_enabled() {
                    let _ = self.buffers.take_debug_snapshot(
                        Arc::clone(&self.device),
                        Arc::clone(&self.queue),
                        &mut vis,
                    );
                }
            }
        }

        // Standard propagation approach
        let result = self.propagate_internal(grid, updated_indices, rules).await;

        // Take final snapshot if debug visualization is enabled
        if let Some(visualizer) = &self.debug_visualizer {
            if let Ok(mut vis) = visualizer.lock() {
                if vis.is_enabled() {
                    let _ = self.buffers.take_debug_snapshot(
                        Arc::clone(&self.device),
                        Arc::clone(&self.queue),
                        &mut vis,
                    );
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        buffers::GpuBuffers, pipeline::ComputePipelines, subgrid::SubgridConfig, GpuError,
    };
    use std::sync::Arc;
    use wfc_core::{grid::PossibilityGrid, BoundaryCondition};
    use wfc_rules::AdjacencyRules;

    // Mock GPU device for testing
    struct MockGpu {
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        buffers: Arc<GpuBuffers>,
        pipelines: Arc<ComputePipelines>,
    }

    // Helper to create a mock GPU environment for testing
    async fn setup_mock_gpu() -> Result<MockGpu, GpuError> {
        // Get a real device and queue for testing (using wgpu's default backends)
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| GpuError::AdapterRequestFailed)?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Test Device"),
                    required_features: wgpu::Features::empty(),
                    // Request higher limits that match our storage buffer usage in propagation
                    required_limits: wgpu::Limits {
                        max_storage_buffers_per_shader_stage: 8,
                        ..wgpu::Limits::downlevel_defaults()
                    },
                },
                None,
            )
            .await
            .map_err(|e| GpuError::DeviceRequestFailed(e))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // For testing, we create minimal grid dimensions and rules
        let grid_dims = (2, 2, 1);
        let width = grid_dims.0;
        let height = grid_dims.1;
        let depth = grid_dims.2;
        let num_tiles = 2;
        let boundary_mode = BoundaryCondition::Finite;

        // Create a simple grid with all possibilities enabled
        let grid = PossibilityGrid::new(width, height, depth, num_tiles);

        // Create simple adjacency rules where all tiles can be adjacent
        let mut allowed_tuples = Vec::new();
        for axis in 0..6 {
            for tile1 in 0..num_tiles {
                for tile2 in 0..num_tiles {
                    allowed_tuples.push((axis, tile1, tile2));
                }
            }
        }
        let rules = AdjacencyRules::from_allowed_tuples(num_tiles, 6, allowed_tuples);

        // Create buffers
        let buffers = GpuBuffers::new(&device, &queue, &grid, &rules, boundary_mode)?;
        let buffers = Arc::new(buffers);

        // Create a custom test pipeline instead of using ComputePipelines::new which may have issues
        let propagation_shader_code = include_str!("./shaders/test_shader.wgsl").to_string();

        // Create the shader module directly instead of using the cache
        let propagation_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Test Propagation Shader"),
            source: wgpu::ShaderSource::Wgsl(propagation_shader_code.into()),
        });

        // Create a minimal bind group layout for testing
        let propagation_bind_group_layout = Arc::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Test Propagation Bind Group Layout"),
                entries: &[
                    // Keep it minimal for testing
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()),
                        },
                        count: None,
                    },
                ],
            },
        ));

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Test Pipeline Layout"),
            bind_group_layouts: &[&propagation_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create test compute pipeline
        let propagation_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Test Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &propagation_shader,
                entry_point: "main_propagate",
                compilation_options: Default::default(),
            },
        ));

        // Create test pipelines object
        let pipelines = Arc::new(ComputePipelines {
            entropy_pipeline: propagation_pipeline.clone(), // Use the same test pipeline for both
            propagation_pipeline,
            entropy_bind_group_layout: propagation_bind_group_layout.clone(),
            propagation_bind_group_layout,
            entropy_workgroup_size: 64,
            propagation_workgroup_size: 64,
        });

        Ok(MockGpu {
            device,
            queue,
            buffers,
            pipelines,
        })
    }

    #[tokio::test]
    async fn test_early_termination_configuration() {
        // This is primarily a compilation test to ensure the with_early_termination method
        // works as expected and the propagator can be configured.

        if let Ok(mock_gpu) = setup_mock_gpu().await {
            let params = GpuParamsUniform {
                grid_width: 2,
                grid_height: 2,
                grid_depth: 1,
                num_tiles: 2,
                num_axes: 6,
                worklist_size: 0,
                boundary_mode: 0,
                _padding1: 0,
            };

            let propagator = GpuConstraintPropagator::new(
                mock_gpu.device,
                mock_gpu.queue,
                mock_gpu.pipelines,
                mock_gpu.buffers,
                (2, 2, 1),
                BoundaryCondition::Finite,
                params,
            );

            // Configure early termination
            let propagator = propagator.with_early_termination(5, 2);

            // Verify configuration
            assert_eq!(propagator.early_termination_threshold, 5);
            assert_eq!(propagator.early_termination_consecutive_passes, 2);
        } else {
            // Skip test if GPU creation fails (headless CI environments may not have GPU)
            println!("Skipping test_early_termination_configuration due to GPU init failure");
        }
    }

    #[test]
    fn test_parallel_subgrid_config() {
        // Create a test subgrid config
        let subgrid_config = SubgridConfig {
            max_subgrid_size: 64,
            overlap_size: 2,
            min_size: 128,
        };

        // Create a default config for comparison
        let default_config = SubgridConfig::default();

        // Check the default values
        assert_eq!(default_config.max_subgrid_size, 64);
        assert_eq!(default_config.overlap_size, 2);
        assert_eq!(default_config.min_size, 128);

        // Check our custom config
        assert_eq!(subgrid_config.max_subgrid_size, 64);
        assert_eq!(subgrid_config.overlap_size, 2);
        assert_eq!(subgrid_config.min_size, 128);
    }
}
