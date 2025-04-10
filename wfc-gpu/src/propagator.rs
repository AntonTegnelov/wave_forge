use crate::{
    buffers::{GpuBuffers, GpuParamsUniform},
    debug_viz::DebugVisualizer,
    pipeline::ComputePipelines,
    subgrid::{
        divide_into_subgrids, extract_subgrid, merge_subgrids, SubgridConfig, SubgridRegion,
    },
    sync::GpuSynchronizer,
};
use async_trait::async_trait;
use log::{debug, error, info};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use wfc_core::{
    grid::PossibilityGrid,
    propagator::{ConstraintPropagator, PropagationError},
    BoundaryCondition,
};
use wfc_rules::AdjacencyRules;
use wgpu;

/// GPU implementation of the ConstraintPropagator trait.
#[derive(Debug, Clone)]
pub struct GpuConstraintPropagator {
    // References to shared GPU resources (Reverted to hold individual components)
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) pipelines: Arc<ComputePipelines>,
    pub(crate) buffers: Arc<GpuBuffers>,
    // State for ping-pong buffer index
    current_worklist_idx: Arc<AtomicUsize>,
    pub(crate) params: GpuParamsUniform,
    // Early termination settings
    early_termination_threshold: u32,
    early_termination_consecutive_passes: u32,
    // Subgrid processing configuration
    subgrid_config: Option<SubgridConfig>,
    // Debug visualization
    debug_visualizer: Option<Arc<std::sync::Mutex<DebugVisualizer>>>,
    synchronizer: Arc<GpuSynchronizer>,
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
            current_worklist_idx: Arc::new(AtomicUsize::new(0)),
            params,
            early_termination_threshold: 10, // Default: if fewer than 10 cells are affected, consider early termination
            early_termination_consecutive_passes: 3, // Default: require 3 consecutive passes below threshold
            subgrid_config: None,                    // Disabled by default
            debug_visualizer: None,                  // Debug visualization disabled by default
            synchronizer,
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

    /// Gets the binding resource for the grid possibilities buffer.
    fn grid_possibilities_binding(&self) -> wgpu::BindingResource {
        self.buffers
            .grid_buffers
            .grid_possibilities_buf
            .as_entire_binding()
    }

    /// Gets the binding resource for the current input worklist buffer.
    fn input_worklist_binding(&self) -> wgpu::BindingResource {
        // Load the current index atomically from the Arc
        let idx = self.current_worklist_idx.load(Ordering::Relaxed);
        if idx == 0 {
            self.buffers
                .worklist_buffers
                .worklist_buf_a
                .as_entire_binding()
        } else {
            self.buffers
                .worklist_buffers
                .worklist_buf_b
                .as_entire_binding()
        }
    }

    /// Gets the binding resource for the current output worklist buffer.
    fn output_worklist_binding(&self) -> wgpu::BindingResource {
        // Load the current index atomically from the Arc
        let idx = self.current_worklist_idx.load(Ordering::Relaxed);
        if idx == 0 {
            self.buffers
                .worklist_buffers
                .worklist_buf_b
                .as_entire_binding()
        } else {
            self.buffers
                .worklist_buffers
                .worklist_buf_a
                .as_entire_binding()
        }
    }

    /// Gets the binding resource for the input worklist count buffer.
    fn input_worklist_count_binding(&self) -> wgpu::BindingResource {
        self.buffers
            .worklist_buffers
            .worklist_count_buf
            .as_entire_binding()
    }

    /// Gets the binding resource for the output worklist count buffer.
    fn output_worklist_count_binding(&self) -> wgpu::BindingResource {
        self.buffers
            .worklist_buffers
            .worklist_count_buf
            .as_entire_binding()
    }

    /// Creates the bind group for the constraint propagation compute pass.
    fn create_propagation_bind_group(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Propagation Bind Group"),
            layout: &self.pipelines.propagation_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffers.params_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.grid_possibilities_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self
                        .buffers
                        .rule_buffers
                        .adjacency_rules_buf
                        .as_entire_binding(),
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
                    resource: self.buffers.contradiction_flag_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.input_worklist_count_binding(),
                },
            ],
        })
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
            boundary_mode: self.params.boundary_mode,
            heuristic_type: self.params.heuristic_type,
            tie_breaking: self.params.tie_breaking,
            max_propagation_steps: self.params.max_propagation_steps,
            contradiction_check_frequency: self.params.contradiction_check_frequency,
            worklist_size: 0, // Will be set during propagation
            grid_element_count: (subgrid.width * subgrid.height * subgrid.depth) as u32,
            _padding: 0,
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
            region.x_offset,
            region.y_offset,
            region.z_offset,
            region.end_x(),
            region.end_y(),
            region.end_z(),
            local_updated_coords.len()
        );

        // Create temporary buffers for this subgrid
        let temp_buffers = Arc::new(
            GpuBuffers::new(
                &self.device,
                &self.queue,
                &subgrid,
                rules,
                BoundaryCondition::Finite,
            )
            .map_err(|e| PropagationError::GpuSetupError(e.to_string()))?,
        );

        // Modify the constraint propagator to use these buffers temporarily
        let original_buffers = self.buffers.clone();
        let original_params = self.params;
        self.buffers = temp_buffers;
        self.params = temp_params;

        // Convert coordinates to indices for propagate_internal
        let (width, height, _depth) = (subgrid.width, subgrid.height, subgrid.depth);
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

    /// The internal propagation implementation (takes &self).
    async fn propagate_internal(
        &self,
        _grid: &mut PossibilityGrid, // Grid not directly modified here anymore
        updated_indices: Vec<u32>,
        _rules: &AdjacencyRules,
    ) -> Result<(), PropagationError> {
        let start_time = std::time::Instant::now();

        let initial_worklist_size = updated_indices.len() as u32;
        if initial_worklist_size == 0 {
            debug!("Initial worklist empty, skipping propagation.");
            return Ok(());
        }

        // 1. Prepare Initial Worklist
        // Create encoder for initial writes
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Initial Worklist Write Encoder"),
            });

        // Determine which buffer is the initial input based on atomic index
        let initial_input_idx = self.current_worklist_idx.load(Ordering::Relaxed);
        let input_worklist_buffer = if initial_input_idx == 0 {
            &self.buffers.worklist_buffers.worklist_buf_a
        } else {
            &self.buffers.worklist_buffers.worklist_buf_b
        };

        // Write initial updated indices to the *input* worklist buffer
        let worklist_data: &[u8] = bytemuck::cast_slice(&updated_indices);
        self.queue
            .write_buffer(input_worklist_buffer, 0, worklist_data);

        // Write the initial worklist size to the *input* count buffer
        let input_count_buffer = &self.buffers.worklist_buffers.worklist_count_buf;
        self.queue.write_buffer(
            input_count_buffer,
            0,
            bytemuck::cast_slice(&[initial_worklist_size]),
        );

        self.queue.submit(Some(encoder.finish()));

        // 2. Propagation Loop
        let mut current_pass = 0;
        let mut consecutive_low_worklist_passes = 0;

        loop {
            if current_pass >= self.params.max_propagation_steps {
                return Err(PropagationError::InternalError(format!(
                    "Propagation exceeded max steps ({})",
                    current_pass
                )));
            }

            // Load current index for this pass
            let pass_input_idx = self.current_worklist_idx.load(Ordering::Relaxed);

            // Create Bind Group (depends on current_worklist_idx)
            let bind_group = self.create_propagation_bind_group_for_pass(pass_input_idx);

            // Create Command Encoder
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("Propagation Pass {} Encoder", current_pass)),
                });

            // --- Read Input Worklist Size ---
            let worklist_size_data = match crate::buffers::download_buffer_data::<u32>(
                Some(self.device.clone()),
                Some(self.queue.clone()),
                &*self.buffers.worklist_buffers.worklist_count_buf,
                &*self.buffers.worklist_buffers.staging_worklist_count_buf,
                std::mem::size_of::<u32>() as u64,
                Some("Download Worklist Count".to_string()),
            )
            .await
            {
                Ok(data) => data,
                Err(e) => return Err(PropagationError::GpuCommunicationError(e.to_string())),
            };

            let worklist_size = worklist_size_data.first().cloned().unwrap_or(0);

            if worklist_size == 0 {
                debug!(
                    "Propagation Pass {}: Worklist empty. Finishing.",
                    current_pass
                );
                break;
            }

            // --- Propagation Compute Pass ---
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Propagation Pass {}", current_pass)),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.pipelines.propagation_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                let workgroup_size = 256;
                let num_workgroups = (worklist_size + workgroup_size - 1) / workgroup_size;

                debug!(
                    "Propagation Pass {}: Dispatching {} workgroups for {} items.",
                    current_pass, num_workgroups, worklist_size
                );

                compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
            }

            // Zero the output worklist count buffer before the next pass uses it as input
            // IMPORTANT: This must happen *after* the compute pass that writes to it,
            // and *before* the next pass reads it.
            let output_count_buffer = &self.buffers.worklist_buffers.worklist_count_buf;
            encoder.clear_buffer(output_count_buffer, 0, None);

            self.queue.submit(Some(encoder.finish()));
            let _ = self.device.poll(wgpu::MaintainBase::Wait);

            // --- Check for contradictions periodically ---
            if current_pass % self.params.contradiction_check_frequency == 0 {
                let contradiction_flag_data = match crate::buffers::download_buffer_data::<u32>(
                    Some(self.device.clone()),
                    Some(self.queue.clone()),
                    &*self.buffers.contradiction_flag_buf,
                    &*self.buffers.staging_contradiction_flag_buf,
                    std::mem::size_of::<u32>() as u64,
                    Some("Contradiction Flag".to_string()),
                )
                .await
                {
                    Ok(data) => data,
                    Err(e) => return Err(PropagationError::GpuCommunicationError(e.to_string())),
                };

                let contradiction_flag = contradiction_flag_data.first().cloned().unwrap_or(0);

                if contradiction_flag != 0 {
                    let contradiction_loc_data = match crate::buffers::download_buffer_data::<u32>(
                        Some(self.device.clone()),
                        Some(self.queue.clone()),
                        &*self.buffers.contradiction_location_buf,
                        &*self.buffers.staging_contradiction_location_buf,
                        3 * std::mem::size_of::<u32>() as u64,
                        Some("Contradiction Location".to_string()),
                    )
                    .await
                    {
                        Ok(data) => data,
                        Err(e) => {
                            return Err(PropagationError::GpuCommunicationError(e.to_string()))
                        }
                    };

                    let contradiction_loc = contradiction_loc_data
                        .first()
                        .cloned()
                        .unwrap_or(std::u32::MAX);
                    let (width, height, _) = self.buffers.grid_dims;
                    let (cx, cy, cz) = index_to_coords(contradiction_loc as usize, width, height);
                    return Err(PropagationError::Contradiction(cx, cy, cz));
                }
            }

            // --- Prepare for next pass: Swap Worklists --- //
            // Atomically swap the index for the next iteration
            self.current_worklist_idx.fetch_xor(1, Ordering::Relaxed);

            // --- Check Early Termination ---
            let output_worklist_size_data = match crate::buffers::download_buffer_data::<u32>(
                Some(self.device.clone()),
                Some(self.queue.clone()),
                &*self.buffers.worklist_buffers.worklist_count_buf,
                &*self.buffers.worklist_buffers.staging_worklist_count_buf,
                std::mem::size_of::<u32>() as u64,
                Some("Output Worklist Count".to_string()),
            )
            .await
            {
                Ok(data) => data,
                Err(e) => return Err(PropagationError::GpuCommunicationError(e.to_string())),
            };

            let output_worklist_size = output_worklist_size_data.first().cloned().unwrap_or(0);

            if output_worklist_size < self.early_termination_threshold {
                consecutive_low_worklist_passes += 1;
                if consecutive_low_worklist_passes >= self.early_termination_consecutive_passes {
                    debug!(
                        "Propagation Pass {}: Early termination condition met ({} passes with < {} items).",
                        current_pass,
                        consecutive_low_worklist_passes,
                        self.early_termination_threshold
                    );
                    break;
                }
            } else {
                consecutive_low_worklist_passes = 0; // Reset counter
            }

            current_pass += 1;
        }

        // Final contradiction check after loop finishes
        let contradiction_flag_data = match crate::buffers::download_buffer_data::<u32>(
            Some(self.device.clone()),
            Some(self.queue.clone()),
            &*self.buffers.contradiction_flag_buf,
            &*self.buffers.staging_contradiction_flag_buf,
            std::mem::size_of::<u32>() as u64,
            Some("Final Contradiction Flag".to_string()),
        )
        .await
        {
            Ok(data) => data,
            Err(e) => return Err(PropagationError::GpuCommunicationError(e.to_string())),
        };

        let contradiction_flag = contradiction_flag_data.first().cloned().unwrap_or(0);

        if contradiction_flag != 0 {
            let contradiction_loc_data = match crate::buffers::download_buffer_data::<u32>(
                Some(self.device.clone()),
                Some(self.queue.clone()),
                &*self.buffers.contradiction_location_buf,
                &*self.buffers.staging_contradiction_location_buf,
                3 * std::mem::size_of::<u32>() as u64,
                Some("Final Contradiction Location".to_string()),
            )
            .await
            {
                Ok(data) => data,
                Err(e) => {
                    error!(
                        "Failed to get contradiction location after flag was set: {}",
                        e
                    );
                    return Err(PropagationError::GpuCommunicationError(e.to_string()));
                }
            };

            let contradiction_loc = contradiction_loc_data
                .first()
                .cloned()
                .unwrap_or(std::u32::MAX);
            let (width, height, _) = self.buffers.grid_dims;
            let (cx, cy, cz) = index_to_coords(contradiction_loc as usize, width, height);
            return Err(PropagationError::Contradiction(cx, cy, cz));
        }

        debug!(
            "GPU Propagation finished in {:.2?} after {} passes.",
            start_time.elapsed(),
            current_pass
        );

        Ok(())
    }

    // Helper to create bind group based on current pass index
    fn create_propagation_bind_group_for_pass(&self, pass_input_idx: usize) -> wgpu::BindGroup {
        let input_binding = if pass_input_idx == 0 {
            self.buffers
                .worklist_buffers
                .worklist_buf_a
                .as_entire_binding()
        } else {
            self.buffers
                .worklist_buffers
                .worklist_buf_b
                .as_entire_binding()
        };
        let output_binding = if pass_input_idx == 0 {
            self.buffers
                .worklist_buffers
                .worklist_buf_b
                .as_entire_binding()
        } else {
            self.buffers
                .worklist_buffers
                .worklist_buf_a
                .as_entire_binding()
        };

        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Propagation Bind Group"),
            layout: &self.pipelines.propagation_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffers.params_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self
                        .buffers
                        .grid_buffers
                        .grid_possibilities_buf
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self
                        .buffers
                        .rule_buffers
                        .adjacency_rules_buf
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: input_binding,
                }, // Input worklist (read)
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_binding,
                }, // Output worklist (write)
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self
                        .buffers
                        .worklist_buffers
                        .worklist_count_buf
                        .as_entire_binding(),
                }, // Worklist count (atomic)
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.buffers.contradiction_flag_buf.as_entire_binding(),
                }, // Contradiction flag (atomic)
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.buffers.contradiction_location_buf.as_entire_binding(),
                }, // Contradiction location (atomic)
            ],
        })
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
            if let Ok(mut _vis) = visualizer.lock() {
                // Commenting out the take_debug_snapshot call
                // let _ = self.buffers.take_debug_snapshot(&mut _vis);
                // TODO: Fix snapshotting (see above)
            }
        }

        // Convert indices back to coordinates for subgrid processing
        let (width, height, _depth) = (grid.width, grid.height, grid.depth);
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

            processed_subgrids.push((*region, processed));

            // Take a snapshot after each subgrid if debug visualization is enabled
            if let Some(visualizer) = &self.debug_visualizer {
                if let Ok(mut _vis) = visualizer.lock() {
                    // Commenting out the take_debug_snapshot call
                    // let _ = self.buffers.take_debug_snapshot(&mut _vis);
                    // TODO: Fix snapshotting (see above)
                }
            }
        }

        // Merge the processed subgrids back into the main grid
        let merged_updates = merge_subgrids(grid, &processed_subgrids, &config)
            .map_err(|e| PropagationError::GpuSetupError(e.to_string()))?;

        info!(
            "Parallel subgrid processing: Merged {} subgrids with {} updated cells",
            processed_subgrids.len(),
            merged_updates.len()
        );

        // Take a snapshot after merging if debug visualization is enabled
        if let Some(visualizer) = &self.debug_visualizer {
            if let Ok(mut _vis) = visualizer.lock() {
                // Commenting out the take_debug_snapshot call
                // let _ = self.buffers.take_debug_snapshot(&mut _vis);
                // TODO: Fix snapshotting (see above)
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
        &self,
        grid: &mut PossibilityGrid,
        updated_coords: Vec<(usize, usize, usize)>,
        rules: &AdjacencyRules,
    ) -> Result<(), PropagationError> {
        debug!("Starting GPU constraint propagation...");
        let (width, height, _depth) = (grid.width, grid.height, grid.depth);

        // Convert coordinates to flat indices
        let updated_indices: Vec<u32> = updated_coords
            .into_iter()
            .map(|(x, y, z)| coords_to_index(x, y, z, width, height))
            .collect();

        // Call the internal async implementation
        let result = self.propagate_internal(grid, updated_indices, rules).await;

        // Note: Grid is not modified directly by propagate_internal anymore.
        // The caller (e.g., GpuAccelerator) is responsible for downloading the final state if needed.

        debug!("Finished GPU constraint propagation.");
        result
    }
}

// Utility function (consider moving to a utils module)
fn coords_to_index(x: usize, y: usize, z: usize, width: usize, height: usize) -> u32 {
    (z * width * height + y * width + x) as u32
}

fn index_to_coords(index: usize, width: usize, height: usize) -> (usize, usize, usize) {
    let z = index / (width * height);
    let y = (index % (width * height)) / width;
    let x = index % width;
    (x, y, z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        accelerator::GridDefinition, buffers::GpuBuffers, entropy::GpuEntropyCalculator,
        test_utils::create_test_device_queue,
    };
    use futures::executor::block_on;
    use std::sync::Arc;
    use wfc_core::{grid::PossibilityGrid, BoundaryCondition};
    use wfc_rules::{AdjacencyRules, TileSet, Transformation};

    #[tokio::test]
    async fn test_propagation_initialization() {
        let (_device, _queue) = create_test_device_queue();
        // ... rest of test ...
    }

    // ... other tests ...
}
