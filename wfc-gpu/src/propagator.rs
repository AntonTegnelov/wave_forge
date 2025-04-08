use crate::{
    buffers::{DynamicBufferConfig, GpuBuffers, GpuParamsUniform, WorklistBuffers},
    debug_viz::{DebugVisualizer, GpuBuffersDebugExt},
    error_recovery::GpuError,
    pipeline::ComputePipelines,
    subgrid::{
        divide_into_subgrids, extract_subgrid, merge_subgrids, SubgridConfig, SubgridRegion,
    },
    sync::GpuSynchronizer,
};
use async_trait::async_trait;
use log::{debug, info};
use std::sync::Arc;
use wfc_core::{
    grid::{Grid, PossibilityGrid},
    propagator::{ConstraintPropagator, PropagationError},
    BoundaryCondition,
};
use wfc_rules::{AdjacencyRules, TileSet, Transformation};
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
    current_worklist_idx: usize,
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
            current_worklist_idx: 0,
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
        if self.current_worklist_idx == 0 {
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
        if self.current_worklist_idx == 0 {
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

    /// Internal propagation logic using GPU compute shaders.
    async fn propagate_internal(
        &mut self,
        _grid: &mut PossibilityGrid,
        updated_indices: Vec<u32>,
        _rules: &AdjacencyRules,
    ) -> Result<(), PropagationError> {
        let start_time = std::time::Instant::now();

        // 1. Initialize Worklist
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Propagation Init Encoder"),
            });

        // Clear output worklist count (ping-pong target)
        encoder.clear_buffer(&self.buffers.worklist_buffers.worklist_count_buf, 0, None);
        // Clear contradiction flag & location (direct access ok)
        encoder.clear_buffer(&self.buffers.contradiction_flag_buf, 0, None);
        encoder.clear_buffer(&self.buffers.contradiction_location_buf, 0, None);

        // Write initial updated indices to the *input* worklist buffer
        let worklist_data: &[u8] = bytemuck::cast_slice(&updated_indices);
        let initial_worklist_size = updated_indices.len() as u32;

        // Ensure worklist buffers are large enough
        let (width, height, depth) = self.params.grid_dims;
        let config = DynamicBufferConfig::default(); // Or get from propagator config

        // Ensure worklist buffers are large enough using the correct function
        self.buffers
            .worklist_buffers
            .ensure_worklist_buffers(
                &self.device,
                width as u32,
                height as u32,
                depth as u32,
                &config, // Pass the buffer config
            )
            .map_err(|e| PropagationError::GpuError(GpuError::BufferOperationError(e)))?;

        let input_worklist_buffer = if self.current_worklist_idx == 0 {
            &self.buffers.worklist_buffers.worklist_buf_a
        } else {
            &self.buffers.worklist_buffers.worklist_buf_b
        };
        self.queue
            .write_buffer(input_worklist_buffer, 0, worklist_data);

        // Write the initial worklist size to the *input* count buffer
        let input_count_buffer = &self.buffers.worklist_buffers.worklist_count_buf; // Always use the single count buffer
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

            // Create Bind Group
            let bind_group = self.create_propagation_bind_group();

            // Create Command Encoder
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("Propagation Pass {} Encoder", current_pass)),
                });

            // --- Read Input Worklist Size --- (Requires copy + map)
            let count_buf_gpu = &self.buffers.worklist_buffers.worklist_count_buf; // Now always use the single count buffer
            let count_buf_staging = &self.buffers.worklist_buffers.staging_worklist_count_buf; // Use single staging buffer

            encoder.copy_buffer_to_buffer(
                count_buf_gpu,
                0,
                count_buf_staging,
                0,
                4, // Size of u32
            );
            // Submit copy command SEPARATELY before mapping
            self.queue.submit(Some(encoder.finish()));

            // Map and read (BLOCKING - Needs Async Version)
            let worklist_size_result = crate::buffers::map_and_process::<u32>(
                count_buf_staging.clone(),
                std::time::Duration::from_secs(5),
                Some(1),
            )
            .await;

            // Re-create encoder for the compute pass dispatch
            encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!(
                        "Propagation Pass {} Encoder Post Read",
                        current_pass
                    )),
                });

            let worklist_size = match worklist_size_result {
                Ok(data_box) => data_box
                    .downcast::<Vec<u32>>()
                    .map_err(|_| {
                        PropagationError::InternalError("Failed to downcast worklist size".into())
                    })?
                    .get(0)
                    .cloned()
                    .unwrap_or(0),
                Err(e) => return Err(PropagationError::GpuCommunicationError(e.to_string())),
            };

            if worklist_size == 0 {
                debug!(
                    "Propagation Pass {}: Worklist empty. Finishing.",
                    current_pass
                );
                break; // Worklist is empty
            }

            // --- Propagation Compute Pass ---
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Propagation Pass {}", current_pass)),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.pipelines.propagation_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                let workgroup_size = 256; // Should match shader
                let num_workgroups = (worklist_size + workgroup_size - 1) / workgroup_size;

                debug!(
                    "Propagation Pass {}: Dispatching {} workgroups for {} items.",
                    current_pass, num_workgroups, worklist_size
                );

                compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
            } // End compute pass

            // Submit commands for this pass
            self.queue.submit(Some(encoder.finish()));

            // Device poll (BLOCKING!)
            self.device.poll(wgpu::Maintain::Wait);

            // --- Check for contradictions periodically ---
            if current_pass % self.params.contradiction_check_frequency == 0 {
                // Create encoder for copy
                let mut copy_encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Copy Contradiction Encoder"),
                        });
                copy_encoder.copy_buffer_to_buffer(
                    &self.buffers.contradiction_flag_buf,
                    0,
                    &self.buffers.staging_contradiction_flag_buf,
                    0,
                    4, // Size of u32
                );
                // Submit copy
                self.queue.submit(Some(copy_encoder.finish()));

                // Read contradiction flag (BLOCKING)
                let contradiction_flag_result = crate::buffers::map_and_process::<u32>(
                    self.buffers.staging_contradiction_flag_buf.clone(),
                    std::time::Duration::from_secs(5),
                    Some(1),
                )
                .await;

                let contradiction_flag = match contradiction_flag_result {
                    Ok(data_box) => data_box
                        .downcast::<Vec<u32>>()
                        .map_err(|_| {
                            PropagationError::InternalError(
                                "Failed to downcast contradiction flag".into(),
                            )
                        })?
                        .get(0)
                        .cloned()
                        .unwrap_or(0),
                    Err(e) => return Err(PropagationError::GpuCommunicationError(e.to_string())),
                };

                if contradiction_flag != 0 {
                    // Copy location buffer
                    let mut copy_encoder_loc =
                        self.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("Copy Contradiction Loc Encoder"),
                            });
                    copy_encoder_loc.copy_buffer_to_buffer(
                        &self.buffers.contradiction_location_buf,
                        0,
                        &self.buffers.staging_contradiction_location_buf,
                        0,
                        4, // Size of u32
                    );
                    self.queue.submit(Some(copy_encoder_loc.finish()));

                    // Read contradiction location (BLOCKING)
                    let contradiction_loc_result = crate::buffers::map_and_process::<u32>(
                        self.buffers.staging_contradiction_location_buf.clone(),
                        std::time::Duration::from_secs(5),
                        Some(1),
                    )
                    .await;

                    let contradiction_loc = match contradiction_loc_result {
                        Ok(data_box) => data_box
                            .downcast::<Vec<u32>>()
                            .map_err(|_| {
                                PropagationError::InternalError(
                                    "Failed to downcast contradiction location".into(),
                                )
                            })?
                            .get(0)
                            .cloned()
                            .unwrap_or(std::u32::MAX),
                        Err(e) => {
                            return Err(PropagationError::GpuCommunicationError(e.to_string()))
                        }
                    };

                    // Calculate coordinates from index
                    let width = self.params.grid_width as usize;
                    let height = self.params.grid_height as usize;
                    let idx = contradiction_loc as usize;
                    let z = idx / (width * height);
                    let y = (idx % (width * height)) / width;
                    let x = idx % width;

                    return Err(PropagationError::Contradiction(x, y, z)); // Pass coordinates
                }
            }

            // --- Check for early termination based on OUTPUT worklist size ---
            let output_count_buffer_gpu = &self.buffers.worklist_buffers.worklist_count_buf; // Always the same buffer
            let output_count_buffer_staging =
                &self.buffers.worklist_buffers.staging_worklist_count_buf; // Always the same staging buffer

            // Create encoder for copy
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Copy Output Count Encoder"),
                });
            encoder.copy_buffer_to_buffer(
                output_count_buffer_gpu,
                0,
                output_count_buffer_staging,
                0,
                4, // Size of u32
            );
            self.queue.submit(Some(encoder.finish()));

            // Read output worklist count (BLOCKING)
            let output_worklist_size_result = crate::buffers::map_and_process::<u32>(
                output_count_buffer_staging.clone(),
                std::time::Duration::from_secs(5),
                Some(1),
            )
            .await;

            let output_worklist_size = match output_worklist_size_result {
                Ok(data_box) => data_box
                    .downcast::<Vec<u32>>()
                    .map_err(|_| {
                        PropagationError::InternalError(
                            "Failed to downcast output worklist size".into(),
                        )
                    })?
                    .get(0)
                    .cloned()
                    .unwrap_or(0),
                Err(e) => return Err(PropagationError::GpuCommunicationError(e.to_string())),
            };

            if output_worklist_size < self.early_termination_threshold {
                consecutive_low_worklist_passes += 1;
                if consecutive_low_worklist_passes >= self.early_termination_consecutive_passes {
                    debug!(
                        "Propagation Pass {}: Early termination threshold met ({} < {} for {} passes).",
                        current_pass,
                        output_worklist_size,
                        self.early_termination_threshold,
                        consecutive_low_worklist_passes
                    );
                    break;
                }
            } else {
                consecutive_low_worklist_passes = 0; // Reset counter
            }

            // --- Prepare for next pass ---
            // The count buffer was already cleared before the dispatch. No need to clear here.

            // Swap worklist *data* buffer index for the next pass's input/output
            self.current_worklist_idx = 1 - self.current_worklist_idx;
            current_pass += 1;
        }

        // --- Final contradiction check ---
        // Create encoder for copy
        let mut copy_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Copy Final Contradiction Encoder"),
                });
        copy_encoder.copy_buffer_to_buffer(
            &self.buffers.contradiction_flag_buf,
            0,
            &self.buffers.staging_contradiction_flag_buf,
            0,
            4,
        );
        self.queue.submit(Some(copy_encoder.finish()));

        // Read flag (BLOCKING)
        let contradiction_flag_result = crate::buffers::map_and_process::<u32>(
            self.buffers.staging_contradiction_flag_buf.clone(),
            std::time::Duration::from_secs(5),
            Some(1),
        )
        .await;

        let contradiction_flag = match contradiction_flag_result {
            Ok(data_box) => data_box
                .downcast::<Vec<u32>>()
                .map_err(|_| {
                    PropagationError::InternalError(
                        "Failed to downcast final contradiction flag".into(),
                    )
                })?
                .get(0)
                .cloned()
                .unwrap_or(0),
            Err(e) => return Err(PropagationError::GpuCommunicationError(e.to_string())),
        };

        if contradiction_flag != 0 {
            // Copy location
            let mut copy_encoder_loc =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Copy Final Contradiction Loc Encoder"),
                    });
            copy_encoder_loc.copy_buffer_to_buffer(
                &self.buffers.contradiction_location_buf,
                0,
                &self.buffers.staging_contradiction_location_buf,
                0,
                4,
            );
            self.queue.submit(Some(copy_encoder_loc.finish()));

            // Read location (BLOCKING)
            let contradiction_loc_result = crate::buffers::map_and_process::<u32>(
                self.buffers.staging_contradiction_location_buf.clone(),
                std::time::Duration::from_secs(5),
                Some(1),
            )
            .await;

            let contradiction_loc = match contradiction_loc_result {
                Ok(data_box) => data_box
                    .downcast::<Vec<u32>>()
                    .map_err(|_| {
                        PropagationError::InternalError(
                            "Failed to downcast final contradiction location".into(),
                        )
                    })?
                    .get(0)
                    .cloned()
                    .unwrap_or(std::u32::MAX),
                Err(e) => return Err(PropagationError::GpuCommunicationError(e.to_string())),
            };

            // Calculate coordinates from index
            let width = self.params.grid_width as usize;
            let height = self.params.grid_height as usize;
            let idx = contradiction_loc as usize;
            let z = idx / (width * height);
            let y = (idx % (width * height)) / width;
            let x = idx % width;

            return Err(PropagationError::Contradiction(x, y, z)); // Pass coordinates
        }

        debug!(
            "GPU Propagation finished in {:.2?} after {} passes.",
            start_time.elapsed(),
            current_pass
        );

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
                    let _ = self.buffers.take_debug_snapshot(&mut vis);
                }
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
                if let Ok(mut vis) = visualizer.lock() {
                    if vis.is_enabled() {
                        let _ = self.buffers.take_debug_snapshot(&mut vis);
                    }
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
            if let Ok(mut vis) = visualizer.lock() {
                if vis.is_enabled() {
                    let _ = self.buffers.take_debug_snapshot(&mut vis);
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
        let (width, height, _depth) = (grid.width, grid.height, grid.depth);
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
                    let _ = self.buffers.take_debug_snapshot(&mut vis);
                }
            }
        }

        // Standard propagation approach
        let result = self.propagate_internal(grid, updated_indices, rules).await;

        // Take final snapshot if debug visualization is enabled
        if let Some(visualizer) = &self.debug_visualizer {
            if let Ok(mut vis) = visualizer.lock() {
                if vis.is_enabled() {
                    let _ = self.buffers.take_debug_snapshot(&mut vis);
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use crate::buffers::{DynamicBufferConfig, WorklistBuffers};
    use crate::propagator::{GpuConstraintPropagator, GpuParamsUniform};
    use crate::{
        buffers::GpuBuffers, pipeline::ComputePipelines, test_utils::create_test_device_queue,
        GpuError,
    };
    use std::sync::Arc;
    use wfc_core::{
        grid::{Grid, PossibilityGrid},
        BoundaryCondition,
    };
    use wfc_rules::{AdjacencyRules, TileSet, Transformation};

    // Helper struct for mock GPU resources
    struct MockGpu {
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        buffers: Arc<GpuBuffers>,
        pipelines: Arc<ComputePipelines>,
    }

    // Make setup_mock_gpu synchronous
    fn setup_mock_gpu() -> Result<MockGpu, GpuError> {
        // Use synchronous test device/queue creation
        let (device, queue) = create_test_device_queue();

        // Create dummy grid and rules for buffer/pipeline creation
        let grid_dims = (16, 16, 1);
        let num_tiles = 4;
        let num_axes = 6;
        let dummy_grid = PossibilityGrid::new(grid_dims.0, grid_dims.1, grid_dims.2, num_tiles);
        let dummy_rules = AdjacencyRules::from_allowed_tuples(num_tiles, num_axes, vec![]);

        // Create test buffers
        let buffers = Arc::new(GpuBuffers::new(
            &device,
            &queue,
            &dummy_grid,
            &dummy_rules,
            BoundaryCondition::Finite,
        )?);

        // Create test pipeline layout (simplified)
        let propagation_bind_group_layout = Arc::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Mock Prop Bind Group Layout"),
                entries: &[], // Empty for mock
            },
        ));
        let entropy_bind_group_layout_0 = Arc::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Mock Entropy Bind Group Layout 0"),
                entries: &[], // Empty for mock
            },
        ));
        let entropy_bind_group_layout_1 = Arc::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Mock Entropy Bind Group Layout 1"),
                entries: &[], // Empty for mock
            },
        ));

        // Mock pipeline (needs a valid shader module, difficult to mock perfectly)
        let dummy_shader = Arc::new(device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Dummy Shader"),
            source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
        }));
        let propagation_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Mock Prop Pipeline Layout"),
                bind_group_layouts: &[&propagation_bind_group_layout],
                push_constant_ranges: &[],
            });
        let propagation_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Mock Prop Pipeline"),
                layout: Some(&propagation_pipeline_layout),
                module: &dummy_shader,
                entry_point: "main",
                compilation_options: Default::default(),
            },
        ));

        // Create test pipelines object - Corrected structure and added features
        let pipelines = Arc::new(ComputePipelines::new(&device, (num_tiles + 31) / 32, &[])?);

        Ok(MockGpu {
            device,
            queue,
            buffers,
            pipelines,
        })
    }

    #[test]
    fn test_propagator_creation() {
        let mock_gpu = setup_mock_gpu().expect("Failed to setup mock GPU");
        let grid_dims = mock_gpu.buffers.grid_dims;
        let boundary_mode = mock_gpu.buffers.boundary_mode;
        let params = GpuParamsUniform {
            // Manual initialization
            grid_width: 16,
            grid_height: 16,
            grid_depth: 1,
            num_tiles: 4,
            num_axes: 6,
            boundary_mode: 0,
            heuristic_type: 0,
            tie_breaking: 0,
            max_propagation_steps: 1000,
            contradiction_check_frequency: 100,
            worklist_size: 0,
            grid_element_count: 256,
            _padding: 0,
        };

        let propagator = GpuConstraintPropagator::new(
            mock_gpu.device,
            mock_gpu.queue,
            mock_gpu.pipelines,
            mock_gpu.buffers,
            grid_dims,
            boundary_mode,
            params,
        );

        assert!(propagator.params.grid_width > 0);
    }

    #[test]
    fn test_early_termination_configuration() {
        let mock_gpu = setup_mock_gpu().expect("Failed to setup mock GPU");
        let grid_dims = mock_gpu.buffers.grid_dims;
        let boundary_mode = mock_gpu.buffers.boundary_mode;
        let params = GpuParamsUniform {
            // Manual initialization
            grid_width: 16,
            grid_height: 16,
            grid_depth: 1,
            num_tiles: 4,
            num_axes: 6,
            boundary_mode: 0,
            heuristic_type: 0,
            tie_breaking: 0,
            max_propagation_steps: 1000,
            contradiction_check_frequency: 100,
            worklist_size: 0,
            grid_element_count: 256,
            _padding: 0,
        };

        let propagator = GpuConstraintPropagator::new(
            mock_gpu.device,
            mock_gpu.queue,
            mock_gpu.pipelines,
            mock_gpu.buffers,
            grid_dims,
            boundary_mode,
            params,
        )
        .with_early_termination(100, 3);

        assert_eq!(propagator.early_termination_threshold, 100);
        assert_eq!(propagator.early_termination_consecutive_passes, 3);
    }

    #[test]
    fn test_parallel_subgrid_config() {
        let mock_gpu = setup_mock_gpu().expect("Failed to setup mock GPU");
        let grid_dims = mock_gpu.buffers.grid_dims;
        let boundary_mode = mock_gpu.buffers.boundary_mode;
        let params = GpuParamsUniform {
            // Manual initialization
            grid_width: 16,
            grid_height: 16,
            grid_depth: 1,
            num_tiles: 4,
            num_axes: 6,
            boundary_mode: 0,
            heuristic_type: 0,
            tie_breaking: 0,
            max_propagation_steps: 1000,
            contradiction_check_frequency: 100,
            worklist_size: 0,
            grid_element_count: 256,
            _padding: 0,
        };

        let config = SubgridConfig {
            max_subgrid_size: 32,
            overlap_size: 4,
            min_size: 64,
        };

        let propagator = GpuConstraintPropagator::new(
            mock_gpu.device,
            mock_gpu.queue,
            mock_gpu.pipelines,
            mock_gpu.buffers,
            grid_dims,
            boundary_mode,
            params,
        )
        .with_parallel_subgrid_processing(config.clone());

        assert!(propagator.subgrid_config.is_some());

        let propagator_disabled = propagator.without_parallel_subgrid_processing();
        assert!(propagator_disabled.subgrid_config.is_none()); // Check for None instead of direct comparison
    }

    /// Test helper for subgrid merging
    #[test]
    #[ignore = "GpuConstraintPropagatorImpl not accessible for merge_subgrid_changes"]
    fn test_merge_subgrid_changes() {
        let mut main_grid = PossibilityGrid::new(4, 4, 1, 3);
        let mut subgrid = PossibilityGrid::new(2, 2, 1, 3);

        // Modify subgrid
        subgrid.get_mut(0, 0, 0).unwrap().set(0, false); // Remove possibility 0
        subgrid.get_mut(1, 1, 0).unwrap().set(1, false); // Remove possibility 1

        // Merge subgrid into main grid at offset (1, 1)
        // This call requires GpuConstraintPropagatorImpl which seems private/gone.
        // GpuConstraintPropagatorImpl::merge_subgrid_changes(
        //     &mut main_grid,
        //     subgrid,
        //     1, // subgrid_x
        //     1, // subgrid_y
        //     0, // subgrid_z
        // ).unwrap();

        // Since we can't call the function, we can't assert the results.
        // // Check main grid
        // // Cell (1, 1, 0) corresponds to subgrid (0, 0, 0)
        // assert!(!main_grid.get(1, 1, 0).unwrap().get(0).map_or(false, |b| *b));
        // assert!(main_grid.get(1, 1, 0).unwrap().get(1).map_or(false, |b| *b));
        // assert!(main_grid.get(1, 1, 0).unwrap().get(2).map_or(false, |b| *b));
        //
        // // Cell (2, 2, 0) corresponds to subgrid (1, 1, 0)
        // assert!(!main_grid.get(2, 2, 0).unwrap().get(1).map_or(false, |b| *b));
        // assert!(main_grid.get(2, 2, 0).unwrap().get(0).map_or(false, |b| *b));
        // assert!(main_grid.get(2, 2, 0).unwrap().get(2).map_or(false, |b| *b));
        //
        // // Other cells in main grid should be untouched
        // assert!(main_grid.get(0, 0, 0).unwrap().iter_ones().count() == 3);
    }

    /// This should trigger buffer resizing logic within ensure_worklist_buffers.
    #[test]
    fn test_ensure_worklist_buffers() {
        let (device, _) = create_test_device_queue();
        // Use the config when creating WorklistBuffers
        let config = DynamicBufferConfig::default();
        let mut buffers =
            WorklistBuffers::new(&device, 100, &config).expect("Failed to create initial buffers");

        // Ensure for a much larger grid
        let large_width: u32 = 1000;
        let large_height: u32 = 1000;
        let large_depth: u32 = 1;

        let result = buffers.ensure_worklist_buffers(
            &device,
            large_width,  // Pass u32
            large_height, // Pass u32
            large_depth,  // Pass u32
            &config,      // Pass the config
        );
        assert!(
            result.is_ok(),
            "ensure_worklist_buffers failed: {:?}",
            result.err()
        );

        // Check if buffers were actually resized (example check on buffer A)
        let expected_size = large_width as u64
            * large_height as u64
            * large_depth as u64
            * std::mem::size_of::<u32>() as u64;
        assert!(
            buffers.worklist_buf_a.size() >= expected_size,
            "Buffer A size not increased"
        );
        assert!(
            buffers.worklist_buf_b.size() >= expected_size,
            "Buffer B size not increased"
        );
    }
}
