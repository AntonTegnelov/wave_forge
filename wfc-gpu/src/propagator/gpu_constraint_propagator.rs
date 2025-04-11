use crate::{
    buffers::{GpuBuffers, GpuParamsUniform},
    debug_viz::DebugVisualizer,
    gpu::sync::GpuSynchronizer,
    shader::pipeline::ComputePipelines,
    subgrid::{
        divide_into_subgrids, extract_subgrid, merge_subgrids, SubgridConfig, SubgridRegion,
    },
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
                    resource: self
                        .buffers
                        .rule_buffers
                        .rule_weights_buf
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.input_worklist_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.output_worklist_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.input_worklist_count_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.buffers.contradiction_flag_buf.as_entire_binding(),
                },
            ],
        })
    }

    async fn propagate_subgrid(
        &mut self,
        _grid: &PossibilityGrid,
        subgrid: PossibilityGrid,
        region: &SubgridRegion,
        updated_coords: Vec<(usize, usize, usize)>,
        rules: &AdjacencyRules,
    ) -> Result<PossibilityGrid, PropagationError> {
        // Create a new GpuConstraintPropagator specifically for this subgrid
        let device = self.device.clone();
        let queue = self.queue.clone();
        let pipelines = self.pipelines.clone();

        // Adjust the updated coordinates to be relative to the subgrid
        let adjusted_coords: Vec<(usize, usize, usize)> = updated_coords
            .into_iter()
            .filter_map(|(x, y, z)| {
                // Check if coordinate is within the region (with overlap)
                if x >= region.x_start
                    && x < region.x_end
                    && y >= region.y_start
                    && y < region.y_end
                    && z >= region.z_start
                    && z < region.z_end
                {
                    // Convert to subgrid-local coordinates
                    Some((x - region.x_start, y - region.y_start, z - region.z_start))
                } else {
                    None
                }
            })
            .collect();

        // Create buffers specifically for this subgrid
        let subgrid_width = region.x_end - region.x_start;
        let subgrid_height = region.y_end - region.y_start;
        let subgrid_depth = region.z_end - region.z_start;

        let mut subgrid_mutable = subgrid.clone();

        if adjusted_coords.is_empty() {
            // No cells to update in this subgrid
            return Ok(subgrid);
        }

        // Directly convert coordinates to indices
        let mut adjusted_indices = Vec::with_capacity(adjusted_coords.len());
        for (x, y, z) in adjusted_coords {
            let index = super::utils::coords_to_index(x, y, z, subgrid_width, subgrid_height);
            adjusted_indices.push(index);
        }

        // Create a temporary GPU constraint propagator for this subgrid
        let subgrid_params = GpuParamsUniform {
            width: subgrid_width as u32,
            height: subgrid_height as u32,
            depth: subgrid_depth as u32,
            pattern_count: self.params.pattern_count,
            boundary_mode: self.params.boundary_mode,
        };

        // Use a fresh set of buffers from the buffer manager for this subgrid
        let subgrid_buffers = GpuBuffers::new(
            device.clone(),
            &subgrid_params,
            rules,
            &subgrid_mutable,
            None, // No custom buffer config for subgrids
        )
        .await
        .map_err(|e| {
            PropagationError::InternalError(format!("Subgrid buffer creation failed: {}", e))
        })?;

        let mut subgrid_propagator = GpuConstraintPropagator::new(
            device,
            queue,
            pipelines,
            Arc::new(subgrid_buffers),
            (subgrid_width, subgrid_height, subgrid_depth),
            BoundaryCondition::Open, // Inside a subgrid, we use open boundaries
            subgrid_params,
        );

        // Process the subgrid with direct propagation
        subgrid_propagator
            .propagate_internal(
                &mut subgrid_mutable, // Not used directly in propagate_internal
                adjusted_indices,
                rules,
            )
            .await?;

        // Download the resulting grid state
        let result = subgrid_propagator
            .synchronizer
            .download_grid_async(&mut subgrid_mutable)
            .await
            .map_err(|e| {
                PropagationError::InternalError(format!("Failed to download subgrid: {}", e))
            })?;

        Ok(result)
    }

    async fn propagate_internal(
        &self,
        _grid: &mut PossibilityGrid, // Grid not directly modified here anymore
        updated_indices: Vec<u32>,
        _rules: &AdjacencyRules,
    ) -> Result<(), PropagationError> {
        // Skip propagation if no cells were updated
        if updated_indices.is_empty() {
            return Ok(());
        }

        let device = &self.device;
        let queue = &self.queue;
        let label = "Propagation";

        debug!(
            "Beginning constraint propagation with {} updated cells",
            updated_indices.len()
        );

        // Reset state for propagation
        queue.write_buffer(
            &self.buffers.contradiction_flag_buf,
            0,
            bytemuck::cast_slice(&[0u32]),
        );

        // Initialize the worklist with the updated indices
        let max_worklist_size = self.buffers.worklist_buffers.max_worklist_size;

        // Ensure we don't exceed the buffer size
        if updated_indices.len() > max_worklist_size {
            return Err(PropagationError::InternalError(format!(
                "Worklist size ({}) exceeds maximum allowable size ({})",
                updated_indices.len(),
                max_worklist_size
            )));
        }

        // Initialize with current index being worklist A
        self.current_worklist_idx.store(0, Ordering::Relaxed);

        // Write the initial worklist
        queue.write_buffer(
            &self.buffers.worklist_buffers.worklist_buf_a,
            0,
            bytemuck::cast_slice(&updated_indices),
        );
        queue.write_buffer(
            &self.buffers.worklist_buffers.worklist_count_buf,
            0,
            bytemuck::cast_slice(&[updated_indices.len() as u32]),
        );

        // Download buffers for checking propagation termination conditions
        let mut contradiction_flag = [0u32; 1];
        let mut worklist_count = [0u32; 1];

        let mut consecutive_low_impact_passes: u32 = 0;
        let mut pass_idx = 0;

        // Ping-pong between worklists until propagation completes
        loop {
            pass_idx += 1;

            // Create a bind group based on the current input/output worklists
            let bind_group = self.create_propagation_bind_group_for_pass(
                self.current_worklist_idx.load(Ordering::Relaxed),
            );

            // Start timestamp for this pass
            let timestamp_start = std::time::Instant::now();

            // Create a command encoder for this pass
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("PropagationPass-{}", pass_idx)),
            });

            {
                // Begin a compute pass for propagation
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("{}ComputePass-{}", label, pass_idx)),
                    timestamp_writes: None,
                });

                // Set pipeline and bind groups
                compute_pass.set_pipeline(&self.pipelines.propagation_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                // Dispatch based on worklist size
                queue.submit(Some(encoder.finish()));

                // Read worklist count before dispatch to determine workgroup count
                let synchronizer = &self.synchronizer;
                worklist_count = synchronizer
                    .download_buffer_async(
                        &self.buffers.worklist_buffers.worklist_count_buf,
                        0,
                        std::mem::size_of::<u32>(),
                    )
                    .await
                    .map_err(|e| {
                        PropagationError::InternalError(format!(
                            "Failed to download worklist count: {}",
                            e
                        ))
                    })?;

                let current_worklist_count = worklist_count[0] as usize;

                // Start a new encoder and compute pass with the right workgroup size
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("PropagationDispatchPass-{}", pass_idx)),
                });

                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some(&format!("{}DispatchPass-{}", label, pass_idx)),
                            timestamp_writes: None,
                        });

                    compute_pass.set_pipeline(&self.pipelines.propagation_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);

                    // Calculate workgroup count
                    let workgroup_size = 64; // Must match the shader
                    let workgroups =
                        (current_worklist_count as f32 / workgroup_size as f32).ceil() as u32;

                    if workgroups > 0 {
                        // Only dispatch if we have items in the worklist
                        compute_pass.dispatch_workgroups(workgroups, 1, 1);
                    }
                }

                // Submit the dispatch command buffer
                queue.submit(Some(encoder.finish()));
            }

            // Check for contradiction after each pass
            contradiction_flag = self
                .synchronizer
                .download_buffer_async(&self.buffers.contradiction_flag_buf, 0, 4)
                .await
                .map_err(|e| {
                    PropagationError::InternalError(format!(
                        "Failed to download contradiction flag: {}",
                        e
                    ))
                })?;

            if contradiction_flag[0] != 0 {
                return Err(PropagationError::Contradiction);
            }

            // Check the output worklist size to determine whether to continue
            let output_idx = 1 - self.current_worklist_idx.load(Ordering::Relaxed);
            let output_worklist_count_buf = if output_idx == 0 {
                &self.buffers.worklist_buffers.worklist_count_buf
            } else {
                &self.buffers.worklist_buffers.worklist_count_buf
            };

            // Download the output worklist count
            worklist_count = self
                .synchronizer
                .download_buffer_async(output_worklist_count_buf, 0, 4)
                .await
                .map_err(|e| {
                    PropagationError::InternalError(format!(
                        "Failed to download worklist count: {}",
                        e
                    ))
                })?;

            let output_worklist_count = worklist_count[0] as u32;

            // Calculate pass duration
            let pass_duration = timestamp_start.elapsed();
            debug!(
                "Propagation pass {} completed in {:?} with {} affected cells",
                pass_idx, pass_duration, output_worklist_count
            );

            // Update the worklist index for the next pass
            self.current_worklist_idx
                .store(output_idx, Ordering::Relaxed);

            // Early termination check
            if output_worklist_count <= self.early_termination_threshold {
                consecutive_low_impact_passes += 1;
                if consecutive_low_impact_passes >= self.early_termination_consecutive_passes {
                    debug!(
                        "Early termination after {} consecutive low-impact passes",
                        consecutive_low_impact_passes
                    );
                    break;
                }
            } else {
                consecutive_low_impact_passes = 0;
            }

            // Done if no more cells to propagate
            if output_worklist_count == 0 {
                break;
            }

            // Safety limit on propagation passes
            if pass_idx > 1000 {
                return Err(PropagationError::InternalError(
                    "Propagation exceeded 1000 passes, possible infinite loop".to_string(),
                ));
            }
        }

        debug!("Propagation complete after {} passes", pass_idx);
        Ok(())
    }

    /// Creates a bind group for a specific propagation pass.
    fn create_propagation_bind_group_for_pass(&self, pass_input_idx: usize) -> wgpu::BindGroup {
        // Get the current input and output worklists
        let input_worklist = if pass_input_idx == 0 {
            &self.buffers.worklist_buffers.worklist_buf_a
        } else {
            &self.buffers.worklist_buffers.worklist_buf_b
        };

        let output_worklist = if pass_input_idx == 0 {
            &self.buffers.worklist_buffers.worklist_buf_b
        } else {
            &self.buffers.worklist_buffers.worklist_buf_a
        };

        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Propagation Bind Group Pass {}", pass_input_idx)),
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
                    resource: self
                        .buffers
                        .rule_buffers
                        .rule_weights_buf
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: input_worklist.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: output_worklist.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self
                        .buffers
                        .worklist_buffers
                        .worklist_count_buf
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.buffers.contradiction_flag_buf.as_entire_binding(),
                },
            ],
        })
    }

    async fn propagate_with_subgrids(
        &mut self,
        grid: &mut PossibilityGrid,
        updated_indices: Vec<u32>,
        rules: &AdjacencyRules,
    ) -> Result<(), PropagationError> {
        debug!(
            "Using subgrid propagation strategy for grid size: {}x{}x{}",
            grid.width(),
            grid.height(),
            grid.depth()
        );

        // Convert updated indices to coordinates for subgrid processing
        let updated_coords = updated_indices
            .iter()
            .map(|&index| {
                super::utils::index_to_coords(index as usize, grid.width(), grid.height())
            })
            .collect::<Vec<_>>();

        // Get configuration for subgrid division
        let config = self
            .subgrid_config
            .as_ref()
            .unwrap_or(&SubgridConfig::default());

        // Divide the grid into subgrids
        let subgrids = divide_into_subgrids(
            grid.width(),
            grid.height(),
            grid.depth(),
            config.max_subgrid_size,
            config.overlap_size,
        );

        debug!("Divided grid into {} subgrids", subgrids.len());

        // Create subgrids from the main grid
        let mut subgrid_results = Vec::with_capacity(subgrids.len());

        for region in &subgrids {
            let subgrid = extract_subgrid(grid, region);

            debug!(
                "Processing subgrid: x={}-{}, y={}-{}, z={}-{}",
                region.x_start,
                region.x_end,
                region.y_start,
                region.y_end,
                region.z_start,
                region.z_end
            );

            // Process this subgrid
            let result = self
                .propagate_subgrid(grid, subgrid, region, updated_coords.clone(), rules)
                .await?;

            subgrid_results.push((region.clone(), result));
        }

        // Merge results back into the main grid
        debug!(
            "Merging {} subgrid results back into main grid",
            subgrid_results.len()
        );
        merge_subgrids(grid, subgrid_results);

        Ok(())
    }

    pub fn init_default() -> Self {
        // This function is not properly implemented yet, make it obvious it should not be used
        unimplemented!("Default initialization not implemented yet")
    }
}

impl Drop for GpuConstraintPropagator {
    /// Clean up any resources when the propagator is dropped.
    fn drop(&mut self) {
        // GPU resources are handled through Arc, so they will be cleaned up
        // automatically when the last reference is dropped.

        // If we need to perform any explicit cleanup beyond reference counting,
        // it would go here.
        debug!("GpuConstraintPropagator dropped");
    }
}

#[async_trait]
impl ConstraintPropagator for GpuConstraintPropagator {
    /// Propagates constraints based on updates to cell possibilities.
    ///
    /// # Arguments
    ///
    /// * `grid` - The grid to propagate constraints through
    /// * `updated_coords` - The coordinates of cells that were updated
    /// * `rules` - The adjacency rules to apply during propagation
    ///
    async fn propagate(
        &self,
        grid: &mut PossibilityGrid,
        updated_coords: Vec<(usize, usize, usize)>,
        rules: &AdjacencyRules,
    ) -> Result<(), PropagationError> {
        // Convert updated coordinates to indices
        let updated_indices = updated_coords
            .iter()
            .map(|(x, y, z)| super::utils::coords_to_index(*x, *y, *z, grid.width(), grid.height()))
            .collect::<Vec<_>>();

        // If subgrid processing is enabled and the grid is large enough, use it
        if let Some(config) = &self.subgrid_config {
            if grid.width() > config.min_grid_size_for_subgrids
                || grid.height() > config.min_grid_size_for_subgrids
                || grid.depth() > config.min_grid_size_for_subgrids
            {
                // Create a mutable clone for subgrid processing
                let mut propagator_clone = self.clone();
                return propagator_clone
                    .propagate_with_subgrids(grid, updated_indices, rules)
                    .await;
            }
        }

        // Otherwise, use direct propagation
        self.propagate_internal(grid, updated_indices, rules).await
    }
}
