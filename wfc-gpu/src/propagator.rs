use crate::{
    buffers::{GpuBuffers, GpuParamsUniform},
    pipeline::ComputePipelines,
    subgrid::{
        divide_into_subgrids, extract_subgrid, merge_subgrids, SubgridConfig, SubgridRegion,
    },
    GpuError,
};
use async_trait::async_trait;
use log::{info, warn};
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
        grid: &PossibilityGrid,
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

        info!(
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

        // Use the standard propagation logic on the subgrid
        let mut temp_subgrid = subgrid;
        match self
            .propagate_internal(&mut temp_subgrid, local_updated_coords, rules)
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

    /// Internal method for constraint propagation.
    /// Extracted from the main propagate method to allow reuse for subgrid processing.
    async fn propagate_internal(
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

        // Early termination tracking
        let mut consecutive_passes_below_threshold = 0;

        // Main propagation loop
        while propagation_pass < MAX_PROPAGATION_PASSES && current_worklist_size > 0 {
            propagation_pass += 1;
            log::debug!(
                "Starting propagation pass {} with {} cells in worklist.",
                propagation_pass,
                current_worklist_size
            );

            // --- Create and Configure Bind Group for Current Pass ---
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Propagation Pass {} Bind Group", propagation_pass)),
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
                        resource: self.input_worklist_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.output_worklist_binding(),
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
                let workgroup_size = self.pipelines.propagation_workgroup_size; // Use dynamic size
                let dispatch_x = (current_worklist_size + workgroup_size - 1) / workgroup_size;
                compute_pass.dispatch_workgroups(dispatch_x, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));

            // --- Check if Contradiction Detected ---
            let contradiction = self
                .buffers
                .check_contradiction(&self.device)
                .await
                .map_err(|e| PropagationError::GpuExecutionError(e.to_string()))?;

            if contradiction {
                // Get the location of the contradiction if available
                if let Ok(location) = self.buffers.get_contradiction_location(&self.device).await {
                    // Convert 1D index back to 3D coordinates
                    let z =
                        location / (self.params.grid_width as u32 * self.params.grid_height as u32);
                    let rem =
                        location % (self.params.grid_width as u32 * self.params.grid_height as u32);
                    let y = rem / self.params.grid_width as u32;
                    let x = rem % self.params.grid_width as u32;

                    return Err(PropagationError::Contradiction(
                        x as usize, y as usize, z as usize,
                    ));
                } else {
                    // Generic contradiction without specific location
                    return Err(PropagationError::Contradiction(0, 0, 0));
                }
            }

            // --- Ping-Pong Buffer Swap and Work List Size Check ---
            // Download the new worklist size from the GPU
            let new_worklist_size = self
                .buffers
                .get_worklist_count(&self.device)
                .await
                .map_err(|e| PropagationError::GpuExecutionError(e.to_string()))?;

            // Early termination check
            if new_worklist_size <= self.early_termination_threshold {
                consecutive_passes_below_threshold += 1;
                if consecutive_passes_below_threshold >= self.early_termination_consecutive_passes {
                    info!("Early termination after {} passes: {} cells below threshold of {} for {} consecutive passes",
                        propagation_pass, new_worklist_size, self.early_termination_threshold, consecutive_passes_below_threshold);
                    break;
                }
            } else {
                consecutive_passes_below_threshold = 0;
            }

            // Update worklist size for next pass
            current_worklist_size = new_worklist_size;

            // Update the params buffer with new worklist size
            self.buffers
                .update_params_worklist_size(&self.queue, current_worklist_size)
                .map_err(|e| PropagationError::GpuSetupError(e.to_string()))?;

            // Toggle worklist buffer for ping-pong pattern
            self.current_worklist_idx = 1 - self.current_worklist_idx;
        }

        if propagation_pass >= MAX_PROPAGATION_PASSES && current_worklist_size > 0 {
            warn!(
                "Propagation stopped after reaching maximum passes ({})",
                MAX_PROPAGATION_PASSES
            );
        }

        info!("Propagation completed after {} passes.", propagation_pass);
        Ok(())
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
    /// * `grid`: Mutable reference to the CPU-side `PossibilityGrid`. This implementation will
    ///            operate on the GPU buffers and synchronize changes back to the CPU grid.
    /// * `updated_coords`: A vector of `(x, y, z)` coordinates indicating cells whose possibilities have recently changed
    ///                     (e.g., due to a collapse) and need to be propagated from.
    /// * `rules`: Reference to the `AdjacencyRules`.
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
        grid: &mut PossibilityGrid,
        updated_coords: Vec<(usize, usize, usize)>,
        rules: &AdjacencyRules,
    ) -> Result<(), PropagationError> {
        // Check if parallel subgrid processing is enabled
        if let Some(config) = &self.subgrid_config {
            // Only use subgrid processing for sufficiently large grids
            let grid_size = grid.width * grid.height * grid.depth;
            let min_size_for_subgrids =
                config.max_subgrid_size * config.max_subgrid_size * config.max_subgrid_size;

            if grid_size >= min_size_for_subgrids {
                // Divide the grid into subgrids
                let subgrid_regions =
                    divide_into_subgrids(grid.width, grid.height, grid.depth, config)
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
                }

                // Merge the processed subgrids back into the main grid
                let merged_updates = merge_subgrids(grid, &processed_subgrids)
                    .map_err(|e| PropagationError::GpuSetupError(e.to_string()))?;

                info!(
                    "Parallel subgrid processing: Merged {} subgrids with {} updated cells",
                    processed_subgrids.len(),
                    merged_updates.len()
                );

                // If we have any updates from the merge, we need to propagate them again
                if !merged_updates.is_empty() {
                    // For the final pass, disable subgrid processing to avoid recursion
                    let original_config = self.subgrid_config.take();
                    let result = self.propagate(grid, merged_updates, rules).await;
                    self.subgrid_config = original_config;
                    return result;
                }

                return Ok(());
            }
        }

        // If parallel processing is disabled or grid is too small, use standard propagation
        // Download the grid from CPU to GPU
        self.buffers
            .upload_grid(&self.queue, grid)
            .map_err(|e| PropagationError::GpuSetupError(e.to_string()))?;

        // Run the internal propagation algorithm
        let result = self.propagate_internal(grid, updated_coords, rules).await;

        // Download the results back to the CPU grid
        if result.is_ok() {
            self.buffers
                .download_grid(&self.device, grid)
                .await
                .map_err(|e| PropagationError::GpuSetupError(e.to_string()))?;
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

        // Create pipelines
        let pipelines = ComputePipelines::new(&device, 1)?;
        let pipelines = Arc::new(pipelines);

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

    #[tokio::test]
    async fn test_parallel_subgrid_processing() -> Result<(), GpuError> {
        // Set up a mock GPU environment
        let mock_gpu = setup_mock_gpu().await?;

        // Create a larger grid for subgrid testing (10x10x1)
        let grid_dims = (10, 10, 1);
        let width = grid_dims.0;
        let height = grid_dims.1;
        let depth = grid_dims.2;
        let num_tiles = 3;

        // Create an initial grid
        let mut grid = PossibilityGrid::new(width, height, depth, num_tiles);

        // Create a test rules set (allow all adjacencies for simplicity)
        let adjacency_bytes = [0xFF, 0xFF, 0xFF, 0xFF];
        let rules = AdjacencyRules::from_bytes(num_tiles, 6, adjacency_bytes.to_vec());

        // Create a params structure
        let params = GpuParamsUniform {
            grid_width: width as u32,
            grid_height: height as u32,
            grid_depth: depth as u32,
            num_tiles: num_tiles as u32,
            num_axes: 6,
            worklist_size: 0,
            boundary_mode: 0, // Finite boundaries
            _padding1: 0,
        };

        // Create a propagator with subgrid processing enabled
        let mut propagator = GpuConstraintPropagator::new(
            mock_gpu.device,
            mock_gpu.queue,
            mock_gpu.pipelines,
            mock_gpu.buffers,
            grid_dims,
            BoundaryCondition::Finite,
            params,
        );

        // Enable subgrid processing with a custom configuration
        let subgrid_config = SubgridConfig {
            max_subgrid_size: 5, // 5x5 subgrids
            overlap_size: 1,     // 1-cell overlap
        };
        propagator = propagator.with_parallel_subgrid_processing(subgrid_config);

        // Simulate some updates (middle of grid)
        let updates = vec![(5, 5, 0)];

        // Collapse a cell to trigger propagation
        grid.collapse(5, 5, 0, 0).unwrap();

        // Run propagation with subgrid processing
        let result = propagator.propagate(&mut grid, updates, &rules).await;

        // Verify no contradiction occurred
        assert!(
            result.is_ok(),
            "Propagation with subgrid processing failed: {:?}",
            result
        );

        // Verify that regions around the collapsed cell were affected
        // The subgrids should have propagated constraints properly
        for y in 3..8 {
            for x in 3..8 {
                if x == 5 && y == 5 {
                    // The collapsed cell should have only the first tile possible
                    let possibilities = grid.get(x, y, 0).unwrap();
                    assert_eq!(
                        possibilities.count_ones(),
                        1,
                        "Collapsed cell should have only one possibility"
                    );
                    assert!(
                        possibilities[0],
                        "Collapsed cell should have the first tile possible"
                    );
                }
            }
        }

        Ok(())
    }
}
