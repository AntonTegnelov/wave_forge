use crate::propagator::propagator_strategy::gpu_error_to_propagation_error;
use crate::{
    buffers::{GpuBuffers, GpuParamsUniform},
    gpu::sync::GpuSynchronizer,
    propagator::{PropagationStrategy, PropagationStrategyFactory},
    shader::pipeline::ComputePipelines,
    utils::debug_viz::DebugVisualizer,
    utils::error_recovery::{GpuError, GridCoord},
    utils::subgrid::{
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
#[derive(Debug)]
pub struct GpuConstraintPropagator {
    // References to shared GPU resources
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) pipelines: Arc<ComputePipelines>,
    pub(crate) buffers: Arc<GpuBuffers>,
    // State
    current_worklist_idx: Arc<AtomicUsize>,
    pub(crate) params: GpuParamsUniform,
    // Debug visualization
    debug_visualizer: Option<Arc<std::sync::Mutex<DebugVisualizer>>>,
    // GPU synchronizer
    synchronizer: Arc<GpuSynchronizer>,
    // Propagation strategy
    strategy: Box<dyn PropagationStrategy>,
}

// Manual Clone implementation that handles strategy cloning by creating a new one
impl Clone for GpuConstraintPropagator {
    fn clone(&self) -> Self {
        // Create a new direct strategy - this is a compromise but we need to implement Clone
        // In practice, a clone is rarely if ever called and users can reconfigure as needed
        let strategy = PropagationStrategyFactory::create_direct(1000);

        Self {
            device: self.device.clone(),
            queue: self.queue.clone(),
            pipelines: self.pipelines.clone(),
            buffers: self.buffers.clone(),
            current_worklist_idx: self.current_worklist_idx.clone(),
            params: self.params.clone(),
            debug_visualizer: self.debug_visualizer.clone(),
            synchronizer: self.synchronizer.clone(),
            strategy,
        }
    }
}

impl GpuConstraintPropagator {
    /// Creates a new `GpuConstraintPropagator`.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        pipelines: Arc<ComputePipelines>,
        buffers: Arc<GpuBuffers>,
        grid_dims: (usize, usize, usize),
        boundary_mode: wfc_core::BoundaryCondition,
        params: GpuParamsUniform,
    ) -> Self {
        let synchronizer = Arc::new(GpuSynchronizer::new(
            device.clone(),
            queue.clone(),
            buffers.clone(),
        ));

        // Create a default strategy based on grid size
        let grid = PossibilityGrid::new(
            grid_dims.0,
            grid_dims.1,
            grid_dims.2,
            params.num_tiles as usize,
        );
        let strategy = PropagationStrategyFactory::create_for_grid(&grid);

        Self {
            device,
            queue,
            pipelines,
            buffers,
            current_worklist_idx: Arc::new(AtomicUsize::new(0)),
            params,
            debug_visualizer: None, // Debug visualization disabled by default
            synchronizer,
            strategy,
        }
    }

    /// Sets the propagation strategy to use.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The propagation strategy to use.
    ///
    /// # Returns
    ///
    /// `Self` for method chaining.
    pub fn with_strategy(mut self, strategy: Box<dyn PropagationStrategy>) -> Self {
        self.strategy = strategy;
        self
    }

    /// Uses direct propagation strategy.
    ///
    /// # Arguments
    ///
    /// * `max_iterations` - The maximum number of propagation iterations.
    ///
    /// # Returns
    ///
    /// `Self` for method chaining.
    pub fn with_direct_propagation(self, max_iterations: u32) -> Self {
        self.with_strategy(PropagationStrategyFactory::create_direct(max_iterations))
    }

    /// Uses subgrid propagation strategy.
    ///
    /// # Arguments
    ///
    /// * `max_iterations` - The maximum number of propagation iterations.
    /// * `subgrid_size` - The size of each subgrid.
    ///
    /// # Returns
    ///
    /// `Self` for method chaining.
    pub fn with_subgrid_propagation(self, max_iterations: u32, subgrid_size: u32) -> Self {
        self.with_strategy(PropagationStrategyFactory::create_subgrid(
            max_iterations,
            subgrid_size,
        ))
    }

    /// Uses adaptive propagation strategy.
    ///
    /// # Arguments
    ///
    /// * `max_iterations` - The maximum number of propagation iterations.
    /// * `subgrid_size` - The size of each subgrid.
    /// * `size_threshold` - The grid size threshold for switching strategies.
    ///
    /// # Returns
    ///
    /// `Self` for method chaining.
    pub fn with_adaptive_propagation(
        self,
        max_iterations: u32,
        subgrid_size: u32,
        size_threshold: usize,
    ) -> Self {
        self.with_strategy(PropagationStrategyFactory::create_adaptive(
            max_iterations,
            subgrid_size,
            size_threshold,
        ))
    }

    /// Uses parallel subgrid processing for large grids.
    ///
    /// This is a legacy method that is now an alias for with_subgrid_propagation.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration for subgrid division and processing.
    ///
    /// # Returns
    ///
    /// `Self` for method chaining.
    pub fn with_parallel_subgrid_processing(self, config: SubgridConfig) -> Self {
        self.with_subgrid_propagation(1000, config.max_subgrid_size as u32)
    }

    /// Disables parallel subgrid processing.
    ///
    /// This is a legacy method that is now an alias for with_direct_propagation.
    ///
    /// # Returns
    ///
    /// `Self` for method chaining.
    pub fn without_parallel_subgrid_processing(self) -> Self {
        self.with_direct_propagation(1000)
    }

    /// Sets the debug visualizer.
    ///
    /// # Arguments
    ///
    /// * `visualizer` - The debug visualizer to use for capturing algorithm state.
    ///
    /// # Returns
    ///
    /// `Self` for method chaining.
    pub fn with_debug_visualizer(mut self, visualizer: DebugVisualizer) -> Self {
        self.debug_visualizer = Some(Arc::new(std::sync::Mutex::new(visualizer)));
        self
    }

    /// Disables debug visualization.
    ///
    /// # Returns
    ///
    /// `Self` for method chaining.
    pub fn without_debug_visualization(mut self) -> Self {
        self.debug_visualizer = None;
        self
    }

    /// Gets a reference to the synchronizer.
    pub fn synchronizer(&self) -> &GpuSynchronizer {
        &self.synchronizer
    }

    /// Gets a reference to the current propagation strategy.
    pub fn strategy(&self) -> &dyn PropagationStrategy {
        self.strategy.as_ref()
    }

    /// Gets a reference to the device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Gets a reference to the queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Gets a reference to the pipelines.
    pub fn pipelines(&self) -> &ComputePipelines {
        &self.pipelines
    }

    /// Gets a reference to the buffers.
    pub fn buffers(&self) -> &GpuBuffers {
        &self.buffers
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
        // Convert updated coordinates to GridCoords
        let updated_cells: Vec<GridCoord> = updated_coords
            .iter()
            .map(|(x, y, z)| GridCoord {
                x: *x,
                y: *y,
                z: *z,
            })
            .collect();

        // Prepare the strategy before use
        self.strategy.prepare(&self.synchronizer)?;

        // Delegate to the strategy for propagation
        log::debug!("Delegating to {} strategy", self.strategy.name());

        // Upload rules to buffers if needed
        // Note: This is normally handled by GpuBuffers::new, but we need to ensure
        // rules are up-to-date before each propagation
        self.synchronizer
            .upload_rules(rules)
            .map_err(gpu_error_to_propagation_error)?;

        // Delegate to the strategy
        let result =
            self.strategy
                .propagate(grid, &updated_cells, &self.buffers, &self.synchronizer);

        // Clean up strategy resources
        self.strategy.cleanup(&self.synchronizer)?;

        result
    }
}
