#![allow(clippy::redundant_field_names)]
use super::{
    backend::{GpuBackend, WgpuBackend},
    sync::GpuSynchronizer,
};
use crate::{
    buffers::{GpuBuffers, GpuParamsUniform},
    coordination::{
        self,
        coordinator::DefaultCoordinator,
        propagation::{
            DirectPropagationCoordinator, PropagationCoordinator, SubgridPropagationCoordinator,
        },
        WfcCoordinator,
    },
    debug_viz::{DebugVisualizationConfig, DebugVisualizer},
    entropy::{EntropyStrategy, EntropyStrategyFactory, GpuEntropyCalculator},
    error_recovery::{GpuError, GridCoord},
    propagator::{GpuConstraintPropagator, PropagationStrategy, PropagationStrategyFactory},
    shader::pipeline::ComputePipelines,
    subgrid::SubgridConfig,
};
use anyhow::Error as AnyhowError;
use log::{error, info, trace};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use wfc_core::propagator::PropagationError;
use wfc_core::{
    adjacency::AdjacencyRules,
    entropy::{EntropyCalculator, EntropyHeuristicType as CoreEntropyHeuristicType},
    grid::{BoundaryCondition, PossibilityGrid},
    progress::ProgressInfo,
    WfcError,
};

/// Grid definition info
#[derive(Debug, Clone)]
pub struct GridDefinition {
    pub dims: (usize, usize, usize),
    pub num_tiles: usize,
}

impl GridDefinition {
    /// Returns the total number of cells in the grid
    pub fn total_cells(&self) -> usize {
        self.dims.0 * self.dims.1 * self.dims.2
    }
}

/// Statistics about the grid state
#[derive(Debug, Clone, Default)]
pub struct GridStats {
    pub iterations: usize,
    pub contradictions: usize,
    pub collapsed_cells: usize,
}

/// Internal state for the GpuAccelerator, managed within an Arc<RwLock<>>.
pub struct AcceleratorInstance {
    backend: Arc<dyn GpuBackend>,
    grid_definition: GridDefinition,
    rules: Arc<AdjacencyRules>,
    boundary_condition: BoundaryCondition,
    pipelines: Arc<ComputePipelines>,
    buffers: Arc<GpuBuffers>,
    sync: Arc<GpuSynchronizer>,
    entropy_calculator: Arc<GpuEntropyCalculator>,
    propagator: Arc<RwLock<GpuConstraintPropagator>>,
    coordinator: Box<dyn WfcCoordinator + Send + Sync>,
    subgrid_config: Option<SubgridConfig>,
    progress_callback:
        Option<Box<dyn FnMut(ProgressInfo) -> Result<bool, AnyhowError> + Send + Sync>>,
    debug_visualizer: Option<DebugVisualizer>,
}

// Custom Debug implementation to handle types that don't implement Debug
impl std::fmt::Debug for AcceleratorInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AcceleratorInstance")
            .field("grid_definition", &self.grid_definition)
            .field("boundary_condition", &self.boundary_condition)
            .field("subgrid_config", &self.subgrid_config)
            .field("has_progress_callback", &self.progress_callback.is_some())
            .field("has_debug_visualizer", &self.debug_visualizer.is_some())
            .finish_non_exhaustive()
    }
}

/// Manages the WGPU context and orchestrates GPU-accelerated WFC operations.
///
/// This struct holds the necessary WGPU resources (instance, adapter, device, queue)
/// and manages the compute pipelines (`ComputePipelines`) and GPU buffers (`GpuBuffers`)
/// required for accelerating entropy calculation and constraint propagation.
///
/// It implements the `EntropyCalculator` and `ConstraintPropagator` traits from `wfc-core`,
/// providing GPU-acceleration.
///
/// # Initialization
///
/// Use the asynchronous `GpuAccelerator::new()` function to initialize the WGPU context
/// and create the necessary resources based on the initial grid state and rules.
///
/// # Usage
///
/// Once initialized, the `GpuAccelerator` instance can be passed to the main WFC `run` function
/// (or used directly) to perform entropy calculation and constraint propagation steps on the GPU.
/// Data synchronization between CPU (`PossibilityGrid`) and GPU (`GpuBuffers`) is handled
/// internally by the respective trait method implementations.
#[derive(Clone)]
pub struct GpuAccelerator {
    instance: Arc<RwLock<AcceleratorInstance>>,
}

impl GpuAccelerator {
    /// Creates a new GPU accelerator for Wave Function Collapse.
    ///
    /// Initializes the GPU device, compute pipelines, and buffers required for the WFC algorithm.
    /// This method performs asynchronous GPU operations and must be awaited.
    ///
    /// # Arguments
    ///
    /// * `initial_grid` - The initial grid state containing all possibilities.
    /// * `rules` - The adjacency rules for the WFC algorithm.
    /// * `boundary_mode` - Whether to use periodic or finite boundary conditions.
    /// * `subgrid_config` - Optional configuration for subgrid processing.
    ///
    /// # Returns
    ///
    /// A `Result` containing either a new `GpuAccelerator` or a `GpuError`.
    ///
    /// # Constraints
    ///
    /// * Dynamically supports arbitrary numbers of unique tile types, limited only by available GPU memory.
    pub async fn new(
        initial_grid: &PossibilityGrid,
        rules: &AdjacencyRules,
        boundary_condition: BoundaryCondition,
        entropy_heuristic: CoreEntropyHeuristicType,
        subgrid_config: Option<SubgridConfig>,
    ) -> Result<Self, GpuError> {
        let start_time = Instant::now();
        info!("Initializing GPU Accelerator...");

        let backend = crate::backend::WgpuBackend::new();
        let device = backend.device();
        let queue = backend.queue();

        info!("Using GPU adapter: {:?}", backend.adapter_info());

        let supported_features = backend.features();
        let mut features_to_enable = wgpu::Features::empty();
        let mut features = vec![];

        if supported_features.contains(wgpu::Features::SHADER_F64) {
            features_to_enable |= wgpu::Features::SHADER_F64;
            features.push("f64".to_string());
            info!("SHADER_F64 feature supported and enabled.");
        }

        let num_tiles = initial_grid.num_tiles();
        let num_tiles_u32 = (num_tiles + 31) / 32;

        let features_ref: Vec<&str> = features.iter().map(|s| s.as_str()).collect();

        let pipelines = Arc::new(ComputePipelines::new(
            &device,
            num_tiles_u32 as u32,
            &features_ref,
        )?);

        let buffers = Arc::new(GpuBuffers::new(
            &device,
            &queue,
            initial_grid,
            rules,
            boundary_condition,
        )?);

        let synchronizer = Arc::new(GpuSynchronizer::new(
            device.clone(),
            queue.clone(),
            buffers.clone(),
        ));

        let total_cells = initial_grid.width * initial_grid.height * initial_grid.depth;

        let params = GpuParamsUniform {
            grid_width: initial_grid.width as u32,
            grid_height: initial_grid.height as u32,
            grid_depth: initial_grid.depth as u32,
            num_tiles: num_tiles as u32,
            num_axes: rules.num_axes() as u32,
            boundary_mode: if boundary_condition == BoundaryCondition::Periodic {
                1
            } else {
                0
            },
            heuristic_type: match entropy_heuristic {
                CoreEntropyHeuristicType::Shannon => 0,
                CoreEntropyHeuristicType::Count => 1,
                CoreEntropyHeuristicType::CountSimple => 2,
                CoreEntropyHeuristicType::WeightedCount => 3,
            },
            tie_breaking: 0,
            max_propagation_steps: 1000,
            contradiction_check_frequency: 100,
            worklist_size: 0,
            grid_element_count: total_cells as u32,
            _padding: 0,
        };

        synchronizer.update_propagation_params(&params)?;

        let mut propagator_concrete = GpuConstraintPropagator::new(
            device.clone(),
            queue.clone(),
            pipelines.clone(),
            buffers.clone(),
            (initial_grid.width, initial_grid.height, initial_grid.depth),
            boundary_condition,
            params,
        );

        if let Some(ref config) = subgrid_config {
            propagator_concrete =
                propagator_concrete.with_parallel_subgrid_processing(config.clone());
        }
        let propagator = Arc::new(RwLock::new(propagator_concrete));

        let mut entropy_calculator_concrete = GpuEntropyCalculator::new(
            device.clone(),
            queue.clone(),
            pipelines.clone(),
            buffers.clone(),
            (initial_grid.width, initial_grid.height, initial_grid.depth),
        );
        entropy_calculator_concrete.set_entropy_heuristic(entropy_heuristic);
        let entropy_calculator = Arc::new(entropy_calculator_concrete);

        info!(
            "GPU Accelerator initialized in {:.2?}",
            start_time.elapsed()
        );

        let default_coord = DefaultCoordinator::new(entropy_calculator.clone(), propagator.clone());

        let grid_definition = GridDefinition {
            dims: (initial_grid.width, initial_grid.height, initial_grid.depth),
            num_tiles: initial_grid.num_tiles(),
        };

        let instance = AcceleratorInstance {
            backend: Arc::new(backend),
            grid_definition,
            rules: Arc::new(rules.clone()),
            boundary_condition,
            pipelines,
            buffers,
            sync: synchronizer,
            entropy_calculator,
            propagator,
            coordinator: Box::new(default_coord),
            subgrid_config,
            progress_callback: None,
            debug_visualizer: None,
        };

        let accelerator = Self {
            instance: Arc::new(RwLock::new(instance)),
        };

        Ok(accelerator)
    }

    pub fn backend(&self) -> Arc<dyn GpuBackend> {
        self.instance.read().unwrap().backend.clone()
    }

    pub fn pipelines(&self) -> Arc<ComputePipelines> {
        self.instance.read().unwrap().pipelines.clone()
    }

    pub fn buffers(&self) -> Arc<GpuBuffers> {
        self.instance.read().unwrap().buffers.clone()
    }

    pub fn grid_definition(&self) -> GridDefinition {
        self.instance.read().unwrap().grid_definition.clone()
    }

    pub fn boundary_condition(&self) -> BoundaryCondition {
        self.instance.read().unwrap().boundary_condition
    }

    pub fn num_tiles(&self) -> usize {
        self.instance.read().unwrap().grid_definition.num_tiles
    }

    pub async fn get_intermediate_result(&self) -> Result<PossibilityGrid, GpuError> {
        let instance = self.instance.read().unwrap();
        let target_grid = PossibilityGrid::new(
            instance.grid_definition.dims.0,
            instance.grid_definition.dims.1,
            instance.grid_definition.dims.2,
            instance.grid_definition.num_tiles,
        );
        instance
            .sync
            .download_grid(&target_grid)
            .await
            .map(|_| target_grid.clone())
    }

    pub fn enable_default_debug_visualization(&mut self) {
        let mut instance = self.instance.write().unwrap();
        if instance.debug_visualizer.is_none() {
            let config = DebugVisualizationConfig::default();
            let sync_clone = instance.sync.clone();
            instance.debug_visualizer = Some(DebugVisualizer::new(config, sync_clone));
            info!("Debug visualization enabled.");
        }
    }

    pub async fn run_with_callback<F>(
        &mut self,
        initial_grid: &PossibilityGrid,
        _rules: &AdjacencyRules,
        max_iterations: u64,
        _progress_callback: F,
        shutdown_signal: Option<tokio::sync::watch::Receiver<bool>>,
    ) -> Result<PossibilityGrid, WfcError>
    where
        F: FnMut(ProgressInfo) -> Result<bool, AnyhowError> + Send + Sync + 'static,
    {
        let start_time = Instant::now();
        let mut instance_guard = self.instance.write().unwrap();

        info!(
            "Running WFC on GPU for grid {}x{}x{} with {} tiles. Max iterations: {}",
            instance_guard.grid_definition.dims.0,
            instance_guard.grid_definition.dims.1,
            instance_guard.grid_definition.dims.2,
            instance_guard.grid_definition.num_tiles,
            max_iterations
        );

        let mut iteration: u64 = 0;
        let total_cells = instance_guard.grid_definition.total_cells();

        let mut run_result = WfcRunResult {
            grid: initial_grid.clone(),
            stats: GridStats::default(),
        };

        trace!("Uploading initial grid state to GPU...");
        instance_guard
            .sync
            .upload_grid(initial_grid)
            .map_err(|e| WfcError::InternalError(e.to_string()))?;

        instance_guard
            .sync
            .reset_contradiction_flag()
            .map_err(|e| WfcError::InternalError(e.to_string()))?;
        instance_guard
            .sync
            .reset_contradiction_location()
            .map_err(|e| WfcError::InternalError(e.to_string()))?;
        instance_guard
            .sync
            .reset_worklist_count()
            .map_err(|e| WfcError::InternalError(e.to_string()))?;

        trace!("Initial grid state uploaded and GPU state reset.");

        loop {
            if let Some(ref signal) = shutdown_signal {
                if *signal.borrow() {
                    info!("Shutdown signal received, stopping WFC execution.");
                    return Err(WfcError::Interrupted);
                }
            }

            if iteration >= max_iterations {
                error!(
                    "Maximum iterations ({}) reached without convergence.",
                    max_iterations
                );
                return Err(WfcError::MaxIterationsReached(max_iterations));
            }

            iteration += 1;
            trace!("--- WFC Iteration {} ---", iteration);

            trace!("Coordinating entropy calculation and selection...");
            let selection_result = instance_guard
                .coordinator
                .coordinate_entropy_and_selection(
                    &instance_guard.entropy_calculator,
                    &instance_guard.buffers,
                    &*instance_guard.backend.device(),
                    &*instance_guard.backend.queue(),
                    &instance_guard.sync,
                )
                .await;

            let selected_coords = match selection_result {
                Ok(Some(coords)) => {
                    trace!("Selected cell for collapse: {:?}", coords);
                    coords
                }
                Ok(None) => {
                    info!(
                        "Converged after {} iterations ({:.2?})",
                        iteration,
                        start_time.elapsed()
                    );
                    run_result.stats.iterations = iteration as usize;
                    break;
                }
                Err(GpuError::ContradictionDetected { coord }) => {
                    error!(
                        "Contradiction detected during entropy calculation at {:?}",
                        coord
                    );
                    run_result.stats.contradictions += 1;
                    run_result.stats.iterations = iteration as usize;
                    if let Some(c) = coord {
                        return Err(WfcError::Contradiction(c.x, c.y, c.z));
                    } else {
                        error!("Contradiction detected by GPU but location unknown.");
                        return Err(WfcError::Interrupted);
                    }
                }
                Err(e) => {
                    error!("GPU error during entropy calculation: {}", e);
                    return Err(WfcError::InternalError(e.to_string()));
                }
            };

            trace!("Coordinating constraint propagation...");
            let update_coords = vec![GridCoord::from(selected_coords)];

            let propagation_result = instance_guard
                .coordinator
                .coordinate_propagation(
                    &instance_guard.propagator,
                    &instance_guard.buffers,
                    &*instance_guard.backend.device(),
                    &*instance_guard.backend.queue(),
                    update_coords,
                )
                .await;

            match propagation_result {
                Ok(_) => {
                    trace!("Propagation successful.");
                    run_result.stats.collapsed_cells += 1;
                }
                Err(PropagationError::Contradiction(x, y, z)) => {
                    error!(
                        "Contradiction detected during propagation at ({}, {}, {})",
                        x, y, z
                    );
                    run_result.stats.contradictions += 1;
                    run_result.stats.iterations = iteration as usize;
                    return Err(WfcError::Contradiction(x, y, z));
                }
                Err(e) => {
                    error!("Propagation error: {}", e);
                    return Err(WfcError::Propagation(e));
                }
            }

            // Extract all the grid information we need before entering the callback blocks
            let grid_width = instance_guard.grid_definition.dims.0;
            let grid_height = instance_guard.grid_definition.dims.1;
            let grid_depth = instance_guard.grid_definition.dims.2;
            let num_tiles = instance_guard.grid_definition.num_tiles;

            // Clone the backend and buffers references outside of any blocks
            let backend_ref = instance_guard.backend.clone();
            let buffers_ref = instance_guard.buffers.clone();

            run_result.stats.iterations = iteration as usize;

            if let Some(cb) = &mut instance_guard.progress_callback {
                trace!("Calling progress callback for iteration {}...", iteration);

                let progress_info = ProgressInfo {
                    iterations: iteration,
                    grid_state: PossibilityGrid::new(
                        grid_width,
                        grid_height,
                        grid_depth,
                        num_tiles,
                    ),
                    collapsed_cells: run_result.stats.collapsed_cells,
                    total_cells: total_cells,
                    elapsed_time: start_time.elapsed(),
                };

                trace!("Calling progress callback...");
                match cb(progress_info) {
                    Ok(true) => {}
                    Ok(false) => {
                        info!("WFC execution cancelled by progress callback.");
                        return Err(WfcError::Interrupted);
                    }
                    Err(e) => {
                        error!("Progress callback failed: {}", e);
                        return Err(WfcError::InternalError(e.to_string()));
                    }
                }
            }

            if let Some(visualizer) = &mut instance_guard.debug_visualizer {
                trace!("Updating debug visualizer for iteration {}...", iteration);

                if let Err(e) = visualizer.update(&*backend_ref, &*buffers_ref).await {
                    error!("Failed to update debug visualizer: {}", e);
                }
            }

            let _progress = run_result.stats.collapsed_cells as f32 / total_cells as f32;
        }

        drop(instance_guard);
        let final_instance_guard = self.instance.read().unwrap();

        trace!("Downloading final grid state from GPU...");
        let final_grid_target = PossibilityGrid::new(
            final_instance_guard.grid_definition.dims.0,
            final_instance_guard.grid_definition.dims.1,
            final_instance_guard.grid_definition.dims.2,
            final_instance_guard.grid_definition.num_tiles,
        );
        let final_grid = final_instance_guard
            .sync
            .download_grid(&final_grid_target)
            .await
            .map_err(|e| WfcError::InternalError(e.to_string()))?;

        let final_collapsed_count = final_grid.count_collapsed_cells();
        run_result.stats.collapsed_cells = final_collapsed_count;

        info!(
            "WFC finished. Iterations: {}, Time: {:.2?}, Collapsed: {}, Contradictions: {}",
            run_result.stats.iterations,
            start_time.elapsed(),
            final_collapsed_count,
            run_result.stats.contradictions
        );

        assert!(final_grid.is_fully_collapsed().unwrap_or(false));

        Ok(final_grid.clone())
    }

    pub fn with_debug_visualization(&mut self, config: DebugVisualizationConfig) {
        let mut instance = self.instance.write().unwrap();
        let sync_clone = instance.sync.clone();
        instance.debug_visualizer = Some(DebugVisualizer::new(config, sync_clone));
        info!("Debug visualization enabled with custom config.");
    }

    pub fn set_progress_callback<F>(&mut self, callback: F)
    where
        F: FnMut(ProgressInfo) -> Result<bool, AnyhowError> + Send + Sync + 'static,
    {
        let mut instance = self.instance.write().unwrap();
        instance.progress_callback = Some(Box::new(callback));
        info!("Progress callback set.");
    }

    /// Creates a new GPU accelerator with a specific entropy strategy
    pub fn with_entropy_strategy<S: EntropyStrategy + 'static>(
        &mut self,
        strategy: S,
    ) -> &mut Self {
        let mut instance = self.instance.write().unwrap();
        let buffers = instance.buffers.clone();

        // Get a mutable reference to the entropy calculator
        let entropy_calculator = Arc::get_mut(&mut instance.entropy_calculator)
            .expect("Could not get exclusive access to entropy calculator");

        // Set the new strategy
        entropy_calculator.with_strategy(Box::new(strategy));

        info!("Custom entropy strategy configured.");
        self
    }

    /// Configure the accelerator with a specific entropy heuristic
    pub fn with_entropy_heuristic(&mut self, heuristic: CoreEntropyHeuristicType) -> &mut Self {
        let instance = self.instance.read().unwrap();

        // Create a new strategy using the factory
        let strategy = EntropyStrategyFactory::create_strategy(
            heuristic,
            instance.grid_definition.num_tiles,
            instance.buffers.grid_buffers.u32s_per_cell,
        );

        // Set the strategy (dropping the read lock first to avoid deadlock)
        drop(instance);

        self.with_entropy_strategy_boxed(strategy)
    }

    /// Lower-level method to set a boxed strategy
    pub fn with_entropy_strategy_boxed(&mut self, strategy: Box<dyn EntropyStrategy>) -> &mut Self {
        let mut instance = self.instance.write().unwrap();

        // Get a mutable reference to the entropy calculator
        let entropy_calculator = Arc::get_mut(&mut instance.entropy_calculator)
            .expect("Could not get exclusive access to entropy calculator");

        // Set the new strategy
        entropy_calculator.with_strategy(strategy);

        info!("Entropy strategy configured using factory.");
        self
    }
}

impl Drop for GpuAccelerator {
    fn drop(&mut self) {
        info!("Dropping GpuAccelerator, releasing GPU resources...");
    }
}

#[derive(Debug)]
pub struct WfcRunResult {
    pub grid: PossibilityGrid,
    pub stats: GridStats,
}

impl WfcRunResult {
    pub fn new(grid: PossibilityGrid, stats: GridStats) -> Self {
        WfcRunResult { grid, stats }
    }
}

// Helper methods not in the core library
trait PossibilityGridExt {
    /// Count the number of cells that are fully collapsed (have only one possibility)
    fn count_collapsed_cells(&self) -> usize;
}

impl PossibilityGridExt for PossibilityGrid {
    fn count_collapsed_cells(&self) -> usize {
        let mut count = 0;
        for z in 0..self.depth {
            for y in 0..self.height {
                for x in 0..self.width {
                    if let Some(cell) = self.get(x, y, z) {
                        if cell.count_ones() == 1 {
                            count += 1;
                        }
                    }
                }
            }
        }
        count
    }
}
