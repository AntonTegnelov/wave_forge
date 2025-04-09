#![allow(clippy::redundant_field_names)]
use crate::{
    backend::GpuBackend,
    buffers::{DownloadRequest, GpuBuffers, GpuParamsUniform},
    coordination::{DefaultCoordinator, WfcCoordinator},
    debug_viz::{DebugVisualizationConfig, DebugVisualizer},
    entropy::{EntropyHeuristicType, GpuEntropyCalculator},
    pipeline::ComputePipelines,
    propagator::GpuConstraintPropagator,
    shader_registry::ShaderRegistry,
    subgrid::SubgridConfig,
    sync::GpuSynchronizer,
    GpuError,
};
use anyhow::Error as AnyhowError;
use log::{error, info, trace, warn};
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Instant;
use wfc_core::{
    entropy::{EntropyCalculator, EntropyError, EntropyHeuristicType as CoreEntropyHeuristicType},
    grid::{GridCoord, GridDefinition, GridStats, GridView, PossibilityGrid},
    propagator::{ConstraintPropagator, PropagationError},
    traits::{self as wfc_traits, CollapseResult, ObserveResult, PropagationResult},
    BoundaryCondition, ProgressInfo, TileId, WfcError,
};
use wfc_rules::AdjacencyRules;
use wgpu::{Device, Queue};

/// Internal state for the GpuAccelerator, managed within an Arc<RwLock<>>.
#[derive(Debug)]
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

        let backend = crate::backend::WgpuBackend::new()?;
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
            device.clone(),
            num_tiles_u32 as u32,
            &features_ref,
        )?);

        let buffers = Arc::new(GpuBuffers::new(
            device.clone(),
            queue.clone(),
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

        if let Some(config) = subgrid_config {
            propagator_concrete = propagator_concrete.with_parallel_subgrid_processing(config);
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
        instance.sync.download_grid(&target_grid).await
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
        rules: &AdjacencyRules,
        max_iterations: u64,
        mut progress_callback: F,
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
            .map_err(|e| WfcError::GpuError(e.to_string()))?;

        instance_guard
            .sync
            .reset_contradiction_flag()
            .map_err(|e| WfcError::GpuError(e.to_string()))?;
        instance_guard
            .sync
            .reset_contradiction_location()
            .map_err(|e| WfcError::GpuError(e.to_string()))?;
        instance_guard
            .sync
            .reset_worklist_count()
            .map_err(|e| WfcError::GpuError(e.to_string()))?;

        trace!("Initial grid state uploaded and GPU state reset.");

        loop {
            if let Some(ref signal) = shutdown_signal {
                if *signal.borrow() {
                    info!("Shutdown signal received, stopping WFC execution.");
                    return Err(WfcError::ExecutionInterrupted(
                        "Shutdown signal".to_string(),
                    ));
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
                        return Err(WfcError::UnknownContradiction);
                    }
                }
                Err(e) => {
                    error!("GPU error during entropy calculation: {}", e);
                    return Err(WfcError::GpuError(e.to_string()));
                }
            };

            trace!("Coordinating constraint propagation...");
            let update_coords = vec![GridCoord::from(selected_coords)];

            let mut propagator_guard = instance_guard.propagator.write().unwrap();

            let propagation_result = instance_guard
                .coordinator
                .coordinate_propagation(
                    &mut *propagator_guard,
                    &instance_guard.buffers,
                    &*instance_guard.backend.device(),
                    &*instance_guard.backend.queue(),
                    update_coords,
                )
                .await;

            drop(propagator_guard);

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
                Err(PropagationError::GpuError(e)) => {
                    error!("GPU error during propagation: {}", e);
                    return Err(WfcError::GpuError(e.to_string()));
                }
                Err(e) => {
                    error!("Propagation error: {}", e);
                    return Err(WfcError::Propagation(e));
                }
            }

            if let Some(cb) = &mut instance_guard.progress_callback {
                trace!("Calling progress callback for iteration {}...", iteration);

                run_result.stats.iterations = iteration as usize;

                let progress = run_result.stats.collapsed_cells as f32 / total_cells as f32;

                let progress_info = ProgressInfo {
                    iterations: iteration,
                    grid_state: None,
                    collapsed_cells: run_result.stats.collapsed_cells,
                    total_cells: total_cells,
                    elapsed_time: start_time.elapsed(),
                };

                trace!("Calling progress callback...");
                match cb(progress_info) {
                    Ok(true) => {}
                    Ok(false) => {
                        info!("WFC execution cancelled by progress callback.");
                        return Err(WfcError::ExecutionCancelled(
                            "Progress callback".to_string(),
                        ));
                    }
                    Err(e) => {
                        error!("Progress callback failed: {}", e);
                        return Err(WfcError::CallbackError(e.to_string()));
                    }
                }
            } else {
                run_result.stats.iterations = iteration as usize;
            }

            if let Some(visualizer) = &mut instance_guard.debug_visualizer {
                trace!("Updating debug visualizer for iteration {}...", iteration);
                if let Err(e) = visualizer
                    .update(&*instance_guard.backend, &*instance_guard.buffers)
                    .await
                {
                    error!("Failed to update debug visualizer: {}", e);
                }
            }
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
            .map_err(|e| WfcError::GpuError(e.to_string()))?;

        let final_collapsed_count = final_grid.count_collapsed_cells();
        run_result.stats.collapsed_cells = final_collapsed_count;

        info!(
            "WFC finished. Iterations: {}, Time: {:.2?}, Collapsed: {}, Contradictions: {}",
            run_result.stats.iterations,
            start_time.elapsed(),
            final_collapsed_count,
            run_result.stats.contradictions
        );

        Ok(final_grid)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::WgpuBackend;
    use wfc_core::entropy::EntropyHeuristicType as CoreEntropyHeuristicType;

    fn setup_basic_test_data() -> (PossibilityGrid, AdjacencyRules) {
        let width = 4;
        let height = 4;
        let depth = 1;
        let num_tiles = 2;

        let mut grid = PossibilityGrid::new(width, height, depth, num_tiles);
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    if let Some(cell) = grid.get_mut(x, y, z) {
                        cell.fill(true);
                    }
                }
            }
        }

        let mut rules = AdjacencyRules::new(num_tiles, 6);
        rules.allow(0, 1, 0);
        rules.allow(1, 0, 0);

        (grid, rules)
    }

    #[tokio::test]
    async fn test_accelerator_creation_and_config() {
        let (grid, rules) = setup_basic_test_data();
        let accelerator = GpuAccelerator::new(
            &grid,
            &rules,
            BoundaryCondition::Finite,
            CoreEntropyHeuristicType::Shannon,
            None,
        )
        .await;

        assert!(accelerator.is_ok());
        let acc = accelerator.unwrap();

        assert_eq!(acc.num_tiles(), grid.num_tiles());
        assert_eq!(acc.boundary_condition(), BoundaryCondition::Finite);
    }

    #[tokio::test]
    async fn test_run_basic_wfc() {
        let (grid, rules) = setup_basic_test_data();
        let mut accelerator = GpuAccelerator::new(
            &grid,
            &rules,
            BoundaryCondition::Finite,
            CoreEntropyHeuristicType::Count,
            None,
        )
        .await
        .expect("Failed to create accelerator");

        let max_iterations = (grid.width * grid.height * grid.depth * 2) as u64;

        let result = accelerator
            .run_with_callback(&grid, &rules, max_iterations, |_info| Ok(true), None)
            .await;

        assert!(result.is_ok(), "WFC run failed: {:?}", result.err());

        if let Ok(final_grid) = result {
            assert_eq!(final_grid.width, grid.width);
            assert_eq!(final_grid.height, grid.height);
            assert_eq!(final_grid.depth, grid.depth);
            assert!(final_grid.is_fully_collapsed());
        }
    }
}
