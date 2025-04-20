#![allow(clippy::redundant_field_names)]
use std::{
    fmt::Debug,
    sync::{Arc, RwLock},
};

use super::{
    backend::{GpuBackend, WgpuBackend},
    sync::GpuSynchronizer,
};

use crate::coordination::strategy;
use crate::{
    buffers::{GpuBuffers, GpuEntropyShaderParams, GpuParamsUniform},
    coordination::{strategy::CoordinationStrategyFactory, DefaultCoordinator, WfcCoordinator},
    entropy::{EntropyStrategy, EntropyStrategyFactory, GpuEntropyCalculator, GpuEntropyStrategy},
    propagator::{GpuConstraintPropagator, PropagationStrategyFactory},
    shader::pipeline::ComputePipelines,
    utils::debug_viz::{DebugVisualizationConfig, DebugVisualizer},
    utils::error::{
        gpu_error::GpuError as NewGpuError, RecoveryAction, RecoveryHookRegistry, WfcError,
    },
    utils::error_recovery::{GpuError, GridCoord},
    utils::subgrid::SubgridConfig,
    utils::RwLock as GpuRwLock,
};

use anyhow::Error as AnyhowError;
use log::{info, trace};
use rand;
use std::time::Instant;
use wfc_core::{
    entropy::{
        EntropyCalculator, EntropyError as CoreEntropyError,
        EntropyHeuristicType as CoreEntropyHeuristicType,
    },
    grid::PossibilityGrid,
    BoundaryCondition, ProgressInfo,
};
use wfc_rules::AdjacencyRules;

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

/// Type alias for the progress callback function
type ProgressCallbackFn = Box<dyn FnMut(ProgressInfo) -> Result<bool, AnyhowError> + Send + Sync>;

/// Internal state for the GpuAccelerator, managed within an Arc<RwLock<>>.
pub struct AcceleratorInstance {
    backend: Arc<dyn GpuBackend>,
    grid_definition: GridDefinition,
    _rules: Arc<AdjacencyRules>,
    boundary_condition: BoundaryCondition,
    pipelines: Arc<ComputePipelines>,
    buffers: Arc<GpuBuffers>,
    sync: Arc<GpuSynchronizer>,
    entropy_calculator: Arc<GpuEntropyCalculator>,
    propagator: Arc<GpuRwLock<GpuConstraintPropagator>>,
    coordinator: Box<dyn WfcCoordinator + Send + Sync>,
    subgrid_config: Option<SubgridConfig>,
    progress_callback: Option<ProgressCallbackFn>,
    debug_visualizer: Option<DebugVisualizer>,
    recovery_hooks: Arc<RwLock<RecoveryHookRegistry>>,
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
    /// A `Result` containing either a new `GpuAccelerator` or a `WfcError`.
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
    ) -> Result<Self, WfcError> {
        let start_time = Instant::now();
        info!("Initializing GPU Accelerator...");

        let backend = WgpuBackend::new();
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

        let pipelines = Arc::new(
            ComputePipelines::new(&device, num_tiles_u32 as u32, &features_ref)
                .map_err(WfcError::Gpu)?,
        );

        let buffers = Arc::new(
            GpuBuffers::new(&device, &queue, initial_grid, rules, boundary_condition)
                .map_err(WfcError::Gpu)?,
        );

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

        // Choose an appropriate propagation strategy based on grid size and subgrid configuration
        if let Some(ref config) = subgrid_config {
            if initial_grid.width * initial_grid.height * initial_grid.depth > 4096 {
                // For large grids with subgrid config, use subgrid propagation
                propagator_concrete = propagator_concrete.with_subgrid_propagation(
                    1000, // Default max iterations
                    config.max_subgrid_size as u32,
                );
                info!(
                    "Using subgrid propagation strategy with subgrid size {}",
                    config.max_subgrid_size
                );
            } else {
                // For smaller grids, even with subgrid config, use direct propagation
                propagator_concrete = propagator_concrete.with_direct_propagation(1000);
                info!("Using direct propagation strategy (grid too small for subgrid)");
            }
        } else if initial_grid.width * initial_grid.height * initial_grid.depth > 4096 {
            // For large grids without explicit subgrid config, use adaptive strategy
            propagator_concrete = propagator_concrete.with_adaptive_propagation(
                1000, // Default max iterations
                16,   // Default subgrid size
                4096, // Default threshold
            );
            info!("Using adaptive propagation strategy");
        } else {
            // For smaller grids, use direct propagation
            propagator_concrete = propagator_concrete.with_direct_propagation(1000);
            info!("Using direct propagation strategy");
        }

        let propagator = Arc::new(GpuRwLock::new(propagator_concrete));

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
            _rules: Arc::new(rules.clone()),
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
            recovery_hooks: Arc::new(RwLock::new(RecoveryHookRegistry::new())),
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
        // Create a read lock on the instance - using a scope to ensure it's dropped before the await
        let grid_definition = {
            let instance = self.instance.read().unwrap();
            instance.grid_definition.clone()
        };

        // Create a grid template with the correct dimensions
        let grid = PossibilityGrid::new(
            grid_definition.dims.0,
            grid_definition.dims.1,
            grid_definition.dims.2,
            grid_definition.num_tiles,
        );

        // Get the synchronizer for the download operation
        let synchronizer = {
            let instance = self.instance.read().unwrap();
            instance.sync.clone()
        };

        // Download the latest grid state from GPU
        // Note: we're using the synchronizer's download_grid method which already handles
        // all the buffer mapping and data transfer logic
        let result = synchronizer.download_grid(&grid).await?;

        Ok(result)
    }

    pub fn enable_default_debug_visualization(&mut self) {
        let mut _instance = self.instance.write().unwrap();
        if _instance.debug_visualizer.is_none() {
            let config = DebugVisualizationConfig::default();
            let sync_clone = _instance.sync.clone();
            _instance.debug_visualizer = Some(DebugVisualizer::new(config, sync_clone));
            info!("Debug visualization enabled.");
        }
    }

    pub async fn run_with_callback<F>(
        &mut self,
        initial_grid: &PossibilityGrid,
        rules: &AdjacencyRules,
        max_iterations: u64,
        mut progress_callback: F,
        _shutdown_signal: Option<tokio::sync::watch::Receiver<bool>>,
    ) -> Result<PossibilityGrid, WfcError>
    where
        F: FnMut(ProgressInfo) -> Result<bool, AnyhowError> + Send + Sync + 'static,
    {
        let start_time = Instant::now();

        // Get the required data while holding the lock briefly
        let grid_definition;
        let synchronizer;
        let device;
        let queue;
        let buffers;
        let entropy_calculator;
        let propagator;
        let coordinator;

        {
            let instance_guard = self.instance.read().unwrap();

            // Clone what we need before dropping the lock
            grid_definition = instance_guard.grid_definition.clone();
            synchronizer = instance_guard.sync.clone();
            device = instance_guard.backend.device();
            queue = instance_guard.backend.queue();
            buffers = instance_guard.buffers.clone();
            entropy_calculator = instance_guard.entropy_calculator.clone();
            propagator = instance_guard.propagator.clone();
            coordinator = instance_guard.coordinator.clone_box();

            info!(
                "Running WFC on GPU for grid {}x{}x{} with {} tiles. Max iterations: {}",
                grid_definition.dims.0,
                grid_definition.dims.1,
                grid_definition.dims.2,
                grid_definition.num_tiles,
                max_iterations
            );
        }

        // Upload initial grid state
        trace!("Uploading initial grid state to GPU...");
        synchronizer
            .upload_grid(initial_grid)
            .map_err(|e| WfcError::other(e.to_string()))?;

        synchronizer
            .reset_contradiction_flag()
            .map_err(|e| WfcError::other(e.to_string()))?;
        synchronizer
            .reset_contradiction_location()
            .map_err(|e| WfcError::other(e.to_string()))?;
        synchronizer
            .reset_worklist_count()
            .map_err(|e| WfcError::other(e.to_string()))?;

        // Create a working copy of the grid
        let mut current_grid = initial_grid.clone();
        let total_cells = grid_definition.total_cells();
        let mut collapsed_cells = 0;
        let mut iterations = 0;

        // Main WFC loop
        while iterations < max_iterations {
            // Calculate entropy and select cell
            let selected_cell = coordinator
                .coordinate_entropy_and_selection(
                    &entropy_calculator,
                    &buffers,
                    &device,
                    &queue,
                    &synchronizer,
                )
                .await
                .map_err(|e| WfcError::other(e.to_string()))?;

            // If no cell was selected, we're done
            if selected_cell.is_none() {
                break;
            }

            let (x, y, z) = selected_cell.unwrap();

            // Collapse the selected cell
            let cell = current_grid.get_mut(x, y, z).unwrap();
            let possible_states = cell.iter_ones().collect::<Vec<_>>();
            if possible_states.is_empty() {
                return Err(WfcError::other(
                    "No possible states for selected cell".to_string(),
                ));
            }

            // Choose a random state from possible states
            let chosen_state = possible_states[rand::random::<usize>() % possible_states.len()];
            cell.clear();
            cell.set(chosen_state, true);
            collapsed_cells += 1;

            // Upload the updated cell state
            synchronizer
                .upload_grid(&current_grid)
                .map_err(|e| WfcError::other(e.to_string()))?;

            // Propagate constraints
            coordinator
                .coordinate_propagation(
                    &propagator,
                    &buffers,
                    &device,
                    &queue,
                    vec![GridCoord { x, y, z }],
                )
                .await
                .map_err(|e| WfcError::other(e.to_string()))?;

            // Download the updated grid state
            current_grid = synchronizer
                .download_grid(&current_grid)
                .await
                .map_err(|e| WfcError::other(e.to_string()))?;

            // Call progress callback
            let progress = ProgressInfo {
                collapsed_cells,
                total_cells,
                elapsed_time: start_time.elapsed(),
                iterations,
                grid_state: current_grid.clone(),
            };

            if let Err(e) = progress_callback(progress) {
                return Err(WfcError::other(format!("Progress callback error: {}", e)));
            }

            iterations += 1;
        }

        // Check if we've fully collapsed
        if !current_grid
            .is_fully_collapsed()
            .map_err(|e| WfcError::other(e.to_string()))?
        {
            return Err(WfcError::other("Failed to fully collapse grid".to_string()));
        }

        Ok(current_grid)
    }

    pub fn with_debug_visualization(&mut self, config: DebugVisualizationConfig) {
        let mut _instance = self.instance.write().unwrap();
        let sync_clone = _instance.sync.clone();
        _instance.debug_visualizer = Some(DebugVisualizer::new(config, sync_clone));
        info!("Debug visualization enabled with custom config.");
    }

    pub fn set_progress_callback<F>(&mut self, callback: F)
    where
        F: FnMut(ProgressInfo) -> Result<bool, AnyhowError> + Send + Sync + 'static,
    {
        let mut _instance = self.instance.write().unwrap();
        _instance.progress_callback = Some(Box::new(callback));
        info!("Progress callback set.");
    }

    /// Creates a new GPU accelerator with a specific entropy strategy
    pub fn with_entropy_strategy<S: GpuEntropyStrategy + 'static>(
        &mut self,
        strategy: S,
    ) -> &mut Self {
        // Scope the instance read lock to avoid borrowing conflicts
        {
            let _instance = self.instance.read().unwrap();
            // Here we could use instance.buffers but we don't need to
        }

        // Set the new strategy using our boxed method
        self.with_entropy_strategy_boxed(Box::new(strategy))
    }

    /// Configure the accelerator with a specific entropy heuristic
    pub fn with_entropy_heuristic(&mut self, heuristic: CoreEntropyHeuristicType) -> &mut Self {
        let _instance = self.instance.read().unwrap();

        // Create a new strategy using the factory
        let base_strategy = EntropyStrategyFactory::create_strategy(
            heuristic,
            _instance.grid_definition.num_tiles,
            _instance.buffers.grid_buffers.u32s_per_cell,
        );

        // Cast the base strategy to GpuEntropyStrategy
        let strategy: Box<dyn GpuEntropyStrategy> = Box::new(EntropyStrategyAdapter(base_strategy));

        // Set the strategy (dropping the read lock first to avoid deadlock)
        drop(_instance);

        self.with_entropy_strategy_boxed(strategy)
    }

    /// Lower-level method to set a boxed strategy
    pub fn with_entropy_strategy_boxed(
        &mut self,
        strategy: Box<dyn GpuEntropyStrategy>,
    ) -> &mut Self {
        // Clone calculator and set the strategy
        let mut calculator = {
            let _instance = self.instance.read().unwrap();
            (*_instance.entropy_calculator).clone()
        };

        calculator.set_strategy_boxed(strategy);

        // Update the instance with the new calculator
        {
            let mut _instance = self.instance.write().unwrap();
            _instance.entropy_calculator = Arc::new(calculator);
        }

        self
    }

    /// Sets the propagation strategy to use.
    ///
    /// This method allows providing a custom implementation of the
    /// AsyncPropagationStrategy trait to be used for constraint propagation.
    ///
    /// # Arguments
    ///
    /// * `strategy` - A type implementing the AsyncPropagationStrategy trait
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining.
    pub fn with_propagation_strategy<S: crate::propagator::AsyncPropagationStrategy + 'static>(
        &mut self,
        strategy: S,
    ) -> &mut Self {
        let boxed_strategy = Box::new(strategy)
            as Box<dyn crate::propagator::AsyncPropagationStrategy + Send + Sync>;
        self.with_propagation_strategy_boxed(boxed_strategy)
    }

    /// Sets the propagation strategy to use.
    ///
    /// This method allows providing a boxed implementation of the
    /// AsyncPropagationStrategy trait to be used for constraint propagation.
    ///
    /// # Arguments
    ///
    /// * `strategy` - A boxed type implementing the AsyncPropagationStrategy trait
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining.
    pub fn with_propagation_strategy_boxed(
        &mut self,
        strategy: Box<dyn crate::propagator::AsyncPropagationStrategy + Send + Sync>,
    ) -> &mut Self {
        // Create a new propagator with the new strategy
        {
            // Lock the instance for writing
            let _instance = self.instance.write().unwrap();

            // Get a mutable reference to the propagator - use async lock in sync context
            let mut propagator_guard =
                futures::executor::block_on(async { _instance.propagator.write().await });

            // Get the needed GPU resources
            let device = _instance.backend.device();
            let queue = _instance.backend.queue();
            let pipelines = _instance.pipelines.clone();
            let buffers = _instance.buffers.clone();

            // Set up configuration parameters for propagator
            let grid_dims = (
                _instance.grid_definition.dims.0,
                _instance.grid_definition.dims.1,
                _instance.grid_definition.dims.2,
            );

            // Create parameters uniform similar to the existing one
            let params = propagator_guard.params;

            // Create a new propagator with the provided strategy
            let new_propagator = crate::propagator::GpuConstraintPropagator::new(
                device.clone(),
                queue.clone(),
                pipelines,
                buffers,
                grid_dims,
                _instance.boundary_condition,
                params,
            )
            .with_strategy(strategy);

            // Replace the old propagator with the new one
            *propagator_guard = new_propagator;
        }
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
    /// `&mut Self` for method chaining.
    pub fn with_direct_propagation(&mut self, max_iterations: u32) -> &mut Self {
        let pipelines = self.instance.read().unwrap().pipelines.clone();
        let strategy = PropagationStrategyFactory::create_direct_async(max_iterations, pipelines);
        self.with_propagation_strategy_boxed(strategy)
    }

    /// Sets the propagation strategy to subgrid propagation.
    ///
    /// # Arguments
    ///
    /// * `max_iterations` - The maximum number of propagation iterations.
    /// * `subgrid_size` - The size of each subgrid.
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining.
    pub fn with_subgrid_propagation(
        &mut self,
        max_iterations: u32,
        subgrid_size: u32,
    ) -> &mut Self {
        let pipelines = self.instance.read().unwrap().pipelines.clone();
        let strategy = PropagationStrategyFactory::create_subgrid_async(
            max_iterations,
            subgrid_size,
            pipelines,
        );
        self.with_propagation_strategy_boxed(strategy)
    }

    /// Sets the propagation strategy to adaptive propagation.
    ///
    /// # Arguments
    ///
    /// * `max_iterations` - The maximum number of propagation iterations.
    /// * `subgrid_size` - The size of each subgrid.
    /// * `size_threshold` - The grid size threshold for switching strategies.
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining.
    pub fn with_adaptive_propagation(
        &mut self,
        max_iterations: u32,
        subgrid_size: u32,
        size_threshold: usize,
    ) -> &mut Self {
        let pipelines = self.instance.read().unwrap().pipelines.clone();
        let strategy = PropagationStrategyFactory::create_adaptive_async(
            max_iterations,
            subgrid_size,
            size_threshold,
            pipelines,
        );
        self.with_propagation_strategy_boxed(strategy)
    }

    /// Sets the propagation strategy based on the grid size.
    ///
    /// This will automatically select an appropriate propagation strategy
    /// based on the size of the grid.
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining.
    pub fn with_auto_propagation(&mut self) -> &mut Self {
        let dims;
        let pipelines;
        {
            let instance = self.instance.read().unwrap();
            dims = instance.grid_definition.dims;
            pipelines = instance.pipelines.clone();
        }

        let num_tiles = self.num_tiles();
        let grid = PossibilityGrid::new(dims.0, dims.1, dims.2, num_tiles);
        let strategy = PropagationStrategyFactory::create_for_grid_async(&grid, pipelines);
        self.with_propagation_strategy_boxed(strategy)
    }

    /// Use an adaptive coordination strategy for WFC algorithm execution.
    /// This strategy selects the most appropriate coordination approach based on grid size.
    pub fn with_adaptive_coordination(&mut self) -> &mut Self {
        // Create the needed components outside the instance lock scope
        let grid_size;
        let entropy_calculator_clone;
        let propagator_clone;

        {
            // Limit the scope of the instance borrow
            let _instance = self.instance.read().unwrap();
            grid_size = _instance.grid_definition.dims;
            entropy_calculator_clone = _instance.entropy_calculator.clone();
            propagator_clone = _instance.propagator.clone();
        }

        // Create the strategy outside the instance lock
        let strategy = CoordinationStrategyFactory::create_adaptive(
            entropy_calculator_clone.clone(),
            propagator_clone.clone(),
            grid_size,
        );

        // Create a new DefaultCoordinator and set the strategy
        let mut coordinator = DefaultCoordinator::new(entropy_calculator_clone, propagator_clone);
        coordinator.with_coordination_strategy_boxed(strategy);

        // Set the coordinator with a separate lock scope
        {
            let mut _instance = self.instance.write().unwrap();
            _instance.coordinator = Box::new(coordinator);
        }

        self
    }

    /// Use the default coordination strategy for WFC algorithm execution.
    pub fn with_default_coordination(&mut self) -> &mut Self {
        // Create the needed components outside the instance lock scope
        let entropy_calculator_clone;
        let propagator_clone;

        {
            // Limit the scope of the instance borrow
            let _instance = self.instance.read().unwrap();
            entropy_calculator_clone = _instance.entropy_calculator.clone();
            propagator_clone = _instance.propagator.clone();
        }

        // Create the strategy outside the instance lock
        let strategy = CoordinationStrategyFactory::create_default(
            entropy_calculator_clone.clone(),
            propagator_clone.clone(),
        );

        // Create a new DefaultCoordinator and set the strategy
        let mut coordinator = DefaultCoordinator::new(entropy_calculator_clone, propagator_clone);
        coordinator.with_coordination_strategy_boxed(strategy);

        // Set the coordinator with a separate lock scope
        {
            let mut _instance = self.instance.write().unwrap();
            _instance.coordinator = Box::new(coordinator);
        }

        self
    }

    /// Sets the coordination strategy to use.
    ///
    /// # Arguments
    ///
    /// * `strategy` - A coordination strategy implementing the CoordinationStrategy trait.
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining.
    pub fn with_coordination_strategy<S: strategy::CoordinationStrategy + 'static>(
        &mut self,
        strategy: S,
    ) -> &mut Self {
        // Create the needed components outside the instance lock scope
        let entropy_calculator_clone;
        let propagator_clone;

        {
            // Limit the scope of the instance borrow
            let _instance = self.instance.read().unwrap();
            entropy_calculator_clone = _instance.entropy_calculator.clone();
            propagator_clone = _instance.propagator.clone();
        }

        // Create a new DefaultCoordinator and set the strategy
        let mut coordinator = DefaultCoordinator::new(entropy_calculator_clone, propagator_clone);
        coordinator.with_coordination_strategy(strategy);

        // Set the coordinator with a separate lock scope
        {
            let mut _instance = self.instance.write().unwrap();
            _instance.coordinator = Box::new(coordinator);
        }

        self
    }

    /// Sets the coordination strategy to use.
    ///
    /// # Arguments
    ///
    /// * `strategy` - Boxed coordination strategy.
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining.
    pub fn with_coordination_strategy_boxed(
        &mut self,
        strategy: Box<dyn strategy::CoordinationStrategy>,
    ) -> &mut Self {
        // Create the needed components outside the instance lock scope
        let entropy_calculator_clone;
        let propagator_clone;

        {
            // Limit the scope of the instance borrow
            let _instance = self.instance.read().unwrap();
            entropy_calculator_clone = _instance.entropy_calculator.clone();
            propagator_clone = _instance.propagator.clone();
        }

        // Create a new DefaultCoordinator and set the strategy
        let mut coordinator = DefaultCoordinator::new(entropy_calculator_clone, propagator_clone);
        coordinator.with_coordination_strategy_boxed(strategy);

        // Set the coordinator with a separate lock scope
        {
            let mut _instance = self.instance.write().unwrap();
            _instance.coordinator = Box::new(coordinator);
        }

        self
    }

    /// Register a user-defined recovery hook for specific error types
    ///
    /// # Arguments
    ///
    /// * `predicate` - A function that determines if the hook should be applied to an error
    /// * `hook` - A function that performs the recovery action
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining
    pub fn register_recovery_hook<P, F>(&mut self, predicate: P, hook: F) -> &mut Self
    where
        P: Fn(&wfc_core::WfcError) -> bool + Send + Sync + 'static,
        F: Fn(&wfc_core::WfcError) -> Option<RecoveryAction> + Send + Sync + 'static,
    {
        // Get the recovery hooks with a limited scope
        {
            // First get a read lock on the instance
            let _instance = self.instance.read().unwrap();
            // Then get a write lock on the hooks
            let mut hooks = _instance.recovery_hooks.write().unwrap();
            hooks.register_for_core_errors(predicate, hook);
            info!("Registered custom recovery hook");
        }

        self
    }

    /// Register a recovery hook specifically for GPU errors
    ///
    /// # Arguments
    ///
    /// * `hook` - A function that performs the recovery action for GPU errors
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining
    pub fn register_gpu_error_hook<F>(&mut self, hook: F) -> &mut Self
    where
        F: Fn(&NewGpuError) -> Option<RecoveryAction> + Send + Sync + 'static,
    {
        // Get the recovery hooks with a limited scope
        {
            // First get a read lock on the instance
            let _instance = self.instance.read().unwrap();
            // Then get a write lock on the hooks
            let mut hooks = _instance.recovery_hooks.write().unwrap();
            hooks.register_for_gpu_errors(hook);
            info!("Registered GPU error recovery hook");
        }

        self
    }

    /// Register a recovery hook for algorithm errors
    ///
    /// # Arguments
    ///
    /// * `hook` - A function that performs the recovery action for algorithm errors
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining
    pub fn register_algorithm_error_hook<F>(&mut self, hook: F) -> &mut Self
    where
        F: Fn(&str) -> Option<RecoveryAction> + Send + Sync + 'static,
    {
        // Get the recovery hooks with a limited scope
        {
            // First get a read lock on the instance
            let _instance = self.instance.read().unwrap();
            // Then get a write lock on the hooks
            let mut hooks = _instance.recovery_hooks.write().unwrap();
            hooks.register_for_algorithm_errors(hook);
            info!("Registered algorithm error recovery hook");
        }

        self
    }

    /// Register a recovery hook for validation errors
    pub fn register_validation_error_hook<F>(&mut self, hook: F) -> &mut Self
    where
        F: Fn(&str) -> Option<RecoveryAction> + Send + Sync + 'static,
    {
        // Get the recovery hooks with a limited scope
        {
            // First get a read lock on the instance
            let _instance = self.instance.read().unwrap();
            // Then get a write lock on the hooks
            let mut hooks = _instance.recovery_hooks.write().unwrap();
            hooks.register_for_validation_errors(hook);
            info!("Registered validation error recovery hook");
        }

        self
    }

    /// Register a recovery hook for configuration errors
    pub fn register_configuration_error_hook<F>(&mut self, hook: F) -> &mut Self
    where
        F: Fn(&str) -> Option<RecoveryAction> + Send + Sync + 'static,
    {
        // Get the recovery hooks with a limited scope
        {
            // First get a read lock on the instance
            let _instance = self.instance.read().unwrap();
            // Then get a write lock on the hooks
            let mut hooks = _instance.recovery_hooks.write().unwrap();
            hooks.register_for_configuration_errors(hook);
            info!("Registered configuration error recovery hook");
        }

        self
    }

    /// Try to handle an error with registered hooks
    ///
    /// This method is intended for internal use within the WFC algorithm.
    ///
    /// # Arguments
    ///
    /// * `error` - The error to handle
    ///
    /// # Returns
    ///
    /// `Option<RecoveryAction>` if a hook was able to handle the error
    pub(crate) fn try_handle_local_error(&self, error: &WfcError) -> Option<RecoveryAction> {
        // Use limited scope for the lock
        let _instance = self.instance.read().unwrap();
        let hooks = _instance.recovery_hooks.read().unwrap();
        hooks.try_handle(error)
    }

    /// Try to handle a core WFC error with registered hooks
    ///
    /// This method bridges between core WfcError and our local WfcError.
    ///
    /// # Arguments
    ///
    /// * `error` - The core error to handle
    ///
    /// # Returns
    ///
    /// `Option<RecoveryAction>` if a hook was able to handle the error
    #[allow(dead_code)]
    pub(crate) fn try_handle_error(&self, error: &wfc_core::WfcError) -> Option<RecoveryAction> {
        // Convert the core error to our local error type
        let local_error = crate::utils::error::WfcError::from_core_error(error);

        // Delegate to the method that handles local errors
        self.try_handle_local_error(&local_error)
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

#[allow(dead_code)]
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

// An adapter to convert EntropyStrategy to GpuEntropyStrategy
struct EntropyStrategyAdapter(Box<dyn EntropyStrategy>);

impl Debug for EntropyStrategyAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EntropyStrategyAdapter")
    }
}

impl GpuEntropyStrategy for EntropyStrategyAdapter {
    fn heuristic_type(&self) -> CoreEntropyHeuristicType {
        self.0.heuristic_type()
    }

    fn configure_shader_params(&self, params: &mut GpuEntropyShaderParams) {
        self.0.configure_shader_params(params)
    }

    fn prepare(&self, synchronizer: &GpuSynchronizer) -> Result<(), CoreEntropyError> {
        self.0.prepare(synchronizer)
    }

    fn calculate_entropy(
        &self,
        _buffers: &GpuBuffers,
        _pipelines: &ComputePipelines,
        _queue: &wgpu::Queue,
        _grid_dims: (usize, usize, usize),
    ) -> Result<(), CoreEntropyError> {
        // Default implementation as this isn't used in the imported strategy
        Ok(())
    }

    fn upload_data(&self, synchronizer: &GpuSynchronizer) -> Result<(), CoreEntropyError> {
        self.0.upload_data(synchronizer)
    }

    fn post_process(&self, synchronizer: &GpuSynchronizer) -> Result<(), CoreEntropyError> {
        self.0.post_process(synchronizer)
    }
}
