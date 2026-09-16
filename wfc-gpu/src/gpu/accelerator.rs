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
    buffers::{CollapseInfoUniform, GpuBuffers, GpuEntropyShaderParams, GpuParamsUniform},
    coordination::{strategy::CoordinationStrategyFactory, DefaultCoordinator, WfcCoordinator},
    entropy::{EntropyStrategy, EntropyStrategyFactory, GpuEntropyCalculator, GpuEntropyStrategy},
    propagator::{GpuConstraintPropagator, PropagationStrategyFactory},
    shader::pipeline::ComputePipelines,
    utils::debug_viz::{DebugVisualizationConfig, DebugVisualizer},
    utils::error::{
        gpu_error::GpuError as NewGpuError, gpu_error::GpuErrorContext, RecoveryAction,
        RecoveryHookRegistry, WfcError,
    },
    utils::error_recovery::{GpuError, GridCoord},
    utils::subgrid::SubgridConfig,
    utils::RwLock as GpuRwLock,
};

use anyhow::Error as AnyhowError;
use log::{info, trace};
use tracing::{Instrument, info_span};
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
/// Cells collapsed per propagation round by default. One is the classic algorithm; larger batches
/// trade a slightly higher contradiction rate for far fewer GPU round-trips.
const DEFAULT_COLLAPSE_BATCH: usize = 1;

/// Once initialized, the `GpuAccelerator` instance can be passed to the main WFC `run` function
/// (or used directly) to perform entropy calculation and constraint propagation steps on the GPU.
/// Data synchronization between CPU (`PossibilityGrid`) and GPU (`GpuBuffers`) is handled
/// internally by the respective trait method implementations.
#[derive(Clone)]
pub struct GpuAccelerator {
    instance: Arc<RwLock<AcceleratorInstance>>,
    /// Relative weight of each tile when collapsing a cell; uniform when `None`.
    tile_weights: Option<Arc<[f32]>>,
    /// Whole-grid constraint enforced on the CPU before every observation; see
    /// [`GpuAccelerator::with_global_constraint`].
    global_constraint: Option<Arc<dyn wfc_core::constraint::GlobalConstraint>>,
    /// How many cells to collapse before propagating; see [`GpuAccelerator::with_collapse_batch`].
    collapse_batch: usize,
    /// Seed for the collapse choice; see [`GpuAccelerator::with_seed`].
    seed: Option<u64>,
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
        if num_tiles > crate::shader::pipeline::MAX_TILES {
            return Err(WfcError::Configuration(format!(
                "rule set has {num_tiles} tile variants, but the GPU propagation shader supports at most {}",
                crate::shader::pipeline::MAX_TILES
            )));
        }
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
            contradiction_check_frequency: 10,
            worklist_size: 0,
            grid_element_count: (total_cells * 4) as u32,
            _padding0: 0,
            _padding1: 0,
            _padding2: 0,
            _padding3: 0,
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

        // Direct propagation is the only strategy that works purely on the GPU buffers;
        // the subgrid and adaptive strategies still rely on a CPU-side grid.
        propagator_concrete = propagator_concrete.with_direct_propagation(1000);
        info!("Using direct propagation strategy");

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
            tile_weights: None,
            global_constraint: None,
            collapse_batch: DEFAULT_COLLAPSE_BATCH,
            seed: None,
        };

        Ok(accelerator)
    }

    /// Returns a reference to the underlying GPU backend.
    pub fn backend(&self) -> Arc<dyn GpuBackend> {
        // Acquire read lock, handle potential poisoning
        let instance = self.instance.read().unwrap_or_else(|poisoned| {
            log::error!("RwLock for AcceleratorInstance poisoned in backend()");
            poisoned.into_inner()
        });
        instance.backend.clone()
    }

    /// Returns a reference to the compute pipelines.
    pub fn pipelines(&self) -> Arc<ComputePipelines> {
        let instance = self.instance.read().unwrap_or_else(|poisoned| {
            log::error!("RwLock for AcceleratorInstance poisoned in pipelines()");
            poisoned.into_inner()
        });
        instance.pipelines.clone()
    }

    /// Returns a reference to the GPU buffers.
    pub fn buffers(&self) -> Arc<GpuBuffers> {
        let instance = self.instance.read().unwrap_or_else(|poisoned| {
            log::error!("RwLock for AcceleratorInstance poisoned in buffers()");
            poisoned.into_inner()
        });
        instance.buffers.clone()
    }

    /// Returns a reference to the GPU synchronizer.
    pub fn synchronizer(&self) -> Arc<GpuSynchronizer> {
        let instance = self.instance.read().unwrap_or_else(|poisoned| {
            log::error!("RwLock for AcceleratorInstance poisoned in synchronizer()");
            poisoned.into_inner()
        });
        instance.sync.clone()
    }

    /// Returns the grid definition (dimensions, tile count).
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
        _rules: &AdjacencyRules,
        max_iterations: u64,
        mut progress_callback: F,
        _shutdown_signal: Option<tokio::sync::watch::Receiver<bool>>,
    ) -> Result<PossibilityGrid, WfcError>
    where
        F: FnMut(ProgressInfo) -> Result<bool, AnyhowError> + Send + Sync + 'static,
    {
        let start_time = Instant::now();

        use rand::RngExt as _;

        // Draw a seed when none was given, and report it: a thrashing run is only useful if it can be
        // replayed (docs/thrashing.md).
        let seed = self.seed.unwrap_or_else(rand::random);
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
        info!("WFC run seed: {seed}");

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

        // Spans for the timeline (see docs/debugging.md). Stages that await are instrumented
        // rather than entered, because an entered span guard must not be held across `.await`.
        let run_span = info_span!(
            "wfc_run",
            width = grid_definition.dims.0,
            height = grid_definition.dims.1,
            depth = grid_definition.dims.2,
            tiles = grid_definition.num_tiles
        );

        // Upload initial grid state
        trace!("Uploading initial grid state to GPU...");
        info_span!(parent: &run_span, "upload_grid")
            .in_scope(|| synchronizer.upload_grid(initial_grid))
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

        // Cells the caller constrained before the run must be propagated before the first
        // observation. A cell pinned to a single tile has zero entropy, so it is never selected
        // and its neighbours would otherwise never learn about it.
        let constrained_cells: Vec<GridCoord> = (0..initial_grid.depth)
            .flat_map(|z| {
                (0..initial_grid.height)
                    .flat_map(move |y| (0..initial_grid.width).map(move |x| (x, y, z)))
            })
            .filter(|&(x, y, z)| {
                initial_grid
                    .get(x, y, z)
                    .is_some_and(|cell| cell.count_ones() < grid_definition.num_tiles)
            })
            .map(|(x, y, z)| GridCoord { x, y, z })
            .collect();
        if !constrained_cells.is_empty() {
            trace!(
                "Propagating {} pre-constrained cells before the first observation",
                constrained_cells.len()
            );
            let cells = constrained_cells.len();
            coordinator
                .coordinate_propagation(&propagator, &buffers, &device, &queue, constrained_cells)
                .instrument(info_span!(parent: &run_span, "initial_propagation", cells))
                .await
                .map_err(|e| WfcError::other(e.to_string()))?;
            current_grid = synchronizer
                .download_grid(&current_grid)
                .instrument(info_span!(parent: &run_span, "download_grid"))
                .await
                .map_err(|e| WfcError::other(e.to_string()))?;
        }
        let total_cells = grid_definition.total_cells();
        let mut collapsed_cells = 0usize;
        let mut iterations = 0;

        // Backtracking (docs/status.md A-9). A contradiction means an earlier choice was wrong, not
        // that the grid is unsolvable, so the run keeps the grid state before every collapse and
        // undoes choices instead of giving up. As in marian42's generator, each consecutive failure
        // undoes twice as many steps; unlike his, the choice that led into the failure is then
        // forbidden, so the search cannot repeat it and always makes progress.
        const MAX_HISTORY: usize = 2048;
        const MAX_UNDO_STEPS: usize = 64;
        const MAX_BACKTRACKS: usize = 50_000;
        /// Manhattan distance kept between cells collapsed in the same batch.
        const BATCH_SPACING: usize = 4;
        /// How far the search of recent choices may widen when one cell keeps failing.
        const MAX_CULPRIT_RADIUS: usize = 8;
        struct Choice {
            grid: PossibilityGrid,
            cell: (usize, usize, usize),
            tile: usize,
        }
        let mut history: std::collections::VecDeque<Choice> = std::collections::VecDeque::new();
        let mut undo_steps = 1usize;
        let mut backtracks = 0usize;
        // Where contradictions surface, and how far the search has to unwind, distinguish a run that is
        // merely slow from one that keeps failing in the same place (docs/thrashing.md, H3).
        let mut failures_by_cell: std::collections::HashMap<(usize, usize, usize), usize> =
            std::collections::HashMap::new();
        let mut undo_depths: Vec<usize> = Vec::new();
        let mut progress_log: Vec<(usize, usize, usize)> = Vec::new();
        // A failure carries where it happened, so the search can jump back to the choice that caused
        // it instead of undoing whatever happened to be most recent.
        let mut failure: Option<(String, Option<(usize, usize, usize)>)> = None;

        // Main WFC loop
        while iterations < max_iterations {
            let iteration_span = info_span!(parent: &run_span, "iteration", iteration = iterations);

            if let Some((reason, culprit)) = failure.take() {
                backtracks += 1;
                if backtracks > MAX_BACKTRACKS {
                    return Err(WfcError::other(format!(
                        "Contradiction: gave up after {backtracks} backtracks; last failure: {reason}"
                    )));
                }
                let backtrack_span =
                    info_span!(parent: &iteration_span, "backtrack", undo_steps, backtracks);
                // Conflict-directed: undo back to the most recent choice made next to where the
                // failure surfaced, because that is what most likely caused it. A contradiction far
                // from any recent choice falls back to undoing a doubling number of steps.
                // How many times this cell has already failed. A cell that keeps failing means the
                // real cause lies further back than its immediate neighbourhood, so widen the search
                // rather than undoing the same single choice again: the previous version matched the
                // choice it had just restored, so recovery undid exactly one step forever and the run
                // thrashed (docs/thrashing.md).
                let repeats = culprit.map_or(0, |cell| *failures_by_cell.get(&cell).unwrap_or(&0));
                let radius = 1 + repeats.min(MAX_CULPRIT_RADIUS);
                let near_culprit = culprit.and_then(|(cx, cy, cz)| {
                    history.iter().rposition(|choice| {
                        let (hx, hy, hz) = choice.cell;
                        hx.abs_diff(cx) <= radius && hy.abs_diff(cy) <= radius && hz.abs_diff(cz) <= radius
                    })
                });
                let steps = match near_culprit {
                    // Undo at least as many steps as this cell has failed, so a repeated failure keeps
                    // reaching further back instead of retrying the same choice.
                    Some(index) => (history.len() - index).max(undo_steps),
                    None => undo_steps,
                };
                let mut restored = None;
                let mut undone = 0usize;
                for _ in 0..steps {
                    match history.pop_back() {
                        Some(choice) => {
                            restored = Some(choice);
                            undone += 1;
                        }
                        None => break,
                    }
                }
                let Some(choice) = restored else {
                    return Err(WfcError::other(format!(
                        "Contradiction: no choices left to undo after {backtracks} backtracks; {reason}"
                    )));
                };
                trace!("Backtracking {undone} step(s) after: {reason}");
                undo_depths.push(undone);
                if let Some(cell) = culprit {
                    *failures_by_cell.entry(cell).or_default() += 1;
                }
                progress_log.push((iterations as usize, collapsed_cells, backtracks));
                undo_steps = if near_culprit.is_some() && repeats == 0 {
                    1
                } else {
                    (undo_steps * 2).min(MAX_UNDO_STEPS)
                };
                collapsed_cells = collapsed_cells.saturating_sub(undone);
                current_grid = choice.grid;
                let (bx, by, bz) = choice.cell;
                let cell = current_grid.get_mut(bx, by, bz).expect("cell from history is in bounds");
                cell.set(choice.tile, false);
                if cell.count_ones() == 0 {
                    // Every tile here has now been ruled out, so the mistake lies further back.
                    failure = Some((format!("no tiles left at ({bx}, {by}, {bz})"), None));
                    continue;
                }

                // The GPU still holds the contradiction from the failed attempt; clear it before
                // propagating the restored state.
                for reset in [
                    synchronizer.reset_contradiction_flag(),
                    synchronizer.reset_contradiction_location(),
                    synchronizer.reset_worklist_count(),
                ] {
                    reset.map_err(|e| WfcError::other(e.to_string()))?;
                }
                backtrack_span
                    .in_scope(|| synchronizer.upload_grid(&current_grid))
                    .map_err(|e| WfcError::other(e.to_string()))?;
                if let Err(e) = coordinator
                    .coordinate_propagation(
                        &propagator,
                        &buffers,
                        &device,
                        &queue,
                        vec![GridCoord { x: bx, y: by, z: bz }],
                    )
                    .instrument(backtrack_span.clone())
                    .await
                {
                    failure = Some((e.to_string(), None));
                    continue;
                }
                current_grid = synchronizer
                    .download_grid(&current_grid)
                    .instrument(info_span!(parent: &backtrack_span, "download_grid"))
                    .await
                    .map_err(|e| WfcError::other(e.to_string()))?;
                continue;
            }

            if let Some(constraint) = &self.global_constraint {
                let mut constraint_failure = None;
                loop {
                    let changed = match info_span!(parent: &iteration_span, "global_constraint")
                        .in_scope(|| constraint.apply(&mut current_grid))
                    {
                        Ok(changed) => changed,
                        Err((x, y, z)) => {
                            constraint_failure = Some((
                                format!("global constraint cannot be satisfied at ({x}, {y}, {z})"),
                                Some((x, y, z)),
                            ));
                            break;
                        }
                    };
                    if changed.is_empty() {
                        break;
                    }
                    let cells = changed.len();
                    info_span!(parent: &iteration_span, "upload_cells", cells).in_scope(|| {
                        for &(x, y, z) in &changed {
                            synchronizer.upload_cell(&current_grid, x, y, z);
                        }
                    });
                    if let Err(e) = coordinator
                        .coordinate_propagation(
                            &propagator,
                            &buffers,
                            &device,
                            &queue,
                            changed.into_iter().map(|(x, y, z)| GridCoord { x, y, z }).collect(),
                        )
                        .instrument(info_span!(parent: &iteration_span, "constraint_propagation", cells))
                        .await
                    {
                        constraint_failure = Some((e.to_string(), None));
                        break;
                    }
                    current_grid = synchronizer
                        .download_grid(&current_grid)
                        .instrument(info_span!(parent: &iteration_span, "download_grid"))
                        .await
                        .map_err(|e| WfcError::other(e.to_string()))?;
                }
                if let Some(constraint_failure) = constraint_failure {
                    failure = Some(constraint_failure);
                    continue;
                }
            }

            // Compute entropy on the GPU so the min-entropy buffer is current before selecting
            entropy_calculator
                .dispatch_entropy_calculation_pass()
                .instrument(info_span!(parent: &iteration_span, "entropy_pass"))
                .await
                .map_err(|e| WfcError::other(e.to_string()))?;

            // Select the lowest-entropy cell
            let selected_cell = coordinator
                .coordinate_entropy_and_selection(
                    &entropy_calculator,
                    &buffers,
                    &device,
                    &queue,
                    &synchronizer,
                )
                .instrument(info_span!(parent: &iteration_span, "select_cell"))
                .await
                .map_err(|e| WfcError::other(e.to_string()))?;

            // If no cell was selected, we're done
            if selected_cell.is_none() {
                break;
            }

            let (x, y, z) = selected_cell.unwrap();
            // The GPU picked the lowest-entropy cell. Fill the rest of the batch from the grid already
            // downloaded, taking fewest-possibility cells that are far enough from those chosen that
            // they are unlikely to constrain each other before propagation runs.
            let mut batch = vec![(x, y, z)];
            if self.collapse_batch > 1 {
                let mut candidates: Vec<(usize, (usize, usize, usize))> = Vec::new();
                for cz in 0..current_grid.depth {
                    for cy in 0..current_grid.height {
                        for cx in 0..current_grid.width {
                            let count = current_grid.get(cx, cy, cz).map_or(0, |cell| cell.count_ones());
                            if count > 1 {
                                candidates.push((count, (cx, cy, cz)));
                            }
                        }
                    }
                }
                candidates.sort_unstable();
                for (_, coord) in candidates {
                    if batch.len() >= self.collapse_batch {
                        break;
                    }
                    let far_enough = batch.iter().all(|&(bx, by, bz)| {
                        bx.abs_diff(coord.0) + by.abs_diff(coord.1) + bz.abs_diff(coord.2) >= BATCH_SPACING
                    });
                    if far_enough {
                        batch.push(coord);
                    }
                }
            }

            // Collapse every cell in the batch, then propagate from all of them at once.
            let mut collapsed_this_round: Vec<GridCoord> = Vec::with_capacity(batch.len());
            let mut batch_failure = None;
            for &(x, y, z) in &batch {
            let cell = current_grid.get_mut(x, y, z).unwrap();
            let possible_states = cell.iter_ones().collect::<Vec<_>>();
            if possible_states.is_empty() {
                // An earlier choice emptied this cell; undo instead of failing the run.
                batch_failure = Some((format!("no tiles left at ({x}, {y}, {z})"), Some((x, y, z))));
                break;
            }
            if possible_states.len() == 1 {
                // An earlier collapse in this batch already decided it.
                continue;
            }

            // Choose a remaining state, in proportion to its weight when weights are set
            let chosen_state = match &self.tile_weights {
                Some(weights) => {
                    use rand::distr::{Distribution, weighted::WeightedIndex};
                    let distribution =
                        WeightedIndex::new(possible_states.iter().map(|&tile| weights[tile]))
                            .map_err(|e| WfcError::other(format!("invalid tile weights: {e}")))?;
                    possible_states[distribution.sample(&mut rng)]
                }
                None => possible_states[rng.random_range(0..possible_states.len())],
            };

            // Remember the state before the collapse so this choice can be undone.
            if history.len() == MAX_HISTORY {
                history.pop_front();
            }
            history.push_back(Choice {
                grid: current_grid.clone(),
                cell: (x, y, z),
                tile: chosen_state,
            });

            // Use the grid's collapse method directly
            current_grid.collapse(x, y, z, chosen_state).map_err(|e| {
                WfcError::other(format!(
                    "Failed to collapse cell ({},{},{}): {}",
                    x, y, z, e
                ))
            })?;
            collapsed_cells += 1;
            collapsed_this_round.push(GridCoord { x, y, z });

            // Only this cell changed; a full upload would repack and rewrite the whole grid.
            info_span!(parent: &iteration_span, "upload_cell", x, y, z)
                .in_scope(|| synchronizer.upload_cell(&current_grid, x, y, z));
            }

            if let Some(reason) = batch_failure {
                failure = Some(reason);
                continue;
            }
            if collapsed_this_round.is_empty() {
                continue;
            }

            // Propagate from every cell collapsed this round: propagation is confluent, so one round
            // over all of them reaches the same fixpoint as a round per cell.
            let batched = collapsed_this_round.len();
            let first = collapsed_this_round[0];
            if let Err(e) = coordinator
                .coordinate_propagation(&propagator, &buffers, &device, &queue, collapsed_this_round)
                .instrument(info_span!(parent: &iteration_span, "propagate", batched))
                .await
            {
                failure = Some((e.to_string(), Some((first.x, first.y, first.z))));
                continue;
            }

            // Download the updated grid state
            current_grid = synchronizer
                .download_grid(&current_grid)
                .instrument(info_span!(parent: &iteration_span, "download_grid"))
                .await
                .map_err(|e| WfcError::other(e.to_string()))?;

            // The collapse held, so the next failure starts undoing from one step again.
            undo_steps = 1;

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
            return Err(WfcError::other(format!(
                "Failed to fully collapse grid: {collapsed_cells} of {total_cells} cells after \
                 {iterations} iterations (limit {max_iterations}) and {backtracks} backtracks"
            )));
        }

        // Search cost is as much a performance number as wall time: a batched run that is fast on
        // average can be thrashing on a bad seed (docs/solver-fit.md).
        info!(
            "WFC run finished: {collapsed_cells} collapses for {total_cells} cells, \
             {iterations} iterations, {backtracks} backtracks"
        );
        if std::env::var("WFC_REPORT_SEARCH").is_ok() {
            let mut worst: Vec<_> = failures_by_cell.iter().map(|(c, n)| (*n, *c)).collect();
            worst.sort_unstable_by(|a, b| b.cmp(a));
            let repeated: usize = worst.iter().filter(|(n, _)| *n > 1).map(|(n, _)| n).sum();
            let deepest = undo_depths.iter().copied().max().unwrap_or(0);
            let mean_depth =
                undo_depths.iter().sum::<usize>() as f64 / undo_depths.len().max(1) as f64;
            eprintln!(
                "search: seed={seed} collapses={collapsed_cells} cells={total_cells} \
                 iterations={iterations} backtracks={backtracks} \
                 distinct_failure_cells={} repeated_failures={repeated} \
                 max_undo={deepest} mean_undo={mean_depth:.1} worst_cells={:?}",
                failures_by_cell.len(),
                worst.iter().take(5).collect::<Vec<_>>()
            );
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
    /// Makes the run reproducible: the same seed, grid, rules and settings produce the same result.
    ///
    /// Propagation is already deterministic (a monotone fixpoint, so the order restrictions are applied
    /// in cannot change it), and cell selection is a deterministic minimum, so the choice of tile was
    /// the only source of run-to-run variation. Without a seed, a run that thrashes cannot be replayed,
    /// bisected, or compared against another configuration: the only difference measured is luck
    /// (docs/thrashing.md, status.md A-6).
    pub fn with_seed(&mut self, seed: u64) -> &mut Self {
        self.seed = Some(seed);
        self
    }

    /// Collapses up to `cells` cells before each propagation round, instead of one.
    ///
    /// Propagation is a monotone fixpoint, so its result does not depend on the order restrictions are
    /// applied: collapsing several cells and then propagating reaches the same fixpoint as propagating
    /// between them. What changes is the *search* — a later choice in the batch is made without seeing
    /// what an earlier one implied, so contradictions become more likely and backtracking absorbs
    /// them. The payoff is that the entropy pass, the selection readback, the grid download and the
    /// propagation round are each paid once per batch rather than once per cell: a traced city run
    /// spent 16 s of 26.7 s propagating, across 23916 passes of which 23156 carried eight cells or
    /// fewer (docs/solver-fit.md).
    ///
    /// Cells in a batch are kept a few cells apart, so they rarely constrain each other before
    /// propagation runs.
    pub fn with_collapse_batch(&mut self, cells: usize) -> &mut Self {
        self.collapse_batch = cells.max(1);
        self
    }

    /// Enforces a constraint on the whole grid that adjacency rules cannot express, such as
    /// connectivity. It runs on the CPU before every observation, on the grid as downloaded after
    /// propagation; the cells it changes are uploaded and propagated until it changes nothing. If
    /// it reports a violation, the run fails with a contradiction.
    ///
    /// This is a stopgap in the current run loop, which already moves the whole grid to the CPU
    /// every iteration; the solver redesign (#7) has to keep the capability.
    pub fn with_global_constraint(
        &mut self,
        constraint: Arc<dyn wfc_core::constraint::GlobalConstraint>,
    ) -> &mut Self {
        self.global_constraint = Some(constraint);
        self
    }

    /// Sets the relative weight of each tile used when collapsing a cell, typically
    /// `TileSet::weights`. Without weights every remaining tile is equally likely.
    ///
    /// # Errors
    ///
    /// Returns [`WfcError::Configuration`] if there is not exactly one weight per tile or a weight
    /// is not a positive finite number.
    pub fn with_tile_weights(&mut self, weights: &[f32]) -> Result<&mut Self, WfcError> {
        let num_tiles = self.num_tiles();
        if weights.len() != num_tiles {
            return Err(WfcError::Configuration(format!(
                "expected {num_tiles} tile weights, got {}",
                weights.len()
            )));
        }
        if let Some((tile, weight)) = weights
            .iter()
            .enumerate()
            .find(|(_, w)| !(w.is_finite() && **w > 0.0))
        {
            return Err(WfcError::Configuration(format!(
                "tile {tile} has weight {weight}; weights must be positive and finite"
            )));
        }
        self.tile_weights = Some(weights.into());
        Ok(self)
    }

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

    /// Collapses a specific cell on the GPU to a chosen tile ID.
    ///
    /// This function dispatches a compute shader (`collapse_cell.wgsl`)
    /// to update the `grid_possibilities_buf` directly on the GPU.
    ///
    /// # Arguments
    ///
    /// * `coords` - The (x, y, z) coordinates of the cell to collapse.
    /// * `chosen_tile_id` - The ID of the tile to collapse the cell to.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the collapse command was successfully submitted.
    /// * `Err(NewGpuError)` if an error occurred during GPU operations.
    pub fn collapse_cell_gpu(
        &self,
        coords: (usize, usize, usize),
        chosen_tile_id: u32,
    ) -> Result<(), NewGpuError> {
        let instance = self.instance.read().map_err(|_| {
            NewGpuError::other(
                "Failed to acquire read lock on AcceleratorInstance",
                GpuErrorContext::default(),
            )
        })?;

        let device = instance.backend.device();
        let queue = instance.backend.queue();
        let buffers = &instance.buffers;
        let pipelines = &instance.pipelines;

        // 1. Prepare uniform data
        let collapse_info = CollapseInfoUniform {
            coord_x: coords.0 as u32,
            coord_y: coords.1 as u32,
            coord_z: coords.2 as u32,
            chosen_tile_id,
        };

        // 2. Write uniform data to buffer
        // Note: Using write_buffer might have sync issues if not careful.
        // Consider using an encoder copy if problems arise.
        queue.write_buffer(
            &buffers.collapse_info_buf,
            0,
            bytemuck::cast_slice(&[collapse_info]),
        );

        // 3. Create encoder and dispatch compute pass
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Collapse Cell Encoder"),
        });

        let collapse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Collapse Bind Group"),
            layout: &pipelines.collapse_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers
                        .grid_buffers
                        .grid_possibilities_buf
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.params_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.collapse_info_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Collapse Cell Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipelines.collapse_pipeline);
            compute_pass.set_bind_group(0, &collapse_bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1); // Dispatch single invocation
        } // compute_pass dropped here

        // 4. Submit commands
        queue.submit(std::iter::once(encoder.finish()));

        Ok(())
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
