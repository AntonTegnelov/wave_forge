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
use log::{error, info, trace, warn};
use rand::Rng;
use std::time::Instant;
use tokio::sync::watch;
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
        println!("[WFC-GPU DEBUG] GpuAccelerator::new called.");
        println!(
            "[WFC-GPU DEBUG]   Initial grid: width={}, height={}, depth={}, num_tiles={}",
            initial_grid.width,
            initial_grid.height,
            initial_grid.depth,
            initial_grid.num_tiles()
        );
        println!(
            "[WFC-GPU DEBUG]   Boundary condition: {:?}, Entropy heuristic: {:?}",
            boundary_condition, entropy_heuristic
        );
        if let Some(ref config) = subgrid_config {
            println!("[WFC-GPU DEBUG]   Subgrid config: {:?}", config);
        } else {
            println!("[WFC-GPU DEBUG]   Subgrid config: None");
        }

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
        println!("[WFC-GPU DEBUG]   ComputePipelines created.");

        let buffers = Arc::new(
            GpuBuffers::new(&device, &queue, initial_grid, rules, boundary_condition)
                .map_err(WfcError::Gpu)?,
        );
        println!("[WFC-GPU DEBUG]   GpuBuffers created.");

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
        println!(
            "[WFC-GPU DEBUG]   GpuParamsUniform created and sent to synchronizer: {:?}",
            params
        );

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
        &self,
        initial_grid: &PossibilityGrid,
        _rules: &AdjacencyRules,
        _grid_definition: GridDefinition,
        max_iterations: usize,
        mut progress_callback: F,
        shutdown_signal: Option<Arc<watch::Receiver<bool>>>,
    ) -> Result<WfcRunResult, WfcError>
    where
        F: FnMut(ProgressInfo) -> Result<bool, AnyhowError> + Send + Sync + 'static,
    {
        println!(
            "[WFC-GPU DEBUG] GpuAccelerator::run_with_callback started. Max iterations: {}, Initial grid: {}x{}x{} ({} tiles)",
            max_iterations,
            initial_grid.width,
            initial_grid.height,
            initial_grid.depth,
            initial_grid.num_tiles()
        );
        let start_time = Instant::now();
        let mut stats = GridStats::default(); // Initialize stats

        let (device, queue, synchronizer, buffers, coordinator, entropy_calculator, propagator) = {
            let instance = self.instance.read().unwrap();
            (
                instance.backend.device(),
                instance.backend.queue(),
                instance.sync.clone(),
                instance.buffers.clone(),
                instance.coordinator.clone_box(),
                instance.entropy_calculator.clone(),
                instance.propagator.clone(),
            )
        };

        info!(
            "Running WFC on GPU for grid {}x{}x{} with {} tiles. Max iterations: {}",
            self.grid_definition().dims.0,
            self.grid_definition().dims.1,
            self.grid_definition().dims.2,
            self.num_tiles(),
            max_iterations
        );

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
        let total_cells = self.grid_definition().total_cells();
        let mut iterations: usize = 0; // Explicitly usize

        println!(
            "[WFC-GPU DEBUG] Starting main WFC loop. Total cells: {}",
            total_cells
        );
        // Main WFC loop
        while iterations < max_iterations {
            println!(
                "[WFC-GPU DEBUG] Iteration {} / {}",
                iterations + 1,
                max_iterations
            );
            trace!("WFC Iteration {} / {}", iterations + 1, max_iterations);
            // Calculate entropy and select cell
            let selected_cell_coords = match coordinator
                .coordinate_entropy_and_selection(
                    &entropy_calculator,
                    &buffers,
                    &device,
                    &queue,
                    &synchronizer,
                )
                .await
            {
                Ok(Some(coords)) => {
                    println!("[WFC-GPU DEBUG] Cell selected by coordinator: {:?}", coords);
                    coords
                }
                Ok(None) => {
                    println!("[WFC-GPU DEBUG] Coordinator found no cell to select. Grid fully collapsed or contradiction. Iteration {}.", iterations);
                    trace!(
                        "Coordinator found no cell to select (grid likely fully collapsed or in contradiction). Iteration {}.",
                        iterations
                    );
                    break; // Exit loop if no cell selected
                }
                Err(e) => {
                    // Convert Box<dyn Error + Send + Sync> to WfcError
                    // This specific conversion might need adjustment based on the actual error type from coordinator
                    println!("[WFC-GPU DEBUG] Error during entropy selection: {:?}", e);
                    return Err(WfcError::other(format!("Entropy selection error: {}", e)));
                }
            };

            let (x, y, z) = selected_cell_coords;

            // Collapse the selected cell on CPU side first
            // This part needs to be done carefully to ensure CPU and GPU states are consistent
            // or that GPU collapse is the source of truth before propagation.
            // The current test seems to assume CPU grid is modified then uploaded.
            let chosen_tile_id = {
                let cell_possibilities = current_grid.get(x, y, z).ok_or_else(|| {
                    WfcError::Algorithm(format!(
                        "Selected cell ({},{},{}) out of bounds during CPU collapse decision",
                        x, y, z
                    ))
                })?;

                let possible_states: Vec<usize> = cell_possibilities.iter_ones().collect();
                if possible_states.is_empty() {
                    // This should ideally be caught by GPU contradiction or entropy selection returning None
                    error!(
                        "Selected cell ({}, {}, {}) has no possible states for CPU collapse. Iteration {}.",
                        x, y, z, iterations
                    );
                    // If this happens, it implies a contradiction not caught by the GPU selection or a desync.
                    // Check GPU contradiction flag here as an early exit.
                    let (gpu_contradicted_early, gpu_loc_early) = synchronizer
                        .download_contradiction_status()
                        .await
                        .map_err(WfcError::from)?;
                    if gpu_contradicted_early {
                        let (gx, gy, gz) = gpu_loc_early.map_or((0, 0, 0), |fi| {
                            let (w, h, _) = self.grid_definition().dims;
                            (
                                fi as usize % w,
                                (fi as usize % (w * h)) / w,
                                fi as usize / (w * h),
                            )
                        });
                        return Err(WfcError::Contradiction {
                            x: gx,
                            y: gy,
                            z: gz,
                        });
                    }
                    return Err(WfcError::Contradiction { x, y, z });
                }

                // Using the simple random choice as in the original loop structure.
                // TODO: Ensure this matches the GPU collapse choice if GPU is authoritative for collapse.
                // For now, assume CPU dictates the choice, which is then uploaded and used for GPU propagation.
                let mut rng = rand::thread_rng();
                let chosen_idx = rng.gen_range(0..possible_states.len());
                let tile_id = possible_states[chosen_idx];
                println!("[WFC-GPU DEBUG] CPU: Collapsing cell ({},{},{}) to tile_id (transformed): {}. ({} possibilities)", x, y, z, tile_id, possible_states.len());
                tile_id
            };

            current_grid
                .collapse(x, y, z, chosen_tile_id)
                .map_err(|e| {
                    WfcError::Algorithm(format!(
                        "Failed to collapse cell ({},{},{}) on CPU: {}",
                        x, y, z, e
                    ))
                })?;
            stats.collapsed_cells += 1; // Update stats

            // Upload the entire grid state
            synchronizer
                .upload_grid(&current_grid) // Removed .await, assuming sync
                .map_err(WfcError::from)?;
            println!("[WFC-GPU DEBUG] Grid state uploaded to GPU for propagation. Cell ({},{},{}) collapsed to tile {}.", x,y,z, chosen_tile_id);

            // Propagate constraints on the GPU
            let propagation_result = coordinator
                .coordinate_propagation(
                    &propagator,
                    &buffers,
                    &device,
                    &queue,
                    vec![GridCoord { x, y, z }],
                )
                .await;

            match propagation_result {
                Ok(_) => {
                    println!(
                        "[WFC-GPU DEBUG] GPU Propagation successful for cell ({},{},{}).",
                        x, y, z
                    );
                    /* continue */
                }
                Err(e) => {
                    // Handle propagation error - potentially a contradiction found by GPU propagation
                    // The error type from coordinate_propagation needs to be mapped to WfcError
                    println!("[WFC-GPU DEBUG] GPU Propagation error: {:?}. Iteration {}. Attempting to download grid state.", e, iterations);
                    error!("GPU Propagation error: {:?}. Iteration {}.", e, iterations);
                    // Attempt to download GPU state for diagnosis
                    current_grid = synchronizer
                        .download_grid(&current_grid) // Reverted to original call signature
                        .await
                        .map_err(WfcError::from)?;
                    // Check GPU contradiction flag directly as propagation might have set it
                    let (gpu_contradicted_prop, gpu_loc_prop) = synchronizer
                        .download_contradiction_status()
                        .await
                        .map_err(WfcError::from)?;
                    if gpu_contradicted_prop {
                        if let Some(flat_index) = gpu_loc_prop {
                            let (w, h, _) = self.grid_definition().dims;
                            let z_gpu = flat_index as usize / (w * h);
                            let y_gpu = (flat_index as usize % (w * h)) / w;
                            let x_gpu = flat_index as usize % w;
                            return Err(WfcError::Contradiction {
                                x: x_gpu,
                                y: y_gpu,
                                z: z_gpu,
                            });
                        } else {
                            return Err(WfcError::GpuContradictionUnknownLocation);
                        }
                    }
                    // If not a specific GPU contradiction, convert the propagation error.
                    // This assumes `e` can be converted or mapped. For now, wrap it.
                    return Err(WfcError::Algorithm(format!("Propagation failed: {}", e)));
                }
            }

            // Download the updated grid state from GPU after propagation
            current_grid = synchronizer
                .download_grid(&current_grid) // Reverted to original call signature
                .await
                .map_err(WfcError::from)?;
            println!("[WFC-GPU DEBUG] Grid state downloaded from GPU after propagation.");

            // Call progress callback
            let progress_info = ProgressInfo {
                total_cells,
                collapsed_cells: stats.collapsed_cells,
                iterations: iterations as u64,
                elapsed_time: start_time.elapsed(),
                grid_state: current_grid.clone(),
            };

            println!(
                "[WFC-GPU DEBUG] Calling progress callback. Iter: {}, Collapsed: {}, Elapsed: {:?}",
                iterations,
                stats.collapsed_cells,
                start_time.elapsed()
            );
            // Allow callback to interrupt
            if !(progress_callback(progress_info)
                .map_err(|e| WfcError::Other(format!("Progress callback error: {}", e)))?)
            {
                info!(
                    "WFC run interrupted by callback at iteration {}.",
                    iterations
                );
                return Err(WfcError::Other("Run interrupted by callback".to_string()));
            }

            if let Some(sd_watch_rx) = &shutdown_signal {
                if *sd_watch_rx.borrow() {
                    // Correct way to read from watch::Receiver
                    info!(
                        "WFC run interrupted by shutdown signal at iteration {}.",
                        iterations
                    );
                    return Err(WfcError::Other(
                        "Run interrupted by shutdown signal".to_string(),
                    ));
                }
            }

            iterations += 1;
            stats.iterations = iterations;
        }

        trace!("WFC loop finished after {} iterations.", iterations);
        println!("[WFC-GPU DEBUG] WFC loop finished. Total iterations: {}. Current collapsed cells (CPU count): {}", iterations, current_grid.count_collapsed_cells());

        // After the loop, check final state
        let (gpu_had_contradiction, gpu_contradiction_flat_index) = synchronizer
            .download_contradiction_status()
            .await
            .map_err(WfcError::from)?;

        match current_grid.is_fully_collapsed() {
            Ok(true) => {
                info!(
                    "WFC successfully completed in {} iterations. Grid fully collapsed.",
                    iterations
                );
                println!("[WFC-GPU DEBUG] WFC success! Grid fully collapsed. Iterations: {}, Final stats: {:?}", iterations, stats);
                // Ensure stats reflects full collapse if loop exited early due to selection returning None
                // but grid was already fully collapsed.
                if stats.collapsed_cells < total_cells {
                    stats.collapsed_cells = total_cells;
                }
                Ok(WfcRunResult::new(current_grid, stats)) // Return WfcRunResult
            }
            Ok(false) => {
                // Not fully collapsed, and is_fully_collapsed() itself reported Ok(false)
                if gpu_had_contradiction {
                    warn!(
                        "WFC loop ended. Grid not fully collapsed, and GPU reported a contradiction. Iterations: {}.",
                        iterations
                    );
                    println!("[WFC-GPU DEBUG] WFC loop ended. Grid not fully collapsed. GPU reported contradiction. Iterations: {}, GPU Contradiction Index: {:?}", iterations, gpu_contradiction_flat_index);
                    if let Some(flat_index) = gpu_contradiction_flat_index {
                        let (w, h, _) = self.grid_definition().dims;
                        let z_gpu = flat_index as usize / (w * h);
                        let y_gpu = (flat_index as usize % (w * h)) / w;
                        let x_gpu = flat_index as usize % w;
                        Err(WfcError::Contradiction {
                            x: x_gpu,
                            y: y_gpu,
                            z: z_gpu,
                        })
                    } else {
                        Err(WfcError::GpuContradictionUnknownLocation)
                    }
                } else {
                    warn!(
                        "WFC loop ended. Grid not fully collapsed, but no GPU contradiction reported. Iterations: {}.",
                        iterations
                    );
                    println!("[WFC-GPU DEBUG] WFC loop ended. Grid not fully collapsed. No GPU contradiction. Iterations: {}", iterations);
                    Err(WfcError::IncompleteCollapse)
                }
            }
            Err(cpu_err_string) => {
                // is_fully_collapsed() returned an Err(String)
                error!(
                    "WFC loop ended. CPU grid check failed: {}. GPU contradiction: {}. Iterations: {}.",
                    cpu_err_string, gpu_had_contradiction, iterations
                );
                println!("[WFC-GPU DEBUG] WFC loop ended. CPU grid check failed: \"{}\". GPU contradiction: {}, Iterations: {}", cpu_err_string, gpu_had_contradiction, iterations);
                if let Some((cx, cy, cz)) = parse_contradiction_string(&cpu_err_string) {
                    Err(WfcError::Contradiction {
                        x: cx,
                        y: cy,
                        z: cz,
                    })
                } else if gpu_had_contradiction {
                    if let Some(flat_index) = gpu_contradiction_flat_index {
                        let (w, h, _) = self.grid_definition().dims;
                        let z_gpu = flat_index as usize / (w * h);
                        let y_gpu = (flat_index as usize % (w * h)) / w;
                        let x_gpu = flat_index as usize % w;
                        Err(WfcError::Contradiction {
                            x: x_gpu,
                            y: y_gpu,
                            z: z_gpu,
                        })
                    } else {
                        Err(WfcError::GpuContradictionUnknownLocation)
                    }
                } else {
                    Err(WfcError::GridStateCheckError(cpu_err_string))
                }
            }
        }
    }

    pub fn with_entropy_heuristic(&mut self, heuristic: CoreEntropyHeuristicType) -> &mut Self {
        println!(
            "[WFC-GPU DEBUG] GpuAccelerator::with_entropy_heuristic called with {:?}",
            heuristic
        );
        let mut instance = self
            .instance
            .write()
            .expect("Failed to acquire write lock on AcceleratorInstance");

        // Update the GpuParamsUniform with the new heuristic
        let params = instance.params.clone();
        let mut new_params = params.clone();
        new_params.heuristic_type = match heuristic {
            CoreEntropyHeuristicType::Shannon => 0,
            CoreEntropyHeuristicType::Count => 1,
            CoreEntropyHeuristicType::CountSimple => 2,
            CoreEntropyHeuristicType::WeightedCount => 3,
        };
        instance.sync.update_propagation_params(&new_params)?;
        instance.params = new_params;
        self
    }

    pub fn with_entropy_strategy_boxed(
        &mut self,
        strategy: Box<dyn GpuEntropyStrategy>,
    ) -> &mut Self {
        println!("[WFC-GPU DEBUG] GpuAccelerator::with_entropy_strategy_boxed called.");
        let mut instance = self
            .instance
            .write()
            .expect("Failed to acquire write lock on AcceleratorInstance");

        let strategy_adapter = Box::new(EntropyStrategyAdapter(strategy));
        instance.entropy_calculator = Arc::new(GpuEntropyCalculator::new(
            instance.backend.device().clone(),
            instance.backend.queue().clone(),
            instance.pipelines.clone(),
            instance.buffers.clone(),
            instance.grid_definition.dims,
        ));
        instance
            .entropy_calculator
            .set_entropy_heuristic(strategy_adapter.entropy_heuristic());
        self
    }

    pub fn with_propagation_strategy_boxed(
        &mut self,
        strategy: Box<dyn crate::propagator::AsyncPropagationStrategy + Send + Sync>,
    ) -> &mut Self {
        println!("[WFC-GPU DEBUG] GpuAccelerator::with_propagation_strategy_boxed called.");
        // Create a new propagator with the new strategy
        {
            // Lock the instance for writing
            let instance = self.instance.write().unwrap();

            // Get a mutable reference to the propagator - use async lock in sync context
            let mut propagator_guard =
                futures::executor::block_on(async { instance.propagator.write().await });

            // Get the needed GPU resources
            let device = instance.backend.device();
            let queue = instance.backend.queue();
            let pipelines = instance.pipelines.clone();
            let buffers = instance.buffers.clone();

            // Set up configuration parameters for propagator
            let grid_dims = (
                instance.grid_definition.dims.0,
                instance.grid_definition.dims.1,
                instance.grid_definition.dims.2,
            );
            // Use the existing params from the instance, but allow strategy to override if needed.
            // This assumes the core params like grid dimensions are fixed after GpuAccelerator::new().
            let params = instance.params.clone();

            // Re-initialize the propagator with the new strategy and existing resources.
            // This is a simplified approach. A more robust one might involve specific
            // methods on GpuConstraintPropagator to update its strategy and re-bind resources if needed.
            *propagator_guard = GpuConstraintPropagator::new_with_strategy(
                device,
                queue,
                pipelines,
                buffers,
                grid_dims,
                instance.boundary_condition,
                params,   // Pass the cloned params
                strategy, // Pass the new strategy
            );
        }
        self
    }

    /// Use direct propagation for WFC algorithm execution.
    ///
    /// `&mut Self` for method chaining.
    pub fn with_direct_propagation(&mut self, max_iterations: u32) -> &mut Self {
        println!(
            "[WFC-GPU DEBUG] GpuAccelerator::with_direct_propagation called. Max iterations: {}",
            max_iterations
        );
        let pipelines = self.instance.read().unwrap().pipelines.clone();
        let strategy = PropagationStrategyFactory::create_direct_async(max_iterations, pipelines);
        self.with_propagation_strategy_boxed(strategy)
    }

    /// Use subgrid propagation for WFC algorithm execution.
    ///
    /// `&mut Self` for method chaining.
    pub fn with_subgrid_propagation(
        &mut self,
        max_iterations: u32,
        subgrid_size: u32,
    ) -> &mut Self {
        println!("[WFC-GPU DEBUG] GpuAccelerator::with_subgrid_propagation called. Max iterations: {}, Subgrid size: {}", max_iterations, subgrid_size);
        let pipelines = self.instance.read().unwrap().pipelines.clone();
        let strategy = PropagationStrategyFactory::create_subgrid_async(
            max_iterations,
            subgrid_size,
            pipelines,
        );
        self.with_propagation_strategy_boxed(strategy)
    }

    /// Use adaptive propagation for WFC algorithm execution.
    ///
    /// `&mut Self` for method chaining.
    pub fn with_adaptive_propagation(
        &mut self,
        max_iterations: u32,
        subgrid_size: u32,
        size_threshold: usize,
    ) -> &mut Self {
        println!("[WFC-GPU DEBUG] GpuAccelerator::with_adaptive_propagation called. Max iterations: {}, Subgrid size: {}, Size threshold: {}", max_iterations, subgrid_size, size_threshold);
        let pipelines = self.instance.read().unwrap().pipelines.clone();
        let strategy = PropagationStrategyFactory::create_adaptive_async(
            max_iterations,
            subgrid_size,
            size_threshold,
            pipelines,
        );
        self.with_propagation_strategy_boxed(strategy)
    }

    /// Automatically select and configure the propagation strategy based on grid size.
    ///
    /// `&mut Self` for method chaining.
    pub fn with_auto_propagation(&mut self) -> &mut Self {
        println!("[WFC-GPU DEBUG] GpuAccelerator::with_auto_propagation called.");
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

    /// Use adaptive coordination for WFC algorithm execution.
    /// This strategy selects the most appropriate coordination approach based on grid size.
    pub fn with_adaptive_coordination(&mut self) -> &mut Self {
        println!("[WFC-GPU DEBUG] GpuAccelerator::with_adaptive_coordination called.");
        // Create the needed components outside the instance lock scope
        let grid_size;
        let entropy_calculator_clone;
        let propagator_clone;

        {
            // Limit the scope of the instance borrow
            let instance = self.instance.read().unwrap();
            grid_size = instance.grid_definition.dims;
            entropy_calculator_clone = instance.entropy_calculator.clone();
            propagator_clone = instance.propagator.clone();
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
            let mut instance = self.instance.write().unwrap();
            instance.coordinator = Box::new(coordinator);
        }

        self
    }

    /// Use the default coordination strategy for WFC algorithm execution.
    pub fn with_default_coordination(&mut self) -> &mut Self {
        println!("[WFC-GPU DEBUG] GpuAccelerator::with_default_coordination called.");
        // Create the needed components outside the instance lock scope
        let entropy_calculator_clone;
        let propagator_clone;

        {
            // Limit the scope of the instance borrow
            let instance = self.instance.read().unwrap();
            entropy_calculator_clone = instance.entropy_calculator.clone();
            propagator_clone = instance.propagator.clone();
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
            let mut instance = self.instance.write().unwrap();
            instance.coordinator = Box::new(coordinator);
        }

        self
    }

    /// Sets a custom coordination strategy for the WFC algorithm.
    pub fn with_coordination_strategy<S: strategy::CoordinationStrategy + 'static>(
        &mut self,
        strategy: S,
    ) -> &mut Self {
        println!("[WFC-GPU DEBUG] GpuAccelerator::with_coordination_strategy called.");
        // Create the needed components outside the instance lock scope
        let entropy_calculator_clone;
        let propagator_clone;

        {
            // Limit the scope of the instance borrow
            let instance = self.instance.read().unwrap();
            entropy_calculator_clone = instance.entropy_calculator.clone();
            propagator_clone = instance.propagator.clone();
        }

        // Create a new DefaultCoordinator and set the strategy
        let mut coordinator = DefaultCoordinator::new(entropy_calculator_clone, propagator_clone);
        coordinator.with_coordination_strategy(strategy);

        // Set the coordinator with a separate lock scope
        {
            let mut instance = self.instance.write().unwrap();
            instance.coordinator = Box::new(coordinator);
        }

        self
    }

    /// Sets a custom boxed coordination strategy for the WFC algorithm.
    pub fn with_coordination_strategy_boxed(
        &mut self,
        strategy: Box<dyn strategy::CoordinationStrategy>,
    ) -> &mut Self {
        println!("[WFC-GPU DEBUG] GpuAccelerator::with_coordination_strategy_boxed called.");
        // Create the needed components outside the instance lock scope
        let entropy_calculator_clone;
        let propagator_clone;

        {
            // Limit the scope of the instance borrow
            let instance = self.instance.read().unwrap();
            entropy_calculator_clone = instance.entropy_calculator.clone();
            propagator_clone = instance.propagator.clone();
        }

        // Create a new DefaultCoordinator and set the strategy
        let mut coordinator = DefaultCoordinator::new(entropy_calculator_clone, propagator_clone);
        coordinator.with_coordination_strategy_boxed(strategy);

        // Set the coordinator with a separate lock scope
        {
            let mut instance = self.instance.write().unwrap();
            instance.coordinator = Box::new(coordinator);
        }

        self
    }

    fn drop(&mut self) {
        // This ensures that the instance is dropped, which in turn should handle
        // the proper cleanup of GPU resources if AcceleratorInstance implements Drop.
        // However, explicit cleanup methods on AcceleratorInstance might be preferred
        // if specific order or error handling is needed for resource deallocation.
        if Arc::strong_count(&self.instance) == 1 {
            if let Ok(instance_guard) = self.instance.try_write() {
                // Perform any explicit cleanup if necessary before the instance is dropped
                println!("[WFC-GPU DEBUG] GpuAccelerator::drop called. Last reference, instance will be dropped.");
                if let Some(visualizer) = &instance_guard.debug_visualizer {
                    visualizer.cleanup(); // Example: ensure visualizer resources are released
                    println!("[WFC-GPU DEBUG] Debug visualizer cleaned up.");
                }
            } else {
                warn!("[WFC-GPU DEBUG] GpuAccelerator::drop: Could not acquire write lock, instance might not be fully cleaned up if it relies on Drop for GPU resources.");
            }
        } else {
            trace!(
                "[WFC-GPU DEBUG] GpuAccelerator::drop called. {} strong references remaining.",
                Arc::strong_count(&self.instance)
            );
        }
    }
}
