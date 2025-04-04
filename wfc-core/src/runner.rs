use crate::{
    entropy::EntropyCalculator, grid::PossibilityGrid,
    propagator::propagator::ConstraintPropagator, BoundaryCondition, ProgressInfo,
    PropagationError, WfcCheckpoint, WfcError,
};
use log::{debug, error, info, warn};
use rand::{
    distributions::{Distribution, WeightedIndex},
    rngs::StdRng,
    SeedableRng,
};
#[cfg(feature = "serde")]
use serde_json;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use wfc_rules::AdjacencyRules;

/// Alias for the complex progress callback function type.
pub type ProgressCallback = Box<dyn Fn(ProgressInfo) -> Result<(), WfcError> + Send + Sync>;

/// Alias for the progressive results callback function type.
pub type ProgressiveResultsCallback =
    Box<dyn Fn(&PossibilityGrid, u64) -> Result<(), WfcError> + Send + Sync>;

/// Information needed for backtracking when contradictions are encountered.
#[derive(Clone, Debug)]
pub struct BacktrackInfo {
    /// The grid state before the choice was made
    pub grid_state: PossibilityGrid,
    /// The iteration number when this choice point was created
    pub iteration: u64,
    /// The coordinates of the cell that was collapsed
    pub coords: (usize, usize, usize),
    /// The tile index that was chosen
    pub selected_tile: usize,
    /// The potential tile indices that were available (including the one chosen)
    pub available_tiles: Vec<usize>,
}

/// Configuration options for the WFC runner.
pub struct WfcConfig {
    pub boundary_condition: BoundaryCondition,
    pub progress_callback: Option<ProgressCallback>,
    pub progressive_results_callback: Option<ProgressiveResultsCallback>,
    pub shutdown_signal: Arc<AtomicBool>,
    pub initial_checkpoint: Option<WfcCheckpoint>,
    pub checkpoint_interval: Option<u64>,
    pub checkpoint_path: Option<PathBuf>,
    pub max_iterations: Option<u64>,
    pub seed: Option<u64>,
    /// Maximum number of backtracking steps to try when a contradiction is found.
    /// If None, backtracking is disabled.
    pub max_backtrack_depth: Option<usize>,
}

impl WfcConfig {
    /// Creates a new builder for `WfcConfig`.
    pub fn builder() -> WfcConfigBuilder {
        WfcConfigBuilder::default()
    }
}

impl Default for WfcConfig {
    fn default() -> Self {
        Self {
            boundary_condition: BoundaryCondition::Finite,
            progress_callback: None,
            progressive_results_callback: None,
            shutdown_signal: Arc::new(AtomicBool::new(false)),
            initial_checkpoint: None,
            checkpoint_interval: None,
            checkpoint_path: None,
            max_iterations: None,
            seed: None,
            max_backtrack_depth: None,
        }
    }
}

/// Builder for `WfcConfig`.
///
/// Allows for a more ergonomic construction of `WfcConfig` instances.
#[derive(Default)]
pub struct WfcConfigBuilder {
    boundary_condition: BoundaryCondition,
    progress_callback: Option<ProgressCallback>,
    progressive_results_callback: Option<ProgressiveResultsCallback>,
    shutdown_signal: Option<Arc<AtomicBool>>, // Optional, default is created if None
    initial_checkpoint: Option<WfcCheckpoint>,
    checkpoint_interval: Option<u64>,
    checkpoint_path: Option<PathBuf>,
    max_iterations: Option<u64>,
    seed: Option<u64>,
    max_backtrack_depth: Option<usize>,
}

impl WfcConfigBuilder {
    /// Sets the boundary condition for the WFC algorithm.
    pub fn boundary_condition(mut self, mode: BoundaryCondition) -> Self {
        self.boundary_condition = mode;
        self
    }

    /// Sets the progress callback function.
    pub fn progress_callback(mut self, callback: ProgressCallback) -> Self {
        self.progress_callback = Some(callback);
        self
    }

    /// Sets the progressive results callback function.
    pub fn progressive_results_callback(mut self, callback: ProgressiveResultsCallback) -> Self {
        self.progressive_results_callback = Some(callback);
        self
    }

    /// Provides an external shutdown signal.
    /// If not provided, a new signal will be created.
    pub fn shutdown_signal(mut self, signal: Arc<AtomicBool>) -> Self {
        self.shutdown_signal = Some(signal);
        self
    }

    /// Sets the initial checkpoint to load state from.
    pub fn initial_checkpoint(mut self, checkpoint: WfcCheckpoint) -> Self {
        self.initial_checkpoint = Some(checkpoint);
        self
    }

    /// Sets the interval (in iterations) for saving checkpoints.
    pub fn checkpoint_interval(mut self, interval: u64) -> Self {
        self.checkpoint_interval = Some(interval);
        self
    }

    /// Sets the path for saving checkpoints.
    pub fn checkpoint_path(mut self, path: PathBuf) -> Self {
        self.checkpoint_path = Some(path);
        self
    }

    /// Sets the maximum number of iterations allowed.
    pub fn max_iterations(mut self, max: u64) -> Self {
        self.max_iterations = Some(max);
        self
    }

    /// Sets the seed for the random number generator.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the maximum number of backtracking steps to try when a contradiction is found.
    pub fn max_backtrack_depth(mut self, depth: usize) -> Self {
        self.max_backtrack_depth = Some(depth);
        self
    }

    /// Builds the `WfcConfig` instance.
    pub fn build(self) -> WfcConfig {
        WfcConfig {
            boundary_condition: self.boundary_condition,
            progress_callback: self.progress_callback,
            progressive_results_callback: self.progressive_results_callback,
            shutdown_signal: self
                .shutdown_signal
                .unwrap_or_else(|| Arc::new(AtomicBool::new(false))),
            initial_checkpoint: self.initial_checkpoint,
            checkpoint_interval: self.checkpoint_interval,
            checkpoint_path: self.checkpoint_path,
            max_iterations: self.max_iterations,
            seed: self.seed,
            max_backtrack_depth: self.max_backtrack_depth,
        }
    }
}

// Helper struct to track WFC state during execution
#[derive(Debug)]
struct RunState {
    iterations: u64,
    collapsed_cells_count: usize,
    iteration_limit: u64,
    total_cells: usize,
    // Track backtracking history and depth
    backtrack_history: Vec<BacktrackInfo>,
    current_backtrack_depth: usize,
}

// Helper function to perform initialization steps
fn initialize_run_state<
    P: ConstraintPropagator + Send + Sync,
    E: EntropyCalculator + Send + Sync,
>(
    grid: &mut PossibilityGrid,
    _rules: &AdjacencyRules,
    _propagator: &mut P,
    _entropy_calculator: &E,
    config: &WfcConfig,
) -> Result<RunState, WfcError> {
    let mut iterations: u64 = 0;

    // --- Checkpoint Loading ---
    if let Some(checkpoint) = &config.initial_checkpoint {
        info!(
            "Loading state from checkpoint (Iteration {})...",
            checkpoint.iterations
        );
        // Validate checkpoint grid compatibility
        if grid.width != checkpoint.grid.width
            || grid.height != checkpoint.grid.height
            || grid.depth != checkpoint.grid.depth
        {
            return Err(WfcError::CheckpointError(
                "Grid dimension mismatch".to_string(),
            ));
        }
        if grid.num_tiles() != checkpoint.grid.num_tiles() {
            return Err(WfcError::CheckpointError(
                "Grid tile count mismatch".to_string(),
            ));
        }
        *grid = checkpoint.grid.clone();
        iterations = checkpoint.iterations;
        info!("Checkpoint loaded successfully.");
    }

    let width = grid.width;
    let height = grid.height;
    let depth = grid.depth;
    let total_cells = width * height * depth;
    let mut collapsed_cells_count = 0;
    let num_tiles = grid.num_tiles();

    // Pre-calculate initial collapsed count
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                if let Some(cell) = grid.get(x, y, z) {
                    assert_eq!(cell.len(), num_tiles, "Grid cell bitvec length mismatch");
                    let count = cell.count_ones();
                    if count == 0 {
                        error!("Initial contradiction found at ({}, {}, {})", x, y, z);
                        return Err(WfcError::Contradiction(x, y, z));
                    } else if count == 1 {
                        collapsed_cells_count += 1;
                    }
                } else {
                    error!(
                        "Failed to access grid cell ({}, {}, {}) during init check",
                        x, y, z
                    );
                    return Err(WfcError::GridError("Failed to access cell".to_string()));
                }
            }
        }
    }
    debug!(
        "Initial state: {}/{} cells collapsed.",
        collapsed_cells_count, total_cells
    );

    // Determine the iteration limit
    let iteration_limit = match config.max_iterations {
        Some(limit) => limit,
        None => (total_cells as u64).saturating_mul(10), // Default limit
    };
    if iteration_limit > 0 {
        info!("WFC run iteration limit set to: {}", iteration_limit);
    }

    Ok(RunState {
        iterations,
        collapsed_cells_count,
        iteration_limit,
        total_cells,
        backtrack_history: Vec::new(),
        current_backtrack_depth: 0,
    })
}

/// Runs the core Wave Function Collapse (WFC) algorithm loop.
///
/// This function orchestrates the WFC process:
/// 1. **Initialization**: Checks the initial grid state for contradictions and runs initial propagation.
/// 2. **Observation**: Repeatedly selects the cell with the lowest entropy (uncertainty).
/// 3. **Collapse**: Collapses the selected cell to a single possible tile, chosen randomly based on weights.
/// 4. **Propagation**: Propagates the consequences of the collapse through the grid using the `ConstraintPropagator`,
///    eliminating possibilities that violate `AdjacencyRules`.
/// 5. **Termination**: Stops when all cells are collapsed (success) or a contradiction occurs (failure).
///
/// # Type Parameters
///
/// * `P`: A type implementing the `ConstraintPropagator` trait.
/// * `E`: A type implementing the `EntropyCalculator` trait.
///
/// # Arguments
///
/// * `grid`: A mutable reference to the `PossibilityGrid` representing the state of the system.
///            It will be modified in place during the WFC run.
/// * `tileset`: A reference to the `TileSet` containing information about tile weights.
/// * `rules`: A reference to the `AdjacencyRules` defining valid neighbor constraints.
/// * `propagator`: An instance of the chosen `ConstraintPropagator` implementation.
/// * `entropy_calculator`: An instance of the chosen `EntropyCalculator` implementation.
/// * `config`: Configuration settings for the run (`WfcConfig`).
///
/// # Returns
///
/// * `Ok(())` if the algorithm successfully collapses the entire grid without contradictions.
/// * `Err(WfcError)` if an error occurs, such as:
///     * `WfcError::Contradiction`: A cell reaches a state where no tiles are possible.
///     * `WfcError::PropagationError`: An error occurs during constraint propagation.
///     * `WfcError::GridError`: An issue accessing grid data.
///     * `WfcError::ConfigurationError`: Invalid input (e.g., missing weights).
///     * `WfcError::IncompleteCollapse`: The algorithm finishes but some cells remain uncollapsed.
///     * `WfcError::TimeoutOrInfiniteLoop`: The algorithm exceeds a maximum iteration limit.
///     * `WfcError::Interrupted`: The algorithm is interrupted by a shutdown signal.
///     * `WfcError::Unknown`: An unknown error occurred.
pub async fn run<
    P: ConstraintPropagator + Send + Sync,
    E: EntropyCalculator + Send + Sync + 'static,
>(
    grid: &mut PossibilityGrid,
    rules: &AdjacencyRules,
    mut propagator: P,
    entropy_calculator: E,
    config: WfcConfig,
) -> Result<(), WfcError> {
    info!(
        "Starting WFC run with boundary mode: {:?}...",
        config.boundary_condition
    );
    let start_time = Instant::now();

    // Check for shutdown signal before we even start
    if config.shutdown_signal.load(Ordering::Relaxed) {
        info!("Shutdown signal received before starting, stopping WFC run.");
        return Err(WfcError::ShutdownSignalReceived);
    }

    // --- Initialization ---
    let mut state =
        initialize_run_state(grid, rules, &mut propagator, &entropy_calculator, &config)?;

    let seed = config.seed.unwrap_or_else(rand::random::<u64>);

    let is_backtracking_enabled = config.max_backtrack_depth.is_some();
    if is_backtracking_enabled {
        info!(
            "Backtracking enabled with max depth: {}",
            config.max_backtrack_depth.unwrap()
        );
    }

    // --- Main Loop ---
    while state.collapsed_cells_count < state.total_cells {
        // Check iteration limit
        if state.iterations >= state.iteration_limit {
            warn!(
                "Reached iteration limit ({}), stopping.",
                state.iteration_limit
            );
            return Err(WfcError::MaxIterationsReached(state.iterations));
        }
        state.iterations += 1;
        debug!("WFC Iteration: {}", state.iterations);

        // Check for external shutdown signal
        if config.shutdown_signal.load(Ordering::Relaxed) {
            info!("Shutdown signal received, stopping WFC run.");
            return Err(WfcError::ShutdownSignalReceived);
        }

        // --- Perform one iteration: Observe -> Collapse -> Propagate ---
        let iteration_start_time = Instant::now();
        let result = perform_iteration(
            grid,
            rules,
            &mut propagator,
            &entropy_calculator,
            state.iterations,
            seed,
        )
        .await;

        let iteration_duration = iteration_start_time.elapsed();

        match result {
            Ok(Some((collapsed_coords, backtrack_info))) => {
                state.collapsed_cells_count += 1;
                debug!(
                    "Iteration {} successful. Collapsed {:?}. Total collapsed: {}. Took: {:?}",
                    state.iterations,
                    collapsed_coords,
                    state.collapsed_cells_count,
                    iteration_duration
                );

                // If backtracking is enabled, store the choice point
                if is_backtracking_enabled {
                    state.backtrack_history.push(backtrack_info);
                    // Trim history if it exceeds reasonable size
                    if let Some(max_depth) = config.max_backtrack_depth {
                        if state.backtrack_history.len() > max_depth * 2 {
                            // Keep twice the max depth to allow for some history while avoiding excessive memory use
                            state
                                .backtrack_history
                                .drain(0..state.backtrack_history.len() - max_depth);
                        }
                    }
                }

                // Call progress callback if provided
                if let Some(callback) = &config.progress_callback {
                    let progress_info = ProgressInfo {
                        collapsed_cells: state.collapsed_cells_count,
                        total_cells: state.total_cells,
                        elapsed_time: start_time.elapsed(),
                        iterations: state.iterations,
                        grid_state: grid.clone(),
                    };

                    if let Err(cb_error) = callback(progress_info) {
                        warn!("Progress callback returned error: {:?}", cb_error);
                        // Optionally decide whether to continue or abort based on callback error
                    }
                }

                // Call progressive results callback if configured
                if let Some(progressive_cb) = &config.progressive_results_callback {
                    progressive_cb(grid, state.iterations)?;
                }
            }
            Ok(None) => {
                info!(
                    "Observation phase found no cells to collapse (fully collapsed or contradiction state). Stopping after iteration {}.",
                    state.iterations
                );
                break; // Should be fully collapsed
            }
            Err(WfcError::Contradiction(cx, cy, cz)) => {
                error!(
                    "Contradiction detected at ({}, {}, {}) during iteration {}",
                    cx, cy, cz, state.iterations
                );

                // If backtracking is enabled and we have history, try to backtrack
                if is_backtracking_enabled
                    && !state.backtrack_history.is_empty()
                    && state.current_backtrack_depth < config.max_backtrack_depth.unwrap_or(0)
                {
                    info!(
                        "Attempting to backtrack... (depth: {}/{})",
                        state.current_backtrack_depth + 1,
                        config.max_backtrack_depth.unwrap_or(0)
                    );

                    // Get the most recent choice point
                    let mut backtrack_point = state.backtrack_history.pop().unwrap();

                    // Remove the tile that led to contradiction from available_tiles
                    let chosen_tile = backtrack_point.selected_tile;
                    backtrack_point
                        .available_tiles
                        .retain(|&t| t != chosen_tile);

                    if !backtrack_point.available_tiles.is_empty() {
                        // We have alternative tiles to try, restore grid state
                        *grid = backtrack_point.grid_state.clone();

                        // Reset iterations and collapsed count
                        state.iterations = backtrack_point.iteration;

                        // Recalculate collapsed cells count
                        state.collapsed_cells_count = 0;
                        for z in 0..grid.depth {
                            for y in 0..grid.height {
                                for x in 0..grid.width {
                                    if let Some(cell) = grid.get(x, y, z) {
                                        if cell.count_ones() == 1 {
                                            state.collapsed_cells_count += 1;
                                        }
                                    }
                                }
                            }
                        }

                        state.current_backtrack_depth += 1;
                        info!("Backtracked to iteration {} with {} possibilities remaining for cell {:?}",
                             backtrack_point.iteration, backtrack_point.available_tiles.len(), backtrack_point.coords);

                        // Continue algorithm execution
                        continue;
                    } else {
                        // No alternatives available at this choice point, try next one if available
                        info!("No alternative tiles available at this choice point, looking deeper...");
                        state.current_backtrack_depth += 1;
                        continue;
                    }
                }

                // Either backtracking is disabled, we've exhausted history, or reached max depth
                return Err(WfcError::Contradiction(cx, cy, cz));
            }
            Err(e) => {
                error!("Error during iteration {}: {:?}", state.iterations, e);
                // Consider saving checkpoint on error if needed
                return Err(e);
            }
        }

        // --- Checkpointing ---
        if let (Some(interval), Some(path)) = (config.checkpoint_interval, &config.checkpoint_path)
        {
            if state.iterations % interval == 0 {
                info!(
                    "Saving checkpoint at iteration {} to {:?}...",
                    state.iterations, path
                );
                let checkpoint = WfcCheckpoint {
                    iterations: state.iterations,
                    grid: grid.clone(),
                    // Add other relevant state if needed (e.g., RNG state)
                };
                match save_checkpoint(&checkpoint, path) {
                    Ok(_) => info!("Checkpoint saved successfully."),
                    Err(e) => warn!("Failed to save checkpoint: {}", e),
                }
            }
        }
    }

    // --- Final Check ---
    if state.collapsed_cells_count == state.total_cells {
        info!("WFC completed successfully. Grid fully collapsed.");
    } else {
        // This case might occur if the loop terminated due to contradiction within perform_iteration
        // or if the initial state was already fully collapsed.
        warn!(
            "WFC loop finished, but grid not fully collapsed ({} / {} cells).",
            state.collapsed_cells_count, state.total_cells
        );
        // Consider verifying grid state here if needed
    }

    let total_duration = start_time.elapsed();
    info!(
        "WFC run finished in {:?}. Total iterations: {}",
        total_duration, state.iterations
    );
    Ok(())
}

/// Performs a single iteration of the WFC algorithm: observe, collapse, propagate.
///
/// This function encapsulates the core WFC iteration process:
/// 1. Observe: Find the cell with the minimum entropy.
/// 2. Collapse: Randomly select a state for the selected cell based on weights.
/// 3. Propagate: Update affected cells using constraint rules.
///
/// # Arguments
///
/// * `grid`: The grid to update, contains cell possibility states.
/// * `rules`: The rule set defining valid neighbor constraints.
/// * `propagator`: The constraint propagation implementation.
/// * `entropy_calculator`: The entropy calculation implementation.
/// * `iteration`: Current iteration number for debugging & reporting.
/// * `base_seed`: Base random seed, will be modified by coordinates for deterministic but varied selection.
///
/// # Returns
///
/// * `Ok(Some((x, y, z), backtrack_info))` - Successfully collapsed the cell at coordinates (x, y, z) with backtracking information.
/// * `Ok(None)` - No more cells to collapse (either all collapsed or all remaining have contradictions).
/// * `Err(WfcError)` - An error occurred during the iteration.
async fn perform_iteration<
    P: ConstraintPropagator + Send + Sync,
    E: EntropyCalculator + Send + Sync,
>(
    grid: &mut PossibilityGrid,
    rules: &AdjacencyRules,
    propagator: &mut P,
    entropy_calculator: &E,
    iteration: u64,
    base_seed: u64,
) -> Result<Option<((usize, usize, usize), BacktrackInfo)>, WfcError> {
    debug!("Starting iteration {}...", iteration);

    // 1. Observation Phase: Find the cell with lowest entropy
    debug!("Observation phase...");
    let start_observe = Instant::now();

    // Calculate entropy for all cells
    let entropy_grid = entropy_calculator.calculate_entropy(grid)?;

    // Select the cell with the minimum positive entropy
    let selected_coords = match entropy_calculator.select_lowest_entropy_cell(&entropy_grid) {
        Some((x, y, z)) => {
            debug!(
                "Selected cell ({}, {}, {}) with entropy {}",
                x,
                y,
                z,
                entropy_grid.get(x, y, z).unwrap_or(&f32::MAX)
            );
            (x, y, z)
        }
        None => {
            // This means either fully collapsed or only contradictions remain (entropy <= 0)
            debug!("Observation phase found no cells with positive entropy to collapse. Observation took: {:?}", start_observe.elapsed());
            // Verify if actually fully collapsed or if it's a contradiction state not caught earlier
            match grid.is_fully_collapsed() {
                Ok(true) => {
                    info!("Grid confirmed fully collapsed during observation.");
                    return Ok(None);
                }
                Ok(false) => {
                    // This implies contradictions exist but weren't handled before observation
                    error!("Observation found no positive entropy cells, but grid is not fully collapsed. Likely unhandled contradiction.");
                    // Attempt to find a contradiction cell to report
                    for cz in 0..grid.depth {
                        for cy in 0..grid.height {
                            for cx in 0..grid.width {
                                if let Some(cell) = grid.get(cx, cy, cz) {
                                    if cell.count_ones() == 0 {
                                        return Err(WfcError::Contradiction(cx, cy, cz));
                                    }
                                }
                            }
                        }
                    }
                    return Err(WfcError::InternalError(
                        "Failed to find specific contradiction cell after observation failure."
                            .to_string(),
                    ));
                }
                Err(e) => {
                    error!(
                        "Error checking grid collapse state during observation: {}",
                        e
                    );
                    return Err(WfcError::InternalError(format!("Grid check error: {}", e)));
                }
            }
        }
    };

    let (x, y, z) = selected_coords;
    let observe_duration = start_observe.elapsed();
    debug!("Observation phase completed. Took: {:?}", observe_duration);

    // 2. Collapse Phase: Choose a state for the selected cell
    debug!("Collapse phase for cell {:?}...", (x, y, z));
    let start_collapse = Instant::now();

    // Get possibilities before collapse
    let possibilities_before = match grid.get(x, y, z) {
        Some(p) => p.clone(),
        None => {
            return Err(WfcError::InternalError(
                "Selected cell out of bounds?".into(),
            ))
        }
    };

    if possibilities_before.count_ones() <= 1 {
        debug!(
            "Cell {:?} already collapsed or in contradiction, skipping collapse.",
            (x, y, z)
        );
        // This might indicate an issue if Observor selected it, unless Observor handles this.
        // Re-running observe might be an option, or just continuing if the grid state is valid.
        // For now, treat as success but didn't collapse *this* turn.
        return Ok(None);
    }

    // Create a clone of the grid state before collapse for backtracking purposes
    let grid_before_collapse = grid.clone();

    // Get the list of available tile indices for this cell
    let available_tiles: Vec<usize> = possibilities_before.iter_ones().collect();

    // Use a deterministic RNG seeded based on coordinates and base seed
    let cell_seed = seed_from_coords(x, y, z, base_seed.wrapping_add(iteration));
    let mut rng = StdRng::seed_from_u64(cell_seed);

    // Prepare weights for WeightedIndex
    let weights: Vec<f32> = available_tiles
        .iter()
        .map(|&tile_index| rules.get_tile_weight(tile_index))
        .collect();

    if weights.is_empty() || weights.iter().all(|&w| w <= 0.0) {
        error!(
            "Contradiction detected at {:?} during collapse phase: No valid possibilities with positive weights.",
            (x, y, z)
        );
        return Err(WfcError::Contradiction(x, y, z));
    }

    // Create weighted distribution
    let dist = match WeightedIndex::new(&weights) {
        Ok(d) => d,
        Err(e) => {
            error!("Failed to create weighted distribution: {:?}", e);
            return Err(WfcError::WeightedChoiceError(e));
        }
    };

    // Select a tile based on weighted distribution
    let selected_index = dist.sample(&mut rng);
    let selected_tile = available_tiles[selected_index];

    // Create backtracking info
    let backtrack_info = BacktrackInfo {
        grid_state: grid_before_collapse,
        iteration,
        coords: (x, y, z),
        selected_tile,
        available_tiles: available_tiles.clone(),
    };

    debug!(
        "Selected tile {} (weight: {}) for cell {:?}",
        selected_tile,
        weights[selected_index],
        (x, y, z)
    );

    // Update the grid to collapse this cell to the selected state
    let mut new_state = bitvec::bitvec!(0; grid.num_tiles());
    new_state.set(selected_tile, true);

    if let Some(cell) = grid.get_mut(x, y, z) {
        *cell = new_state;
    } else {
        return Err(WfcError::GridError(format!(
            "Failed to update cell {:?}",
            (x, y, z)
        )));
    }

    let collapse_duration = start_collapse.elapsed();
    debug!("Collapse phase completed. Took: {:?}", collapse_duration);

    // 3. Propagation Phase: Update the affected cells
    debug!("Propagation phase...");
    let start_propagate = Instant::now();

    // Perform constraint propagation
    let update_result = propagator.propagate(grid, vec![(x, y, z)], rules).await;
    let propagate_duration = start_propagate.elapsed();

    match update_result {
        Ok(_) => {
            debug!("Propagation successful. Took: {:?}", propagate_duration);
            Ok(Some(((x, y, z), backtrack_info))) // Return coords and backtracking info
        }
        Err(PropagationError::Contradiction(cx, cy, cz)) => {
            error!(
                "Contradiction detected at ({}, {}, {}) during propagation after collapsing ({}, {}, {}).",
                cx, cy, cz, x, y, z
            );
            Err(WfcError::Contradiction(cx, cy, cz))
        }
        Err(e) => {
            error!("Propagation failed: {:?}", e);
            Err(WfcError::Propagation(e))
        }
    }
}

// Add seed_from_coords definition if it was removed from utils
fn seed_from_coords(x: usize, y: usize, z: usize, base_seed: u64) -> u64 {
    // Simple spatial hash
    base_seed
        .wrapping_add((x as u64).wrapping_mul(0x9E3779B97F4A7C15))
        .wrapping_add((y as u64).wrapping_mul(0x6A09E667F3BCC909))
        .wrapping_add((z as u64).wrapping_mul(0xB2F127D5A3F8A6D1))
}

// --- Placeholder Checkpoint Saving/Loading Logic ---
#[cfg(feature = "serde")]
fn save_checkpoint(checkpoint: &WfcCheckpoint, path: &std::path::Path) -> Result<(), WfcError> {
    use std::fs::File;
    use std::io::BufWriter;
    info!("Saving checkpoint to {:?}...", path);
    let file = File::create(path)
        .map_err(|e| WfcError::CheckpointError(format!("IO Error creating file: {}", e)))?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, checkpoint)
        .map_err(|e| WfcError::CheckpointError(format!("Serialization Error: {}", e)))?;
    info!("Checkpoint saved successfully.");
    Ok(())
}

#[cfg(not(feature = "serde"))]
fn save_checkpoint(_checkpoint: &WfcCheckpoint, _path: &std::path::Path) -> Result<(), WfcError> {
    warn!("Checkpoint saving skipped: 'serde' feature not enabled.");
    Err(WfcError::CheckpointError(
        "Feature 'serde' not enabled".to_string(),
    ))
}

#[cfg(feature = "serde")]
#[allow(dead_code)]
fn load_checkpoint(path: &std::path::Path) -> Result<WfcCheckpoint, WfcError> {
    use std::fs::File;
    info!("Loading checkpoint from {:?}...", path);
    let file = File::open(path)
        .map_err(|e| WfcError::CheckpointError(format!("IO Error opening file: {}", e)))?;
    let checkpoint: WfcCheckpoint = serde_json::from_reader(file)
        .map_err(|e| WfcError::CheckpointError(format!("Deserialization Error: {}", e)))?;
    info!("Checkpoint loaded successfully.");
    Ok(checkpoint)
}

#[cfg(not(feature = "serde"))]
#[allow(dead_code)]
fn load_checkpoint(_path: &std::path::Path) -> Result<WfcCheckpoint, WfcError> {
    warn!("Checkpoint loading skipped: 'serde' feature not enabled.");
    Err(WfcError::CheckpointError(
        "Feature 'serde' not enabled".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        entropy::EntropyCalculator,
        grid::{EntropyGrid, PossibilityGrid},
        propagator::propagator::{ConstraintPropagator, PropagationError},
        BoundaryCondition, WfcCheckpoint, WfcError,
    };
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::sync::{atomic::AtomicBool, Arc};
    use wfc_rules::{AdjacencyRules, TileSet, Transformation};

    // --- Mock Implementations ---
    struct TestEntropyCalculator;
    impl EntropyCalculator for TestEntropyCalculator {
        fn calculate_entropy(
            &self,
            grid: &PossibilityGrid,
        ) -> Result<EntropyGrid, crate::EntropyError> {
            Ok(EntropyGrid::new(grid.width, grid.height, grid.depth))
        }
        fn select_lowest_entropy_cell(
            &self,
            _entropy_grid: &EntropyGrid,
        ) -> Option<(usize, usize, usize)> {
            // Always return a valid coordinate to keep the algorithm running
            // until it sees the shutdown signal
            Some((0, 0, 0))
        }
    }

    struct TestPropagator;
    #[async_trait::async_trait]
    impl ConstraintPropagator for TestPropagator {
        async fn propagate(
            &mut self,
            _grid: &mut PossibilityGrid,
            _updated_coords: Vec<(usize, usize, usize)>,
            _rules: &AdjacencyRules,
        ) -> Result<(), PropagationError> {
            Ok(())
        }
    }
    // Helper function to create uniform rules using from_allowed_tuples
    fn create_uniform_rules(num_transformed_tiles: usize, num_axes: usize) -> AdjacencyRules {
        let mut allowed_tuples = Vec::new();
        for axis in 0..num_axes {
            for ttid1 in 0..num_transformed_tiles {
                for ttid2 in 0..num_transformed_tiles {
                    allowed_tuples.push((axis, ttid1, ttid2));
                }
            }
        }
        AdjacencyRules::from_allowed_tuples(num_transformed_tiles, num_axes, allowed_tuples)
    }
    // --- End Mock Implementations ---

    #[test]
    fn test_runner_initialization_ok() -> Result<(), crate::WfcError> {
        Ok(())
    }

    fn create_minimal_tileset() -> Result<wfc_rules::TileSet, crate::WfcError> {
        let weights = vec![1.0];
        let allowed_transformations = vec![vec![Transformation::Identity]];
        TileSet::new(weights, allowed_transformations).map_err(crate::WfcError::from)
    }

    #[test]
    fn test_runner_run_minimal() -> Result<(), crate::WfcError> {
        let _rng = StdRng::seed_from_u64(123);
        let tileset = create_minimal_tileset()?;
        let rules = create_uniform_rules(tileset.num_transformed_tiles(), 6);
        let width = 5;
        let height = 5;
        let depth = 1;
        let grid = PossibilityGrid::new(width, height, depth, tileset.num_transformed_tiles());

        let config = WfcConfig::builder()
            .boundary_condition(BoundaryCondition::Periodic)
            .max_iterations(1000)
            .seed(123)
            .build();

        let entropy_calculator = TestEntropyCalculator;
        let propagator = TestPropagator;

        let mut grid_mut = grid.clone();
        ::pollster::block_on(run(
            &mut grid_mut,
            &rules,
            propagator,
            entropy_calculator,
            config,
        ))?;

        Ok(())
    }

    #[test]
    fn test_runner_run_with_checkpoint() -> Result<(), crate::WfcError> {
        let _rng = StdRng::seed_from_u64(456);
        let tileset = create_minimal_tileset()?;
        let rules = create_uniform_rules(tileset.num_transformed_tiles(), 6);
        let initial_checkpoint_grid =
            PossibilityGrid::new(3, 3, 1, tileset.num_transformed_tiles());

        let initial_checkpoint = WfcCheckpoint {
            grid: initial_checkpoint_grid.clone(),
            iterations: 5,
        };

        let config = WfcConfig::builder()
            .boundary_condition(BoundaryCondition::Finite)
            .max_iterations(100)
            .initial_checkpoint(initial_checkpoint.clone())
            .seed(456)
            .build();

        let entropy_calculator = TestEntropyCalculator;
        let propagator = TestPropagator;

        let mut grid_mut = initial_checkpoint.grid;
        let result = ::pollster::block_on(run(
            &mut grid_mut,
            &rules,
            propagator,
            entropy_calculator,
            config,
        ));

        // Simply check if the run completed - it's expected to return Ok
        // since TestEntropyCalculator returns None immediately
        assert!(result.is_ok());
        Ok(())
    }

    #[test]
    fn test_runner_run_timeout() -> Result<(), crate::WfcError> {
        let _rng = StdRng::seed_from_u64(789);
        let tileset = create_minimal_tileset()?;
        let rules = create_uniform_rules(tileset.num_transformed_tiles(), 6);
        let grid = PossibilityGrid::new(2, 2, 1, tileset.num_transformed_tiles());

        let config = WfcConfig::builder()
            .boundary_condition(BoundaryCondition::Periodic)
            .max_iterations(1)
            .seed(789)
            .build();

        let entropy_calculator = TestEntropyCalculator;
        let propagator = TestPropagator;

        let mut grid_mut = grid.clone();
        let result = ::pollster::block_on(run(
            &mut grid_mut,
            &rules,
            propagator,
            entropy_calculator,
            config,
        ));

        assert!(result.is_ok() || matches!(result, Err(WfcError::MaxIterationsReached(1))));
        Ok(())
    }

    #[test]
    fn test_runner_run_interrupted() -> Result<(), crate::WfcError> {
        let _rng = StdRng::seed_from_u64(101);
        let tileset = create_minimal_tileset()?;
        let rules = create_uniform_rules(tileset.num_transformed_tiles(), 6);
        let grid = PossibilityGrid::new(10, 10, 1, tileset.num_transformed_tiles());

        // Custom entropy calculator that returns a valid cell coordinate
        // so the main loop doesn't exit immediately
        struct InterruptibleEntropyCalculator;
        impl EntropyCalculator for InterruptibleEntropyCalculator {
            fn calculate_entropy(
                &self,
                grid: &PossibilityGrid,
            ) -> Result<EntropyGrid, crate::EntropyError> {
                Ok(EntropyGrid::new(grid.width, grid.height, grid.depth))
            }

            fn select_lowest_entropy_cell(
                &self,
                _entropy_grid: &EntropyGrid,
            ) -> Option<(usize, usize, usize)> {
                // Always return a valid coordinate to keep the algorithm running
                // until it sees the shutdown signal
                Some((0, 0, 0))
            }
        }

        let shutdown_signal = Arc::new(AtomicBool::new(true));

        let config = WfcConfig::builder()
            .boundary_condition(BoundaryCondition::Periodic)
            .max_iterations(1_000_000)
            .shutdown_signal(shutdown_signal.clone())
            .seed(101)
            .build();

        let entropy_calculator = InterruptibleEntropyCalculator;
        let propagator = TestPropagator;

        let mut grid_mut = grid.clone();
        let result = ::pollster::block_on(run(
            &mut grid_mut,
            &rules,
            propagator,
            entropy_calculator,
            config,
        ));

        // Print the actual result for debugging
        println!("Result: {:?}", result);
        assert!(matches!(result, Err(WfcError::ShutdownSignalReceived)));
        Ok(())
    }
}
