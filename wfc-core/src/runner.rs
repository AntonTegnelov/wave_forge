use crate::{
    entropy::EntropyCalculator, propagator::ConstraintPropagator, BoundaryMode, PossibilityGrid,
    ProgressInfo, PropagationError, WfcCheckpoint, WfcError,
};
use log::{debug, error, info, warn};
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;
use serde_json;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use wfc_rules::{AdjacencyRules, TileSet};

/// Alias for the complex progress callback function type.
pub type ProgressCallback = Box<dyn Fn(ProgressInfo) -> Result<(), WfcError> + Send + Sync>;

/// Configuration options for the WFC runner.
pub struct WfcConfig {
    pub boundary_mode: BoundaryMode,
    pub progress_callback: Option<ProgressCallback>,
    pub shutdown_signal: Arc<AtomicBool>,
    pub initial_checkpoint: Option<WfcCheckpoint>,
    pub checkpoint_interval: Option<u64>,
    pub checkpoint_path: Option<PathBuf>,
    pub max_iterations: Option<u64>,
    pub seed: Option<u64>,
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
            boundary_mode: BoundaryMode::Clamped,
            progress_callback: None,
            shutdown_signal: Arc::new(AtomicBool::new(false)),
            initial_checkpoint: None,
            checkpoint_interval: None,
            checkpoint_path: None,
            max_iterations: None,
            seed: None,
        }
    }
}

/// Builder for `WfcConfig`.
///
/// Allows for a more ergonomic construction of `WfcConfig` instances.
#[derive(Default)]
pub struct WfcConfigBuilder {
    boundary_mode: BoundaryMode,
    progress_callback: Option<ProgressCallback>,
    shutdown_signal: Option<Arc<AtomicBool>>, // Optional, default is created if None
    initial_checkpoint: Option<WfcCheckpoint>,
    checkpoint_interval: Option<u64>,
    checkpoint_path: Option<PathBuf>,
    max_iterations: Option<u64>,
    seed: Option<u64>,
}

impl WfcConfigBuilder {
    /// Sets the boundary mode for the WFC algorithm.
    pub fn boundary_mode(mut self, mode: BoundaryMode) -> Self {
        self.boundary_mode = mode;
        self
    }

    /// Sets the progress callback function.
    pub fn progress_callback(mut self, callback: ProgressCallback) -> Self {
        self.progress_callback = Some(callback);
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

    /// Builds the `WfcConfig` instance.
    pub fn build(self) -> WfcConfig {
        WfcConfig {
            boundary_mode: self.boundary_mode,
            progress_callback: self.progress_callback,
            shutdown_signal: self
                .shutdown_signal
                .unwrap_or_else(|| Arc::new(AtomicBool::new(false))),
            initial_checkpoint: self.initial_checkpoint,
            checkpoint_interval: self.checkpoint_interval,
            checkpoint_path: self.checkpoint_path,
            max_iterations: self.max_iterations,
            seed: self.seed,
        }
    }
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
/// * `P`: A type implementing the `ConstraintPropagator` trait (e.g., `CpuConstraintPropagator`).
/// * `E`: A type implementing the `EntropyCalculator` trait (e.g., `CpuEntropyCalculator`).
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
pub fn run(
    grid: &mut PossibilityGrid,
    tileset: &TileSet,
    rules: &AdjacencyRules,
    mut propagator: Box<dyn ConstraintPropagator + Send + Sync>,
    entropy_calculator: Box<dyn EntropyCalculator + Send + Sync>,
    config: &WfcConfig,
) -> Result<(), WfcError> {
    info!(
        "Starting WFC run with boundary mode: {:?}...",
        config.boundary_mode
    );
    let start_time = Instant::now();
    let mut iterations = 0;

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
            error!(
                "Checkpoint grid dimensions ({}, {}, {}) mismatch the target grid dimensions ({}, {}, {}).",
                checkpoint.grid.width, checkpoint.grid.height, checkpoint.grid.depth,
                grid.width, grid.height, grid.depth
            );
            return Err(WfcError::CheckpointError(
                "Grid dimension mismatch".to_string(),
            ));
        }
        if grid.num_tiles() != checkpoint.grid.num_tiles() {
            error!(
                "Checkpoint grid num_tiles ({}) mismatch the rules num_tiles ({}).",
                checkpoint.grid.num_tiles(),
                grid.num_tiles()
            );
            return Err(WfcError::CheckpointError(
                "Grid tile count mismatch".to_string(),
            ));
        }

        *grid = checkpoint.grid.clone(); // Load grid state (clone from config)
        iterations = checkpoint.iterations; // Load iteration count
        info!("Checkpoint loaded successfully.");
    }
    // --- End Checkpoint Loading ---

    let width = grid.width;
    let height = grid.height;
    let depth = grid.depth;
    let total_cells = width * height * depth;
    let mut collapsed_cells_count = 0;
    let num_tiles = grid.num_tiles();

    // Pre-calculate initial collapsed count using local dimensions
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

    // Run initial propagation based on any pre-collapsed cells or initial constraints.
    // Use local dimensions for coordinate generation
    let all_coords: Vec<(usize, usize, usize)> = (0..depth)
        .flat_map(|z| (0..height).flat_map(move |y| (0..width).map(move |x| (x, y, z))))
        .collect();
    debug!(
        "Running initial propagation for all {} cells...",
        all_coords.len()
    );
    // Clone all_coords for the initial propagation
    if let Err(prop_err) = propagator.propagate(grid, all_coords.clone(), rules) {
        error!("Initial propagation failed: {:?}", prop_err);
        if let PropagationError::Contradiction(cx, cy, cz) = prop_err {
            return Err(WfcError::Contradiction(cx, cy, cz));
        }
        return Err(WfcError::from(prop_err));
    }
    // Recalculate collapsed count after initial propagation
    {
        let current_grid = &*grid; // Borrow immutably
        collapsed_cells_count = (0..depth)
            .flat_map(|z| {
                (0..height).flat_map(move |y| {
                    (0..width).map(move |x| {
                        current_grid // Use immutable borrow
                            .get(x, y, z)
                            .map_or(0, |c| if c.count_ones() == 1 { 1 } else { 0 })
                    })
                })
            })
            .sum();
    }
    debug!(
        "After initial setup/load/propagation: {}/{} cells collapsed. Starting main loop at iter {}.",
        collapsed_cells_count, total_cells, iterations
    );

    // Determine the iteration limit from config
    let iteration_limit = match config.max_iterations {
        Some(limit) => limit,
        None => (total_cells as u64).saturating_mul(10), // Default limit calculation
    };
    if iteration_limit > 0 {
        info!("WFC run iteration limit set to: {}", iteration_limit);
    }

    loop {
        iterations += 1;

        // --- Checkpoint Saving Logic ---
        if let (Some(interval), Some(path)) = (config.checkpoint_interval, &config.checkpoint_path)
        {
            if interval > 0 && iterations % interval == 0 {
                debug!(
                    "Iter {}: Reached checkpoint interval. Saving state...",
                    iterations
                );
                let checkpoint_data = WfcCheckpoint {
                    grid: grid.clone(), // Clone the current grid state
                    iterations,
                };
                match File::create(path) {
                    Ok(file) => {
                        let writer = BufWriter::new(file);
                        match serde_json::to_writer_pretty(writer, &checkpoint_data) {
                            Ok(_) => info!(
                                "Checkpoint saved successfully to {:?} at iteration {}",
                                path, iterations
                            ),
                            Err(e) => warn!(
                                "Failed to serialize and save checkpoint to {:?}: {}",
                                path, e
                            ),
                        }
                    }
                    Err(e) => warn!("Failed to create checkpoint file at {:?}: {}", path, e),
                }
            }
        }
        // --- End Checkpoint Saving Logic ---

        // --- Check for shutdown signal from config ---
        if config.shutdown_signal.load(Ordering::Relaxed) {
            warn!("Shutdown signal received, stopping WFC run prematurely.");
            return Err(WfcError::Interrupted);
        }

        // --- Check if finished (before iteration) ---
        if collapsed_cells_count >= total_cells {
            info!("All cells collapsed.");
            // Final callback before breaking
            if let Some(ref callback) = config.progress_callback {
                let progress_info = ProgressInfo {
                    total_cells,
                    collapsed_cells: collapsed_cells_count,
                    iterations: iterations - 1, // Use previous iter count
                    elapsed_time: start_time.elapsed(),
                    grid_state: grid.clone(),
                };
                callback(progress_info)?;
            }
            break;
        }

        // --- Perform one iteration ---
        match perform_iteration(
            grid,
            tileset,
            rules,
            &mut propagator,
            &entropy_calculator, // Pass by reference
            iterations,
        ) {
            Ok(Some(_collapsed_coords)) => {
                // Recalculate collapsed count after successful iteration
                let current_grid = &*grid; // Borrow immutably
                collapsed_cells_count = (0..depth)
                    .flat_map(|z| {
                        (0..height).flat_map(move |y| {
                            (0..width).map(move |x| {
                                current_grid // Use immutable borrow
                                    .get(x, y, z)
                                    .map_or(0, |c| if c.count_ones() == 1 { 1 } else { 0 })
                            })
                        })
                    })
                    .sum();

                // --- Progress Callback ---
                if let Some(ref callback) = config.progress_callback {
                    let progress_info = ProgressInfo {
                        total_cells,
                        collapsed_cells: collapsed_cells_count,
                        iterations,
                        elapsed_time: start_time.elapsed(),
                        grid_state: grid.clone(),
                    };
                    callback(progress_info)?;
                }
            }
            Ok(None) => {
                // No cell found to collapse, check if grid is actually fully collapsed
                if collapsed_cells_count >= total_cells {
                    info!(
                        "Iter {}: No cell to collapse, grid fully collapsed. Finalizing.",
                        iterations
                    );
                    // Final callback before breaking
                    if let Some(ref callback) = config.progress_callback {
                        let progress_info = ProgressInfo {
                            total_cells,
                            collapsed_cells: collapsed_cells_count,
                            iterations,
                            elapsed_time: start_time.elapsed(),
                            grid_state: grid.clone(),
                        };
                        callback(progress_info)?;
                    }
                    break; // Success
                } else {
                    warn!(
                        "Iter {}: No lowest entropy cell found, but {} cells remain uncollapsed. Checking state...",
                        iterations, total_cells - collapsed_cells_count
                    );
                    // Re-run propagation on all cells to ensure consistency
                    if let Err(prop_err) = propagator.propagate(grid, all_coords.clone(), rules) {
                        error!(
                            "Iter {}: Propagation failed during final check: {:?}",
                            iterations, prop_err
                        );
                        return Err(WfcError::from(prop_err));
                    }
                    let current_grid = &*grid; // Borrow immutably
                    let final_collapsed_count: usize = (0..depth)
                        .flat_map(|z| {
                            (0..height).flat_map(move |y| {
                                (0..width).map(move |x| {
                                    current_grid // USE current_grid here
                                        .get(x, y, z)
                                        .map_or(0, |c| if c.count_ones() == 1 { 1 } else { 0 })
                                })
                            })
                        })
                        .sum();
                    if final_collapsed_count >= total_cells {
                        info!("Iter {}: Grid confirmed fully collapsed after final propagation check.", iterations);
                        break; // Success
                    } else {
                        error!(
                             "Iter {}: Incomplete collapse: {} cells remain uncollapsed after final check.",
                             iterations, total_cells - final_collapsed_count
                         );
                        return Err(WfcError::IncompleteCollapse);
                    }
                }
            }
            Err(e) => return Err(e), // Propagate errors (Contradiction, Config, etc.)
        }

        // Safeguard against infinite loops
        if iteration_limit > 0 && iterations > iteration_limit {
            // Check against the determined limit
            error!(
                "Maximum iterations ({}) exceeded. Assuming infinite loop.",
                iteration_limit
            );
            return Err(WfcError::TimeoutOrInfiniteLoop);
        }
    }

    info!(
        "WFC run finished in {:?} after {} iterations.",
        start_time.elapsed(),
        iterations
    );
    Ok(())
}

/// Performs a single iteration of the WFC algorithm: observe, collapse, propagate.
fn perform_iteration(
    grid: &mut PossibilityGrid,
    tileset: &TileSet,
    rules: &AdjacencyRules,
    propagator: &mut Box<dyn ConstraintPropagator + Send + Sync>,
    entropy_calculator: &Box<dyn EntropyCalculator + Send + Sync>,
    iteration: u64,
) -> Result<Option<(usize, usize, usize)>, WfcError> {
    debug!("Starting WFC iteration...");

    // 1. Observation: Find the cell with the lowest entropy
    debug!("Calculating entropy grid...");
    let entropy_grid = entropy_calculator.calculate_entropy(grid)?;
    debug!("Selecting lowest entropy cell...");
    let lowest_entropy_coords = entropy_calculator.select_lowest_entropy_cell(&entropy_grid);

    if let Some((x, y, z)) = lowest_entropy_coords {
        debug!("Found lowest entropy cell at ({}, {}, {})", x, y, z);
        let cell_to_collapse = grid.get_mut(x, y, z).ok_or_else(|| {
            WfcError::GridError(format!(
                "Lowest entropy cell ({},{},{}) out of bounds",
                x, y, z
            ))
        })?;

        let possible_tile_indices: Vec<usize> = cell_to_collapse.iter_ones().collect();

        if possible_tile_indices.is_empty() {
            return Err(WfcError::Contradiction(x, y, z));
        }

        if possible_tile_indices.len() > 1 {
            // Collect weights corresponding to the *base tiles* of the possible transformed tiles.
            let weights_result: Result<Vec<f32>, WfcError> = possible_tile_indices
                .iter()
                .map(|&ttid| { // ttid is the TransformedTileId (index in BitVec)
                    // 1. Get the base tile ID for this transformed tile ID
                    let (base_id, _transform) = tileset.get_base_tile_and_transform(ttid)
                        .ok_or_else(|| WfcError::InternalError(format!(
                            "Failed to map transformed tile ID {} back to base tile at ({},{},{}).",
                            ttid, x, y, z
                        )))?;
                    // 2. Get the weight associated with that base tile ID
                    tileset.get_weight(base_id)
                        .ok_or_else(|| WfcError::ConfigurationError(format!(
                            "Weight missing for base tile ID {} (derived from ttid {}) at ({}, {}, {}).",
                            base_id.0, ttid, x, y, z
                        )))
                })
                .collect();
            let weights = weights_result?;

            if weights.is_empty() || weights.iter().all(|&w| w <= 0.0) {
                return Err(WfcError::ConfigurationError(
                    "No valid positive weights for collapse choice".to_string(),
                ));
            }

            let dist = WeightedIndex::new(&weights)?;
            let mut rng = thread_rng();
            let chosen_weighted_index = dist.sample(&mut rng);
            let chosen_tile_index = possible_tile_indices[chosen_weighted_index];

            debug!(
                "Iter {}: Collapsing cell ({}, {}, {}) to tile index {} (weight {})",
                iteration, x, y, z, chosen_tile_index, weights[chosen_weighted_index]
            );

            cell_to_collapse.fill(false);
            cell_to_collapse.set(chosen_tile_index, true);

            debug!(
                "Iter {}: Propagating constraints from ({}, {}, {})...",
                iteration, x, y, z
            );
            propagator.propagate(grid, vec![(x, y, z)], rules)?;
            debug!("Iter {}: Propagation successful.", iteration);
            Ok(Some((x, y, z))) // Indicate a collapse happened
        } else {
            // Already collapsed
            debug!(
                "Iter {}: Cell ({}, {}, {}) was already collapsed. Skipping.",
                iteration, x, y, z
            );
            Ok(Some((x, y, z))) // Still considered progress, return coords
        }
    } else {
        Ok(None) // No cell found to collapse
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{entropy::cpu::CpuEntropyCalculator, propagator::cpu::CpuConstraintPropagator};
    use std::{fs, io::Read};
    use tempfile::tempdir;
    use wfc_rules::{TileSetError, Transformation};

    // --- Test Setup Helpers ---
    const TEST_GRID_DIM: usize = 3;
    const TEST_NUM_TILES: usize = 2;

    fn setup_grid() -> PossibilityGrid {
        PossibilityGrid::new(TEST_GRID_DIM, TEST_GRID_DIM, TEST_GRID_DIM, TEST_NUM_TILES)
    }

    fn create_simple_tileset(num_base_tiles: usize) -> Result<TileSet, TileSetError> {
        let weights = vec![1.0; num_base_tiles];
        let allowed_transforms = vec![vec![Transformation::Identity]; num_base_tiles];
        TileSet::new(weights, allowed_transforms)
    }

    fn setup_tileset() -> TileSet {
        create_simple_tileset(TEST_NUM_TILES).expect("Failed to create test tileset")
    }

    fn create_uniform_rules(tileset: &TileSet) -> AdjacencyRules {
        let num_tiles = tileset.num_transformed_tiles();
        let num_axes = 6;
        let mut allowed_tuples = Vec::new();
        for axis in 0..num_axes {
            for ttid1 in 0..num_tiles {
                for ttid2 in 0..num_tiles {
                    allowed_tuples.push((axis, ttid1, ttid2));
                }
            }
        }
        AdjacencyRules::from_allowed_tuples(num_tiles, num_axes, allowed_tuples)
    }

    fn setup_rules() -> AdjacencyRules {
        let tileset = setup_tileset();
        create_uniform_rules(&tileset)
    }

    fn checkpoint_test_config(path: PathBuf, interval: u64) -> WfcConfig {
        WfcConfig {
            checkpoint_path: Some(path),
            checkpoint_interval: Some(interval),
            ..Default::default()
        }
    }

    fn load_checkpoint_test_config(checkpoint: WfcCheckpoint) -> WfcConfig {
        WfcConfig {
            initial_checkpoint: Some(checkpoint),
            ..Default::default()
        }
    }

    #[test]
    fn test_checkpoint_saving() {
        let grid_dim = TEST_GRID_DIM;
        let num_transformed_tiles = TEST_NUM_TILES;
        let mut grid = PossibilityGrid::new(grid_dim, grid_dim, grid_dim, num_transformed_tiles);
        let tileset = setup_tileset();
        let rules = create_uniform_rules(&tileset);
        let propagator = Box::new(CpuConstraintPropagator::new(BoundaryMode::Clamped));
        let entropy_calculator = Box::new(CpuEntropyCalculator::new(
            Arc::new(tileset.clone()),
            crate::entropy::SelectionStrategy::FirstMinimum,
        ));
        let temp_dir = tempdir().unwrap();
        let checkpoint_path = temp_dir.path().join("save_checkpoint.bin");
        let config = checkpoint_test_config(checkpoint_path.clone(), 1);

        // Partially collapse grid to have state to save
        if let Some(cell) = grid.get_mut(0, 0, 0) {
            cell.set(0, false); // Make it not fully random
        }

        let result = run(
            &mut grid,
            &tileset,
            &rules,
            propagator,
            entropy_calculator,
            &config,
        );

        assert!(result.is_ok());
        assert!(checkpoint_path.exists());

        // Verify content (basic check)
        let mut file = fs::File::open(checkpoint_path).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();
        assert!(!buffer.is_empty());
        let checkpoint: WfcCheckpoint = serde_json::from_slice(&buffer).unwrap();
        assert!(checkpoint.iterations > 0);
        assert_eq!(checkpoint.grid.width, grid_dim);
    }

    #[test]
    fn test_checkpoint_loading() {
        let grid_dim = TEST_GRID_DIM;
        let num_transformed_tiles = TEST_NUM_TILES;
        let grid_orig = PossibilityGrid::new(grid_dim, grid_dim, grid_dim, num_transformed_tiles);
        let tileset = setup_tileset();
        let rules = create_uniform_rules(&tileset);

        // Create a checkpoint state
        let mut checkpoint_grid = grid_orig.clone();
        checkpoint_grid.get_mut(1, 1, 1).unwrap().set(0, false); // Modify state
        let checkpoint = WfcCheckpoint {
            grid: checkpoint_grid.clone(),
            iterations: 5, // Example iteration count
        };

        // Run with loaded checkpoint
        let mut grid_load = setup_grid(); // Fresh grid
        let propagator = Box::new(CpuConstraintPropagator::new(BoundaryMode::Clamped));
        let entropy_calculator = Box::new(CpuEntropyCalculator::new(
            Arc::new(tileset.clone()),
            crate::entropy::SelectionStrategy::FirstMinimum,
        ));
        let config = load_checkpoint_test_config(checkpoint);

        let result = run(
            &mut grid_load,
            &tileset,
            &rules,
            propagator,
            entropy_calculator,
            &config,
        );

        assert!(result.is_ok());
        // Check if the grid state started from the checkpoint state
        assert_eq!(grid_load.get(1, 1, 1), checkpoint_grid.get(1, 1, 1));
        // TODO: More robust check? Maybe check iteration count via progress?
    }

    #[test]
    fn test_run_success() {
        let mut grid = setup_grid();
        let tileset = setup_tileset();
        let rules = setup_rules();
        let propagator = Box::new(CpuConstraintPropagator::new(BoundaryMode::Clamped));
        let entropy_calculator = Box::new(CpuEntropyCalculator::new(
            Arc::new(tileset.clone()),
            crate::entropy::SelectionStrategy::FirstMinimum,
        ));
        let config = WfcConfig::default();

        let result = run(
            &mut grid,
            &tileset,
            &rules,
            propagator,
            entropy_calculator,
            &config,
        );
        assert!(result.is_ok());
        // Check if fully collapsed (every cell has exactly 1 possibility)
        let is_fully_collapsed = (0..grid.depth).all(|z| {
            (0..grid.height).all(|y| {
                (0..grid.width).all(|x| grid.get(x, y, z).map_or(false, |c| c.count_ones() == 1))
            })
        });
        assert!(is_fully_collapsed, "Grid was not fully collapsed");
    }

    #[test]
    fn test_run_contradiction() {
        let mut grid = setup_grid();
        let tileset = setup_tileset();
        // Create rules where T0 cannot be next to T0 (guaranteed contradiction in 1x1x1)
        let rules = AdjacencyRules::from_allowed_tuples(TEST_NUM_TILES, 6, vec![(0, 1, 1)]); // Only T1->T1 on +X
        let propagator = Box::new(CpuConstraintPropagator::new(BoundaryMode::Clamped));
        let entropy_calculator = Box::new(CpuEntropyCalculator::new(
            Arc::new(tileset.clone()),
            crate::entropy::SelectionStrategy::FirstMinimum,
        ));
        let config = WfcConfig {
            max_iterations: Some(10),
            ..Default::default()
        }; // Limit iterations

        // Force initial state to T0
        grid.get_mut(0, 0, 0).unwrap().fill(false);
        grid.get_mut(0, 0, 0).unwrap().set(0, true);

        let result = run(
            &mut grid,
            &tileset,
            &rules,
            propagator,
            entropy_calculator,
            &config,
        );
        assert!(matches!(
            result,
            Err(WfcError::Contradiction(_, _, _)) | Err(WfcError::PropagationError(_))
        ));
    }

    // Add BoundaryMode to other test setups as needed...
    // test_checkpoint_loading_dimension_mismatch
    // test_checkpoint_loading_tile_count_mismatch
    // test_run_with_checkpoint_load_success
    // test_run_max_iterations
}
