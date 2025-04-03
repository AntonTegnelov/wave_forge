use crate::{
    entropy::EntropyCalculator,
    grid::PossibilityGrid,
    propagator::{ConstraintPropagator, PropagationError},
    BoundaryMode, ProgressInfo, WfcCheckpoint, WfcError,
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
use wfc_rules::{AdjacencyRules, TileSet, TileSetError, Transformation};

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
    // --- Imports ---
    use super::*;
    use crate::{
        entropy::{EntropyCalculator, EntropyError},
        grid::{EntropyGrid, PossibilityGrid},
        propagator::{ConstraintPropagator, PropagationError},
        ProgressInfo, WfcCheckpoint, WfcError,
    };
    use mockall::{mock, predicate::*};
    use std::path::PathBuf;
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    };
    use tempfile::tempdir;
    use wfc_rules::{AdjacencyRules, TileSet};

    // --- Mocks ---
    mock! {
        pub EntropyCalculator {}
        impl EntropyCalculator for EntropyCalculator {
             fn calculate_entropy(
                &self,
                grid: &PossibilityGrid,
             ) -> Result<EntropyGrid, EntropyError>;
             fn select_lowest_entropy_cell(
                &self,
                entropy_grid: &EntropyGrid,
            ) -> Option<(usize, usize, usize)>;
        }
    }

    mock! {
        pub Propagator {}
        impl ConstraintPropagator for Propagator {
            fn propagate(
                &mut self,
                grid: &mut PossibilityGrid,
                updated_coords: Vec<(usize, usize, usize)>,
                rules: &AdjacencyRules,
            ) -> Result<(), PropagationError>;
        }
    }

    // --- Test Setup Helpers (Basic versions) ---
    fn setup_grid() -> PossibilityGrid {
        PossibilityGrid::new(4, 4, 1, 2) // Example: 4x4 grid, 2 tiles
    }

    fn setup_tileset() -> TileSet {
        create_simple_tileset(2).unwrap()
    }

    fn setup_rules() -> AdjacencyRules {
        let ts = setup_tileset();
        create_uniform_rules(&ts)
    }

    // Helper to create a uniform AdjacencyRules
    fn create_uniform_rules(tileset: &TileSet) -> AdjacencyRules {
        let num_tiles = tileset.num_transformed_tiles();
        let num_axes = 6; // Assuming 3D
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

    // Helper to create a simple TileSet
    fn create_simple_tileset(num_base_tiles: usize) -> Result<TileSet, TileSetError> {
        let weights = vec![1.0; num_base_tiles];
        let allowed_transforms = vec![vec![Transformation::Identity]; num_base_tiles];
        TileSet::new(weights, allowed_transforms)
    }

    // Helper to create config with specific path/interval
    fn checkpoint_test_config(path: PathBuf, interval: u64) -> WfcConfig {
        WfcConfig {
            checkpoint_path: Some(path),
            checkpoint_interval: Some(interval),
            max_iterations: Some(10), // Add a limit for timeout test
            ..Default::default()
        }
    }

    // Helper to create config with initial checkpoint
    fn load_checkpoint_test_config(checkpoint: WfcCheckpoint) -> WfcConfig {
        WfcConfig {
            initial_checkpoint: Some(checkpoint),
            ..Default::default()
        }
    }

    // Helper to create config with progress callback
    fn progress_test_config(callback: ProgressCallback) -> WfcConfig {
        WfcConfig {
            progress_callback: Some(callback),
            max_iterations: Some(2), // Limit iterations for test
            ..Default::default()
        }
    }

    // Helper to create config with max iterations
    fn max_iter_test_config(max_iters: u64) -> WfcConfig {
        WfcConfig {
            max_iterations: Some(max_iters),
            ..Default::default()
        }
    }

    #[test]
    fn test_checkpoint_saving() {
        let grid_dim = 2;
        let tileset = create_simple_tileset(2).expect("Failed to create tileset");
        let num_transformed_tiles = tileset.num_transformed_tiles();
        let mut grid = PossibilityGrid::new(grid_dim, grid_dim, grid_dim, num_transformed_tiles);
        let rules = create_uniform_rules(&tileset);
        let mut propagator = MockPropagator::new();
        let mut entropy_calculator = MockEntropyCalculator::new();
        let _shutdown = Arc::new(AtomicBool::new(false));
        let _checkpoint_interval = Some(5u64);
        let dir = tempdir().unwrap();
        let checkpoint_path = dir.path().join("checkpoint.bin");
        let config = checkpoint_test_config(checkpoint_path.clone(), 5);

        // Mocks: Force timeout
        entropy_calculator
            .expect_calculate_entropy()
            .returning(move |_| Ok(EntropyGrid::new(grid_dim, grid_dim, grid_dim)));
        entropy_calculator
            .expect_select_lowest_entropy_cell()
            .returning(|_| Some((0, 0, 0))); // Always find a cell
        propagator
            .expect_propagate()
            // Expect propagate to be called (with some Vec, we don't care which here)
            .returning(|_, _, _| Ok(())); // Return Ok(()) to continue loop

        let result = run(
            &mut grid,
            &tileset,
            &rules,
            Box::new(propagator),
            Box::new(entropy_calculator),
            &config,
        );
        assert!(
            matches!(result, Err(WfcError::TimeoutOrInfiniteLoop)),
            "Expected Timeout error, got {:?}",
            result
        );
        assert!(checkpoint_path.exists(), "Checkpoint file was not created");
    }

    #[test]
    fn test_checkpoint_loading() {
        let grid_dim = 2;
        let tileset = create_simple_tileset(2).expect("Failed to create tileset");
        let num_transformed_tiles = tileset.num_transformed_tiles();
        let mut target_grid =
            PossibilityGrid::new(grid_dim, grid_dim, grid_dim, num_transformed_tiles);
        let rules = create_uniform_rules(&tileset);
        let mut propagator = MockPropagator::new();
        let mut entropy_calculator = MockEntropyCalculator::new();
        let _shutdown = Arc::new(AtomicBool::new(false));

        let initial_iterations = 10u64;
        let checkpoint_grid =
            PossibilityGrid::new(grid_dim, grid_dim, grid_dim, num_transformed_tiles);
        let checkpoint_data = WfcCheckpoint {
            grid: checkpoint_grid,
            iterations: initial_iterations,
        };
        let config = load_checkpoint_test_config(checkpoint_data);

        // Mocks: Complete after loading
        entropy_calculator
            .expect_calculate_entropy()
            .times(1)
            .returning(move |_| Ok(EntropyGrid::new(grid_dim, grid_dim, grid_dim)));
        entropy_calculator
            .expect_select_lowest_entropy_cell()
            .times(1)
            .returning(|_| None); // Return None to finish
        propagator
            .expect_propagate()
            .times(2) // Expect 2 calls (initial + final check)
            .returning(|_, _, _| Ok(()));

        let result = run(
            &mut target_grid,
            &tileset,
            &rules,
            Box::new(propagator),
            Box::new(entropy_calculator),
            &config,
        );
        assert!(
            matches!(result, Err(WfcError::IncompleteCollapse)),
            "Expected Err(IncompleteCollapse) after loading empty checkpoint and immediate mock stop, got {:?}",
            result
        );
    }

    #[test]
    fn test_checkpoint_loading_dimension_mismatch() {
        let grid_dim = 2;
        let target_grid_dim = 3;
        let tileset = create_simple_tileset(2).expect("Failed to create tileset");
        let num_transformed_tiles = tileset.num_transformed_tiles();

        let mut target_grid = PossibilityGrid::new(
            target_grid_dim,
            target_grid_dim,
            target_grid_dim,
            num_transformed_tiles,
        );
        let rules = create_uniform_rules(&tileset);
        let propagator = MockPropagator::new();
        let entropy_calculator = MockEntropyCalculator::new();
        let _shutdown = Arc::new(AtomicBool::new(false));

        let checkpoint_grid =
            PossibilityGrid::new(grid_dim, grid_dim, grid_dim, num_transformed_tiles);
        let checkpoint_data = WfcCheckpoint {
            grid: checkpoint_grid,
            iterations: 10,
        };
        let config = load_checkpoint_test_config(checkpoint_data);

        let result = run(
            &mut target_grid,
            &tileset,
            &rules,
            Box::new(propagator),
            Box::new(entropy_calculator),
            &config,
        );
        assert!(
            matches!(result, Err(WfcError::CheckpointError(_))),
            "Expected CheckpointError for dimension mismatch, got {:?}",
            result
        );
    }

    #[test]
    fn test_checkpoint_loading_tile_count_mismatch() {
        let grid_dim = 2;
        let tileset_chkp = create_simple_tileset(2).expect("Tileset Chkp Failed");
        let num_tiles_chkp = tileset_chkp.num_transformed_tiles();
        let tileset_target = create_simple_tileset(3).expect("Tileset Target Failed");
        let num_tiles_target = tileset_target.num_transformed_tiles();

        let mut target_grid = PossibilityGrid::new(grid_dim, grid_dim, grid_dim, num_tiles_target);
        let rules_target = create_uniform_rules(&tileset_target);
        let propagator = MockPropagator::new();
        let entropy_calculator = MockEntropyCalculator::new();
        let _shutdown = Arc::new(AtomicBool::new(false));

        let checkpoint_grid = PossibilityGrid::new(grid_dim, grid_dim, grid_dim, num_tiles_chkp);
        let checkpoint_data = WfcCheckpoint {
            grid: checkpoint_grid,
            iterations: 10,
        };
        let config = load_checkpoint_test_config(checkpoint_data);

        let result = run(
            &mut target_grid,
            &tileset_target,
            &rules_target,
            Box::new(propagator),
            Box::new(entropy_calculator),
            &config,
        );
        assert!(
            matches!(result, Err(WfcError::CheckpointError(_))),
            "Expected CheckpointError for tile count mismatch, got {:?}",
            result
        );
    }

    #[test]
    fn test_progress_callback_with_grid() {
        let grid_dim = 2;
        let tileset = create_simple_tileset(2).expect("Failed to create tileset");
        let num_transformed_tiles = tileset.num_transformed_tiles();
        let mut grid = PossibilityGrid::new(grid_dim, grid_dim, grid_dim, num_transformed_tiles);
        let rules = create_uniform_rules(&tileset);
        let mut propagator = MockPropagator::new();
        let mut entropy_calculator = MockEntropyCalculator::new();
        let _shutdown = Arc::new(AtomicBool::new(false));
        let progress_called = Arc::new(AtomicBool::new(false));
        let progress_called_clone = progress_called.clone();

        // Mocks
        entropy_calculator
            .expect_calculate_entropy()
            .times(2) // Expect 2 calls
            .returning(move |_| Ok(EntropyGrid::new(grid_dim, grid_dim, grid_dim)));

        let mut select_calls = 0;
        entropy_calculator
            .expect_select_lowest_entropy_cell()
            .times(2) // Expect 2 calls
            .returning(move |_| {
                select_calls += 1;
                if select_calls <= 1 {
                    Some((0, 0, 0)) // Find a cell on first call
                } else {
                    None // Finish on second call
                }
            });
        propagator
            .expect_propagate()
            .times(3) // Expect 3 calls (initial + iteration 1 + final check)
            .returning(|_, _, _| Ok(())); // Expect call, return Ok

        let progress_callback: ProgressCallback = Box::new(move |info: ProgressInfo| {
            println!(
                "Progress CB: Iter {}, Cells {}/{}",
                info.iterations, info.collapsed_cells, info.total_cells
            );
            progress_called_clone.store(true, Ordering::SeqCst);
            Ok(())
        });
        let config = progress_test_config(progress_callback);

        let _result = run(
            &mut grid,
            &tileset,
            &rules,
            Box::new(propagator),
            Box::new(entropy_calculator),
            &config,
        );
        assert!(matches!(_result, Err(WfcError::IncompleteCollapse)));
        assert!(progress_called.load(Ordering::SeqCst));
    }

    #[test]
    fn test_max_iterations_limit() {
        let grid_dim = 2;
        let tileset = create_simple_tileset(2).expect("Failed to create tileset");
        let num_transformed_tiles = tileset.num_transformed_tiles();
        let mut grid = PossibilityGrid::new(grid_dim, grid_dim, grid_dim, num_transformed_tiles);
        let rules = create_uniform_rules(&tileset);
        let mut propagator = MockPropagator::new();
        let mut entropy_calculator = MockEntropyCalculator::new();
        let _shutdown = Arc::new(AtomicBool::new(false));
        let config = max_iter_test_config(5);

        // Mocks: Always find cell, never finish
        entropy_calculator
            .expect_calculate_entropy()
            .returning(move |_| Ok(EntropyGrid::new(grid_dim, grid_dim, grid_dim)));
        entropy_calculator
            .expect_select_lowest_entropy_cell()
            .returning(|_| Some((0, 0, 0)));
        propagator.expect_propagate().returning(|_, _, _| Ok(()));

        let result = run(
            &mut grid,
            &tileset,
            &rules,
            Box::new(propagator),
            Box::new(entropy_calculator),
            &config,
        );
        assert!(
            matches!(result, Err(WfcError::TimeoutOrInfiniteLoop)),
            "Expected TimeoutOrInfiniteLoop error, got {:?}",
            result
        );
    }

    #[test]
    fn test_max_iterations_disabled() {
        let grid_dim = 2;
        let tileset = create_simple_tileset(2).expect("Failed to create tileset");
        let num_transformed_tiles = tileset.num_transformed_tiles();
        let mut grid = PossibilityGrid::new(grid_dim, grid_dim, grid_dim, num_transformed_tiles);
        let rules = create_uniform_rules(&tileset);
        let mut propagator = MockPropagator::new();
        let mut entropy_calculator = MockEntropyCalculator::new();
        let _shutdown = Arc::new(AtomicBool::new(false));
        let config = WfcConfig {
            max_iterations: None,
            ..Default::default()
        };

        // Mocks: Finish after one iteration
        entropy_calculator
            .expect_calculate_entropy()
            .times(1) // Expect 1 call
            .returning(move |_| Ok(EntropyGrid::new(grid_dim, grid_dim, grid_dim)));

        entropy_calculator
            .expect_select_lowest_entropy_cell()
            .times(1) // Expect 1 call
            .returning(move |_| {
                None // Return None immediately to finish
            });

        propagator
            .expect_propagate()
            .times(2) // Expect 2 calls (initial + final check)
            .returning(|_, _, _| Ok(()));

        let result = run(
            &mut grid,
            &tileset,
            &rules,
            Box::new(propagator),
            Box::new(entropy_calculator),
            &config,
        );
        assert!(
            matches!(result, Err(WfcError::IncompleteCollapse)),
            "Expected IncompleteCollapse result, got {:?}",
            result
        );
    }

    #[test]
    fn test_run_success() {
        let _shutdown = Arc::new(AtomicBool::new(false));
        let _checkpoint_interval = Some(5u64);
        let temp_dir = tempdir().unwrap();
        let _checkpoint_path = temp_dir.path().join("checkpoint.bin");

        // TODO: Add actual setup and assertions for success case
    }

    #[test]
    fn test_run_contradiction() {
        let _shutdown = Arc::new(AtomicBool::new(false));

        // TODO: Add actual setup and assertions for contradiction case
    }

    #[test]
    fn test_run_with_checkpoint_load_success() {
        let _shutdown = Arc::new(AtomicBool::new(false));
        let temp_dir = tempdir().unwrap();
        let _checkpoint_path = temp_dir.path().join("checkpoint_load.bin");
        let _checkpoint_interval = Some(1u64);

        let _first_run_shutdown = Arc::new(AtomicBool::new(false));

        let _second_run_shutdown = Arc::new(AtomicBool::new(false));

        // TODO: Add actual setup, execution, and assertions
    }

    #[test]
    fn test_run_max_iterations() {
        let _shutdown = Arc::new(AtomicBool::new(false));
        let _max_iterations = Some(3u64);

        // TODO: Add actual setup and assertions for max iterations case
    }

    #[test]
    fn test_run_with_progress_callback() {
        let _shutdown = Arc::new(AtomicBool::new(false));
        let _progress_counter = Arc::new(AtomicUsize::new(0));

        // TODO: Add setup, progress callback, execution, assertions
    }

    #[test]
    fn test_run_shutdown_signal() {
        let mut grid = setup_grid();
        let tileset = setup_tileset();
        let rules = setup_rules();
        let mut propagator = MockPropagator::new();
        let mut entropy_calculator = MockEntropyCalculator::new();

        let shutdown_signal = Arc::new(AtomicBool::new(false));
        let progress_counter = Arc::new(AtomicUsize::new(0));
        let shutdown_clone = shutdown_signal.clone();
        let progress_counter_clone_for_assert = progress_counter.clone(); // Clone for assertion

        // Mock expectations
        entropy_calculator
            .expect_calculate_entropy()
            .times(1..)
            .returning(move |_| Ok(EntropyGrid::new(grid.width, grid.height, grid.depth)));
        entropy_calculator
            .expect_select_lowest_entropy_cell()
            .times(1..)
            .returning(|_| Some((0, 0, 0)));

        propagator
            .expect_propagate()
            .times(1..)
            .returning(|_, _, _| Ok(()));

        let progress_callback: ProgressCallback = Box::new(move |info: ProgressInfo| {
            let count = progress_counter.fetch_add(1, Ordering::SeqCst);
            println!(
                "Progress: Iteration {}, Collapsed {}/{}, Time {:?}",
                info.iterations, info.collapsed_cells, info.total_cells, info.elapsed_time
            );
            if count >= 2 {
                shutdown_clone.store(true, Ordering::SeqCst);
                println!("Signaling shutdown...");
            }
            Ok(())
        });

        let config = WfcConfig {
            boundary_mode: BoundaryMode::Clamped,
            progress_callback: Some(progress_callback),
            shutdown_signal: shutdown_signal.clone(),
            ..Default::default()
        };

        let handle = std::thread::spawn(move || {
            let result = run(
                &mut grid,
                &tileset,
                &rules,
                Box::new(propagator),
                Box::new(entropy_calculator),
                &config,
            );
            println!("Thread finished with result: {:?}", result);
            result
        });

        let result = handle.join().expect("Thread panicked");

        assert!(
            matches!(result, Err(WfcError::Interrupted)),
            "Expected Interrupted error, got {:?}",
            result
        );
        assert!(
            progress_counter_clone_for_assert.load(Ordering::SeqCst) > 1,
            "Progress callback should have run at least twice"
        );
    }

    #[test]
    fn test_run_invalid_checkpoint_path() {
        let mut grid = setup_grid();
        let tileset = setup_tileset();
        let rules = setup_rules();
        let propagator = MockPropagator::new();
        let entropy_calculator = MockEntropyCalculator::new();
        let shutdown = Arc::new(AtomicBool::new(false));
        let invalid_path = PathBuf::from("/non_existent_dir/checkpoint.bin");

        let config = WfcConfig {
            checkpoint_path: Some(invalid_path),
            checkpoint_interval: Some(1),
            shutdown_signal: shutdown.clone(),
            ..Default::default()
        };

        let result = run(
            &mut grid,
            &tileset,
            &rules,
            Box::new(propagator),
            Box::new(entropy_calculator),
            &config,
        );

        assert!(matches!(result, Err(WfcError::CheckpointError(_))));
    }
}
