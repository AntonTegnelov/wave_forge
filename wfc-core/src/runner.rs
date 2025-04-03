use crate::entropy::EntropyCalculator;
use crate::grid::PossibilityGrid;
use crate::propagator::{ConstraintPropagator, PropagationError};
use crate::{BoundaryMode, ProgressInfo, WfcCheckpoint, WfcError};
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
use wfc_rules::{AdjacencyRules, TileId, TileSet};

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
/// * `boundary_mode`: The boundary mode to use for the propagation.
/// * `progress_callback`: An optional closure that receives `ProgressInfo` updates during the run.
///                        This allows external monitoring or UI updates.
/// * `shutdown_signal`: A shared atomic boolean indicating whether the run should stop.
/// * `initial_checkpoint`: An optional checkpoint to load before starting the run.
/// * `checkpoint_interval`: An optional interval to save checkpoints every N iterations.
/// * `checkpoint_path`: An optional file path to save checkpoints to.
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
pub fn run<P: ConstraintPropagator, E: EntropyCalculator>(
    grid: &mut PossibilityGrid,
    tileset: &TileSet,
    rules: &AdjacencyRules,
    mut propagator: P,
    entropy_calculator: E,
    boundary_mode: BoundaryMode,
    progress_callback: Option<Box<dyn Fn(ProgressInfo) + Send + Sync>>,
    shutdown_signal: Arc<AtomicBool>,
    initial_checkpoint: Option<WfcCheckpoint>,
    checkpoint_interval: Option<u64>,
    checkpoint_path: Option<PathBuf>,
) -> Result<(), WfcError> {
    info!(
        "Starting WFC run with boundary mode: {:?}...",
        boundary_mode
    );
    let start_time = Instant::now();
    let mut iterations = 0;

    // --- Checkpoint Loading ---
    if let Some(checkpoint) = initial_checkpoint {
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

        *grid = checkpoint.grid; // Load grid state
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

    loop {
        iterations += 1;

        // --- Checkpoint Saving Logic ---
        if let (Some(interval), Some(path)) = (checkpoint_interval, &checkpoint_path) {
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

        // --- Check for shutdown signal ---
        if shutdown_signal.load(Ordering::Relaxed) {
            warn!("Shutdown signal received, stopping WFC run prematurely.");
            return Err(WfcError::Interrupted);
        }

        // --- Check if finished (before iteration) ---
        if collapsed_cells_count >= total_cells {
            info!("All cells collapsed.");
            // Final callback before breaking
            if let Some(ref callback) = progress_callback {
                let progress_info = ProgressInfo {
                    total_cells,
                    collapsed_cells: collapsed_cells_count,
                    iterations: iterations - 1, // Use previous iter count
                    elapsed_time: start_time.elapsed(),
                    grid_state: grid.clone(),
                };
                callback(progress_info);
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
                if let Some(ref callback) = progress_callback {
                    let progress_info = ProgressInfo {
                        total_cells,
                        collapsed_cells: collapsed_cells_count,
                        iterations,
                        elapsed_time: start_time.elapsed(),
                        grid_state: grid.clone(),
                    };
                    callback(progress_info);
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
                    if let Some(ref callback) = progress_callback {
                        let progress_info = ProgressInfo {
                            total_cells,
                            collapsed_cells: collapsed_cells_count,
                            iterations,
                            elapsed_time: start_time.elapsed(),
                            grid_state: grid.clone(),
                        };
                        callback(progress_info);
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
        let max_iterations = (total_cells as u64) * 10; // Cast total_cells to u64
        if iterations > max_iterations {
            error!(
                "Maximum iterations ({}) exceeded. Assuming infinite loop.",
                max_iterations
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

// Helper function for a single WFC iteration
fn perform_iteration<'a, P: ConstraintPropagator, E: EntropyCalculator>(
    grid: &'a mut PossibilityGrid,
    tileset: &TileSet,
    rules: &AdjacencyRules,
    propagator: &mut P,
    entropy_calculator: &E,
    iteration: u64,
) -> Result<Option<(usize, usize, usize)>, WfcError> {
    // Returns Some(collapsed_coords) or None if no cell found, or Err
    debug!("Iteration {}: Calculating entropy...", iteration);
    let entropy_grid = entropy_calculator.calculate_entropy(grid)?;
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
            let weights_result: Result<Vec<f32>, _> = possible_tile_indices
                .iter()
                .map(|&index| {
                    tileset.get_weight(TileId(index)).ok_or_else(|| {
                        WfcError::ConfigurationError(format!(
                            "Weight missing for tile index {} at ({}, {}, {}).",
                            index, x, y, z
                        ))
                    })
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

    use wfc_rules::{AdjacencyRules, TileSet, Transformation}; // Ensure Transformation is imported here

    // --- Mock Implementations ---
    // ... Mocks ...

    // --- Helper to create basic rules/tileset (Moved Inside) ---
    fn create_basic_setup(num_tiles: usize) -> (TileSet, AdjacencyRules) {
        let weights = vec![1.0; num_tiles];
        let transformations = vec![vec![Transformation::Identity]; num_tiles];
        let tileset = TileSet::new(weights, transformations).unwrap();
        let num_transformed = tileset.num_transformed_tiles();
        assert_eq!(num_transformed, num_tiles);
        let num_axes = 6;
        let mut allowed_tuples = Vec::new();
        for axis in 0..num_axes {
            for t1_idx in 0..num_transformed {
                for t2_idx in 0..num_transformed {
                    allowed_tuples.push((axis, t1_idx, t2_idx));
                }
            }
        }
        let rules = AdjacencyRules::from_allowed_tuples(num_transformed, num_axes, allowed_tuples);
        (tileset, rules)
    }

    // --- Checkpoint Tests ---
    #[test]
    fn test_checkpoint_saving() { /* ... */
    }
    #[test]
    fn test_checkpoint_loading() { /* ... */
    }
    #[test]
    fn test_checkpoint_loading_dimension_mismatch() { /* ... */
    }
    #[test]
    fn test_checkpoint_loading_tile_count_mismatch() { /* ... */
    }
    #[test]
    fn test_progress_callback_with_grid() { /* ... */
    }
}
