use crate::entropy::EntropyCalculator;
use crate::grid::PossibilityGrid;
use crate::propagator::{ConstraintPropagator, PropagationError};
use crate::{ProgressInfo, WfcCheckpoint, WfcError};
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
    progress_callback: Option<Box<dyn Fn(ProgressInfo) + Send + Sync>>,
    shutdown_signal: Arc<AtomicBool>,
    initial_checkpoint: Option<WfcCheckpoint>,
    checkpoint_interval: Option<u64>,
    checkpoint_path: Option<PathBuf>,
) -> Result<(), WfcError> {
    info!("Starting WFC run...");
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
        let current_grid = &*grid; // Reborrow immutably
                                   // Use local dimensions in recalculation
        collapsed_cells_count = (0..depth)
            .flat_map(|z| {
                (0..height).flat_map(move |y| {
                    (0..width).map(move |x| {
                        current_grid
                            .get(x, y, z)
                            .map_or(0, |c| if c.count_ones() == 1 { 1 } else { 0 })
                    })
                })
            })
            .sum();
    }
    debug!(
        "After initial setup/load/propagation: {}/{} cells collapsed.",
        collapsed_cells_count, total_cells
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

        // --- Check if finished ---
        if collapsed_cells_count >= total_cells {
            info!("All cells collapsed.");
            break;
        }

        // --- 1. Find Lowest Entropy Cell ---
        debug!("Iteration {}: Calculating entropy...", iterations);
        let entropy_grid = entropy_calculator
            .calculate_entropy(grid)
            .map_err(|e| WfcError::InternalError(format!("Entropy calculation failed: {}", e)))?;
        let lowest_entropy_coords = entropy_calculator.select_lowest_entropy_cell(&entropy_grid);

        if let Some((x, y, z)) = lowest_entropy_coords {
            debug!("Found lowest entropy cell at ({}, {}, {})", x, y, z);

            // --- 2. Collapse the Cell ---
            let cell_to_collapse = grid.get_mut(x, y, z).ok_or_else(|| {
                error!(
                    "Internal Error: Lowest entropy coords ({},{},{}) are out of bounds!",
                    x, y, z
                );
                WfcError::GridError("Lowest entropy cell coordinates out of bounds".to_string())
            })?;
            assert_eq!(
                cell_to_collapse.len(),
                num_tiles,
                "Grid cell bitvec length mismatch"
            );

            let possible_tile_indices: Vec<usize> = cell_to_collapse.iter_ones().collect();

            if possible_tile_indices.is_empty() {
                error!("Contradiction detected at ({}, {}, {}): No possible tiles left (collapse phase).", x, y, z);
                return Err(WfcError::Contradiction(x, y, z));
            }

            if possible_tile_indices.len() > 1 {
                // Choose a tile to collapse to using weights from the TileSet.
                let weights_result: Result<Vec<f32>, _> = possible_tile_indices
                    .iter()
                    .map(|&index| {
                        tileset.get_weight(TileId(index)).ok_or_else(|| {
                            error!(
                                "Internal Error: TileId({}) out of bounds for TileSet weights.",
                                index
                            );
                            WfcError::ConfigurationError(format!(
                                "Weight missing for possible tile index {} at ({}, {}, {}).",
                                index, x, y, z
                            ))
                        })
                    })
                    .collect();

                let weights = weights_result?; // Propagate potential error

                if weights.is_empty() || weights.iter().all(|&w| w <= 0.0) {
                    error!("Internal Error: No valid positive weights for possible tiles {:?} at ({}, {}, {}).", possible_tile_indices, x, y, z);
                    return Err(WfcError::ConfigurationError(
                        "No valid positive weights for collapse choice".to_string(),
                    ));
                }

                // Use WeightedIndex distribution for weighted random choice
                let dist = WeightedIndex::new(&weights).map_err(|e| {
                    error!(
                        "Failed to create WeightedIndex distribution at ({}, {}, {}): {}",
                        x, y, z, e
                    );
                    WfcError::InternalError(format!("WeightedIndex creation failed: {}", e))
                })?;

                let mut rng = thread_rng();
                let chosen_weighted_index = dist.sample(&mut rng);
                let chosen_tile_index = possible_tile_indices[chosen_weighted_index];

                debug!(
                    "Iter {}: Collapsing cell ({}, {}, {}) to tile index {} (weight {})",
                    iterations, x, y, z, chosen_tile_index, weights[chosen_weighted_index]
                );

                // Update the grid cell state directly
                cell_to_collapse.fill(false);
                cell_to_collapse.set(chosen_tile_index, true);
                collapsed_cells_count += 1;

                // --- 3. Propagate Constraints ---
                debug!(
                    "Iter {}: Propagating constraints from ({}, {}, {})...",
                    iterations, x, y, z
                );
                if let Err(prop_err) = propagator.propagate(grid, vec![(x, y, z)], rules) {
                    error!(
                        "Iter {}: Propagation failed after collapsing ({}, {}, {}): {:?}",
                        iterations, x, y, z, prop_err
                    );
                    if let PropagationError::Contradiction(cx, cy, cz) = prop_err {
                        return Err(WfcError::Contradiction(cx, cy, cz));
                    }
                    return Err(WfcError::from(prop_err));
                }
                debug!("Iter {}: Propagation successful.", iterations);

                // --- Progress Callback ---
                if let Some(ref callback) = progress_callback {
                    let progress_info = ProgressInfo {
                        total_cells,
                        collapsed_cells: collapsed_cells_count,
                        iterations,
                        elapsed_time: start_time.elapsed(),
                    };
                    callback(progress_info);
                }
            } else if possible_tile_indices.len() == 1 {
                debug!(
                    "Iter {}: Cell ({}, {}, {}) was already collapsed. Skipping collapse.",
                    iterations, x, y, z
                );
            } else {
                error!(
                    "Iter {}: Contradiction detected at ({}, {}, {}) during cell processing.",
                    iterations, x, y, z
                );
                return Err(WfcError::Contradiction(x, y, z));
            }
        } else {
            if collapsed_cells_count >= total_cells {
                info!(
                    "Iter {}: All cells confirmed collapsed. Finalizing.",
                    iterations
                );
                break;
            } else {
                warn!(
                    "Iter {}: No lowest entropy cell found, but {} cells remain uncollapsed. Checking state...",
                    iterations, total_cells - collapsed_cells_count
                );
                // Re-run propagation on *all* cells to ensure consistency before erroring
                debug!(
                    "Iter {}: Re-running propagation on all cells before potential error.",
                    iterations
                );
                if let Err(prop_err) = propagator.propagate(grid, all_coords.clone(), rules) {
                    error!(
                        "Iter {}: Propagation failed during final check: {:?}",
                        iterations, prop_err
                    );
                    if let PropagationError::Contradiction(cx, cy, cz) = prop_err {
                        return Err(WfcError::Contradiction(cx, cy, cz));
                    }
                    return Err(WfcError::from(prop_err));
                }
                // Re-check collapsed count after final propagation
                let current_grid = &*grid; // Borrow immutably before the closure
                let final_collapsed_count: usize = (0..depth)
                    .flat_map(|z| {
                        (0..height).flat_map(move |y| {
                            (0..width).map(move |x| {
                                current_grid
                                    .get(x, y, z) // Use the immutable reference
                                    .map_or(0, |c| if c.count_ones() == 1 { 1 } else { 0 })
                            })
                        })
                    })
                    .sum();

                if final_collapsed_count >= total_cells {
                    info!(
                        "Iter {}: All cells confirmed collapsed after final propagation check. Finalizing.",
                        iterations
                    );
                    break;
                } else {
                    error!(
                        "Iter {}: No lowest entropy cell found, and {} cells remain uncollapsed after final check.",
                        iterations,
                        total_cells - final_collapsed_count
                    );
                    return Err(WfcError::IncompleteCollapse);
                }
            }
        }

        debug!(
            "End of iteration {}. Collapsed cells: {}/{}",
            iterations, collapsed_cells_count, total_cells
        );

        // Safeguard against infinite loops (optional, adjust limit as needed)
        let max_iterations = (total_cells as u64) * 10; // Cast total_cells to u64
        if iterations > max_iterations {
            error!(
                "Maximum iterations ({}) exceeded. Assuming infinite loop.",
                max_iterations
            );
            return Err(WfcError::TimeoutOrInfiniteLoop);
        }
    }

    // Final check after loop exits
    if collapsed_cells_count < total_cells {
        error!(
            "WFC finished, but {} cells remain uncollapsed.",
            total_cells - collapsed_cells_count
        );
        return Err(WfcError::IncompleteCollapse);
    }

    info!(
        "WFC run finished in {:?} after {} iterations.",
        start_time.elapsed(),
        iterations
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    // ... tests ...
}
