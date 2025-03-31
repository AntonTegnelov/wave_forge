use crate::{
    entropy::EntropyCalculator,
    grid::PossibilityGrid,
    propagator::ConstraintPropagator,
    rules::AdjacencyRules,
    tile::{TileId, TileSet},
    ProgressInfo, WfcError,
};
use log::{debug, error, info};
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;

/// Runs the Wave Function Collapse algorithm.
///
/// # Arguments
///
/// * `grid` - The mutable grid of possibilities.
/// * `tileset` - Information about the tiles (weights, etc.).
/// * `rules` - Adjacency constraints between tiles.
/// * `propagator` - The constraint propagation implementation.
/// * `entropy_calculator` - The entropy calculation implementation.
/// * `progress_callback` - An optional callback for reporting progress.
///
/// # Returns
///
/// `Ok(())` on successful collapse, `Err(WfcError)` otherwise.
pub fn run<P: ConstraintPropagator, E: EntropyCalculator>(
    grid: &mut PossibilityGrid,
    tileset: &TileSet,
    rules: &AdjacencyRules,
    mut propagator: P,
    entropy_calculator: E,
    progress_callback: Option<Box<dyn Fn(ProgressInfo) + Send + Sync>>,
) -> Result<(), WfcError> {
    info!("Starting WFC run...");
    let mut iterations = 0;
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
                        return Err(WfcError::Contradiction);
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
    propagator.propagate(grid, all_coords, rules).map_err(|e| {
        error!("Initial propagation failed: {:?}", e);
        WfcError::from(e)
    })?;
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
        "After initial propagation: {}/{} cells collapsed.",
        collapsed_cells_count, total_cells
    );

    loop {
        // --- Check if finished ---
        if collapsed_cells_count >= total_cells {
            info!("All cells collapsed.");
            break;
        }

        // --- 1. Find Lowest Entropy Cell ---
        debug!("Iteration {}: Calculating entropy...", iterations);
        let entropy_grid = entropy_calculator.calculate_entropy(grid);
        let lowest_entropy_coords = entropy_calculator.find_lowest_entropy(&entropy_grid);

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
                return Err(WfcError::Contradiction);
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
                    "Collapsing cell ({}, {}, {}) to tile index {} (weight {})",
                    x, y, z, chosen_tile_index, weights[chosen_weighted_index]
                );

                // Update the grid cell state directly
                cell_to_collapse.fill(false);
                cell_to_collapse.set(chosen_tile_index, true);

                // --- 3. Propagate Constraints ---
                debug!("Propagating constraints from ({}, {}, {})...", x, y, z);
                propagator
                    .propagate(grid, vec![(x, y, z)], rules)
                    .map_err(|e| {
                        error!(
                            "Propagation failed after collapsing ({}, {}, {}): {:?}",
                            x, y, z, e
                        );
                        WfcError::from(e)
                    })?;
                debug!("Propagation successful.");
                // Recalculate collapsed count AFTER propagation
                {
                    let current_grid = &*grid; // Reborrow immutably
                                               // Use local dimensions in recalculation
                    collapsed_cells_count = (0..depth)
                        .flat_map(|z| {
                            (0..height).flat_map(move |y| {
                                (0..width).map(move |x| {
                                    current_grid.get(x, y, z).map_or(0, |c| {
                                        if c.count_ones() == 1 {
                                            1
                                        } else {
                                            0
                                        }
                                    })
                                })
                            })
                        })
                        .sum();
                } // Immutable borrow ends
            } else {
                debug!(
                    "Cell ({}, {}, {}) was already collapsed, skipping collapse/propagation.",
                    x, y, z
                );
            }
        } else {
            // No cell with positive entropy found, but not all cells are collapsed.
            // This indicates an issue, potentially a state where propagation finished
            // but couldn't fully collapse the grid, or an error in entropy calculation/finding.
            if collapsed_cells_count < total_cells {
                error!("WFC finished prematurely: No positive entropy cells found, but {} cells remain uncollapsed.", total_cells - collapsed_cells_count);
                // Depending on desired behavior, could return Ok or a specific error.
                return Err(WfcError::IncompleteCollapse);
            }
            // This case should be caught by the check at the start of the loop now.
            info!("No cells with positive entropy found and all cells are collapsed.");
            break;
        }

        // --- 4. Progress Reporting (Optional) ---
        if let Some(ref callback) = progress_callback {
            let progress = ProgressInfo {
                iteration: iterations,
                collapsed_cells: collapsed_cells_count,
                total_cells, // Use local total_cells
                contradictions: None,
            };
            callback(progress);
            debug!(
                "Progress: Iteration {}, Collapsed {}/{}",
                iterations, collapsed_cells_count, total_cells
            );
        }

        iterations += 1;
        // Safeguard against infinite loops (optional, adjust limit as needed)
        if iterations > total_cells * 10 {
            // Use local total_cells
            error!(
                "WFC exceeded maximum iterations ({}), assuming infinite loop.",
                total_cells * 10
            );
            return Err(WfcError::TimeoutOrInfiniteLoop);
        }
    }

    info!(
        "WFC run finished successfully after {} iterations.",
        iterations
    );
    Ok(())
}
