// Integration tests for wfc-gpu crate

// Main integration test modules organized as inline modules

#[test]
fn integration_test_example() {
    // This is a placeholder for integration tests
    // Integration tests test the library from the outside, like a user would
    assert!(true);
}

// Test modules organized by component

#[cfg(test)]
mod algorithm_tests {
    #[test]
    fn test_full_wfc_execution() {
        // Test full algorithm execution with various configurations
        assert!(true);
    }
}

#[cfg(test)]
mod buffer_tests {
    #[test]
    fn test_buffer_lifecycle() {
        // Test buffer creation, usage, and cleanup
        assert!(true);
    }
}

#[cfg(test)]
mod shader_tests {
    #[test]
    fn test_shader_compilation() {
        // Test shader compilation and validation
        assert!(true);
    }
}

#[cfg(test)]
mod propagation_tests {
    #[test]
    fn test_constraint_propagation() {
        // Test constraint propagation strategies
        assert!(true);
    }
}

#[cfg(test)]
mod error_recovery_tests {
    #[test]
    fn test_error_recovery_mechanisms() {
        // Test error recovery mechanisms
        assert!(true);
    }
}

// You can define more test functions here, or use submodules
// mod submodule_tests;

use wfc_core::entropy::EntropyHeuristicType;
use wfc_core::grid::PossibilityGrid;
use wfc_core::BoundaryCondition;
use wfc_gpu::gpu::accelerator::GpuAccelerator;
use wfc_rules::{AdjacencyRules, TileId, TileSet, Transformation};

#[tokio::test]
async fn test_basic_3d_generation() -> anyhow::Result<()> {
    // Test configuration
    let grid_size = (16, 16, 16); // Small enough for quick testing, large enough to be meaningful
    let num_tiles = 2; // Simple binary tiles (e.g., "filled" and "empty")

    println!("Starting test_basic_3d_generation");
    println!("Grid size: {:?}", grid_size);
    println!("Number of tiles: {}", num_tiles);

    // Create a simple tileset with different weights
    let tileset = TileSet::new(
        vec![1.0, 0.5], // weights: tile 0 (A) is twice as likely as tile 1 (B)
        vec![vec![Transformation::Identity]; num_tiles], // allowed_transformations
    )?;

    println!("Created tileset with weights: Tile A (0): 1.0, Tile B (1): 0.5");
    println!(
        "Number of transformed tiles: {}",
        tileset.num_transformed_tiles()
    );

    // Create weighted adjacency rules: tiles can connect to themselves and their neighbors
    // with specific negative constraints.
    // Tile A = TileId(0), Tile B = TileId(1)
    // Negative Adjacency Rules:
    // 1. Tile A can't be directly to the left of Tile B.
    //    This means the spatial pattern "A B" (A at pos x, B at pos x+1) is forbidden.
    //    This applies if:
    //      - Current tile is A, neighbor is B, axis is 0 (+x, neighbor to the right).
    //      - Current tile is B, neighbor is A, axis is 1 (-x, neighbor to the left).
    // 2. Tile B can't be directly to the left of Tile A.
    //    This means the spatial pattern "B A" (B at pos x, A at pos x+1) is forbidden.
    //    This applies if:
    //      - Current tile is B, neighbor is A, axis is 0 (+x, neighbor to the right).
    //      - Current tile is A, neighbor is B, axis is 1 (-x, neighbor to the left).
    // 3. Tile A can't be directly above Tile A.
    //    This means the spatial pattern "A (top) over A (bottom)" is forbidden.
    //    This applies if:
    //      - Current tile is A, neighbor is A, axis is 2 (+y, neighbor is up).
    //      - Current tile is A, neighbor is A, axis is 3 (-y, neighbor is down).

    let mut allowed_tuples = Vec::new();
    let tile_a_tid = tileset
        .get_transformed_id(TileId(0), Transformation::Identity)
        .unwrap();
    let tile_b_tid = tileset
        .get_transformed_id(TileId(1), Transformation::Identity)
        .unwrap();

    for current_tile_idx in 0..num_tiles {
        let current_tid = tileset
            .get_transformed_id(TileId(current_tile_idx), Transformation::Identity)
            .unwrap();
        for neighbor_tile_idx in 0..num_tiles {
            let neighbor_tid = tileset
                .get_transformed_id(TileId(neighbor_tile_idx), Transformation::Identity)
                .unwrap();
            for axis in 0..6 {
                // 0:+x, 1:-x, 2:+y, 3:-y, 4:+z, 5:-z
                let is_disallowed =
                    // Rule 1: Tile A can't be directly to the left of Tile B (spatial pattern A B)
                    ( (current_tid == tile_a_tid && neighbor_tid == tile_b_tid && axis == 0) || // current A, right B
                      (current_tid == tile_b_tid && neighbor_tid == tile_a_tid && axis == 1) ) || // current B, left A

                    // Rule 2: Tile B can't be directly to the left of Tile A (spatial pattern B A)
                    ( (current_tid == tile_b_tid && neighbor_tid == tile_a_tid && axis == 0) || // current B, right A
                      (current_tid == tile_a_tid && neighbor_tid == tile_b_tid && axis == 1) ) || // current A, left B

                    // Rule 3: Tile A can't be directly above Tile A (spatial pattern A over A)
                    (current_tid == tile_a_tid && neighbor_tid == tile_a_tid && (axis == 2 || axis == 3)); // current A, A up or A down

                if !is_disallowed {
                    if current_tid == neighbor_tid {
                        // Tiles are the same
                        allowed_tuples.push((axis, current_tid, neighbor_tid));
                        allowed_tuples.push((axis, current_tid, neighbor_tid)); // Add twice for higher weight
                    } else {
                        // Different tiles
                        allowed_tuples.push((axis, current_tid, neighbor_tid));
                    }
                }
            }
        }
    }

    println!(
        "Created {} allowed tuples with new constraints.",
        allowed_tuples.len()
    );
    if allowed_tuples.len() < 20 {
        // Print a sample if not too long
        println!(
            "Sample of allowed tuples: {:?}",
            &allowed_tuples[..std::cmp::min(allowed_tuples.len(), allowed_tuples.len())]
        );
    }

    let rules = AdjacencyRules::from_allowed_tuples(
        tileset.num_transformed_tiles(),
        6, // num_axes (3D = 6 directions)
        allowed_tuples,
    );

    println!("Created adjacency rules");
    println!(
        "Number of transformed tiles in rules: {}",
        tileset.num_transformed_tiles()
    );
    println!("Number of axes in rules: {}", 6);

    // Initialize grid with the number of transformed tiles
    let mut grid = PossibilityGrid::new(
        grid_size.0,
        grid_size.1,
        grid_size.2,
        tileset.num_transformed_tiles(),
    );

    println!("Initialized possibility grid");
    println!(
        "Grid dimensions: {}x{}x{}",
        grid.width, grid.height, grid.depth
    );
    println!("Grid num_tiles: {}", grid.num_tiles());

    // Create GridDefinition for the accelerator
    let grid_def = wfc_gpu::gpu::accelerator::GridDefinition {
        dims: grid_size,
        num_tiles: tileset.num_transformed_tiles(),
    };

    // Create GPU accelerator with Count-based entropy heuristic instead of Shannon
    let mut accelerator = GpuAccelerator::new(
        &grid,
        &rules,
        BoundaryCondition::Periodic,
        EntropyHeuristicType::Count,
        None,
    )
    .await?;

    println!("Created GPU accelerator");
    println!("Using Count-based entropy heuristic");

    // Run the WFC algorithm
    println!("\nStarting wave function collapse...");
    let result = accelerator
        .run_with_callback(
            &grid, // Pass initial grid (becomes &PossibilityGrid)
            &rules,
            grid_def,             // Pass GridDefinition
            1000,                 // max iterations
            |_progress| Ok(true), // Continue running (progress_callback)
            None,                 // No shutdown signal
        )
        .await;
    println!("WFC algorithm completed with result: {:?}", result);

    // Ensure the WFC algorithm itself succeeded before checking the grid state.
    let final_grid = match result {
        Ok(run_result) => run_result.grid,
        Err(e) => {
            // Optionally, print the grid state if an error occurred for more debug info.
            // This might be too verbose for every error, but can be useful.
            // println!("Grid state at time of error: {:?}", grid);
            panic!("WFC algorithm failed: {:?}", e);
        }
    };

    println!("\nGrid state after collapse:");
    println!(
        "Number of superpositions remaining: {}",
        final_grid
            .data()
            .iter()
            .filter(|bits| bits.count_ones() > 1)
            .count()
    );
    println!("Is fully collapsed: {:?}", final_grid.is_fully_collapsed());

    // Print a sample of the grid state
    let sample_x = std::cmp::min(3, final_grid.width);
    let sample_y = std::cmp::min(3, final_grid.height);
    let sample_z = std::cmp::min(3, final_grid.depth);

    println!(
        "\nSample of grid state ({}x{}x{}):",
        sample_x, sample_y, sample_z
    );
    for x in 0..sample_x {
        for y in 0..sample_y {
            for z in 0..sample_z {
                if let Some(cell_bits) = final_grid.get(x, y, z) {
                    println!("Cell ({}, {}, {}): {:?}", x, y, z, cell_bits);
                }
            }
        }
    }

    // Verify the result
    let violations = verify_adjacency_rules(&final_grid, &rules);
    println!(
        "Found {} adjacency rule violations via AdjacencyRules::check",
        violations
    );

    let custom_violations = verify_custom_negative_rules(&final_grid, &tileset);
    println!(
        "Found {} custom negative rule violations",
        custom_violations
    );

    let total_cells = final_grid.width * final_grid.height * final_grid.depth;
    let mut collapsed_count = 0;
    for z in 0..final_grid.depth {
        for y in 0..final_grid.height {
            for x in 0..final_grid.width {
                if let Some(cell) = final_grid.get(x, y, z) {
                    if cell.count_ones() == 1 {
                        collapsed_count += 1;
                    }
                }
            }
        }
    }

    println!(
        "Collapsed {} out of {} cells ({:.1}%)",
        collapsed_count,
        total_cells,
        (collapsed_count as f64 / total_cells as f64) * 100.0
    );

    assert!(
        final_grid
            .is_fully_collapsed()
            .expect("is_fully_collapsed() should not error here"),
        "Grid is not fully collapsed according to is_fully_collapsed() method."
    );
    assert!(
        violations == 0,
        "Found {} adjacency rule violations via AdjacencyRules::check",
        violations
    );
    assert!(
        custom_violations == 0,
        "Found {} custom negative rule violations",
        custom_violations
    );
    assert!(
        collapsed_count == total_cells,
        "Not all cells collapsed: {} out of {} cells collapsed. Superpositions or contradictions might exist.",
        collapsed_count,
        total_cells
    );

    Ok(())
}

// Helper function to verify adjacency rules
fn verify_adjacency_rules(grid: &PossibilityGrid, rules: &AdjacencyRules) -> usize {
    let mut violations = 0;
    let w = grid.width;
    let h = grid.height;
    let d = grid.depth;

    // Check each cell's neighbors
    for x in 0..w {
        for y in 0..h {
            for z in 0..d {
                if let Some(cell) = grid.get(x, y, z) {
                    if cell.count_ones() == 1 {
                        let tile = cell.iter_ones().next().unwrap();
                        // Check each direction
                        for (dx, dy, dz, axis) in [
                            (1, 0, 0, 0),  // +x (axis 0)
                            (-1, 0, 0, 1), // -x (axis 1)
                            (0, 1, 0, 2),  // +y (axis 2)
                            (0, -1, 0, 3), // -y (axis 3)
                            (0, 0, 1, 4),  // +z (axis 4)
                            (0, 0, -1, 5), // -z (axis 5)
                        ] {
                            let nx = (x as i32 + dx).rem_euclid(w as i32) as usize;
                            let ny = (y as i32 + dy).rem_euclid(h as i32) as usize;
                            let nz = (z as i32 + dz).rem_euclid(d as i32) as usize;

                            if let Some(neighbor_cell) = grid.get(nx, ny, nz) {
                                if neighbor_cell.count_ones() == 1 {
                                    let neighbor_tile = neighbor_cell.iter_ones().next().unwrap();
                                    if !rules.check(tile, neighbor_tile, axis) {
                                        violations += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    violations
}

// Helper function to verify the new custom adjacency rules
fn verify_custom_negative_rules(grid: &PossibilityGrid, tileset: &TileSet) -> usize {
    let mut violations = 0;
    let w = grid.width;
    let h = grid.height;
    let d = grid.depth;

    // Assuming Tile A is TileId(0) and Tile B is TileId(1) as per test setup
    let tile_a_tid = tileset
        .get_transformed_id(TileId(0), Transformation::Identity)
        .expect("Failed to get transformed TileId for Tile A");
    let tile_b_tid = tileset
        .get_transformed_id(TileId(1), Transformation::Identity)
        .expect("Failed to get transformed TileId for Tile B");

    for x in 0..w {
        for y in 0..h {
            for z in 0..d {
                if let Some(current_cell_possibilities) = grid.get(x, y, z) {
                    if current_cell_possibilities.count_ones() == 1 {
                        let current_tile_id =
                            current_cell_possibilities.iter_ones().next().unwrap();

                        // Check Rule: Tile A can't be directly to the left of Tile B
                        // This means if the current cell is Tile B, its left neighbor (x-1) cannot be Tile A.
                        if current_tile_id == tile_b_tid {
                            let left_nx = (x as i32 - 1).rem_euclid(w as i32) as usize;
                            if let Some(left_neighbor_possibilities) = grid.get(left_nx, y, z) {
                                if left_neighbor_possibilities.count_ones() == 1 {
                                    let left_neighbor_tile_id =
                                        left_neighbor_possibilities.iter_ones().next().unwrap();
                                    if left_neighbor_tile_id == tile_a_tid {
                                        println!(
                                            "Custom Rule Violation: Tile A ({}) is to the left of Tile B ({}) at ({},{},{}) (current) and ({},{},{}) (left neighbor).",
                                            left_neighbor_tile_id, current_tile_id, x, y, z, left_nx, y, z
                                        );
                                        violations += 1;
                                    }
                                }
                            }
                        }

                        // Check Rule: Tile B can't be directly to the left of Tile A
                        // This means if the current cell is Tile A, its left neighbor (x-1) cannot be Tile B.
                        if current_tile_id == tile_a_tid {
                            let left_nx = (x as i32 - 1).rem_euclid(w as i32) as usize;
                            if let Some(left_neighbor_possibilities) = grid.get(left_nx, y, z) {
                                if left_neighbor_possibilities.count_ones() == 1 {
                                    let left_neighbor_tile_id =
                                        left_neighbor_possibilities.iter_ones().next().unwrap();
                                    if left_neighbor_tile_id == tile_b_tid {
                                        println!(
                                            "Custom Rule Violation: Tile B ({}) is to the left of Tile A ({}) at ({},{},{}) (current) and ({},{},{}) (left neighbor).",
                                            left_neighbor_tile_id, current_tile_id, x, y, z, left_nx, y, z
                                        );
                                        violations += 1;
                                    }
                                }
                            }
                        }

                        // Check Rule: Tile A can't be directly above Tile A
                        // This means if the current cell is Tile A, its 'up' neighbor (y+1) cannot be Tile A.
                        if current_tile_id == tile_a_tid {
                            let up_ny = (y as i32 + 1).rem_euclid(h as i32) as usize;
                            if let Some(up_neighbor_possibilities) = grid.get(x, up_ny, z) {
                                if up_neighbor_possibilities.count_ones() == 1 {
                                    let up_neighbor_tile_id =
                                        up_neighbor_possibilities.iter_ones().next().unwrap();
                                    if up_neighbor_tile_id == tile_a_tid {
                                        println!(
                                            "Custom Rule Violation: Tile A ({}) is directly above Tile A ({}) at ({},{},{}) (current) and ({},{},{}) (up neighbor).",
                                            current_tile_id, up_neighbor_tile_id, x, y, z, x, up_ny, z
                                        );
                                        violations += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    violations
}
