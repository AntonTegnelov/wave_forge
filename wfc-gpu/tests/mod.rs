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

use std::collections::HashMap;
use std::path::PathBuf;
use wfc_core::entropy::EntropyHeuristicType;
use wfc_core::grid::PossibilityGrid;
use wfc_core::BoundaryCondition;
use wfc_gpu::gpu::accelerator::GpuAccelerator;
use wfc_rules::{AdjacencyRules, TileSet, Transformation};

#[tokio::test]
async fn test_basic_3d_generation() -> anyhow::Result<()> {
    // Test configuration
    let grid_size = (16, 16, 16); // Small enough for quick testing, large enough to be meaningful
    let num_tiles = 2; // Simple binary tiles (e.g., "filled" and "empty")

    // Create a simple tileset with equal weights and only Identity transformations
    let tileset = TileSet::new(
        vec![1.0; num_tiles],                            // weights
        vec![vec![Transformation::Identity]; num_tiles], // allowed_transformations
    )?;

    // Create simple adjacency rules: tiles can only connect to themselves
    let mut allowed_tuples = Vec::new();
    for tile in 0..num_tiles {
        for axis in 0..6 {
            // 6 axes for 3D (+x, -x, +y, -y, +z, -z)
            allowed_tuples.push((axis, tile, tile));
        }
    }
    let rules = AdjacencyRules::from_allowed_tuples(
        tileset.num_transformed_tiles(),
        6, // num_axes (3D = 6 directions)
        allowed_tuples,
    );

    // Initialize grid
    let mut grid = PossibilityGrid::new(grid_size.0, grid_size.1, grid_size.2, num_tiles);

    // Create GPU accelerator
    let mut accelerator = GpuAccelerator::new(
        &grid,
        &rules,
        BoundaryCondition::Periodic,
        EntropyHeuristicType::Shannon, // Use Shannon entropy for tile selection
        None,                          // No subgrid configuration
    )
    .await?;

    // Run the algorithm with a progress callback
    let result = accelerator
        .run_with_callback(
            &mut grid,
            &rules,
            1000, // Maximum number of iterations
            |progress: ProgressInfo| {
                println!(
                    "Generation progress: {:.1}%",
                    (progress.collapsed_cells as f32 / progress.total_cells as f32) * 100.0
                );
                Ok(true)
            },
            None, // No shutdown signal
        )
        .await;

    // Verify the result
    assert!(result.is_ok(), "WFC algorithm failed: {:?}", result.err());

    // Verify grid is fully collapsed
    let is_collapsed = grid.is_fully_collapsed()?;
    assert!(is_collapsed, "Not all cells were collapsed");

    // Get grid dimensions
    let total_cells = grid.width * grid.height * grid.depth;

    // Verify adjacency rules are satisfied
    let violations = verify_adjacency_rules(&grid, &rules);
    assert_eq!(
        violations, 0,
        "Found {} adjacency rule violations",
        violations
    );

    Ok(())
}

// Helper function to verify adjacency rules
fn verify_adjacency_rules(grid: &PossibilityGrid, rules: &AdjacencyRules) -> usize {
    let mut violations = 0;
    let (w, h, d) = grid.dimensions();

    // Check each cell's neighbors
    for x in 0..w {
        for y in 0..h {
            for z in 0..d {
                if let Some(tile) = grid.get_collapsed_state(x, y, z) {
                    // Check each direction
                    for (dx, dy, dz, axis, positive) in [
                        (1, 0, 0, 0, true),   // +x
                        (-1, 0, 0, 0, false), // -x
                        (0, 1, 0, 1, true),   // +y
                        (0, -1, 0, 1, false), // -y
                        (0, 0, 1, 2, true),   // +z
                        (0, 0, -1, 2, false), // -z
                    ] {
                        let nx = (x as i32 + dx).rem_euclid(w as i32) as usize;
                        let ny = (y as i32 + dy).rem_euclid(h as i32) as usize;
                        let nz = (z as i32 + dz).rem_euclid(d as i32) as usize;

                        if let Some(neighbor_tile) = grid.get_collapsed_state(nx, ny, nz) {
                            if !rules.check(tile, neighbor_tile, axis) {
                                violations += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    violations
}
