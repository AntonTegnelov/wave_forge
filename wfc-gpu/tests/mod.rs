// Integration tests for wfc-gpu. They run on any Vulkan, Metal or DX12 adapter, including
// Mesa's software llvmpipe, so they also work in containers without a GPU.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use wfc_core::entropy::EntropyHeuristicType;
use wfc_core::grid::PossibilityGrid;
use wfc_core::BoundaryCondition;
use wfc_gpu::gpu::accelerator::GpuAccelerator;
use wfc_rules::{AdjacencyRules, TileId, TileSet, Transformation};

/// Constraints set before the run must shape the result even though constrained cells are
/// never selected for collapse. Regression test for docs/status.md A-8: a cell pinned to one
/// tile has zero entropy, so it used to be skipped and its neighbours ignored it.
#[tokio::test]
async fn pre_constrained_cells_propagate_before_first_collapse() -> anyhow::Result<()> {
    // Three tiles that may only touch themselves: pinning a single cell forces the whole grid,
    // so a correct solver finishes without collapsing anything itself.
    let num_tiles = 3;
    let tileset = TileSet::new(
        vec![1.0; num_tiles],
        vec![vec![Transformation::Identity]; num_tiles],
    )?;
    let tuples: Vec<(usize, usize, usize)> = (0..6)
        .flat_map(|axis| (0..num_tiles).map(move |tile| (axis, tile, tile)))
        .collect();
    let rules = AdjacencyRules::from_allowed_tuples(tileset.num_transformed_tiles(), 6, tuples);

    let mut grid = PossibilityGrid::new(4, 4, 4, num_tiles);
    grid.collapse(1, 2, 3, 2).map_err(anyhow::Error::msg)?;

    let mut accelerator = GpuAccelerator::new(
        &grid,
        &rules,
        BoundaryCondition::Finite,
        EntropyHeuristicType::Count,
        None,
    )
    .await?;

    // The callback runs once per collapse; counting calls tells us whether the solver had to
    // pick tiles itself, independent of which tiles it would have picked.
    let collapses = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&collapses);
    let result = accelerator
        .run_with_callback(
            &grid,
            &rules,
            1000,
            move |_| {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(true)
            },
            None,
        )
        .await
        .map_err(|e| anyhow::anyhow!("WFC run failed: {e}"))?;

    assert_eq!(collapses.load(Ordering::SeqCst), 0, "solver collapsed cells that were already forced");
    for z in 0..result.depth {
        for y in 0..result.height {
            for x in 0..result.width {
                let tiles: Vec<usize> = result.get(x, y, z).expect("cell in bounds").iter_ones().collect();
                assert_eq!(tiles, vec![2], "cell ({x}, {y}, {z})");
            }
        }
    }
    Ok(())
}

/// The entropy pass must evaluate every cell, not just those in the first workgroup of each
/// dispatch. Regression test: the host once dispatched with a larger workgroup size than the
/// shader declares, so cells beyond the first 8 columns and rows were never selected.
#[tokio::test]
async fn grids_larger_than_one_workgroup_fully_collapse() -> anyhow::Result<()> {
    let num_tiles = 2;
    let tileset = TileSet::new(
        vec![1.0; num_tiles],
        vec![vec![Transformation::Identity]; num_tiles],
    )?;
    let tuples: Vec<(usize, usize, usize)> = (0..6)
        .flat_map(|axis| (0..num_tiles).flat_map(move |a| (0..num_tiles).map(move |b| (axis, a, b))))
        .collect();
    let rules = AdjacencyRules::from_allowed_tuples(tileset.num_transformed_tiles(), 6, tuples);
    let grid = PossibilityGrid::new(20, 18, 2, num_tiles);

    let mut accelerator = GpuAccelerator::new(
        &grid,
        &rules,
        BoundaryCondition::Finite,
        EntropyHeuristicType::Count,
        None,
    )
    .await?;
    let result = accelerator
        .run_with_callback(&grid, &rules, 20 * 18 * 2 * 2, |_| Ok(true), None)
        .await
        .map_err(|e| anyhow::anyhow!("WFC run failed: {e}"))?;
    assert_eq!(result.is_fully_collapsed(), Ok(true));
    Ok(())
}

#[tokio::test]
async fn test_basic_3d_generation() -> anyhow::Result<()> {
    // Test configuration
    let grid_size = (8, 8, 8); // Small enough for quick testing on software adapters
    let num_tiles = 2; // Simple binary tiles (e.g., "filled" and "empty")

    println!("Starting test_basic_3d_generation");
    println!("Grid size: {:?}", grid_size);
    println!("Number of tiles: {}", num_tiles);

    // Create a simple tileset with different weights
    let tileset = TileSet::new(
        vec![1.0, 0.5], // weights: tile 0 is twice as likely as tile 1
        vec![vec![Transformation::Identity]; num_tiles], // allowed_transformations
    )?;

    println!("Created tileset with weights: {:?}", vec![1.0, 0.5]);
    println!(
        "Number of transformed tiles: {}",
        tileset.num_transformed_tiles()
    );

    // Create weighted adjacency rules: tiles can connect to themselves and their neighbors
    let mut allowed_tuples = Vec::new();
    for tile1 in 0..num_tiles {
        let transformed_tile_id1 = tileset
            .get_transformed_id(TileId(tile1), Transformation::Identity)
            .expect("Failed to get transformed tile ID");
        for tile2 in 0..num_tiles {
            let transformed_tile_id2 = tileset
                .get_transformed_id(TileId(tile2), Transformation::Identity)
                .expect("Failed to get transformed tile ID");
            for axis in 0..6 {
                // Allow tiles to connect to themselves with higher weight
                if tile1 == tile2 {
                    allowed_tuples.push((axis, transformed_tile_id1, transformed_tile_id2));
                    allowed_tuples.push((axis, transformed_tile_id1, transformed_tile_id2));
                // Add twice for higher weight
                } else {
                    // Allow different tiles to connect with lower weight
                    allowed_tuples.push((axis, transformed_tile_id1, transformed_tile_id2));
                }
            }
        }
    }

    println!("Created {} allowed tuples", allowed_tuples.len());
    println!(
        "Sample of allowed tuples: {:?}",
        &allowed_tuples[..std::cmp::min(5, allowed_tuples.len())]
    );

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
            &mut grid,
            &rules,
            (grid_size.0 * grid_size.1 * grid_size.2 * 2) as u64, // enough iterations to collapse every cell
            |_progress| Ok(true), // Continue running
            None,                 // No shutdown signal
        )
        .await;
    println!("WFC algorithm completed with result: {:?}", result);

    let final_grid = result.map_err(|e| anyhow::anyhow!("WFC algorithm failed: {e}"))?;
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
    println!("Found {} adjacency rule violations", violations);

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
        violations == 0,
        "Found {} adjacency rule violations",
        violations
    );
    assert!(
        collapsed_count == total_cells,
        "Not all cells collapsed: {} out of {} cells collapsed",
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
