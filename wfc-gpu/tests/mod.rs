// Integration tests for wfc-gpu. They run on any Vulkan, Metal or DX12 adapter, including
// Mesa's software llvmpipe, so they also work in containers without a GPU.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use wfc_core::BoundaryCondition;
use wfc_core::entropy::EntropyHeuristicType;
use wfc_core::grid::PossibilityGrid;
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

    assert_eq!(
        collapses.load(Ordering::SeqCst),
        0,
        "solver collapsed cells that were already forced"
    );
    for z in 0..result.depth {
        for y in 0..result.height {
            for x in 0..result.width {
                let tiles: Vec<usize> = result
                    .get(x, y, z)
                    .expect("cell in bounds")
                    .iter_ones()
                    .collect();
                assert_eq!(tiles, vec![2], "cell ({x}, {y}, {z})");
            }
        }
    }
    Ok(())
}

/// Rule sets with more than 32 tile variants need several possibility words per cell.
/// Regression test: the propagation shader used to read and write only the first word, so tiles
/// 32 and above were silently ignored.
#[tokio::test]
async fn tile_sets_larger_than_32_tiles_propagate_across_words() -> anyhow::Result<()> {
    // 70 tiles (three words) that may only touch themselves; pinning one cell to tile 69, which
    // lives in the third word, forces the whole grid.
    let num_tiles = 70;
    let tileset = TileSet::new(
        vec![1.0; num_tiles],
        vec![vec![Transformation::Identity]; num_tiles],
    )?;
    let tuples: Vec<(usize, usize, usize)> = (0..6)
        .flat_map(|axis| (0..num_tiles).map(move |tile| (axis, tile, tile)))
        .collect();
    let rules = AdjacencyRules::from_allowed_tuples(tileset.num_transformed_tiles(), 6, tuples);

    let mut grid = PossibilityGrid::new(4, 3, 2, num_tiles);
    grid.collapse(2, 1, 1, 69).map_err(anyhow::Error::msg)?;

    let mut accelerator = GpuAccelerator::new(
        &grid,
        &rules,
        BoundaryCondition::Finite,
        EntropyHeuristicType::Count,
        None,
    )
    .await?;
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

    assert_eq!(
        collapses.load(Ordering::SeqCst),
        0,
        "solver collapsed cells that were already forced"
    );
    for z in 0..result.depth {
        for y in 0..result.height {
            for x in 0..result.width {
                let tiles: Vec<usize> = result
                    .get(x, y, z)
                    .expect("cell in bounds")
                    .iter_ones()
                    .collect();
                assert_eq!(tiles, vec![69], "cell ({x}, {y}, {z})");
            }
        }
    }
    Ok(())
}

/// Rule sets larger than the shader supports are rejected up front with a clear error instead of
/// producing silently wrong propagation.
#[tokio::test]
async fn tile_sets_beyond_the_shader_limit_are_rejected() {
    let num_tiles = wfc_gpu::shader::pipeline::MAX_TILES + 1;
    let rules =
        AdjacencyRules::from_allowed_tuples(num_tiles, 6, Vec::<(usize, usize, usize)>::new());
    let grid = PossibilityGrid::new(1, 1, 1, num_tiles);
    let Err(error) = GpuAccelerator::new(
        &grid,
        &rules,
        BoundaryCondition::Finite,
        EntropyHeuristicType::Count,
        None,
    )
    .await
    else {
        panic!("accelerator must refuse the rule set");
    };
    assert!(
        error.to_string().contains("at most"),
        "unexpected error: {error}"
    );
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
        .flat_map(|axis| {
            (0..num_tiles).flat_map(move |a| (0..num_tiles).map(move |b| (axis, a, b)))
        })
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
            |_progress| Ok(true),                                 // Continue running
            None,                                                 // No shutdown signal
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

/// Tile weights must bias collapse (docs/status.md A-3): with every adjacency allowed, a tile a
/// thousand times heavier than the other should fill nearly every cell.
#[tokio::test]
async fn tile_weights_bias_collapse() -> anyhow::Result<()> {
    let num_tiles = 2;
    let tuples: Vec<(usize, usize, usize)> = (0..6)
        .flat_map(|axis| {
            (0..num_tiles).flat_map(move |a| (0..num_tiles).map(move |b| (axis, a, b)))
        })
        .collect();
    let rules = AdjacencyRules::from_allowed_tuples(num_tiles, 6, tuples);
    let grid = PossibilityGrid::new(4, 4, 4, num_tiles);

    let mut accelerator = GpuAccelerator::new(
        &grid,
        &rules,
        BoundaryCondition::Finite,
        EntropyHeuristicType::Count,
        None,
    )
    .await?;
    accelerator.with_tile_weights(&[1.0, 1000.0])?;
    let result = accelerator
        .run_with_callback(&grid, &rules, 1000, |_| Ok(true), None)
        .await
        .map_err(|e| anyhow::anyhow!("WFC run failed: {e}"))?;

    let mut heavy = 0;
    for z in 0..result.depth {
        for y in 0..result.height {
            for x in 0..result.width {
                let tiles: Vec<usize> = result
                    .get(x, y, z)
                    .expect("cell in bounds")
                    .iter_ones()
                    .collect();
                assert_eq!(tiles.len(), 1, "cell ({x}, {y}, {z}) not collapsed");
                heavy += usize::from(tiles[0] == 1);
            }
        }
    }
    // Expected light cells: 64 / 1001 ≈ 0.06; more than 8 is astronomically unlikely unless
    // weights are ignored (then about 32).
    assert!(heavy >= 56, "only {heavy}/64 cells chose the heavy tile");
    Ok(())
}

/// Weights that do not match the tile set are rejected instead of panicking mid-run.
#[tokio::test]
async fn invalid_tile_weights_are_rejected() -> anyhow::Result<()> {
    let rules = AdjacencyRules::from_allowed_tuples(2, 6, Vec::<(usize, usize, usize)>::new());
    let grid = PossibilityGrid::new(1, 1, 1, 2);
    let mut accelerator = GpuAccelerator::new(
        &grid,
        &rules,
        BoundaryCondition::Finite,
        EntropyHeuristicType::Count,
        None,
    )
    .await?;
    assert!(
        accelerator.with_tile_weights(&[1.0]).is_err(),
        "wrong length"
    );
    assert!(
        accelerator.with_tile_weights(&[1.0, 0.0]).is_err(),
        "zero weight"
    );
    assert!(
        accelerator.with_tile_weights(&[1.0, f32::NAN]).is_err(),
        "NaN weight"
    );
    assert!(accelerator.with_tile_weights(&[1.0, 2.0]).is_ok());
    Ok(())
}

/// Bans one tile everywhere, one cell at a time, as a global constraint would.
struct BanTile(usize);

impl wfc_core::constraint::GlobalConstraint for BanTile {
    fn apply(
        &self,
        grid: &mut PossibilityGrid,
    ) -> Result<Vec<(usize, usize, usize)>, (usize, usize, usize)> {
        let mut changed = Vec::new();
        for z in 0..grid.depth {
            for y in 0..grid.height {
                for x in 0..grid.width {
                    let cell = grid.get_mut(x, y, z).expect("cell in bounds");
                    if cell[self.0] && cell.count_ones() > 1 {
                        cell.set(self.0, false);
                        changed.push((x, y, z));
                    }
                }
            }
        }
        Ok(changed)
    }
}

/// A global constraint's bans must hold in the result: the solver may only choose among what the
/// constraint left, and the cells it changed are propagated.
#[tokio::test]
async fn global_constraint_bans_apply_before_collapsing() -> anyhow::Result<()> {
    let num_tiles = 2;
    let tuples: Vec<(usize, usize, usize)> = (0..6)
        .flat_map(|axis| {
            (0..num_tiles).flat_map(move |a| (0..num_tiles).map(move |b| (axis, a, b)))
        })
        .collect();
    let rules = AdjacencyRules::from_allowed_tuples(num_tiles, 6, tuples);
    let grid = PossibilityGrid::new(4, 4, 2, num_tiles);
    let mut accelerator = GpuAccelerator::new(
        &grid,
        &rules,
        BoundaryCondition::Finite,
        EntropyHeuristicType::Count,
        None,
    )
    .await?;
    accelerator.with_global_constraint(Arc::new(BanTile(1)));
    let result = accelerator
        .run_with_callback(&grid, &rules, 1000, |_| Ok(true), None)
        .await
        .map_err(|e| anyhow::anyhow!("WFC run failed: {e}"))?;
    for z in 0..result.depth {
        for y in 0..result.height {
            for x in 0..result.width {
                assert_eq!(
                    result.get(x, y, z).unwrap().iter_ones().collect::<Vec<_>>(),
                    vec![0],
                    "({x}, {y}, {z})"
                );
            }
        }
    }
    Ok(())
}

struct Unsatisfiable;

impl wfc_core::constraint::GlobalConstraint for Unsatisfiable {
    fn apply(
        &self,
        _: &mut PossibilityGrid,
    ) -> Result<Vec<(usize, usize, usize)>, (usize, usize, usize)> {
        Err((1, 2, 3))
    }
}

/// A violated global constraint ends the run as a contradiction, so callers can restart.
#[tokio::test]
async fn unsatisfiable_global_constraint_is_a_contradiction() -> anyhow::Result<()> {
    let rules = AdjacencyRules::from_allowed_tuples(
        2,
        6,
        (0..6).flat_map(|axis| [(axis, 0, 0), (axis, 1, 1)]),
    );
    let grid = PossibilityGrid::new(2, 2, 2, 2);
    let mut accelerator = GpuAccelerator::new(
        &grid,
        &rules,
        BoundaryCondition::Finite,
        EntropyHeuristicType::Count,
        None,
    )
    .await?;
    accelerator.with_global_constraint(Arc::new(Unsatisfiable));
    let Err(error) = accelerator
        .run_with_callback(&grid, &rules, 100, |_| Ok(true), None)
        .await
    else {
        panic!("run must fail");
    };
    assert!(
        error.to_string().contains("Contradiction") && error.to_string().contains("(1, 2, 3)"),
        "{error}"
    );
    Ok(())
}

/// Rejects any grid where a cell has already collapsed to the forbidden tile. Nothing stops the
/// solver from choosing it, so the run can only finish by undoing that choice.
struct ForbidsCollapsedTile(usize);

impl wfc_core::constraint::GlobalConstraint for ForbidsCollapsedTile {
    fn apply(
        &self,
        grid: &mut PossibilityGrid,
    ) -> Result<Vec<(usize, usize, usize)>, (usize, usize, usize)> {
        for z in 0..grid.depth {
            for y in 0..grid.height {
                for x in 0..grid.width {
                    let cell = grid.get(x, y, z).expect("cell in bounds");
                    if cell.count_ones() == 1 && cell[self.0] {
                        return Err((x, y, z));
                    }
                }
            }
        }
        Ok(Vec::new())
    }
}

/// A contradiction must undo earlier choices rather than end the run (docs/status.md A-9).
/// Both tiles are always allowed by the rules, so only backtracking can produce a grid without
/// the forbidden tile.
#[tokio::test]
async fn contradictions_backtrack_instead_of_failing_the_run() -> anyhow::Result<()> {
    let num_tiles = 2;
    let tuples: Vec<(usize, usize, usize)> = (0..6)
        .flat_map(|axis| {
            (0..num_tiles).flat_map(move |a| (0..num_tiles).map(move |b| (axis, a, b)))
        })
        .collect();
    let rules = AdjacencyRules::from_allowed_tuples(num_tiles, 6, tuples);
    let grid = PossibilityGrid::new(3, 3, 2, num_tiles);

    let mut accelerator = GpuAccelerator::new(
        &grid,
        &rules,
        BoundaryCondition::Finite,
        EntropyHeuristicType::Count,
        None,
    )
    .await?;
    accelerator.with_global_constraint(Arc::new(ForbidsCollapsedTile(1)));
    let result = accelerator
        .run_with_callback(&grid, &rules, 1000, |_| Ok(true), None)
        .await
        .map_err(|e| anyhow::anyhow!("WFC run failed: {e}"))?;

    for z in 0..result.depth {
        for y in 0..result.height {
            for x in 0..result.width {
                assert_eq!(
                    result.get(x, y, z).unwrap().iter_ones().collect::<Vec<_>>(),
                    vec![0],
                    "({x}, {y}, {z})"
                );
            }
        }
    }
    Ok(())
}
