// wfc-core/tests/integration_tests.rs
use bitvec::prelude::*;
use wfc_core::{
    run, AdjacencyRules, CpuConstraintPropagator, CpuEntropyCalculator, PossibilityGrid, TileSet,
    WfcError,
};

// Helper to create AdjacencyRules where identical tiles can be adjacent
fn create_simple_rules(num_tiles: usize, num_axes: usize) -> AdjacencyRules {
    let mut allowed = vec![false; num_axes * num_tiles * num_tiles];
    for axis in 0..num_axes {
        for t1_idx in 0..num_tiles {
            for t2_idx in 0..num_tiles {
                let index = axis * num_tiles * num_tiles + t1_idx * num_tiles + t2_idx;
                // Rule: Allow only if tiles are the same
                if t1_idx == t2_idx {
                    allowed[index] = true;
                }
            }
        }
    }
    AdjacencyRules::new(num_tiles, num_axes, allowed)
}

// Helper to create a grid initialized with all possibilities using new constructor
fn create_initial_grid(
    width: usize,
    height: usize,
    depth: usize,
    num_tiles: usize,
) -> PossibilityGrid {
    PossibilityGrid::new(width, height, depth, num_tiles)
}

// Helper to check if a grid is fully collapsed
fn is_fully_collapsed(grid: &PossibilityGrid) -> bool {
    for z in 0..grid.depth {
        for y in 0..grid.height {
            for x in 0..grid.width {
                if let Some(cell) = grid.get(x, y, z) {
                    if cell.count_ones() != 1 {
                        return false; // Found an uncollapsed cell
                    }
                } else {
                    return false; // Error case, should not happen in valid grid
                }
            }
        }
    }
    true
}

#[test]
fn test_run_simple_2x1x1_success() {
    let num_tiles = 2;
    let num_axes = 6; // 3D checks, even for 2x1x1 grid
    let weights = vec![1.0, 1.0]; // Equal weights for Tile 0 and Tile 1
    let tileset = TileSet::new(weights);
    let rules = create_simple_rules(num_tiles, num_axes);
    let propagator = CpuConstraintPropagator::new();
    let entropy_calculator = CpuEntropyCalculator::new();
    let mut grid = create_initial_grid(2, 1, 1, num_tiles);

    // Run the WFC algorithm
    let result = run(
        &mut grid,
        &tileset,
        &rules,
        propagator,
        entropy_calculator,
        None, // No progress callback
    );

    // Assert successful completion
    assert!(result.is_ok(), "WFC run failed: {:?}", result.err());

    // Assert the grid is fully collapsed
    assert!(is_fully_collapsed(&grid), "Grid was not fully collapsed");

    // Assert the final state is valid according to the rules
    // In this simple case, both cells must have the same TileId
    let cell0_option = grid.get(0, 0, 0);
    let cell1_option = grid.get(1, 0, 0);

    assert!(cell0_option.is_some());
    assert!(cell1_option.is_some());

    let cell0 = cell0_option.unwrap();
    let cell1 = cell1_option.unwrap();

    // Find the single set bit (TileId) in each cell
    let tile_id_0 = cell0.iter_ones().next().expect("Cell 0 has no set bit");
    let tile_id_1 = cell1.iter_ones().next().expect("Cell 1 has no set bit");

    assert_eq!(
        tile_id_0, tile_id_1,
        "Cells (0,0,0) and (1,0,0) have different TileIds, violating the rule"
    );
}

// Test setup: Rules are 0 allows 1, 1 allows 0.
// Initial Grid state: [Tile 0 only, Tile 0 only]
// This violates the rules. Propagation from either cell should lead to contradiction.
#[test]
fn test_run_forced_contradiction() {
    let num_tiles = 2;
    let num_axes = 6;
    let weights = vec![1.0, 1.0];
    let tileset = TileSet::new(weights);
    // Rule: Tile 0 cannot be next to Tile 0 (only Tile 1)
    // Rule: Tile 1 cannot be next to Tile 1 (only Tile 0)
    let mut allowed = vec![false; num_axes * num_tiles * num_tiles];
    for axis in 0..num_axes {
        allowed[axis * num_tiles * num_tiles + 0 * num_tiles + 1] = true; // 0 allows 1
        allowed[axis * num_tiles * num_tiles + 1 * num_tiles + 0] = true; // 1 allows 0
    }
    let rules = AdjacencyRules::new(num_tiles, num_axes, allowed);

    let propagator = CpuConstraintPropagator::new();
    let entropy_calculator = CpuEntropyCalculator::new();
    // Create a 2x1x1 grid
    let mut grid = PossibilityGrid::new(2, 1, 1, num_tiles);

    // Force an initial contradictory state [Tile 0, Tile 0]
    *grid.get_mut(0, 0, 0).unwrap() = bitvec![1, 0];
    *grid.get_mut(1, 0, 0).unwrap() = bitvec![1, 0];

    // Run the WFC algorithm.
    // It should start, find lowest entropy (either cell), try to "collapse" (already done),
    // then propagate, which should immediately detect the rule violation.
    let result = run(
        &mut grid,
        &tileset,
        &rules,
        propagator,
        entropy_calculator,
        None,
    );

    // Assert that the run resulted in a Contradiction or Propagation error
    assert!(result.is_err(), "Expected run to fail, but it succeeded.");
    match result {
        Err(WfcError::Contradiction) | Err(WfcError::PropagationError(_)) => {
            // Expected error type
        }
        Err(e) => panic!(
            "Expected Contradiction or PropagationError, but got {:?}",
            e
        ),
        Ok(()) => panic!("Expected WFC run to fail with a forced contradiction, but it succeeded."),
    }
}

// TODO: Add tests for different grid sizes, more complex rules, TileSet weights, progress reporting.
