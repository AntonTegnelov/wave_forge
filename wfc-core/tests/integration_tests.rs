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
    let tileset = TileSet::new(weights).expect("Failed to create TileSet for simple success test");
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
    let tileset = TileSet::new(weights).expect("Failed to create TileSet for contradiction test");
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
        Err(WfcError::Contradiction(_, _, _)) | Err(WfcError::PropagationError(_)) => {
            // Expected error type
            println!("Correctly detected contradiction or propagation error.");
        }
        Err(e) => panic!(
            "Expected Contradiction or PropagationError, but got {:?}",
            e
        ),
        Ok(()) => panic!("Expected WFC run to fail with a forced contradiction, but it succeeded."),
    }
}

// Helper to create simple rules: Tile `i` can only be adjacent to Tile `i`.
fn create_identity_rules(num_tiles: usize) -> (TileSet, AdjacencyRules) {
    let tileset = TileSet::new(vec![1.0; num_tiles]).unwrap();
    let num_axes = 6;
    let mut allowed = vec![false; num_axes * num_tiles * num_tiles];
    for axis in 0..num_axes {
        for tile_id in 0..num_tiles {
            let index = axis * num_tiles * num_tiles + tile_id * num_tiles + tile_id;
            allowed[index] = true;
        }
    }
    let rules = AdjacencyRules::new(num_tiles, num_axes, allowed);
    (tileset, rules)
}

#[test]
fn test_runner_simple_success() {
    let (tileset, rules) = create_identity_rules(2);
    let mut grid = PossibilityGrid::new(2, 1, 1, rules.num_tiles());
    let propagator = CpuConstraintPropagator::new();
    let entropy_calculator = CpuEntropyCalculator::new();

    let result = run(
        &mut grid,
        &tileset,
        &rules,
        propagator,
        entropy_calculator,
        None, // No progress callback for this test
    );

    assert!(result.is_ok());
    // Check if grid is fully collapsed (all cells have count 1)
    let collapsed_count = (0..grid.depth)
        .flat_map(move |z| {
            (0..grid.height).flat_map(move |y| (0..grid.width).map(move |x| (x, y, z)))
        })
        .filter(|(x, y, z)| grid.get(*x, *y, *z).map_or(0, |c| c.count_ones()) == 1)
        .count();
    assert_eq!(collapsed_count, grid.width * grid.height * grid.depth);
}

#[test]
fn test_runner_immediate_contradiction() {
    // Rule: Tile 0 is NOT allowed next to Tile 0 along X axis.
    let num_tiles = 1;
    let tileset = TileSet::new(vec![1.0; num_tiles]).unwrap();
    let num_axes = 6;
    let n_sq = num_tiles * num_tiles;
    // Start with all allowed, then disallow T0->T0 on X
    let mut allowed = vec![true; num_axes * n_sq];
    let axis_pos_x = 0;
    let axis_neg_x = 1;
    allowed[axis_pos_x * n_sq + 0 * num_tiles + 0] = false; // T0 !-> T0 (+X)
    allowed[axis_neg_x * n_sq + 0 * num_tiles + 0] = false; // T0 !-> T0 (-X)

    let rules = AdjacencyRules::new(num_tiles, num_axes, allowed);

    // Use a 2x1x1 grid, both cells default to Tile 0 possible
    let mut grid = PossibilityGrid::new(2, 1, 1, rules.num_tiles());
    let propagator = CpuConstraintPropagator::new();
    let entropy_calculator = CpuEntropyCalculator::new();

    // Run should fail during initial propagation because cell 0 forbids cell 1 (and vice versa)
    let result = run(
        &mut grid,
        &tileset,
        &rules,
        propagator,
        entropy_calculator,
        None,
    );

    assert!(result.is_err());
    match result {
        Err(WfcError::Contradiction(x, y, z)) => {
            // Contradiction could occur at (0,0,0) or (1,0,0) depending on propagation order
            assert!((x, y, z) == (0, 0, 0) || (x, y, z) == (1, 0, 0));
        }
        Err(e) => panic!("Expected Contradiction error, got {:?}", e),
        Ok(_) => panic!("Expected error, got Ok"),
    }
}

#[test]
fn test_runner_propagation_contradiction() {
    // Rules: T0 -> T1 (+X) ONLY.
    let num_tiles = 2;
    let tileset = TileSet::new(vec![1.0; num_tiles]).unwrap();
    let num_axes = 6;
    let mut allowed = vec![false; num_axes * num_tiles * num_tiles];

    // Only rule: T0 -> T1 (+X)
    let axis_pos_x = 0;
    allowed[axis_pos_x * num_tiles * num_tiles + 0 * num_tiles + 1] = true;

    let rules = AdjacencyRules::new(num_tiles, num_axes, allowed);

    let mut grid = PossibilityGrid::new(2, 1, 1, rules.num_tiles()); // 2x1x1 grid

    // Manually collapse cell (0,0,0) to Tile 0
    let cell0 = grid.get_mut(0, 0, 0).unwrap();
    *cell0 = bitvec![1, 0];

    let propagator = CpuConstraintPropagator::new();
    let entropy_calculator = CpuEntropyCalculator::new();

    let result = run(
        &mut grid,
        &tileset,
        &rules,
        propagator,
        entropy_calculator,
        None,
    );

    // Collapse of cell 0 to Tile 0 forces cell 1 to Tile 1.
    // Propagation from cell 1 back to cell 0 along -X finds no allowed tiles for cell 0,
    // because T1 -> T0 (-X) is not defined.
    assert!(result.is_err());
    match result {
        Err(WfcError::Contradiction(x, y, z)) => {
            assert_eq!((x, y, z), (0, 0, 0)); // Contradiction expected back at the first cell
        }
        Err(e) => panic!("Expected Contradiction error, got {:?}", e),
        Ok(_) => panic!("Expected error, got Ok"),
    }
}

// Renamed existing test to avoid conflict
#[test]
fn test_run_forced_contradiction_setup() {
    // Setup: 2 tiles, Tile 0 and Tile 1
    // Rules: Only Tile 0 -> Tile 0 and Tile 1 -> Tile 1 allowed horizontally (+X)
    let num_tiles = 2;
    let tileset = TileSet::new(vec![1.0, 1.0]).unwrap();
    let num_axes = 6;
    let mut allowed = vec![false; num_axes * num_tiles * num_tiles];
    let axis_pos_x = 0;
    let axis_neg_x = 1;
    allowed[axis_pos_x * num_tiles * num_tiles + 0 * num_tiles + 0] = true; // T0 -> T0 (+X)
    allowed[axis_pos_x * num_tiles * num_tiles + 1 * num_tiles + 1] = true; // T1 -> T1 (+X)
    allowed[axis_neg_x * num_tiles * num_tiles + 0 * num_tiles + 0] = true; // T0 -> T0 (-X)
    allowed[axis_neg_x * num_tiles * num_tiles + 1 * num_tiles + 1] = true; // T1 -> T1 (-X)
    let rules = AdjacencyRules::new(num_tiles, num_axes, allowed);

    let mut grid = PossibilityGrid::new(2, 1, 1, rules.num_tiles()); // 2x1 grid
    let propagator = CpuConstraintPropagator::new();
    let entropy_calculator = CpuEntropyCalculator::new();

    // Manually create a contradiction: Set cell 0 to only Tile 0, cell 1 to only Tile 1
    // This violates the horizontal rule.
    grid.get_mut(0, 0, 0).unwrap().fill(false);
    grid.get_mut(0, 0, 0).unwrap().set(0, true); // Only Tile 0
    grid.get_mut(1, 0, 0).unwrap().fill(false);
    grid.get_mut(1, 0, 0).unwrap().set(1, true); // Only Tile 1

    // Run WFC - It should detect the contradiction during initial propagation.
    let result = run(
        &mut grid,
        &tileset,
        &rules,
        propagator,
        entropy_calculator,
        None,
    );

    assert!(result.is_err(), "WFC should return an error");

    // Check that the error is a Contradiction or related PropagationError
    match result {
        Err(WfcError::Contradiction(_, _, _)) | Err(WfcError::PropagationError(_)) => {
            // Expected error type
            println!("Correctly detected contradiction or propagation error.");
        }
        Ok(_) => panic!("Expected WFC to fail with a contradiction, but it succeeded."),
        Err(e) => panic!(
            "Expected Contradiction or PropagationError, but got a different error: {:?}",
            e
        ),
    }
}
