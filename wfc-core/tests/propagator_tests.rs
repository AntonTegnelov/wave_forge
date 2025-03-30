// wfc-core/tests/propagator_tests.rs
use bitvec::prelude::*;
use wfc_core::grid::PossibilityGrid;
use wfc_core::propagator::{ConstraintPropagator, CpuConstraintPropagator, PropagationError};
use wfc_core::rules::AdjacencyRules;

// Helper to create AdjacencyRules for simple cases
// Rules: Tile 0 can be next to Tile 0 (all axes)
//        Tile 1 can be next to Tile 1 (all axes)
//        Tile 0 and 1 cannot be next to each other
fn create_simple_rules(num_tiles: usize, num_axes: usize) -> AdjacencyRules {
    let mut allowed = vec![false; num_axes * num_tiles * num_tiles];
    for axis in 0..num_axes {
        for t1_idx in 0..num_tiles {
            for t2_idx in 0..num_tiles {
                let index = axis * num_tiles * num_tiles + t1_idx * num_tiles + t2_idx;
                // Simple rule: Allow only if tiles are the same
                if t1_idx == t2_idx {
                    allowed[index] = true;
                }
                // Example for Tile 0 and Tile 1 rule:
                // if t1_idx == 0 && t2_idx == 0 { allowed[index] = true; }
                // if t1_idx == 1 && t2_idx == 1 { allowed[index] = true; }
            }
        }
    }
    AdjacencyRules::new(num_tiles, num_axes, allowed)
}

// Helper to create a grid initialized with all possibilities
fn create_initial_grid(
    width: usize,
    height: usize,
    depth: usize,
    num_tiles: usize,
) -> PossibilityGrid {
    let mut grid = PossibilityGrid::new(width, height, depth);
    let all_possible = bitvec![1; num_tiles];
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                if let Some(cell) = grid.get_mut(x, y, z) {
                    *cell = all_possible.clone();
                }
            }
        }
    }
    grid
}

#[test]
fn test_propagate_simple_case() {
    let num_tiles = 2;
    let num_axes = 6; // 3D
    let rules = create_simple_rules(num_tiles, num_axes);
    let mut propagator = CpuConstraintPropagator::new();
    let mut grid = create_initial_grid(3, 1, 1, num_tiles); // Simple 1D strip

    // Collapse cell (1,0,0) to only allow Tile 0
    let mut initial_update_coords = Vec::new();
    if let Some(cell) = grid.get_mut(1, 0, 0) {
        *cell = bitvec![1, 0]; // Only Tile 0 allowed
        initial_update_coords.push((1, 0, 0));
    } else {
        panic!("Could not get mutable cell (1,0,0)");
    }

    // Propagate the change
    let result = propagator.propagate(&mut grid, initial_update_coords, &rules);
    assert!(result.is_ok());

    // Check neighbors of (1,0,0), which are (0,0,0) and (2,0,0)
    // They should now also only allow Tile 0, because Tile 0 cannot be next to Tile 1
    let expected_neighbor_possibilities = bitvec![1, 0];

    // Check cell (0,0,0)
    if let Some(cell) = grid.get(0, 0, 0) {
        assert_eq!(
            cell, &expected_neighbor_possibilities,
            "Cell (0,0,0) has wrong possibilities"
        );
    } else {
        panic!("Could not get cell (0,0,0)");
    }

    // Check cell (2,0,0)
    if let Some(cell) = grid.get(2, 0, 0) {
        assert_eq!(
            cell, &expected_neighbor_possibilities,
            "Cell (2,0,0) has wrong possibilities"
        );
    } else {
        panic!("Could not get cell (2,0,0)");
    }

    // Cell (1,0,0) should remain unchanged by propagation
    if let Some(cell) = grid.get(1, 0, 0) {
        assert_eq!(cell, &bitvec![1, 0], "Cell (1,0,0) changed unexpectedly");
    } else {
        panic!("Could not get cell (1,0,0)");
    }
}

#[test]
fn test_propagate_contradiction() {
    let num_tiles = 2;
    let num_axes = 6;
    let rules = create_simple_rules(num_tiles, num_axes); // Tiles must be same
    let mut propagator = CpuConstraintPropagator::new();
    let mut grid = create_initial_grid(2, 1, 1, num_tiles); // Two adjacent cells

    // Collapse cell (0,0,0) to Tile 0
    if let Some(cell) = grid.get_mut(0, 0, 0) {
        *cell = bitvec![1, 0];
    }
    // Collapse cell (1,0,0) to Tile 1
    if let Some(cell) = grid.get_mut(1, 0, 0) {
        *cell = bitvec![0, 1];
    }

    // Propagating from (0,0,0) should force (1,0,0) to become Tile 0,
    // but it's already Tile 1, leading to a contradiction.
    let initial_update_coords = vec![(0, 0, 0)];
    let result = propagator.propagate(&mut grid, initial_update_coords, &rules);

    assert!(result.is_err());
    match result {
        Err(PropagationError::Contradiction(x, y, z)) => {
            // The contradiction should be detected at the neighbor (1,0,0)
            assert_eq!(
                (x, y, z),
                (1, 0, 0),
                "Contradiction reported at wrong location"
            );
        }
        Ok(()) => panic!("Expected Contradiction error, but got Ok"),
    }

    // Test propagating from the other cell too, should yield same contradiction
    let mut grid2 = create_initial_grid(2, 1, 1, num_tiles);
    if let Some(cell) = grid2.get_mut(0, 0, 0) {
        *cell = bitvec![1, 0];
    }
    if let Some(cell) = grid2.get_mut(1, 0, 0) {
        *cell = bitvec![0, 1];
    }
    let initial_update_coords2 = vec![(1, 0, 0)];
    let result2 = propagator.propagate(&mut grid2, initial_update_coords2, &rules);
    assert!(result2.is_err());
    match result2 {
        Err(PropagationError::Contradiction(x, y, z)) => {
            // Contradiction detected at neighbor (0,0,0) this time
            assert_eq!(
                (x, y, z),
                (0, 0, 0),
                "Contradiction reported at wrong location"
            );
        }
        Ok(()) => panic!("Expected Contradiction error, but got Ok"),
    }
}

#[test]
fn test_propagate_no_change() {
    let num_tiles = 2;
    let num_axes = 6;
    let rules = create_simple_rules(num_tiles, num_axes);
    let mut propagator = CpuConstraintPropagator::new();
    let mut grid = create_initial_grid(2, 1, 1, num_tiles);
    let original_grid = grid.clone(); // Keep copy for comparison (Grid needs Clone)

    // Collapse cell (0,0,0) to Tile 0
    let mut initial_update_coords = Vec::new();
    if let Some(cell) = grid.get_mut(0, 0, 0) {
        *cell = bitvec![1, 0];
        initial_update_coords.push((0, 0, 0));
    }

    // Propagate once
    let result1 = propagator.propagate(&mut grid, initial_update_coords.clone(), &rules);
    assert!(result1.is_ok());
    let grid_after_first_prop = grid.clone(); // Clone the grid state

    // Propagate again with the same starting point - should cause no further changes
    let result2 = propagator.propagate(&mut grid, initial_update_coords, &rules);
    assert!(result2.is_ok());

    // Compare grid state after first and second propagation cell by cell
    assert_eq!(grid.width, grid_after_first_prop.width);
    assert_eq!(grid.height, grid_after_first_prop.height);
    assert_eq!(grid.depth, grid_after_first_prop.depth);
    let mut changed_after_second_prop = false;
    for z in 0..grid.depth {
        for y in 0..grid.height {
            for x in 0..grid.width {
                if grid.get(x, y, z) != grid_after_first_prop.get(x, y, z) {
                    changed_after_second_prop = true;
                    break;
                }
            }
            if changed_after_second_prop {
                break;
            }
        }
        if changed_after_second_prop {
            break;
        }
    }
    assert!(
        !changed_after_second_prop,
        "Grid state changed after second propagation"
    );

    // Ensure the grid changed from the original state cell by cell
    let mut changed_from_original = false;
    for z in 0..grid.depth {
        for y in 0..grid.height {
            for x in 0..grid.width {
                if grid.get(x, y, z) != original_grid.get(x, y, z) {
                    changed_from_original = true;
                    break;
                }
            }
            if changed_from_original {
                break;
            }
        }
        if changed_from_original {
            break;
        }
    }
    assert!(
        changed_from_original,
        "Grid state did not change after first propagation"
    );
}

// TODO: Add more complex tests (larger grids, more complex rules, multi-step propagation)
