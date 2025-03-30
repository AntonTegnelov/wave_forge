// wfc-core/tests/entropy_tests.rs
use bitvec::prelude::*;
use wfc_core::entropy::{CpuEntropyCalculator, EntropyCalculator};
use wfc_core::grid::{EntropyGrid, PossibilityGrid};

// Helper to create a simple PossibilityGrid for testing using public API
fn create_test_possibility_grid(
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
fn test_calculate_entropy_initial() {
    let calculator = CpuEntropyCalculator::new();
    let num_tiles = 8;
    let grid = create_test_possibility_grid(2, 2, 1, num_tiles);
    let entropy_grid = calculator.calculate_entropy(&grid);

    assert_eq!(entropy_grid.width, 2);
    assert_eq!(entropy_grid.height, 2);
    assert_eq!(entropy_grid.depth, 1);
    // All cells should have entropy equal to num_tiles initially
    for z in 0..entropy_grid.depth {
        for y in 0..entropy_grid.height {
            for x in 0..entropy_grid.width {
                let entropy = entropy_grid.get(x, y, z).unwrap();
                assert!(
                    (*entropy - num_tiles as f32).abs() < f32::EPSILON,
                    "Cell ({},{},{}) has entropy {} != {}",
                    x,
                    y,
                    z,
                    entropy,
                    num_tiles
                );
            }
        }
    }
}

#[test]
fn test_calculate_entropy_varied() {
    let calculator = CpuEntropyCalculator::new();
    let num_tiles = 4;
    let mut grid = create_test_possibility_grid(2, 1, 1, num_tiles);

    // Set specific possibilities
    *grid.get_mut(0, 0, 0).unwrap() = bitvec![1, 0, 1, 0]; // 2 possibilities -> entropy 2.0
    *grid.get_mut(1, 0, 0).unwrap() = bitvec![0, 0, 0, 1]; // 1 possibility -> entropy 0.0 (collapsed)

    let entropy_grid = calculator.calculate_entropy(&grid);

    assert!((entropy_grid.get(0, 0, 0).unwrap() - 2.0).abs() < f32::EPSILON);
    assert!(entropy_grid.get(1, 0, 0).unwrap().abs() < f32::EPSILON); // Should be 0.0
}

#[test]
fn test_find_lowest_entropy() {
    let calculator = CpuEntropyCalculator::new();
    let mut entropy_grid = EntropyGrid::new(3, 2, 1);

    // Fill with some entropy values
    *entropy_grid.get_mut(0, 0, 0).unwrap() = 5.0;
    *entropy_grid.get_mut(1, 0, 0).unwrap() = 3.0; // Potential minimum
    *entropy_grid.get_mut(2, 0, 0).unwrap() = 0.0; // Collapsed, should be ignored
    *entropy_grid.get_mut(0, 1, 0).unwrap() = 4.0;
    *entropy_grid.get_mut(1, 1, 0).unwrap() = 3.0; // Tie with (1,0,0)
    *entropy_grid.get_mut(2, 1, 0).unwrap() = 6.0;

    let lowest_coords = calculator.find_lowest_entropy(&entropy_grid);

    // The result could be either (1,0,0) or (1,1,0) due to the tie.
    // The specific result depends on rayon's internal scheduling, so we accept either.
    assert!(lowest_coords.is_some());
    let coords = lowest_coords.unwrap();
    assert!(coords == (1, 0, 0) || coords == (1, 1, 0));
    // Ensure the entropy at the chosen coordinate is indeed 3.0
    assert!((entropy_grid.get(coords.0, coords.1, coords.2).unwrap() - 3.0).abs() < f32::EPSILON);
}

#[test]
fn test_find_lowest_entropy_all_zero() {
    let calculator = CpuEntropyCalculator::new();
    let mut entropy_grid = EntropyGrid::new(2, 1, 1);
    *entropy_grid.get_mut(0, 0, 0).unwrap() = 0.0;
    *entropy_grid.get_mut(1, 0, 0).unwrap() = 0.0;

    let lowest_coords = calculator.find_lowest_entropy(&entropy_grid);
    assert!(
        lowest_coords.is_none(),
        "Should return None when all cells have zero entropy"
    );
}

#[test]
fn test_find_lowest_entropy_single_cell() {
    let calculator = CpuEntropyCalculator::new();
    let mut entropy_grid = EntropyGrid::new(1, 1, 1);
    *entropy_grid.get_mut(0, 0, 0).unwrap() = 5.5;

    let lowest_coords = calculator.find_lowest_entropy(&entropy_grid);
    assert_eq!(lowest_coords, Some((0, 0, 0)));
}
