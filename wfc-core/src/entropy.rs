use crate::grid::{EntropyGrid, PossibilityGrid};

/// Trait defining the interface for calculating cell entropy and finding the minimum.
///
/// Entropy is a measure of uncertainty or "mixedness" within a cell's possibility state.
/// Lower entropy generally indicates fewer possible tiles, making the cell a good candidate
/// for collapsing next in the WFC algorithm.
pub trait EntropyCalculator {
    /// Calculates the entropy for every cell in the `PossibilityGrid`.
    ///
    /// Returns an `EntropyGrid` (a `Grid<f32>`) where each cell contains the calculated
    /// entropy value based on the corresponding cell's `BitVec` in the input `grid`.
    /// The specific entropy calculation method depends on the implementing type.
    ///
    /// # Arguments
    ///
    /// * `grid` - The `PossibilityGrid` containing the current possibility state.
    ///
    /// # Returns
    ///
    /// An `EntropyGrid` with the calculated entropy for each cell.
    #[must_use]
    fn calculate_entropy(&self, grid: &PossibilityGrid) -> EntropyGrid;

    /// Finds the coordinates of the cell with the lowest positive entropy.
    ///
    /// Iterates through the `entropy_grid`, ignoring cells with entropy <= 0 (already collapsed
    /// or potentially in a contradictory state), and returns the `(x, y, z)` coordinates
    /// of the cell with the minimum positive entropy value.
    ///
    /// Returns `None` if all cells have entropy <= 0 (i.e., the grid is fully collapsed or in an error state).
    /// Ties may be broken arbitrarily (often by picking the first one found).
    ///
    /// # Arguments
    ///
    /// * `entropy_grid` - The grid containing pre-calculated entropy values.
    ///
    /// # Returns
    ///
    /// * `Some((x, y, z))` - Coordinates of the cell with the lowest positive entropy.
    /// * `None` - If no cell with positive entropy is found.
    #[must_use]
    fn find_lowest_entropy(&self, entropy_grid: &EntropyGrid) -> Option<(usize, usize, usize)>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{EntropyGrid, PossibilityGrid};
    use bitvec::prelude::*;

    // --- Test cases for entropy logic (CPU or GPU) ---
    // Note: These tests focus on the expected outcome (entropy grid state),
    // not the specific implementation (CPU/GPU). They require a concrete
    // calculator implementation to run.

    // Mock Calculator (if needed for tests independent of CPU/GPU impl)
    struct MockEntropyCalculator;
    impl EntropyCalculator for MockEntropyCalculator {
        fn calculate_entropy(&self, grid: &PossibilityGrid) -> EntropyGrid {
            // Simple mock: return grid where entropy = count_ones()
            let mut entropy_grid = EntropyGrid::new(grid.width, grid.height, grid.depth);
            for z in 0..grid.depth {
                for y in 0..grid.height {
                    for x in 0..grid.width {
                        if let Some(cell) = grid.get(x, y, z) {
                            let count = cell.count_ones();
                            *entropy_grid.get_mut(x, y, z).unwrap() = count as f32;
                        }
                    }
                }
            }
            entropy_grid
        }

        fn find_lowest_entropy(&self, entropy_grid: &EntropyGrid) -> Option<(usize, usize, usize)> {
            // Simple mock: find first cell with entropy > 1.0
            for z in 0..entropy_grid.depth {
                for y in 0..entropy_grid.height {
                    for x in 0..entropy_grid.width {
                        if entropy_grid.get(x, y, z).map_or(false, |&e| e > 1.0) {
                            return Some((x, y, z));
                        }
                    }
                }
            }
            None
        }
    }

    #[test]
    fn test_mock_calculate_entropy() {
        let calculator = MockEntropyCalculator;
        let mut grid = PossibilityGrid::new(2, 1, 1, 3);
        *grid.get_mut(0, 0, 0).unwrap() = bitvec![1, 0, 0]; // 1 possibility
        *grid.get_mut(1, 0, 0).unwrap() = bitvec![0, 1, 1]; // 2 possibilities

        let entropy_grid = calculator.calculate_entropy(&grid);
        assert_eq!(entropy_grid.get(0, 0, 0), Some(&1.0));
        assert_eq!(entropy_grid.get(1, 0, 0), Some(&2.0));
    }

    #[test]
    fn test_mock_find_lowest_entropy() {
        let calculator = MockEntropyCalculator;
        let mut entropy_grid = EntropyGrid::new(2, 2, 1);
        *entropy_grid.get_mut(0, 0, 0).unwrap() = 1.0;
        *entropy_grid.get_mut(1, 0, 0).unwrap() = 0.0;
        *entropy_grid.get_mut(0, 1, 0).unwrap() = 2.0; // First > 1.0
        *entropy_grid.get_mut(1, 1, 0).unwrap() = 3.0;

        let lowest = calculator.find_lowest_entropy(&entropy_grid);
        assert_eq!(lowest, Some((0, 1, 0)));

        // Test case where all are <= 1.0
        *entropy_grid.get_mut(0, 1, 0).unwrap() = 1.0;
        *entropy_grid.get_mut(1, 1, 0).unwrap() = 1.0;
        let lowest = calculator.find_lowest_entropy(&entropy_grid);
        assert_eq!(lowest, None);
    }

    // Add more tests here that verify the *behavior* of entropy calculation
    // and lowest entropy finding, potentially using a test-specific
    // GpuAccelerator instance if integration testing is set up.
}
