use crate::grid::{EntropyGrid, PossibilityGrid};

/// Trait for selecting the next cell to collapse based on entropy.
///
/// Implementors of this trait define a strategy for choosing the cell
/// with the lowest entropy, which is a core step in the WFC algorithm.
pub trait EntropyCalculator: Send + Sync {
    /// Calculates the entropy for each cell in the grid.
    ///
    /// # Arguments
    ///
    /// * `grid` - The current state of the possibility grid.
    ///
    /// # Returns
    ///
    /// * `Result<EntropyGrid, String>` - An `EntropyGrid` containing the calculated
    ///   entropy for each cell, or an error message if calculation fails.
    fn calculate_entropy(&self, grid: &PossibilityGrid) -> Result<EntropyGrid, String>;

    /// Selects the cell with the lowest non-zero entropy.
    ///
    /// # Arguments
    ///
    /// * `entropy_grid` - The grid containing pre-calculated entropy values.
    ///
    /// # Returns
    ///
    /// * `Option<(usize, usize, usize)>` - The coordinates (x, y, z) of the cell
    ///   with the lowest entropy, or `None` if all cells are collapsed (entropy 0)
    ///   or if the grid is empty.
    fn select_lowest_entropy_cell(
        &self,
        entropy_grid: &EntropyGrid,
    ) -> Option<(usize, usize, usize)>;
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
        fn calculate_entropy(&self, grid: &PossibilityGrid) -> Result<EntropyGrid, String> {
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
            Ok(entropy_grid)
        }

        fn select_lowest_entropy_cell(
            &self,
            entropy_grid: &EntropyGrid,
        ) -> Option<(usize, usize, usize)> {
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

        let entropy_grid = calculator.calculate_entropy(&grid).unwrap();
        assert_eq!(entropy_grid.get(0, 0, 0), Some(&1.0));
        assert_eq!(entropy_grid.get(1, 0, 0), Some(&2.0));
    }

    #[test]
    fn test_mock_select_lowest_entropy_cell() {
        let calculator = MockEntropyCalculator;
        let mut entropy_grid = EntropyGrid::new(2, 2, 1);
        *entropy_grid.get_mut(0, 0, 0).unwrap() = 1.0;
        *entropy_grid.get_mut(1, 0, 0).unwrap() = 0.0;
        *entropy_grid.get_mut(0, 1, 0).unwrap() = 2.0; // First > 1.0
        *entropy_grid.get_mut(1, 1, 0).unwrap() = 3.0;

        let lowest = calculator.select_lowest_entropy_cell(&entropy_grid);
        assert_eq!(lowest, Some((0, 1, 0)));

        // Test case where all are <= 1.0
        *entropy_grid.get_mut(0, 1, 0).unwrap() = 1.0;
        *entropy_grid.get_mut(1, 1, 0).unwrap() = 1.0;
        let lowest = calculator.select_lowest_entropy_cell(&entropy_grid);
        assert_eq!(lowest, None);
    }

    // Add more tests here that verify the *behavior* of entropy calculation
    // and lowest entropy finding, potentially using a test-specific
    // GpuAccelerator instance if integration testing is set up.
}
