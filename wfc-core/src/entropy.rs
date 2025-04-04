use crate::grid::{EntropyGrid, PossibilityGrid};
use thiserror::Error;

/// Strategy for selecting among cells with the same lowest entropy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionStrategy {
    /// Choose the first cell encountered with the minimum entropy (deterministic).
    FirstMinimum,
    /// Choose randomly among all cells sharing the minimum positive entropy.
    RandomLowest,
    // TODO: HilbertCurve,
}

/// Errors related to entropy calculation or selection.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum EntropyError {
    #[error("Failed to access grid cell during entropy calculation at ({0}, {1}, {2})")]
    GridAccessError(usize, usize, usize),
    #[error("Logarithm of zero encountered during entropy calculation for cell ({0}, {1}, {2})")]
    LogOfZero(usize, usize, usize),
    #[error(
        "Invalid weight encountered (<= 0) during entropy calculation for cell ({0}, {1}, {2})"
    )]
    InvalidWeight(usize, usize, usize),
    #[error("Entropy calculation resulted in NaN for cell ({0}, {1}, {2})")]
    NaNResult(usize, usize, usize),
    #[error("Other entropy calculation error: {0}")]
    Other(String),
}

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
    /// * `Result<EntropyGrid, EntropyError>` - An `EntropyGrid` containing the calculated
    ///   entropy for each cell, or an error message if calculation fails.
    fn calculate_entropy(&self, grid: &PossibilityGrid) -> Result<EntropyGrid, EntropyError>;

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

    // Mock Calculator
    struct MockEntropyCalculator;
    impl EntropyCalculator for MockEntropyCalculator {
        fn calculate_entropy(&self, grid: &PossibilityGrid) -> Result<EntropyGrid, EntropyError> {
            let mut entropy_grid = EntropyGrid::new(grid.width, grid.height, grid.depth);
            for z in 0..grid.depth {
                for y in 0..grid.height {
                    for x in 0..grid.width {
                        let cell = grid
                            .get(x, y, z)
                            .ok_or(EntropyError::GridAccessError(x, y, z))?;
                        let count = cell.count_ones();
                        let entropy_cell =
                            entropy_grid
                                .get_mut(x, y, z)
                                .ok_or(EntropyError::Other(format!(
                                    "Failed to access entropy grid cell ({},{},{})",
                                    x, y, z
                                )))?;
                        *entropy_cell = count as f32;
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

        let entropy_grid_result = calculator.calculate_entropy(&grid);
        assert!(entropy_grid_result.is_ok());
        let entropy_grid = entropy_grid_result.unwrap();
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
