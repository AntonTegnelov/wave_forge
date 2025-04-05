use crate::grid::{EntropyGrid, PossibilityGrid};
use bitvec::prelude::*;
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

/// Defines the entropy calculation heuristic to use
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntropyHeuristicType {
    /// Standard Shannon entropy (log2 of possibilities count)
    Shannon,
    /// Simple possibility count (linear weighting)
    Count,
    /// Just count possibilities, no logarithm (simpler and potentially more stable)
    CountSimple,
    /// Weighted count considering tile weights
    WeightedCount,
}

impl Default for EntropyHeuristicType {
    fn default() -> Self {
        EntropyHeuristicType::Shannon
    }
}

/// Trait for strategies to calculate entropy values for cells
pub trait EntropyHeuristic: Send + Sync {
    /// Calculate the entropy value for a single cell
    ///
    /// # Arguments
    ///
    /// * `cell` - The bit vector representing possible states for this cell
    /// * `weights` - Optional slice of weights for each possible state (if available)
    ///
    /// # Returns
    ///
    /// * `f32` - The calculated entropy value (lower values indicate higher certainty)
    fn calculate_cell_entropy(&self, cell: &BitSlice, weights: Option<&[f32]>) -> f32;

    /// Get the heuristic type
    fn get_type(&self) -> EntropyHeuristicType;
}

/// Shannon entropy heuristic: H = log2(n) where n is the number of possibilities
#[derive(Debug, Clone, Copy)]
pub struct ShannonEntropyHeuristic;

impl EntropyHeuristic for ShannonEntropyHeuristic {
    fn calculate_cell_entropy(&self, cell: &BitSlice, _weights: Option<&[f32]>) -> f32 {
        let count = cell.count_ones();
        if count == 0 {
            return -1.0; // Contradiction: negative entropy
        } else if count == 1 {
            return 0.0; // Fully collapsed: zero entropy
        } else {
            // Shannon entropy for uniform distribution: log2(n)
            return (count as f32).log2();
        }
    }

    fn get_type(&self) -> EntropyHeuristicType {
        EntropyHeuristicType::Shannon
    }
}

/// Count-based entropy: just use the count of possibilities directly
#[derive(Debug, Clone, Copy)]
pub struct CountEntropyHeuristic;

impl EntropyHeuristic for CountEntropyHeuristic {
    fn calculate_cell_entropy(&self, cell: &BitSlice, _weights: Option<&[f32]>) -> f32 {
        let count = cell.count_ones();
        if count == 0 {
            return -1.0; // Contradiction: negative entropy
        } else if count == 1 {
            return 0.0; // Fully collapsed: zero entropy
        } else {
            // Just return the count (minus 1 to make it start from 1.0)
            return (count - 1) as f32;
        }
    }

    fn get_type(&self) -> EntropyHeuristicType {
        EntropyHeuristicType::Count
    }
}

/// Weighted count-based entropy: uses weights if available
#[derive(Debug, Clone, Copy)]
pub struct WeightedCountEntropyHeuristic;

impl EntropyHeuristic for WeightedCountEntropyHeuristic {
    fn calculate_cell_entropy(&self, cell: &BitSlice, weights: Option<&[f32]>) -> f32 {
        let count = cell.count_ones();
        if count == 0 {
            return -1.0; // Contradiction: negative entropy
        } else if count == 1 {
            return 0.0; // Fully collapsed: zero entropy
        } else if let Some(weights) = weights {
            // Sum the weights of possible states
            let mut sum = 0.0;
            for i in cell.iter_ones() {
                if i < weights.len() {
                    sum += weights[i];
                }
            }
            return sum; // Higher sum = higher entropy
        } else {
            // Fall back to simple count if no weights
            return (count - 1) as f32;
        }
    }

    fn get_type(&self) -> EntropyHeuristicType {
        EntropyHeuristicType::WeightedCount
    }
}

/// Factory function to create entropy heuristic by type
pub fn create_entropy_heuristic(heuristic_type: EntropyHeuristicType) -> Box<dyn EntropyHeuristic> {
    match heuristic_type {
        EntropyHeuristicType::Shannon => Box::new(ShannonEntropyHeuristic),
        EntropyHeuristicType::Count => Box::new(CountEntropyHeuristic),
        EntropyHeuristicType::CountSimple => Box::new(CountEntropyHeuristic), // For now, reuse Count with simple count
        EntropyHeuristicType::WeightedCount => Box::new(WeightedCountEntropyHeuristic),
    }
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

    /// Sets the entropy calculation heuristic to use
    ///
    /// Default implementations will use the specified heuristic if they support it,
    /// or fall back to their default if not supported.
    ///
    /// # Returns
    ///
    /// `bool` - true if the heuristic was applied successfully, false if unsupported
    fn set_entropy_heuristic(&mut self, _heuristic_type: EntropyHeuristicType) -> bool {
        false // Default implementation doesn't support changing heuristic
    }

    /// Gets the current entropy calculation heuristic
    fn get_entropy_heuristic(&self) -> EntropyHeuristicType {
        EntropyHeuristicType::Shannon // Default to Shannon entropy
    }
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

    #[test]
    fn test_create_entropy_heuristic() {
        // Test creating entropy heuristics
        let shannon = create_entropy_heuristic(EntropyHeuristicType::Shannon);
        let count = create_entropy_heuristic(EntropyHeuristicType::Count);
        let count_simple = create_entropy_heuristic(EntropyHeuristicType::CountSimple);
        let weighted = create_entropy_heuristic(EntropyHeuristicType::WeightedCount);

        // Verify the type of each heuristic
        assert_eq!(shannon.get_type(), EntropyHeuristicType::Shannon);
        assert_eq!(count.get_type(), EntropyHeuristicType::Count);

        // For CountSimple we currently reuse the Count implementation
        assert_eq!(count_simple.get_type(), EntropyHeuristicType::Count);

        assert_eq!(weighted.get_type(), EntropyHeuristicType::WeightedCount);
    }

    #[test]
    fn test_entropy_heuristic_calculations() {
        // Test factory function with different heuristic types
        let shannon = create_entropy_heuristic(EntropyHeuristicType::Shannon);
        let count = create_entropy_heuristic(EntropyHeuristicType::Count);
        let count_simple = create_entropy_heuristic(EntropyHeuristicType::CountSimple);
        let weighted = create_entropy_heuristic(EntropyHeuristicType::WeightedCount);

        // Create a test bit vector with 4 possibilities
        let mut bits = bitvec![usize, Lsb0; 0; 8];
        bits.set(0, true);
        bits.set(1, true);
        bits.set(2, true);
        bits.set(3, true);

        // Count the actual number of set bits to adjust expectations
        let actual_count = bits.count_ones();

        // Calculate entropy using each heuristic
        let shannon_entropy = shannon.calculate_cell_entropy(bits.as_bitslice(), None);
        let count_entropy = count.calculate_cell_entropy(bits.as_bitslice(), None);
        let count_simple_entropy = count_simple.calculate_cell_entropy(bits.as_bitslice(), None);
        let weighted_entropy = weighted.calculate_cell_entropy(bits.as_bitslice(), None);

        // Expected values based on actual implementation
        let expected_shannon = f32::log2(actual_count as f32);
        // Count returns (count - 1)
        let expected_count = (actual_count - 1) as f32;
        // CountSimple currently reuses Count implementation
        let expected_count_simple = expected_count;
        // Weighted without weights returns (count - 1)
        let expected_weighted = (actual_count - 1) as f32;

        // Assert that each heuristic calculates entropy correctly
        assert_eq!(shannon_entropy, expected_shannon);
        assert_eq!(count_entropy, expected_count);
        assert_eq!(count_simple_entropy, expected_count_simple);
        assert_eq!(weighted_entropy, expected_weighted);

        // Print values for debugging
        println!(
            "Shannon: {}, Count: {}, CountSimple: {}, Weighted: {}",
            shannon_entropy, count_entropy, count_simple_entropy, weighted_entropy
        );
        println!(
            "Actual count: {}, bits length: {}",
            actual_count,
            bits.len()
        );
    }

    // Add more tests here that verify the *behavior* of entropy calculation
    // and lowest entropy finding, potentially using a test-specific
    // GpuAccelerator instance if integration testing is set up.
}
