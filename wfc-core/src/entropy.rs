use crate::grid::{EntropyGrid, PossibilityGrid};
use float_ord::FloatOrd;
use rayon::prelude::*;

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

/// A basic, parallel CPU implementation of the `EntropyCalculator` trait.
///
/// This implementation uses `rayon` for parallel computation of entropy across grid cells.
/// Note: Currently uses a simple heuristic for entropy (count of possibilities).
#[derive(Debug, Clone)]
pub struct CpuEntropyCalculator;

impl CpuEntropyCalculator {
    /// Creates a new `CpuEntropyCalculator`.
    pub fn new() -> Self {
        Self
    }
}

// Implement Default as suggested by clippy
impl Default for CpuEntropyCalculator {
    /// Creates a default `CpuEntropyCalculator`.
    fn default() -> Self {
        Self::new()
    }
}

impl EntropyCalculator for CpuEntropyCalculator {
    /// Calculates entropy for each cell based on the number of remaining possibilities.
    ///
    /// Uses `rayon` to parallelize the calculation across cells.
    /// Cells with 1 or fewer possibilities (already collapsed) are assigned an entropy of 0.0.
    /// For cells with > 1 possibility, the entropy is currently calculated simply as the
    /// count of set bits (number of possible tiles).
    ///
    /// TODO: Implement a more sophisticated entropy calculation (e.g., Shannon entropy based on tile weights).
    fn calculate_entropy(&self, grid: &PossibilityGrid) -> EntropyGrid {
        let mut entropy_grid = EntropyGrid::new(grid.width, grid.height, grid.depth);
        let width = grid.width;
        let height = grid.height;
        // let num_tiles = grid.num_tiles(); // num_tiles is not strictly needed for count_ones

        // Parallel calculation using rayon - iterate over the output entropy grid's data
        entropy_grid
            .data // Access data of EntropyGrid (which is Grid<f32>, so data is Vec<f32>)
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, entropy_cell)| {
                // Calculate 3D coordinates from 1D index
                let z = index / (width * height);
                let temp = index % (width * height);
                let y = temp / width;
                let x = temp % width;

                // Get the corresponding possibility cell using public API
                if let Some(possibility_cell) = grid.get(x, y, z) {
                    // Ensure the bitvec length matches num_tiles if needed for operations, though count_ones is fine.
                    // assert_eq!(possibility_cell.len(), num_tiles);
                    let possibilities_count = possibility_cell.count_ones();
                    *entropy_cell = if possibilities_count <= 1 {
                        0.0
                    } else {
                        // More sophisticated entropy calculations can go here.
                        // For now, just use the count as a proxy.
                        // Adding a small amount of noise can help break ties.
                        // TODO: Replace with proper Shannon entropy or similar.
                        possibilities_count as f32 // + small random noise?
                    };
                } else {
                    // Should not happen if grids are consistent, but handle defensively
                    *entropy_cell = f32::NAN; // Or some other error indicator
                }
            });

        entropy_grid
    }

    /// Finds the coordinates of the cell with the minimum positive entropy using a parallel search.
    ///
    /// Uses `rayon` to iterate over the `entropy_grid` in parallel.
    /// Filters out cells with entropy <= 0.0.
    /// Uses `float_ord::FloatOrd` for reliable comparison of `f32` entropy values.
    /// Finds the index of the minimum element and converts it back to `(x, y, z)` coordinates.
    ///
    /// Returns `None` if no cells with positive entropy exist.
    fn find_lowest_entropy(&self, entropy_grid: &EntropyGrid) -> Option<(usize, usize, usize)> {
        // Use parallel iteration and reduction to find the minimum non-zero entropy.
        // We need to find the *index* of the minimum element, not just the value.
        // We ignore cells with entropy 0.0 (already collapsed or contradiction).
        // Need to handle potential floating point precision issues and NaNs if using f32 directly for min.
        // Using FloatOrd helps here.

        let result = entropy_grid
            .data
            .par_iter()
            .enumerate()
            .filter_map(|(index, &entropy)| {
                if entropy > 0.0 {
                    // Wrap f32 in FloatOrd for comparison
                    Some((index, FloatOrd(entropy)))
                } else {
                    None // Ignore zero entropy cells
                }
            })
            .min_by_key(|&(_, entropy)| entropy); // Find the minimum based on FloatOrd<f32>

        result.map(|(index, _)| {
            // Convert the 1D index back to 3D coordinates
            let z = index / (entropy_grid.width * entropy_grid.height);
            let temp = index % (entropy_grid.width * entropy_grid.height);
            let y = temp / entropy_grid.width;
            let x = temp % entropy_grid.width;
            (x, y, z)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{EntropyGrid, PossibilityGrid};
    use bitvec::prelude::*;

    #[test]
    fn test_calculate_entropy_basic() {
        let calculator = CpuEntropyCalculator::new();
        let mut grid = PossibilityGrid::new(2, 1, 1, 3); // 2 cells, 3 tiles

        // Initial state: all possible [1, 1, 1]
        let entropy_grid = calculator.calculate_entropy(&grid);
        assert_eq!(entropy_grid.get(0, 0, 0), Some(&3.0));
        assert_eq!(entropy_grid.get(1, 0, 0), Some(&3.0));

        // Collapse one cell: [1, 0, 0]
        *grid.get_mut(0, 0, 0).unwrap() = bitvec![1, 0, 0];
        let entropy_grid = calculator.calculate_entropy(&grid);
        assert_eq!(entropy_grid.get(0, 0, 0), Some(&0.0)); // Entropy becomes 0 for collapsed
        assert_eq!(entropy_grid.get(1, 0, 0), Some(&3.0));

        // Partially collapse other cell: [0, 1, 1]
        *grid.get_mut(1, 0, 0).unwrap() = bitvec![0, 1, 1];
        let entropy_grid = calculator.calculate_entropy(&grid);
        assert_eq!(entropy_grid.get(0, 0, 0), Some(&0.0));
        assert_eq!(entropy_grid.get(1, 0, 0), Some(&2.0)); // Entropy is count > 1

        // Collapse other cell: [0, 1, 0]
        *grid.get_mut(1, 0, 0).unwrap() = bitvec![0, 1, 0];
        let entropy_grid = calculator.calculate_entropy(&grid);
        assert_eq!(entropy_grid.get(0, 0, 0), Some(&0.0));
        assert_eq!(entropy_grid.get(1, 0, 0), Some(&0.0));
    }

    #[test]
    fn test_calculate_entropy_empty_grid() {
        let calculator = CpuEntropyCalculator::new();
        let grid = PossibilityGrid::new(0, 1, 1, 3);
        let entropy_grid = calculator.calculate_entropy(&grid);
        assert_eq!(entropy_grid.width, 0);
        assert!(entropy_grid.data.is_empty());
    }

    #[test]
    fn test_find_lowest_entropy_basic() {
        let calculator = CpuEntropyCalculator::new();
        let mut entropy_grid = EntropyGrid::new(2, 2, 1);
        // Set some entropy values
        *entropy_grid.get_mut(0, 0, 0).unwrap() = 3.0;
        *entropy_grid.get_mut(1, 0, 0).unwrap() = 0.0; // Collapsed
        *entropy_grid.get_mut(0, 1, 0).unwrap() = 2.0; // Lowest positive
        *entropy_grid.get_mut(1, 1, 0).unwrap() = 4.0;

        let lowest = calculator.find_lowest_entropy(&entropy_grid);
        assert_eq!(lowest, Some((0, 1, 0)));
    }

    #[test]
    fn test_find_lowest_entropy_tie() {
        let calculator = CpuEntropyCalculator::new();
        let mut entropy_grid = EntropyGrid::new(2, 1, 1);
        *entropy_grid.get_mut(0, 0, 0).unwrap() = 2.0;
        *entropy_grid.get_mut(1, 0, 0).unwrap() = 2.0;

        let lowest = calculator.find_lowest_entropy(&entropy_grid);
        // Should return one of the coordinates with the lowest entropy
        assert!(lowest == Some((0, 0, 0)) || lowest == Some((1, 0, 0)));
    }

    #[test]
    fn test_find_lowest_entropy_all_zero() {
        let calculator = CpuEntropyCalculator::new();
        let mut entropy_grid = EntropyGrid::new(2, 1, 1);
        *entropy_grid.get_mut(0, 0, 0).unwrap() = 0.0;
        *entropy_grid.get_mut(1, 0, 0).unwrap() = 0.0;

        let lowest = calculator.find_lowest_entropy(&entropy_grid);
        assert_eq!(lowest, None); // No positive entropy found
    }

    #[test]
    fn test_find_lowest_entropy_single_cell() {
        let calculator = CpuEntropyCalculator::new();
        let mut entropy_grid = EntropyGrid::new(1, 1, 1);
        *entropy_grid.get_mut(0, 0, 0).unwrap() = 5.0;
        let lowest = calculator.find_lowest_entropy(&entropy_grid);
        assert_eq!(lowest, Some((0, 0, 0)));

        *entropy_grid.get_mut(0, 0, 0).unwrap() = 0.0;
        let lowest = calculator.find_lowest_entropy(&entropy_grid);
        assert_eq!(lowest, None);
    }

    #[test]
    fn test_find_lowest_entropy_empty_grid() {
        let calculator = CpuEntropyCalculator::new();
        let entropy_grid = EntropyGrid::new(0, 0, 0);
        let lowest = calculator.find_lowest_entropy(&entropy_grid);
        assert_eq!(lowest, None);
    }
}
