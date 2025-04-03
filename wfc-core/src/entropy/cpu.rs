use crate::entropy::SelectionStrategy; // Correct import path
use crate::{EntropyCalculator, EntropyError, EntropyGrid, PossibilityGrid};
use bitvec::prelude::{BitSlice, Lsb0};
use rand::seq::SliceRandom; // Uncommented
use rand::thread_rng;
use std::sync::Arc;
use wfc_rules::TileSet; // Import Arc // Uncommented

/// CPU implementation of the EntropyCalculator trait.
#[derive(Debug, Clone)] // Cannot be Copy if it holds an Arc
pub struct CpuEntropyCalculator {
    tileset: Arc<TileSet>,       // Store TileSet for weight access
    strategy: SelectionStrategy, // Add strategy field
}

impl CpuEntropyCalculator {
    pub fn new(tileset: Arc<TileSet>, strategy: SelectionStrategy) -> Self {
        Self { tileset, strategy }
    }

    // Helper to calculate Shannon entropy for a single cell's possibilities
    // H = - Sum( P(i) * log2(P(i)) ) where P(i) is probability of tile i
    // Probability P(i) is proportional to weight w(i). P(i) = w(i) / Sum(w)
    // H = log2(Sum(w)) - Sum( w(i) * log2(w(i)) ) / Sum(w)
    fn calculate_cell_entropy(
        &self,
        possibilities: &BitSlice<usize, Lsb0>,
    ) -> Result<f32, EntropyError> {
        let possible_tile_indices: Vec<_> = possibilities.iter_ones().collect();
        let count = possible_tile_indices.len();

        if count == 0 {
            return Ok(f32::NEG_INFINITY); // Contradiction, assign very low entropy
        } else if count == 1 {
            return Ok(0.0); // Fully collapsed, zero entropy
        }

        let mut sum_of_weights = 0.0f32;
        let mut sum_of_weight_log_weight = 0.0f32;

        for ttid in possible_tile_indices {
            let (base_id, _) = self
                .tileset
                .get_base_tile_and_transform(ttid)
                .ok_or_else(|| {
                    EntropyError::Other(format!("Failed to map ttid {} to base tile", ttid))
                })?;
            let weight = self
                .tileset
                .get_weight(base_id)
                .ok_or(EntropyError::InvalidWeight(0, 0, 0))?;

            if weight <= 0.0 {
                return Err(EntropyError::InvalidWeight(0, 0, 0));
            }

            sum_of_weights += weight;
            // Use weight.ln() which is often faster than log2, can convert later if needed,
            // but for comparison, the base doesn't matter.
            sum_of_weight_log_weight += weight * weight.ln();
        }

        if sum_of_weights <= 0.0 {
            // Only possible if all possible tiles had zero/negative weight, which should be an error state
            return Err(EntropyError::InvalidWeight(0, 0, 0));
        }

        // Entropy H = ln(Sum(w)) - Sum(w*ln(w)) / Sum(w)
        let entropy = sum_of_weights.ln() - (sum_of_weight_log_weight / sum_of_weights);

        if entropy.is_nan() {
            Err(EntropyError::NaNResult(0, 0, 0))
        } else if entropy < 0.0 {
            // Due to floating point inaccuracies, entropy might be slightly negative near 0.
            // Clamp it to 0.0 for collapsed states.
            Ok(0.0)
        } else {
            Ok(entropy)
        }
    }
}

impl EntropyCalculator for CpuEntropyCalculator {
    fn calculate_entropy(&self, grid: &PossibilityGrid) -> Result<EntropyGrid, EntropyError> {
        let mut entropy_grid = EntropyGrid::new(grid.width, grid.height, grid.depth);
        for z in 0..grid.depth {
            for y in 0..grid.height {
                for x in 0..grid.width {
                    let possibilities = grid
                        .get(x, y, z)
                        .ok_or(EntropyError::GridAccessError(x, y, z))?;

                    // Use .as_bitslice() to convert &BitVec to &BitSlice
                    let entropy_val = self.calculate_cell_entropy(possibilities.as_bitslice())?;

                    let entropy_cell = entropy_grid.get_mut(x, y, z).ok_or_else(|| {
                        EntropyError::Other(format!(
                            "Entropy grid access failed at ({},{},{})",
                            x, y, z
                        ))
                    })?;
                    *entropy_cell = entropy_val;
                }
            }
        }
        Ok(entropy_grid)
    }

    fn select_lowest_entropy_cell(
        &self,
        entropy_grid: &EntropyGrid,
    ) -> Option<(usize, usize, usize)> {
        let mut min_entropy = f32::MAX;
        let mut lowest_cells = Vec::new(); // Store all cells with the current min entropy

        // First pass: Find the minimum positive entropy
        for z in 0..entropy_grid.depth {
            for y in 0..entropy_grid.height {
                for x in 0..entropy_grid.width {
                    if let Some(&entropy) = entropy_grid.get(x, y, z) {
                        // Consider only positive entropy values
                        if entropy > 0.0 && entropy < min_entropy {
                            min_entropy = entropy;
                        }
                    }
                }
            }
        }

        // Check if any positive entropy was found
        if min_entropy == f32::MAX {
            return None; // Grid is fully collapsed, empty, or contains only contradictions
        }

        // Second pass: Collect all cells with the minimum entropy (within a small tolerance for floats)
        let tolerance = 1e-6; // Adjust tolerance as needed
        for z in 0..entropy_grid.depth {
            for y in 0..entropy_grid.height {
                for x in 0..entropy_grid.width {
                    if let Some(&entropy) = entropy_grid.get(x, y, z) {
                        if (entropy - min_entropy).abs() < tolerance {
                            lowest_cells.push((x, y, z));
                        }
                    }
                }
            }
        }

        // Select based on strategy
        match self.strategy {
            SelectionStrategy::FirstMinimum => {
                // Return the first one found (consistent with previous behavior if iteration order is stable)
                lowest_cells.first().copied()
            }
            SelectionStrategy::RandomLowest => {
                // Choose randomly from the collected cells
                let mut rng = thread_rng();
                lowest_cells.choose(&mut rng).copied()
            } // TODO: Add HilbertCurve case
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::PossibilityGrid;
    use bitvec::prelude::bitvec;
    use std::sync::Arc;
    use wfc_rules::{TileSet, TileSetError, Transformation};

    // Helper to create a simple TileSet for testing
    fn create_test_tileset(weights: Vec<f32>) -> Result<TileSet, TileSetError> {
        let num_base_tiles = weights.len();
        let allowed_transforms = vec![vec![Transformation::Identity]; num_base_tiles];
        TileSet::new(weights, allowed_transforms)
    }

    #[test]
    fn test_calculate_cell_entropy_multiple_possibilities() {
        let tileset = Arc::new(create_test_tileset(vec![1.0, 1.0, 1.0]).unwrap()); // 3 tiles, equal weight
        let calculator = CpuEntropyCalculator::new(tileset, SelectionStrategy::FirstMinimum);
        let possibilities = bitvec![usize, Lsb0; 1, 1, 1]; // All 3 possible
        let entropy = calculator.calculate_cell_entropy(&possibilities);
        assert!(entropy.is_ok());
        // Expected: log2(3) approx 1.58496
        // Using ln(): ln(3) approx 1.0986
        assert!((entropy.unwrap() - 1.0986).abs() < 1e-4);

        let possibilities_two = bitvec![usize, Lsb0; 1, 0, 1]; // Tiles 0 and 2 possible
        let entropy_two = calculator.calculate_cell_entropy(&possibilities_two);
        assert!(entropy_two.is_ok());
        // Expected: log2(2) = 1.0
        // Using ln(): ln(2) approx 0.6931
        assert!((entropy_two.unwrap() - 0.6931).abs() < 1e-4);
    }

    #[test]
    fn test_calculate_cell_entropy_collapsed() {
        let tileset = Arc::new(create_test_tileset(vec![1.0, 2.0]).unwrap());
        let calculator = CpuEntropyCalculator::new(tileset, SelectionStrategy::FirstMinimum);
        let possibilities = bitvec![usize, Lsb0; 0, 1]; // Only Tile 1 possible
        let entropy = calculator.calculate_cell_entropy(&possibilities);
        assert!(entropy.is_ok());
        assert_eq!(entropy.unwrap(), 0.0);
    }

    #[test]
    fn test_calculate_cell_entropy_contradiction() {
        let tileset = Arc::new(create_test_tileset(vec![1.0, 1.0]).unwrap());
        let calculator = CpuEntropyCalculator::new(tileset, SelectionStrategy::FirstMinimum);
        let possibilities = bitvec![usize, Lsb0; 0, 0]; // No possibilities
        let entropy_result = calculator.calculate_cell_entropy(&possibilities);
        assert!(entropy_result.is_ok());
        let entropy_value = entropy_result.unwrap();
        assert!(entropy_value.is_infinite() && entropy_value.is_sign_negative());
    }

    #[test]
    fn test_calculate_cell_entropy_weighted() {
        let tileset = Arc::new(create_test_tileset(vec![1.0, 3.0]).unwrap()); // Tile 1 is 3x more likely
        let calculator = CpuEntropyCalculator::new(tileset, SelectionStrategy::FirstMinimum);
        let possibilities = bitvec![usize, Lsb0; 1, 1]; // Both possible
        let entropy = calculator.calculate_cell_entropy(&possibilities);
        assert!(entropy.is_ok());
        // Expected: - (1/4 * log2(1/4) + 3/4 * log2(3/4)) approx 0.81128
        // Using ln(): ln(1+3) - (1*ln(1) + 3*ln(3))/(1+3) = ln(4) - (3*ln(3))/4 approx 0.5623
        assert!((entropy.unwrap() - 0.5623).abs() < 1e-4);
    }

    #[test]
    fn test_calculate_entropy_full_grid() {
        let tileset = Arc::new(create_test_tileset(vec![1.0, 1.0]).unwrap());
        let calculator =
            CpuEntropyCalculator::new(tileset.clone(), SelectionStrategy::FirstMinimum);
        let mut grid = PossibilityGrid::new(2, 1, 1, 2);
        *grid.get_mut(0, 0, 0).unwrap() = bitvec![usize, Lsb0; 1, 0]; // Collapsed
        *grid.get_mut(1, 0, 0).unwrap() = bitvec![usize, Lsb0; 1, 1]; // Both possible

        let entropy_grid_result = calculator.calculate_entropy(&grid);
        assert!(entropy_grid_result.is_ok());
        let entropy_grid = entropy_grid_result.unwrap();
        assert_eq!(entropy_grid.get(0, 0, 0), Some(&0.0));
        assert!((entropy_grid.get(1, 0, 0).unwrap() - 0.6931).abs() < 1e-4); // ln(2)
    }

    #[test]
    fn test_select_lowest_first_minimum() {
        let tileset = Arc::new(create_test_tileset(vec![1.0; 3]).unwrap());
        let calculator = CpuEntropyCalculator::new(tileset, SelectionStrategy::FirstMinimum);
        let mut entropy_grid = EntropyGrid::new(2, 2, 1);
        *entropy_grid.get_mut(0, 0, 0).unwrap() = 0.5;
        *entropy_grid.get_mut(1, 0, 0).unwrap() = 0.2; // Lowest
        *entropy_grid.get_mut(0, 1, 0).unwrap() = 0.2; // Also lowest
        *entropy_grid.get_mut(1, 1, 0).unwrap() = 0.8;

        let selection = calculator.select_lowest_entropy_cell(&entropy_grid);
        // Iteration order is likely X, Y, Z. So (1,0,0) should be found first.
        assert_eq!(selection, Some((1, 0, 0)));
    }

    #[test]
    fn test_select_lowest_random() {
        let tileset = Arc::new(create_test_tileset(vec![1.0; 3]).unwrap());
        let calculator = CpuEntropyCalculator::new(tileset, SelectionStrategy::RandomLowest);
        let mut entropy_grid = EntropyGrid::new(2, 2, 1);
        *entropy_grid.get_mut(0, 0, 0).unwrap() = 0.5;
        *entropy_grid.get_mut(1, 0, 0).unwrap() = 0.2; // Lowest
        *entropy_grid.get_mut(0, 1, 0).unwrap() = 0.2; // Also lowest
        *entropy_grid.get_mut(1, 1, 0).unwrap() = 0.8;

        // Run multiple times to increase chance of seeing both lowest cells selected
        let mut seen_1_0_0 = false;
        let mut seen_0_1_0 = false;
        for _ in 0..100 {
            let selection = calculator.select_lowest_entropy_cell(&entropy_grid);
            assert!(selection.is_some());
            let coords = selection.unwrap();
            assert!(coords == (1, 0, 0) || coords == (0, 1, 0));
            if coords == (1, 0, 0) {
                seen_1_0_0 = true;
            }
            if coords == (0, 1, 0) {
                seen_0_1_0 = true;
            }
            if seen_1_0_0 && seen_0_1_0 {
                break;
            }
        }
        assert!(
            seen_1_0_0 && seen_0_1_0,
            "RandomLowest did not select both minimum cells over 100 trials"
        );
    }

    #[test]
    fn test_select_lowest_all_collapsed_or_contradiction() {
        let tileset = Arc::new(create_test_tileset(vec![1.0; 2]).unwrap());
        let calculator = CpuEntropyCalculator::new(tileset, SelectionStrategy::FirstMinimum);
        let mut entropy_grid = EntropyGrid::new(2, 1, 1);
        *entropy_grid.get_mut(0, 0, 0).unwrap() = 0.0; // Collapsed
        *entropy_grid.get_mut(1, 0, 0).unwrap() = f32::NEG_INFINITY; // Contradiction

        let selection = calculator.select_lowest_entropy_cell(&entropy_grid);
        assert_eq!(selection, None);

        // Test with only collapsed cells
        *entropy_grid.get_mut(1, 0, 0).unwrap() = 0.0;
        let selection_all_zero = calculator.select_lowest_entropy_cell(&entropy_grid);
        assert_eq!(selection_all_zero, None);
    }
}
