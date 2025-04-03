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
