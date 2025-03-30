use crate::grid::{EntropyGrid, PossibilityGrid};
use float_ord::FloatOrd;
use rayon::prelude::*;

pub trait EntropyCalculator {
    #[must_use]
    fn calculate_entropy(&self, grid: &PossibilityGrid) -> EntropyGrid;

    #[must_use]
    fn find_lowest_entropy(&self, entropy_grid: &EntropyGrid) -> Option<(usize, usize, usize)>;
}

// Basic CPU implementation
#[derive(Debug, Clone)]
pub struct CpuEntropyCalculator;

impl CpuEntropyCalculator {
    pub fn new() -> Self {
        Self
    }
}

// Implement Default as suggested by clippy
impl Default for CpuEntropyCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl EntropyCalculator for CpuEntropyCalculator {
    fn calculate_entropy(&self, grid: &PossibilityGrid) -> EntropyGrid {
        let mut entropy_grid = EntropyGrid::new(grid.width, grid.height, grid.depth);

        // Parallel calculation using rayon
        entropy_grid
            .data
            .par_iter_mut() // Parallel mutable iterator over output entropy data
            .zip(grid.data.par_iter()) // Zip with parallel iterator over input possibility data
            .for_each(|(entropy_cell, possibility_cell)| {
                // Simple placeholder entropy: number of possibilities (set bits)
                // Lower non-zero count = lower entropy (more constrained)
                // Cells with 0 or 1 possibility have entropy 0.0 (already collapsed or contradiction)
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
            });

        entropy_grid
    }

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
