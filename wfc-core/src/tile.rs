#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileId(pub usize); // Simple wrapper for now

use crate::TileSetError; // Import the new error type

#[derive(Debug, Clone)]
pub struct TileSet {
    /// Relative weights for each tile. Higher weight means higher probability of being chosen.
    /// The index corresponds to the TileId(index).
    pub weights: Vec<f32>,
    // pub symmetries: Vec<SymmetryInfo>, // Future use
}

impl TileSet {
    /// Creates a new TileSet with the given weights.
    /// Returns TileSetError if validation fails.
    pub fn new(weights: Vec<f32>) -> Result<Self, TileSetError> {
        if weights.is_empty() {
            return Err(TileSetError::EmptyWeights);
        }
        // Check for non-positive weights
        for (index, &weight) in weights.iter().enumerate() {
            if weight <= 0.0 {
                return Err(TileSetError::NonPositiveWeight(index, weight.to_string()));
            }
        }
        // All checks passed
        Ok(Self { weights })
    }

    /// Gets the weight for a specific TileId.
    /// Returns None if the TileId is out of bounds.
    pub fn get_weight(&self, tile_id: TileId) -> Option<f32> {
        self.weights.get(tile_id.0).copied()
    }
}
