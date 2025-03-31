use thiserror::Error;

/// Represents a unique identifier for a tile.
///
/// Often used as an index into tile-related data structures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileId(pub usize); // Simple wrapper for now

/// Errors that can occur during TileSet creation or validation.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum TileSetError {
    /// Error indicating the provided list of weights was empty.
    #[error("TileSet weights cannot be empty.")]
    EmptyWeights,
    /// Error indicating a non-positive weight was found for a tile.
    /// Weights must be > 0.
    #[error("TileSet weights must be positive. Found non-positive weight at index {0}: {1}")]
    NonPositiveWeight(usize, String),
}

/// Stores information about the set of tiles used in WFC.
///
/// Currently, this primarily includes the weights associated with each tile,
/// which influence the probability of choosing a tile during collapse.
#[derive(Debug, Clone)]
pub struct TileSet {
    /// Relative weights for each tile. Higher weight means higher probability of being chosen.
    /// The index corresponds to the `TileId(index)`.
    pub weights: Vec<f32>,
    // pub symmetries: Vec<SymmetryInfo>, // Future use
}

impl TileSet {
    /// Creates a new `TileSet` with the given weights.
    ///
    /// Validates that the weights vector is not empty and all weights are positive.
    ///
    /// # Errors
    ///
    /// Returns `TileSetError::EmptyWeights` if `weights` is empty.
    /// Returns `TileSetError::NonPositiveWeight` if any weight is `<= 0.0`.
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

    /// Gets the weight for a specific `TileId`.
    ///
    /// Returns `None` if the `TileId` is out of bounds.
    pub fn get_weight(&self, tile_id: TileId) -> Option<f32> {
        self.weights.get(tile_id.0).copied()
    }
}
