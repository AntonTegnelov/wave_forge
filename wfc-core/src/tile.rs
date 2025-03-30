#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileId(pub usize); // Simple wrapper for now

#[derive(Debug, Clone)]
pub struct TileSet {
    /// Relative weights for each tile. Higher weight means higher probability of being chosen.
    /// The index corresponds to the TileId(index).
    pub weights: Vec<f32>,
    // pub symmetries: Vec<SymmetryInfo>, // Future use
}

impl TileSet {
    /// Creates a new TileSet with the given weights.
    ///
    /// # Panics
    /// Panics if the weights vector is empty or contains non-positive values.
    pub fn new(weights: Vec<f32>) -> Self {
        assert!(!weights.is_empty(), "TileSet weights cannot be empty.");
        assert!(
            weights.iter().all(|&w| w > 0.0),
            "TileSet weights must be positive."
        );
        Self { weights }
    }

    /// Gets the weight for a specific TileId.
    /// Returns None if the TileId is out of bounds.
    pub fn get_weight(&self, tile_id: TileId) -> Option<f32> {
        self.weights.get(tile_id.0).copied()
    }
}
