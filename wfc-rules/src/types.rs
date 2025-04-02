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

/// Represents adjacency rules between tiles for different axes.
///
/// Stores the rules efficiently in a flattened boolean vector for fast lookup
/// and easy transfer to GPU buffers.
/// The indexing scheme assumes `allowed[axis][tile1][tile2]` layout.
#[derive(Debug, Clone)]
pub struct AdjacencyRules {
    num_tiles: usize,
    num_axes: usize,
    /// Flattened vector storing allowed adjacencies.
    /// Indexing: `axis * num_tiles * num_tiles + tile1.0 * num_tiles + tile2.0`
    allowed: Vec<bool>,
}

impl AdjacencyRules {
    /// Creates new `AdjacencyRules`.
    ///
    /// Initializes the rules based on the provided `allowed` vector,
    /// which must be pre-flattened according to the scheme:
    /// `allowed[axis * num_tiles * num_tiles + tile1.0 * num_tiles + tile2.0]`
    ///
    /// # Arguments
    ///
    /// * `num_tiles` - The total number of unique tile types.
    /// * `num_axes` - The number of axes (directions) rules are defined for (e.g., 6 for 3D).
    /// * `allowed` - The flattened boolean vector representing allowed adjacencies.
    ///
    /// # Panics
    ///
    /// Panics if the length of `allowed` is not equal to `num_axes * num_tiles * num_tiles`.
    pub fn new(num_tiles: usize, num_axes: usize, allowed: Vec<bool>) -> Self {
        assert_eq!(
            allowed.len(),
            num_axes * num_tiles * num_tiles,
            "Provided 'allowed' vector has incorrect size."
        );
        Self {
            num_tiles,
            num_axes,
            allowed,
        }
    }

    /// Gets the number of different tile types these rules apply to.
    pub fn num_tiles(&self) -> usize {
        self.num_tiles
    }

    /// Gets the number of axes/directions these rules are defined for (e.g., 6 for 3D +/- X/Y/Z).
    pub fn num_axes(&self) -> usize {
        self.num_axes
    }

    /// Provides read-only access to the internal flattened boolean vector representing allowed adjacencies.
    ///
    /// This is primarily intended for scenarios like GPU buffer packing where direct access to the raw rule data is needed.
    pub fn get_allowed_rules(&self) -> &Vec<bool> {
        &self.allowed
    }

    /// Checks if `tile2` is allowed to be placed adjacent to `tile1` along the specified `axis`.
    ///
    /// Performs bounds checks internally and returns `false` if indices are out of range.
    /// Uses inline attribute for potential performance optimization in tight loops.
    #[inline]
    pub fn check(&self, tile1: TileId, tile2: TileId, axis: usize) -> bool {
        // Use checked indexing instead of asserts to avoid panics in release builds if used incorrectly.
        if tile1.0 >= self.num_tiles || tile2.0 >= self.num_tiles || axis >= self.num_axes {
            // Consider logging a warning here in debug builds?
            return false; // Treat out-of-bounds as disallowed.
        }

        let index = axis * self.num_tiles * self.num_tiles + tile1.0 * self.num_tiles + tile2.0;

        // Use .get() for safe access to the vector.
        *self.allowed.get(index).unwrap_or(&false)
    }
}
