use thiserror::Error;

/// Represents a unique identifier for a tile.
///
/// Often used as an index into tile-related data structures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileId(pub usize); // Simple wrapper for now

/// Represents a symmetry transformation applied to a tile.
/// Currently supports rotations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Transformation {
    Identity,
    Rot90,
    Rot180,
    Rot270,
    // Future: FlipX, FlipY, etc.
}

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
    /// Error indicating mismatch between weights and transformation list lengths.
    #[error(
        "Number of allowed_transformations lists ({0}) does not match number of weights ({1})."
    )]
    TransformationWeightMismatch(usize, usize),
    /// Error indicating an empty list of allowed transformations for a tile.
    /// Each tile must at least have the Identity transformation.
    #[error(
        "Tile {0} has an empty list of allowed transformations. Must include at least Identity."
    )]
    EmptyTransformations(usize),
    /// Error indicating the Identity transformation is missing for a tile.
    #[error("Tile {0}'s allowed transformations list does not include Transformation::Identity.")]
    MissingIdentityTransformation(usize),
}

/// Stores information about the set of tiles used in WFC.
///
/// Includes the weights associated with each tile and the allowed symmetry
/// transformations for each tile.
#[derive(Debug, Clone)]
pub struct TileSet {
    /// Relative weights for each tile. Higher weight means higher probability of being chosen.
    /// The index corresponds to the `TileId(index)`. Assumed to be the weight of the base tile.
    pub weights: Vec<f32>,
    /// Allowed symmetry transformations for each base tile.
    /// The outer index corresponds to the `TileId(index)`.
    /// Each inner list must contain at least `Transformation::Identity`.
    pub allowed_transformations: Vec<Vec<Transformation>>,
}

impl TileSet {
    /// Creates a new `TileSet` with the given weights and allowed transformations.
    ///
    /// Validates:
    /// - Weights vector is not empty.
    /// - All weights are positive.
    /// - The number of transformation lists matches the number of weights.
    /// - Each transformation list is non-empty and contains `Transformation::Identity`.
    ///
    /// # Arguments
    ///
    /// * `weights` - Relative weights for each base tile.
    /// * `allowed_transformations` - A list where each element is a list of allowed
    ///   `Transformation`s for the corresponding base tile.
    ///
    /// # Errors
    ///
    /// Returns `TileSetError` variants for validation failures.
    pub fn new(
        weights: Vec<f32>,
        allowed_transformations: Vec<Vec<Transformation>>,
    ) -> Result<Self, TileSetError> {
        if weights.is_empty() {
            return Err(TileSetError::EmptyWeights);
        }
        if weights.len() != allowed_transformations.len() {
            return Err(TileSetError::TransformationWeightMismatch(
                allowed_transformations.len(),
                weights.len(),
            ));
        }

        // Check weights and transformations
        for (index, (weight, transformations)) in weights
            .iter()
            .zip(allowed_transformations.iter())
            .enumerate()
        {
            // Check weight
            if *weight <= 0.0 {
                return Err(TileSetError::NonPositiveWeight(index, weight.to_string()));
            }
            // Check transformations
            if transformations.is_empty() {
                return Err(TileSetError::EmptyTransformations(index));
            }
            if !transformations.contains(&Transformation::Identity) {
                return Err(TileSetError::MissingIdentityTransformation(index));
            }
            // Future: Add validation for duplicate transformations?
        }

        // All checks passed
        Ok(Self {
            weights,
            allowed_transformations,
        })
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

#[cfg(test)]
mod tests {
    use super::*; // Import items from parent module

    #[test]
    fn tileset_new_valid() {
        let weights = vec![1.0, 2.0];
        let transformations = vec![
            vec![Transformation::Identity],
            vec![Transformation::Identity, Transformation::Rot90],
        ];
        let tileset = TileSet::new(weights.clone(), transformations.clone());
        assert!(tileset.is_ok());
        let ts = tileset.unwrap();
        assert_eq!(ts.weights, weights);
        assert_eq!(ts.allowed_transformations, transformations);
    }

    #[test]
    fn tileset_new_empty_weights() {
        let weights = vec![];
        let transformations = vec![];
        let result = TileSet::new(weights, transformations);
        assert!(matches!(result, Err(TileSetError::EmptyWeights)));
    }

    #[test]
    fn tileset_new_non_positive_weight() {
        let weights = vec![1.0, 0.0];
        let transformations = vec![
            vec![Transformation::Identity],
            vec![Transformation::Identity],
        ];
        let result = TileSet::new(weights, transformations);
        assert!(matches!(result, Err(TileSetError::NonPositiveWeight(1, _))));
    }

    #[test]
    fn tileset_new_weight_transformation_mismatch() {
        let weights = vec![1.0];
        let transformations = vec![
            vec![Transformation::Identity],
            vec![Transformation::Identity],
        ]; // Mismatch
        let result = TileSet::new(weights, transformations);
        assert!(matches!(
            result,
            Err(TileSetError::TransformationWeightMismatch(2, 1))
        ));
    }

    #[test]
    fn tileset_new_empty_transformations_for_tile() {
        let weights = vec![1.0, 1.0];
        let transformations = vec![vec![Transformation::Identity], vec![]]; // Empty list for second tile
        let result = TileSet::new(weights, transformations);
        assert!(matches!(result, Err(TileSetError::EmptyTransformations(1))));
    }

    #[test]
    fn tileset_new_missing_identity_transformation() {
        let weights = vec![1.0, 1.0];
        let transformations = vec![vec![Transformation::Identity], vec![Transformation::Rot90]]; // Missing Identity
        let result = TileSet::new(weights, transformations);
        assert!(matches!(
            result,
            Err(TileSetError::MissingIdentityTransformation(1))
        ));
    }

    #[test]
    fn adjacency_check_valid() {
        let rules = AdjacencyRules::new(
            3,
            2,
            vec![
                // 3 tiles, 2 axes
                // Axis 0
                // T0 -> T0, T1, T2
                true, true, false, // T1 -> T0, T1, T2
                true, true, true, // T2 -> T0, T1, T2
                false, true, true, // Axis 1
                // T0 -> T0, T1, T2
                true, false, false, // T1 -> T0, T1, T2
                false, true, false, // T2 -> T0, T1, T2
                false, false, true,
            ],
        );

        assert!(rules.check(TileId(0), TileId(1), 0)); // Axis 0: T0 -> T1 (true)
        assert!(!rules.check(TileId(0), TileId(2), 0)); // Axis 0: T0 -> T2 (false)
        assert!(rules.check(TileId(1), TileId(2), 0)); // Axis 0: T1 -> T2 (true)

        assert!(!rules.check(TileId(0), TileId(1), 1)); // Axis 1: T0 -> T1 (false)
        assert!(rules.check(TileId(2), TileId(2), 1)); // Axis 1: T2 -> T2 (true)
    }

    #[test]
    fn adjacency_check_out_of_bounds() {
        let rules = AdjacencyRules::new(2, 1, vec![true, false, false, true]); // 2 tiles, 1 axis

        assert!(!rules.check(TileId(0), TileId(2), 0)); // Tile 2 out of bounds
        assert!(!rules.check(TileId(2), TileId(0), 0)); // Tile 2 out of bounds
        assert!(!rules.check(TileId(0), TileId(0), 1)); // Axis 1 out of bounds
    }

    // Add more tests for AdjacencyRules if needed
}
