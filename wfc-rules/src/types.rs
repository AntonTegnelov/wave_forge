use std::collections::HashMap;
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

impl Transformation {
    /// Calculates the resulting axis index after applying the transformation.
    /// Assumes a standard 3D axis convention:
    /// 0: +X, 1: -X, 2: +Y, 3: -Y, 4: +Z, 5: -Z
    /// Only handles Z-axis rotations currently.
    /// Panics if axis index is out of bounds (0-5).
    pub fn transform_axis(self, axis: usize) -> usize {
        match self {
            Transformation::Identity => axis, // No change
            Transformation::Rot90 => match axis {
                0 => 2,        // +X -> +Y
                1 => 3,        // -X -> -Y
                2 => 1,        // +Y -> -X
                3 => 0,        // -Y -> +X
                4 | 5 => axis, // Z axes unchanged
                _ => panic!("Invalid axis index: {}", axis),
            },
            Transformation::Rot180 => match axis {
                0 => 1,        // +X -> -X
                1 => 0,        // -X -> +X
                2 => 3,        // +Y -> -Y
                3 => 2,        // -Y -> +Y
                4 | 5 => axis, // Z axes unchanged
                _ => panic!("Invalid axis index: {}", axis),
            },
            Transformation::Rot270 => match axis {
                0 => 3,        // +X -> -Y
                1 => 2,        // -X -> +Y
                2 => 0,        // +Y -> +X
                3 => 1,        // -Y -> -X
                4 | 5 => axis, // Z axes unchanged
                _ => panic!("Invalid axis index: {}", axis),
            },
            // Add cases for FlipX, FlipY etc. later
        }
    }

    /// Returns the inverse transformation.
    pub fn inverse(self) -> Transformation {
        match self {
            Transformation::Identity => Transformation::Identity,
            Transformation::Rot90 => Transformation::Rot270,
            Transformation::Rot180 => Transformation::Rot180,
            Transformation::Rot270 => Transformation::Rot90,
            // Add inverses for other transformations later
        }
    }

    /// Combines this transformation with another (applies other then self).
    /// This is equivalent to matrix multiplication order: self * other.
    pub fn combine(self, other: Transformation) -> Transformation {
        // For rotations, this is simple addition modulo 4
        // Identity=0, Rot90=1, Rot180=2, Rot270=3
        let self_val = self as usize;
        let other_val = other as usize;
        let combined_val = (self_val + other_val) % 4;
        match combined_val {
            0 => Transformation::Identity,
            1 => Transformation::Rot90,
            2 => Transformation::Rot180,
            3 => Transformation::Rot270,
            _ => unreachable!(), // Should not happen due to modulo 4
        }
        // This logic needs extension if non-rotation transforms are added.
    }
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
/// Includes the weights associated with each tile, the allowed symmetry
/// transformations for each tile, and mappings for transformed tile states.
#[derive(Debug, Clone)]
pub struct TileSet {
    /// Relative weights for each base tile.
    pub weights: Vec<f32>,
    /// Allowed symmetry transformations for each base tile.
    pub allowed_transformations: Vec<Vec<Transformation>>,
    /// Total number of unique transformed tile states (base tile + allowed transformation).
    num_transformed_tiles: usize,
    /// Maps (Base TileId, Transformation) to a unique TransformedTileId (usize index).
    transformed_tile_map: HashMap<(TileId, Transformation), usize>,
    /// Maps a TransformedTileId (usize index) back to (Base TileId, Transformation).
    reverse_transformed_map: Vec<(TileId, Transformation)>,
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

        let mut num_transformed_tiles = 0;
        let mut transformed_tile_map = HashMap::new();
        let mut reverse_transformed_map = Vec::new();

        // Check weights and transformations, and build mappings
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

            // Add mappings for this base tile and its allowed transformations
            let base_tile_id = TileId(index);
            for &transformation in transformations {
                // TODO: Check for duplicate transformations within the inner vec?
                let transformed_id = num_transformed_tiles;
                if transformed_tile_map
                    .insert((base_tile_id, transformation), transformed_id)
                    .is_some()
                {
                    // This should not happen if input `allowed_transformations` has unique transforms per tile
                    // but could indicate an issue if the input is malformed.
                    // Consider adding a specific error variant for this internal consistency check.
                    panic!("Internal error: Duplicate transformation detected during mapping.");
                }
                reverse_transformed_map.push((base_tile_id, transformation));
                num_transformed_tiles += 1;
            }
        }

        Ok(Self {
            weights,
            allowed_transformations,
            num_transformed_tiles,
            transformed_tile_map,
            reverse_transformed_map,
        })
    }

    /// Gets the total number of unique transformed tile states.
    pub fn num_transformed_tiles(&self) -> usize {
        self.num_transformed_tiles
    }

    /// Gets the unique `TransformedTileId` (usize index) for a given base tile and transformation.
    /// Returns `None` if the transformation is not allowed for the base tile.
    pub fn get_transformed_id(
        &self,
        base_id: TileId,
        transformation: Transformation,
    ) -> Option<usize> {
        self.transformed_tile_map
            .get(&(base_id, transformation))
            .copied()
    }

    /// Gets the base tile and transformation corresponding to a `TransformedTileId` (usize index).
    /// Returns `None` if the index is out of bounds.
    pub fn get_base_tile_and_transform(
        &self,
        transformed_id: usize,
    ) -> Option<(TileId, Transformation)> {
        self.reverse_transformed_map.get(transformed_id).copied()
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
    pub fn new(num_transformed_tiles: usize, num_axes: usize, allowed: Vec<bool>) -> Self {
        assert_eq!(
            allowed.len(),
            num_axes * num_transformed_tiles * num_transformed_tiles,
            "Provided 'allowed' vector has incorrect size for transformed tiles."
        );
        Self {
            num_tiles: num_transformed_tiles, // Store the total transformed count here
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

    /// Checks if `transformed_tile2_id` is allowed to be placed adjacent to `transformed_tile1_id` along the specified `axis`.
    ///
    /// Performs bounds checks internally and returns `false` if indices are out of range.
    /// Uses inline attribute for potential performance optimization in tight loops.
    #[inline]
    pub fn check(
        &self,
        transformed_tile1_id: usize,
        transformed_tile2_id: usize,
        axis: usize,
    ) -> bool {
        if transformed_tile1_id >= self.num_tiles
            || transformed_tile2_id >= self.num_tiles
            || axis >= self.num_axes
        {
            return false; // Treat out-of-bounds as disallowed.
        }
        let index = axis * self.num_tiles * self.num_tiles
            + transformed_tile1_id * self.num_tiles
            + transformed_tile2_id;
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

        assert!(rules.check(0, 1, 0)); // Axis 0: T0 -> T1 (true)
        assert!(!rules.check(0, 2, 0)); // Axis 0: T0 -> T2 (false)
        assert!(rules.check(1, 2, 0)); // Axis 0: T1 -> T2 (true)

        assert!(!rules.check(0, 1, 1)); // Axis 1: T0 -> T1 (false)
        assert!(rules.check(2, 2, 1)); // Axis 1: T2 -> T2 (true)
    }

    #[test]
    fn adjacency_check_out_of_bounds() {
        let rules = AdjacencyRules::new(2, 1, vec![true, false, false, true]); // 2 tiles, 1 axis

        assert!(!rules.check(0, 2, 0)); // Tile 2 out of bounds
        assert!(!rules.check(2, 0, 0)); // Tile 2 out of bounds
        assert!(!rules.check(0, 0, 1)); // Axis 1 out of bounds
    }

    // Add more tests for AdjacencyRules if needed
}
