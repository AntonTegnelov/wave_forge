use std::collections::HashMap;
use thiserror::Error;

/// Represents a unique identifier for a tile.
///
/// Often used as an index into tile-related data structures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileId(pub usize); // Simple wrapper for now

/// Represents a symmetry transformation applied to a tile.
/// Currently supports rotations and reflections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Transformation {
    Identity,
    Rot90,
    Rot180,
    Rot270,
    FlipX,
    FlipY,
    FlipZ,
    // Future: FlipX, FlipY, etc.
}

impl Transformation {
    /// Calculates the resulting axis index after applying the transformation.
    /// Assumes a standard 3D axis convention:
    /// 0: +X, 1: -X, 2: +Y, 3: -Y, 4: +Z, 5: -Z
    pub fn transform_axis(self, axis: usize) -> usize {
        match self {
            Transformation::Identity => axis,
            Transformation::Rot90 => match axis {
                0 => 2,
                1 => 3,
                2 => 1,
                3 => 0,
                4 | 5 => axis,
                _ => panic!("Invalid axis"),
            },
            Transformation::Rot180 => match axis {
                0 => 1,
                1 => 0,
                2 => 3,
                3 => 2,
                4 | 5 => axis,
                _ => panic!("Invalid axis"),
            },
            Transformation::Rot270 => match axis {
                0 => 3,
                1 => 2,
                2 => 0,
                3 => 1,
                4 | 5 => axis,
                _ => panic!("Invalid axis"),
            },
            Transformation::FlipX => match axis {
                0 => 1,
                1 => 0,
                2..=5 => axis,
                _ => panic!("Invalid axis"),
            },
            Transformation::FlipY => match axis {
                2 => 3,
                3 => 2,
                0 | 1 | 4 | 5 => axis,
                _ => panic!("Invalid axis"),
            },
            Transformation::FlipZ => match axis {
                4 => 5,
                5 => 4,
                0..=3 => axis,
                _ => panic!("Invalid axis"),
            },
        }
    }

    /// Returns the inverse transformation.
    pub fn inverse(self) -> Transformation {
        match self {
            Transformation::Identity => Transformation::Identity,
            Transformation::Rot90 => Transformation::Rot270,
            Transformation::Rot180 => Transformation::Rot180,
            Transformation::Rot270 => Transformation::Rot90,
            Transformation::FlipX => Transformation::FlipX, // Reflections are self-inverse
            Transformation::FlipY => Transformation::FlipY,
            Transformation::FlipZ => Transformation::FlipZ,
        }
    }

    /// Combines this transformation with another (applies `other` then `self`).
    /// Equivalent to matrix multiplication: `self * other`.
    /// NOTE: Combination logic for reflections is currently unimplemented.
    pub fn combine(self, other: Transformation) -> Transformation {
        match (self, other) {
            // Identity cases
            (Transformation::Identity, _) => other,
            (_, Transformation::Identity) => self,

            // Rotation * Rotation
            (
                Transformation::Rot90 | Transformation::Rot180 | Transformation::Rot270,
                Transformation::Rot90 | Transformation::Rot180 | Transformation::Rot270,
            ) => {
                let self_val = match self {
                    Transformation::Rot90 => 1,
                    Transformation::Rot180 => 2,
                    Transformation::Rot270 => 3,
                    _ => 0,
                };
                let other_val = match other {
                    Transformation::Rot90 => 1,
                    Transformation::Rot180 => 2,
                    Transformation::Rot270 => 3,
                    _ => 0,
                };
                match (self_val + other_val) % 4 {
                    0 => Transformation::Identity,
                    1 => Transformation::Rot90,
                    2 => Transformation::Rot180,
                    3 => Transformation::Rot270,
                    _ => unreachable!(),
                }
            }

            // Any combination involving a Flip is currently unimplemented
            (Transformation::FlipX, _)
            | (_, Transformation::FlipX)
            | (Transformation::FlipY, _)
            | (_, Transformation::FlipY)
            | (Transformation::FlipZ, _)
            | (_, Transformation::FlipZ) => {
                panic!("Transformation::combine is not implemented for reflections yet.");
            }
        }
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
/// Stores the rules efficiently for potentially sparse rule sets using a HashMap.
#[derive(Debug, Clone)]
pub struct AdjacencyRules {
    num_tiles: usize, // Still stores total number of *transformed* tiles for validation
    num_axes: usize,
    /// Stores only the allowed adjacencies `(axis, ttid1, ttid2) -> true`.
    /// Absence means the adjacency is disallowed.
    allowed: HashMap<(usize, usize, usize), bool>,
    // For faster iteration over allowed neighbours (optional optimization later):
    // allowed_neighbors: HashMap<(usize, usize), HashSet<usize>>, // (axis, ttid1) -> HashSet<ttid2>
}

impl AdjacencyRules {
    /// Creates new `AdjacencyRules` from a list of allowed tuples.
    ///
    /// # Arguments
    ///
    /// * `num_transformed_tiles` - The total number of unique transformed tile states.
    /// * `num_axes` - The number of axes (directions) rules are defined for.
    /// * `allowed_tuples` - An iterator providing tuples `(axis, transformed_tile1_id, transformed_tile2_id)`
    ///   for all allowed adjacencies.
    ///
    /// # Returns
    ///
    /// A new `AdjacencyRules` instance.
    pub fn from_allowed_tuples(
        num_transformed_tiles: usize,
        num_axes: usize,
        allowed_tuples: impl IntoIterator<Item = (usize, usize, usize)>,
    ) -> Self {
        let mut allowed = HashMap::new();
        for (axis, ttid1, ttid2) in allowed_tuples {
            // Optional validation (could also be done during generation):
            // assert!(axis < num_axes, "Axis index out of bounds");
            // assert!(ttid1 < num_transformed_tiles, "ttid1 out of bounds");
            // assert!(ttid2 < num_transformed_tiles, "ttid2 out of bounds");
            allowed.insert((axis, ttid1, ttid2), true);
        }
        Self {
            num_tiles: num_transformed_tiles,
            num_axes,
            allowed,
        }
    }

    /// Gets the total number of unique transformed tile states used in these rules.
    pub fn num_tiles(&self) -> usize {
        self.num_tiles
    }

    /// Gets the number of axes/directions these rules are defined for.
    pub fn num_axes(&self) -> usize {
        self.num_axes
    }

    /// Provides read-only access to the internal HashMap storing allowed adjacencies.
    /// Useful for debugging or advanced analysis.
    pub fn get_allowed_rules_map(&self) -> &HashMap<(usize, usize, usize), bool> {
        &self.allowed
    }

    /// Checks if `transformed_tile2_id` is allowed to be placed adjacent to `transformed_tile1_id` along the specified `axis`.
    ///
    /// Performs bounds checks internally and returns `false` if indices are out of range.
    /// Returns `true` only if the specific rule `(axis, ttid1, ttid2)` exists in the map.
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
            return false;
        }
        // Check if the key (rule) exists in the map. Absence means false.
        self.allowed
            .contains_key(&(axis, transformed_tile1_id, transformed_tile2_id))
    }

    /// Returns the opposite axis index.
    /// Assumes standard 3D axis convention:
    /// 0: +X, 1: -X, 2: +Y, 3: -Y, 4: +Z, 5: -Z
    /// Panics if the input axis is invalid (>= num_axes, usually 6).
    #[inline]
    pub fn opposite_axis(&self, axis: usize) -> usize {
        match axis {
            0 => 1,
            1 => 0,
            2 => 3,
            3 => 2,
            4 => 5,
            5 => 4,
            _ => panic!("Invalid axis index: {}", axis),
        }
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
    fn adjacency_check_valid_sparse() {
        let num_tiles = 3;
        let num_axes = 2;
        let allowed_rules = vec![
            (0, 0, 1), // Axis 0: T0 -> T1
            (0, 1, 0), // Axis 0: T1 -> T0
            (1, 2, 2), // Axis 1: T2 -> T2
        ];
        let rules = AdjacencyRules::from_allowed_tuples(num_tiles, num_axes, allowed_rules);

        assert!(rules.check(0, 1, 0));
        assert!(rules.check(1, 0, 0));
        assert!(rules.check(2, 2, 1));

        // Check disallowed (not present in map)
        assert!(!rules.check(0, 0, 0));
        assert!(!rules.check(1, 1, 0));
        assert!(!rules.check(0, 1, 1));
        assert!(!rules.check(2, 1, 1));
    }

    #[test]
    fn adjacency_check_out_of_bounds_sparse() {
        let rules = AdjacencyRules::from_allowed_tuples(2, 1, vec![(0, 0, 1)]); // 2 tiles, 1 axis

        assert!(!rules.check(0, 2, 0)); // Tile 2 out of bounds
        assert!(!rules.check(2, 0, 0)); // Tile 2 out of bounds
        assert!(!rules.check(0, 0, 1)); // Axis 1 out of bounds
    }

    // Add more tests for AdjacencyRules if needed
}
