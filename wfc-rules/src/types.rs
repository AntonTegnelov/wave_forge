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
    #[must_use] pub fn transform_axis(self, axis: usize) -> usize {
        match self {
            Self::Identity => axis,
            Self::Rot90 => match axis {
                0 => 2,
                1 => 3,
                2 => 1,
                3 => 0,
                4 | 5 => axis,
                _ => panic!("Invalid axis"),
            },
            Self::Rot180 => match axis {
                0 => 1,
                1 => 0,
                2 => 3,
                3 => 2,
                4 | 5 => axis,
                _ => panic!("Invalid axis"),
            },
            Self::Rot270 => match axis {
                0 => 3,
                1 => 2,
                2 => 0,
                3 => 1,
                4 | 5 => axis,
                _ => panic!("Invalid axis"),
            },
            Self::FlipX => match axis {
                0 => 1,
                1 => 0,
                2..=5 => axis,
                _ => panic!("Invalid axis"),
            },
            Self::FlipY => match axis {
                2 => 3,
                3 => 2,
                0 | 1 | 4 | 5 => axis,
                _ => panic!("Invalid axis"),
            },
            Self::FlipZ => match axis {
                4 => 5,
                5 => 4,
                0..=3 => axis,
                _ => panic!("Invalid axis"),
            },
        }
    }

    /// Returns the inverse transformation.
    #[must_use] pub const fn inverse(self) -> Self {
        match self {
            Self::Identity => Self::Identity,
            Self::Rot90 => Self::Rot270,
            Self::Rot180 => Self::Rot180,
            Self::Rot270 => Self::Rot90,
            Self::FlipX => Self::FlipX, // Reflections are self-inverse
            Self::FlipY => Self::FlipY,
            Self::FlipZ => Self::FlipZ,
        }
    }

    /// Combines this transformation with another (applies `other` then `self`).
    /// Equivalent to matrix multiplication: `self * other`.
    /// NOTE: Combination logic for reflections is currently unimplemented.
    #[must_use] pub fn combine(self, other: Self) -> Self {
        match (self, other) {
            // Identity cases
            (Self::Identity, _) => other,
            (_, Self::Identity) => self,

            // Rotation * Rotation
            (
                Self::Rot90 | Self::Rot180 | Self::Rot270,
                Self::Rot90 | Self::Rot180 | Self::Rot270,
            ) => {
                let self_val = match self {
                    Self::Rot90 => 1,
                    Self::Rot180 => 2,
                    Self::Rot270 => 3,
                    _ => 0,
                };
                let other_val = match other {
                    Self::Rot90 => 1,
                    Self::Rot180 => 2,
                    Self::Rot270 => 3,
                    _ => 0,
                };
                match (self_val + other_val) % 4 {
                    0 => Self::Identity,
                    1 => Self::Rot90,
                    2 => Self::Rot180,
                    3 => Self::Rot270,
                    _ => unreachable!(),
                }
            }

            // Any combination involving a Flip is currently unimplemented
            (Self::FlipX | Self::FlipY | Self::FlipZ, _) |
(_, Self::FlipX | Self::FlipY | Self::FlipZ) => {
                panic!("Transformation::combine is not implemented for reflections yet.");
            }
        }
    }
}

/// Errors that can occur during `TileSet` creation or validation.
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
    /// Maps (Base `TileId`, Transformation) to a unique `TransformedTileId` (usize index).
    transformed_tile_map: HashMap<(TileId, Transformation), usize>,
    /// Maps a `TransformedTileId` (usize index) back to (Base `TileId`, Transformation).
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
    #[must_use] pub const fn num_transformed_tiles(&self) -> usize {
        self.num_transformed_tiles
    }

    /// Gets the unique `TransformedTileId` (usize index) for a given base tile and transformation.
    /// Returns `None` if the transformation is not allowed for the base tile.
    #[must_use] pub fn get_transformed_id(
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
    #[must_use] pub fn get_base_tile_and_transform(
        &self,
        transformed_id: usize,
    ) -> Option<(TileId, Transformation)> {
        self.reverse_transformed_map.get(transformed_id).copied()
    }

    /// Gets the weight for a specific `TileId`.
    ///
    /// Returns `None` if the `TileId` is out of bounds.
    #[must_use] pub fn get_weight(&self, tile_id: TileId) -> Option<f32> {
        self.weights.get(tile_id.0).copied()
    }
}

/// Represents adjacency rules between tiles for different axes.
///
/// Stores the rules efficiently for potentially sparse rule sets using a `HashMap`.
#[derive(Debug, Clone)]
pub struct AdjacencyRules {
    num_tiles: usize,
    num_axes: usize,
    allowed: HashMap<(usize, usize, usize), bool>,
    weighted: HashMap<(usize, usize, usize), f32>,
}

impl AdjacencyRules {
    /// Creates new `AdjacencyRules` from a list of allowed tuples.
    pub fn from_allowed_tuples(
        num_transformed_tiles: usize,
        num_axes: usize,
        allowed_tuples: impl IntoIterator<Item = (usize, usize, usize)>,
    ) -> Self {
        let mut allowed = HashMap::new();
        for (axis, ttid1, ttid2) in allowed_tuples {
            if axis < num_axes && ttid1 < num_transformed_tiles && ttid2 < num_transformed_tiles {
                allowed.insert((axis, ttid1, ttid2), true);
            }
        }
        Self {
            num_tiles: num_transformed_tiles,
            num_axes,
            allowed,
            weighted: HashMap::new(),
        }
    }

    /// Creates new `AdjacencyRules` with weighted rules from a list of allowed tuples with weights.
    ///
    /// # Arguments
    ///
    /// * `num_transformed_tiles` - Total number of unique transformed tile states used in these rules.
    /// * `num_axes` - Number of axes/directions these rules are defined for.
    /// * `weighted_tuples` - An iterator of tuples (axis, `tile1_id`, `tile2_id`, weight) where weight is a value in [0.0, 1.0].
    ///
    /// # Returns
    ///
    /// A new `AdjacencyRules` instance with weighted rules.
    pub fn from_weighted_tuples(
        num_transformed_tiles: usize,
        num_axes: usize,
        weighted_tuples: impl IntoIterator<Item = (usize, usize, usize, f32)>,
    ) -> Self {
        let mut allowed = HashMap::new();
        let mut weighted = HashMap::new();

        for (axis, ttid1, ttid2, weight) in weighted_tuples {
            if axis < num_axes && ttid1 < num_transformed_tiles && ttid2 < num_transformed_tiles {
                // Only add to allowed rules if weight > 0
                if weight > 0.0 {
                    allowed.insert((axis, ttid1, ttid2), true);

                    // Only store weights different from 1.0 (default)
                    if weight < 1.0 {
                        weighted.insert((axis, ttid1, ttid2), weight);
                    }
                }
            }
        }

        Self {
            num_tiles: num_transformed_tiles,
            num_axes,
            allowed,
            weighted,
        }
    }

    /// Gets the total number of unique transformed tile states used in these rules.
    #[must_use] pub const fn num_tiles(&self) -> usize {
        self.num_tiles
    }

    /// Gets the number of axes/directions these rules are defined for.
    #[must_use] pub const fn num_axes(&self) -> usize {
        self.num_axes
    }

    /// Provides read-only access to the internal `HashMap` storing allowed adjacencies.
    /// Useful for debugging or advanced analysis.
    #[must_use] pub const fn get_allowed_rules_map(&self) -> &HashMap<(usize, usize, usize), bool> {
        &self.allowed
    }

    /// Checks if `transformed_tile2_id` is allowed to be placed adjacent to `transformed_tile1_id` along the specified `axis`.
    ///
    /// Performs bounds checks internally and returns `false` if indices are out of range.
    /// Returns `true` only if the specific rule `(axis, ttid1, ttid2)` exists in the map.
    #[inline]
    #[must_use] pub fn check(
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
    /// Panics if the input axis is invalid (>= `num_axes`, usually 6).
    #[inline]
    #[must_use] pub fn opposite_axis(&self, axis: usize) -> usize {
        match axis {
            0 => 1,
            1 => 0,
            2 => 3,
            3 => 2,
            4 => 5,
            5 => 4,
            _ => panic!("Invalid axis index: {axis}"),
        }
    }

    /// Gets the weight of a specific adjacency rule.
    ///
    /// Returns a value in the range [0.0, 1.0] where:
    /// - 0.0 means the adjacency is not allowed
    /// - 1.0 means the adjacency is fully allowed (default)
    /// - Values between 0.0 and 1.0 represent weighted constraints
    ///
    /// # Arguments
    ///
    /// * `transformed_tile1_id` - The ID of the first tile.
    /// * `transformed_tile2_id` - The ID of the second tile.
    /// * `axis` - The axis/direction of the adjacency.
    ///
    /// # Returns
    ///
    /// A value between 0.0 and 1.0 representing the weight of the rule.
    #[inline]
    #[must_use] pub fn get_weight(
        &self,
        transformed_tile1_id: usize,
        transformed_tile2_id: usize,
        axis: usize,
    ) -> f32 {
        if transformed_tile1_id >= self.num_tiles
            || transformed_tile2_id >= self.num_tiles
            || axis >= self.num_axes
        {
            return 0.0;
        }

        // If the adjacency is not allowed, return 0.0
        if !self
            .allowed
            .contains_key(&(axis, transformed_tile1_id, transformed_tile2_id))
        {
            return 0.0;
        }

        // Return the weight or 1.0 if no weight is specified
        self.weighted
            .get(&(axis, transformed_tile1_id, transformed_tile2_id))
            .copied()
            .unwrap_or(1.0)
    }

    /// Provides read-only access to the internal `HashMap` storing weighted adjacencies.
    /// Useful for debugging or advanced analysis.
    #[must_use] pub const fn get_weighted_rules_map(&self) -> &HashMap<(usize, usize, usize), f32> {
        &self.weighted
    }

    /// Returns the weight associated with the *base tile* underlying the `transformed_tile_id`.
    /// **Placeholder:** Currently returns 1.0. Requires access to `TileSet` for correct implementation.
    #[must_use] pub const fn get_tile_weight(&self, transformed_tile_id: usize) -> f32 {
        // TODO: Implement correctly using TileSet access.
        // This requires either storing Arc<TileSet> or passing TileSet reference.
        if transformed_tile_id >= self.num_tiles {
            return 0.0; // Invalid ID
        }
        // Placeholder:
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Helper Functions ---

    /// Creates a basic `TileSet` with a specified number of base tiles and Identity transformation.
    fn create_basic_tileset(num_base_tiles: usize) -> Result<TileSet, TileSetError> {
        let weights = vec![1.0; num_base_tiles];
        // Each base tile has only the Identity transformation
        let allowed_transforms = vec![vec![Transformation::Identity]; num_base_tiles];
        TileSet::new(weights, allowed_transforms)
    }

    /// Creates simple adjacency rules: Tile `i` can only be adjacent to Tile `i` along any axis.
    fn create_simple_rules(tileset: &TileSet) -> AdjacencyRules {
        let num_transformed_tiles = tileset.num_transformed_tiles();
        let num_axes = 6; // Assuming 3D
        let mut allowed_tuples = Vec::new();
        for axis in 0..num_axes {
            for ttid in 0..num_transformed_tiles {
                allowed_tuples.push((axis, ttid, ttid));
            }
        }
        AdjacencyRules::from_allowed_tuples(num_transformed_tiles, num_axes, allowed_tuples)
    }

    /// Creates weighted adjacency rules: Tile `i` can be adjacent to Tile `i` and `i+1` with different weights
    fn create_weighted_rules(tileset: &TileSet) -> AdjacencyRules {
        let num_transformed_tiles = tileset.num_transformed_tiles();
        let num_axes = 6; // Assuming 3D
        let mut weighted_tuples = Vec::new();
        for axis in 0..num_axes {
            for ttid in 0..num_transformed_tiles {
                // Same tile has weight 1.0 (full probability)
                weighted_tuples.push((axis, ttid, ttid, 1.0));

                // Next tile has weight 0.5 (partial probability)
                if ttid + 1 < num_transformed_tiles {
                    weighted_tuples.push((axis, ttid, ttid + 1, 0.5));
                }
            }
        }
        AdjacencyRules::from_weighted_tuples(num_transformed_tiles, num_axes, weighted_tuples)
    }

    #[test]
    fn test_from_allowed_tuples_simple() {
        let tileset = create_basic_tileset(2).unwrap();
        let rules = create_simple_rules(&tileset);

        // Tile 0 can be adjacent to tile 0 along axis 0
        assert!(rules.check(0, 0, 0));
        // Tile 1 can be adjacent to tile 1 along axis a
        assert!(rules.check(1, 1, 1));
        // Tile 0 cannot be adjacent to tile 1 along axis 0
        assert!(!rules.check(0, 1, 0));
    }

    #[test]
    fn test_weighted_rules() {
        let tileset = create_basic_tileset(3).unwrap();
        let rules = create_weighted_rules(&tileset);

        // Check rule existence
        assert!(rules.check(0, 0, 0)); // Same tile rule exists
        assert!(rules.check(0, 1, 0)); // Next tile rule exists
        assert!(!rules.check(0, 2, 0)); // Skip tile rule doesn't exist

        // Check weights
        assert_eq!(rules.get_weight(0, 0, 0), 1.0); // Same tile has weight 1.0
        assert_eq!(rules.get_weight(0, 1, 0), 0.5); // Next tile has weight 0.5
        assert_eq!(rules.get_weight(0, 2, 0), 0.0); // Non-existent rule has weight 0.0

        // Check weighted_rules map
        let weighted_map = rules.get_weighted_rules_map();
        assert!(!weighted_map.contains_key(&(0, 0, 0))); // Weight 1.0 not stored explicitly
        assert!(weighted_map.contains_key(&(0, 0, 1))); // Weight 0.5 is stored
        assert_eq!(weighted_map.get(&(0, 0, 1)), Some(&0.5f32)); // Correct weight value
    }
}

// --- Proptest section ---
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    // Strategy to generate a valid AdjacencyRules instance
    fn arb_adjacency_rules() -> impl Strategy<Value = AdjacencyRules> {
        // Define ranges for num_tiles and num_axes
        (1..10usize, 1..=6usize).prop_flat_map(|(num_tiles, num_axes)| {
            // Strategy for generating allowed tuples (axis, t1, t2)
            let tuple_strategy = (
                0..num_axes,  // axis
                0..num_tiles, // t1
                0..num_tiles, // t2
            );

            // Generate a vector of these tuples
            proptest::collection::vec(tuple_strategy, 0..num_tiles * num_tiles * num_axes).prop_map(
                move |tuples| {
                    let _tileset = TileSet {
                        weights: vec![1.0; num_tiles],
                        allowed_transformations: vec![
                            vec![Transformation::Identity; num_tiles];
                            num_tiles
                        ],
                        num_transformed_tiles: num_tiles,
                        transformed_tile_map: HashMap::new(),
                        reverse_transformed_map: vec![
                            (TileId(0), Transformation::Identity);
                            num_tiles
                        ],
                    };
                    AdjacencyRules::from_allowed_tuples(num_tiles, num_axes, tuples)
                },
            )
        })
    }

    proptest! {
        // Property: If check(t1, t2, axis) is true, then check(t2, t1, opposite_axis) should also be true *IF* the reverse rule was explicitly added.
        // Note: This property only holds if the rule generation process ensures symmetry.
        // The current from_allowed_tuples doesn't enforce symmetry, so we test the check method itself.
        // Property: opposite_axis(opposite_axis(axis)) == axis
        #[test]
        fn test_opposite_axis_symmetry(rules in arb_adjacency_rules(), axis in 0..6usize) {
            prop_assume!(axis < rules.num_axes);
            let opposite = rules.opposite_axis(axis);
            prop_assume!(opposite < rules.num_axes); // Ensure opposite is also valid
            assert_eq!(rules.opposite_axis(opposite), axis);
        }

        // Property: check() should return false for invalid tile IDs or axes
        #[test]
        fn test_check_invalid_inputs(
            rules in arb_adjacency_rules(),
            t1 in any::<usize>(),
            t2 in any::<usize>(),
            axis in any::<usize>()
        ) {
            let is_valid_t1 = t1 < rules.num_tiles;
            let is_valid_t2 = t2 < rules.num_tiles;
            let is_valid_axis = axis < rules.num_axes;

            if !is_valid_t1 || !is_valid_t2 || !is_valid_axis {
                prop_assert!(!rules.check(t1, t2, axis), "check should be false for invalid inputs");
            }
            // No assertion if inputs are valid, as the result depends on the specific rules
        }

        // Property: If a rule (axis, t1, t2) was explicitly added via from_allowed_tuples,
        // then check(t1, t2, axis) should return true.
        // This tests the construction and checking consistency.
        #[test]
        fn test_construction_consistency(
            num_tiles in 1..10usize,
            num_axes in 1..=6usize,
            allowed_tuples in proptest::collection::vec((0..6usize, 0..10usize, 0..10usize), 0..50)
        )
        {
            // Filter tuples to be valid for the generated num_tiles and num_axes
            let valid_tuples: Vec<(usize, usize, usize)> = allowed_tuples
                .into_iter()
                .filter(|(ax, t1, t2)| *ax < num_axes && *t1 < num_tiles && *t2 < num_tiles)
                .collect();

            let _tileset = TileSet {
                weights: vec![1.0; num_tiles],
                allowed_transformations: vec![vec![Transformation::Identity; num_tiles]; num_tiles],
                num_transformed_tiles: num_tiles,
                transformed_tile_map: HashMap::new(),
                reverse_transformed_map: vec![(TileId(0), Transformation::Identity); num_tiles],
            };
            let rules = AdjacencyRules::from_allowed_tuples(num_tiles, num_axes, valid_tuples.clone());

            for (axis, t1, t2) in valid_tuples {
                prop_assert!(rules.check(t1, t2, axis), "Rule added via constructor should be checkable");
            }
        }
    }
}
