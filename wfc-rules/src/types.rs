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
            // Validate inputs before inserting
            if axis < num_axes && ttid1 < num_transformed_tiles && ttid2 < num_transformed_tiles {
                allowed.insert((axis, ttid1, ttid2), true);
            } else {
                // Optional: Log a warning about skipped invalid tuples
                // log::warn!("Skipping invalid rule tuple: ({}, {}, {}) for num_tiles={}, num_axes={}", axis, ttid1, ttid2, num_transformed_tiles, num_axes);
            }
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
    use super::*;
    use crate::TileId; // Make sure TileId is in scope
    use std::collections::HashSet;

    // Helper to create a basic AdjacencyRules for testing
    fn create_test_rules() -> AdjacencyRules {
        // Example: 2 tiles, 6 axes (3D)
        // T0 <-> T1 on X (+0 / -1)
        // T0 <-> T0 on Y (+2 / -3)
        // T1 <-> T1 on Y (+2 / -3)
        // T0 <-> T0 on Z (+4 / -5)
        // T1 <-> T1 on Z (+4 / -5)
        let allowed_rules = vec![
            (0, 0, 1), // +X: T0 -> T1
            (1, 1, 0), // -X: T1 -> T0
            (2, 0, 0), // +Y: T0 -> T0
            (2, 1, 1), // +Y: T1 -> T1
            (3, 0, 0), // -Y: T0 -> T0
            (3, 1, 1), // -Y: T1 -> T1
            (4, 0, 0), // +Z: T0 -> T0
            (4, 1, 1), // +Z: T1 -> T1
            (5, 0, 0), // -Z: T0 -> T0
            (5, 1, 1), // -Z: T1 -> T1
        ];
        AdjacencyRules::from_allowed_tuples(2, 6, allowed_rules)
    }

    #[test]
    fn test_rule_check() {
        let rules = create_test_rules();

        // Check allowed rules
        assert!(rules.check(0, 1, 0)); // +X: T0 -> T1
        assert!(rules.check(1, 0, 1)); // -X: T1 -> T0
        assert!(rules.check(0, 0, 2)); // +Y: T0 -> T0
        assert!(rules.check(1, 1, 5)); // -Z: T1 -> T1

        // Check disallowed rules
        assert!(!rules.check(0, 0, 0)); // +X: T0 -> T0 (Not allowed by rule)
        assert!(!rules.check(1, 1, 0)); // +X: T1 -> T1
        assert!(!rules.check(0, 1, 2)); // +Y: T0 -> T1
        assert!(!rules.check(1, 0, 3)); // -Y: T1 -> T0

        // Check invalid axis/tile (should default to false)
        assert!(!rules.check(0, 0, 6)); // Invalid axis
        assert!(!rules.check(2, 0, 0)); // Invalid tile ID
    }

    #[test]
    fn test_opposite_axis() {
        let rules = create_test_rules();
        assert_eq!(rules.opposite_axis(0), 1); // +X -> -X
        assert_eq!(rules.opposite_axis(1), 0); // -X -> +X
        assert_eq!(rules.opposite_axis(2), 3); // +Y -> -Y
        assert_eq!(rules.opposite_axis(3), 2); // -Y -> +Y
        assert_eq!(rules.opposite_axis(4), 5); // +Z -> -Z
        assert_eq!(rules.opposite_axis(5), 4); // -Z -> +Z
    }

    #[test]
    #[should_panic]
    fn test_opposite_axis_panic() {
        let rules = create_test_rules();
        rules.opposite_axis(6); // Invalid axis
    }

    #[test]
    fn test_from_allowed_tuples_empty() {
        let rules = AdjacencyRules::from_allowed_tuples(0, 0, vec![]);
        assert_eq!(rules.num_tiles, 0);
        assert_eq!(rules.num_axes, 0);
        assert!(rules.allowed.is_empty());
    }

    #[test]
    fn test_from_allowed_tuples_duplicates() {
        // Ensures duplicates are handled (HashSet should deduplicate)
        let allowed_rules = vec![
            (0, 0, 1),
            (0, 0, 1), // Duplicate
            (1, 1, 0),
        ];
        let rules = AdjacencyRules::from_allowed_tuples(2, 6, allowed_rules);
        assert_eq!(rules.allowed.len(), 2);
        assert!(rules.check(0, 1, 0));
        assert!(rules.check(1, 0, 1));
    }

    #[test]
    fn test_from_allowed_tuples_invalid_tile_ids() {
        // Behavior with invalid tile IDs in tuples (should they be ignored? error?)
        // Current implementation ignores them because HashSet insert won't happen if num_tiles check fails
        let allowed_rules = vec![
            (0, 0, 1),
            (1, 2, 0), // Invalid tile ID 2 (num_tiles=2)
            (1, 1, 3), // Invalid tile ID 3
        ];
        let rules = AdjacencyRules::from_allowed_tuples(2, 6, allowed_rules);
        assert_eq!(rules.allowed.len(), 1); // Only (0,0,1) should be added
        assert!(rules.check(0, 1, 0));
        assert!(!rules.check(1, 0, 1)); // (1,1,0) was not implicitly added
        assert!(!rules.check(2, 0, 1)); // Invalid rule involving tile 2
        assert!(!rules.check(1, 3, 1)); // Invalid rule involving tile 3
    }

    #[test]
    fn test_from_allowed_tuples_invalid_axis() {
        // Behavior with invalid axis IDs (should be ignored)
        let allowed_rules = vec![
            (0, 0, 1),
            (6, 1, 0), // Invalid axis 6
        ];
        let rules = AdjacencyRules::from_allowed_tuples(2, 6, allowed_rules);
        assert_eq!(rules.allowed.len(), 1);
        assert!(rules.check(0, 1, 0));
        assert!(!rules.check(1, 0, 6)); // Check with invalid axis is false
    }

    // Add more tests for AdjacencyRules if needed
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
                move |tuples| AdjacencyRules::from_allowed_tuples(num_tiles, num_axes, tuples),
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

            let rules = AdjacencyRules::from_allowed_tuples(num_tiles, num_axes, valid_tuples.clone());

            for (axis, t1, t2) in valid_tuples {
                prop_assert!(rules.check(t1, t2, axis), "Rule added via constructor should be checkable");
            }
        }
    }
}
