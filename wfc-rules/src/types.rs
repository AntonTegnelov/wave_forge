use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during `TileSet` creation or validation.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum TileSetError {
    /// Error indicating the provided list of weights was empty.
    #[error("TileSet weights cannot be empty.")]
    EmptyWeights,
    /// A tile's weight is not a positive finite number.
    #[error("TileSet weights are positive finite numbers. Tile {0} has {1}")]
    NonPositiveWeight(usize, String),
}

/// The tiles of a rule set: the relative weight of each.
#[derive(Debug, Clone)]
pub struct TileSet {
    /// Relative weights for each tile.
    pub weights: Vec<f32>,
}

impl TileSet {
    /// A `TileSet` of these weights.
    ///
    /// # Errors
    ///
    /// [`TileSetError::EmptyWeights`] when there are none, and
    /// [`TileSetError::NonPositiveWeight`] for the first that is not a positive finite number.
    pub fn new(weights: Vec<f32>) -> Result<Self, TileSetError> {
        if weights.is_empty() {
            return Err(TileSetError::EmptyWeights);
        }
        if let Some((index, weight)) = weights
            .iter()
            .enumerate()
            .find(|(_, weight)| !(weight.is_finite() && **weight > 0.0))
        {
            return Err(TileSetError::NonPositiveWeight(index, weight.to_string()));
        }
        Ok(Self { weights })
    }

    /// How many tiles there are.
    #[must_use]
    pub fn num_tiles(&self) -> usize {
        self.weights.len()
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
}

impl AdjacencyRules {
    /// Creates new `AdjacencyRules` from a list of allowed tuples.
    pub fn from_allowed_tuples(
        num_tiles: usize,
        num_axes: usize,
        allowed_tuples: impl IntoIterator<Item = (usize, usize, usize)>,
    ) -> Self {
        let mut allowed = HashMap::new();
        for (axis, tile1, tile2) in allowed_tuples {
            if axis < num_axes && tile1 < num_tiles && tile2 < num_tiles {
                allowed.insert((axis, tile1, tile2), true);
            }
        }
        Self {
            num_tiles,
            num_axes,
            allowed,
        }
    }

    /// How many tiles these rules are for.
    #[must_use]
    pub const fn num_tiles(&self) -> usize {
        self.num_tiles
    }

    /// Gets the number of axes/directions these rules are defined for.
    #[must_use]
    pub const fn num_axes(&self) -> usize {
        self.num_axes
    }

    /// Checks if `tile2` is allowed to be placed adjacent to `tile1` along the specified `axis`.
    ///
    /// Performs bounds checks internally and returns `false` if indices are out of range.
    /// Returns `true` only if the specific rule `(axis, tile1, tile2)` exists in the map.
    #[inline]
    #[must_use]
    pub fn check(&self, tile1: usize, tile2: usize, axis: usize) -> bool {
        if tile1 >= self.num_tiles || tile2 >= self.num_tiles || axis >= self.num_axes {
            return false;
        }
        // Check if the key (rule) exists in the map. Absence means false.
        self.allowed.contains_key(&(axis, tile1, tile2))
    }

    /// Returns the opposite axis index.
    /// Assumes standard 3D axis convention:
    /// 0: +X, 1: -X, 2: +Y, 3: -Y, 4: +Z, 5: -Z
    /// Panics if the input axis is invalid (>= `num_axes`, usually 6).
    #[inline]
    #[must_use]
    pub fn opposite_axis(&self, axis: usize) -> usize {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_allowed_tuples_simple() {
        let tuples = (0..6).flat_map(|axis| (0..2).map(move |tile| (axis, tile, tile)));
        let rules = AdjacencyRules::from_allowed_tuples(2, 6, tuples);

        assert!(rules.check(0, 0, 0));
        assert!(rules.check(1, 1, 1));
        assert!(!rules.check(0, 1, 0));
    }

    #[test]
    fn a_weight_that_is_not_a_positive_finite_number_names_its_tile() {
        for weight in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let result = TileSet::new(vec![1.0, weight]);

            assert!(
                matches!(&result, Err(TileSetError::NonPositiveWeight(1, _))),
                "{weight}: {result:?}"
            );
        }
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
