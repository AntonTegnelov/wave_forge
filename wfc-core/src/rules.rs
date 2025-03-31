use crate::tile::TileId;

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
