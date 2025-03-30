use crate::tile::TileId;

// Represents adjacency rules using a flattened vector for efficiency.
// The vector stores boolean values indicating if tile2 is allowed next to tile1 along a specific axis.
// Indexing: allowed[axis * num_tiles * num_tiles + tile1.0 * num_tiles + tile2.0]
#[derive(Debug, Clone)]
pub struct AdjacencyRules {
    num_tiles: usize,
    num_axes: usize,
    /// Flattened vector: allowed[axis][tile1][tile2]
    allowed: Vec<bool>,
}

impl AdjacencyRules {
    /// Creates new AdjacencyRules, initializing all adjacencies based on the provided `allowed` vector.
    ///
    /// # Panics
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

    /// Gets the number of different tile types the rules are defined for.
    pub fn num_tiles(&self) -> usize {
        self.num_tiles
    }

    /// Gets the number of axes/directions the rules are defined for (e.g., 6 for 3D +/- X/Y/Z).
    pub fn num_axes(&self) -> usize {
        self.num_axes
    }

    /// Checks if `tile2` is allowed to be adjacent to `tile1` along the specified `axis`.
    ///
    /// # Panics
    /// Panics if `axis` is out of bounds or if `tile1` or `tile2` IDs are out of bounds.
    pub fn check(&self, tile1: TileId, tile2: TileId, axis: usize) -> bool {
        assert!(axis < self.num_axes, "Axis index out of bounds");
        assert!(tile1.0 < self.num_tiles, "Tile1 ID out of bounds");
        assert!(tile2.0 < self.num_tiles, "Tile2 ID out of bounds");

        let index = axis * self.num_tiles * self.num_tiles + tile1.0 * self.num_tiles + tile2.0;
        // Use get() for safety, though asserts should prevent panics.
        // Default to false if index is somehow out of bounds (shouldn't happen with asserts).
        *self.allowed.get(index).unwrap_or(&false)
    }
}
