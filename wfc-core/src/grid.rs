// use crate::error::GridError; // REMOVED
// use crate::GridError; // REMOVED
use bitvec::prelude::{bitvec, BitVec, Lsb0};
#[cfg(feature = "serde")] // Guard serde imports
use serde::{Deserialize, Serialize};

/// A generic 3-dimensional grid structure holding data of type `T`.
///
/// Used as a basic building block for various grid-based representations
/// in the WFC algorithm, such as the `PossibilityGrid` and `EntropyGrid`.
#[derive(Debug, Clone)] // Keep Clone for tests, consider removing if large grids are cloned often
pub struct Grid<T> {
    /// The width of the grid (X dimension).
    pub width: usize,
    /// The height of the grid (Y dimension).
    pub height: usize,
    /// The depth of the grid (Z dimension).
    pub depth: usize,
    /// The flattened 1D vector storing the grid data.
    /// Data is stored in row-major order (x varies fastest, then y, then z).
    /// Use `get` and `get_mut` for safe access via 3D coordinates.
    pub data: Vec<T>,
}

impl<T: Clone + Default> Grid<T> {
    /// Creates a new generic grid with the given dimensions, initialized with default values of type `T`.
    ///
    /// # Arguments
    ///
    /// * `width` - Grid width (X dimension).
    /// * `height` - Grid height (Y dimension).
    /// * `depth` - Grid depth (Z dimension).
    ///
    /// # Returns
    ///
    /// A new `Grid<T>` instance with dimensions `width`x`height`x`depth`,
    /// where each cell contains `T::default()`.
    pub fn new(width: usize, height: usize, depth: usize) -> Self {
        let size = width * height * depth;
        // Ensure size is not zero to prevent Vec::with_capacity(0)
        if size == 0 {
            return Self {
                width,
                height,
                depth,
                data: Vec::new(),
            };
        }
        let data = vec![T::default(); size];
        Self {
            width,
            height,
            depth,
            data,
        }
    }

    /// Calculates the 1D index into the `data` vector for the given 3D coordinates.
    ///
    /// Returns `None` if the coordinates `(x, y, z)` are outside the grid boundaries.
    /// The indexing scheme is `z * width * height + y * width + x`.
    #[inline] // Make indexing inline for performance
    fn index(&self, x: usize, y: usize, z: usize) -> Option<usize> {
        if x < self.width && y < self.height && z < self.depth {
            Some(z * self.width * self.height + y * self.width + x)
        } else {
            None
        }
    }

    /// Returns an immutable reference to the element at the given 3D coordinates `(x, y, z)`.
    ///
    /// Returns `None` if the coordinates are out of bounds.
    #[inline] // Make access inline
    pub fn get(&self, x: usize, y: usize, z: usize) -> Option<&T> {
        self.index(x, y, z).and_then(|idx| self.data.get(idx))
    }

    /// Returns a mutable reference to the element at the given 3D coordinates `(x, y, z)`.
    ///
    /// Returns `None` if the coordinates are out of bounds.
    #[inline] // Make access inline
    pub fn get_mut(&mut self, x: usize, y: usize, z: usize) -> Option<&mut T> {
        self.index(x, y, z)
            .and_then(move |idx| self.data.get_mut(idx))
    }
}

// --- PossibilityGrid Definition ---

/// A specialized 3D grid storing the possibility state for each cell in the WFC algorithm.
///
/// Each cell contains a `BitVec`, where the index corresponds to a `TileId`.
/// If the bit at index `i` is set (`true`), it means `TileId(i)` is still considered
/// a possible tile for that cell. If the bit is unset (`false`), the tile has been eliminated.
#[derive(Debug, Clone, PartialEq)] // Removed direct Serialize/Deserialize
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))] // Guard derive macros
pub struct PossibilityGrid {
    /// The width of the grid (X dimension).
    pub width: usize,
    /// The height of the grid (Y dimension).
    pub height: usize,
    /// The depth of the grid (Z dimension).
    pub depth: usize,
    /// The total number of unique tile types the grid is configured for.
    /// This determines the length of the `BitVec` in each cell.
    num_tiles: usize,
    /// The flattened 1D vector storing the `BitVec` possibility state for each cell.
    data: Vec<BitVec>,
}

impl PossibilityGrid {
    /// Creates a new `PossibilityGrid` initialized with all tiles possible for every cell.
    ///
    /// # Arguments
    ///
    /// * `width` - Grid width (X dimension).
    /// * `height` - Grid height (Y dimension).
    /// * `depth` - Grid depth (Z dimension).
    /// * `num_tiles` - The total number of unique tile types.
    ///
    /// # Returns
    ///
    /// A new `PossibilityGrid` where every bit in every cell's `BitVec` is set to `true`.
    ///
    /// # Panics
    ///
    /// Panics if `num_tiles` is 0, as a grid must have at least one tile type.
    pub fn new(width: usize, height: usize, depth: usize, num_tiles: usize) -> Self {
        assert!(num_tiles > 0, "num_tiles must be greater than 0");
        let size = width * height * depth;
        if size == 0 {
            return Self {
                width,
                height,
                depth,
                num_tiles,
                data: Vec::new(),
            };
        }
        // Create a template BitVec with all possibilities set
        let mut all_possible = BitVec::with_capacity(num_tiles);
        all_possible.resize(num_tiles, true);
        // Initialize the data vector by cloning the template
        let data = vec![all_possible; size];
        Self {
            width,
            height,
            depth,
            num_tiles,
            data,
        }
    }

    /// Returns the number of unique tile types this grid is configured for.
    /// This corresponds to the length of the `BitVec` in each cell.
    pub fn num_tiles(&self) -> usize {
        self.num_tiles
    }

    /// Provides read-only access to the internal flat vector containing the `BitVec`
    /// possibility state for each cell.
    ///
    /// Primarily intended for advanced use cases like direct buffer manipulation (e.g., for GPU transfer),
    /// where accessing the underlying data structure is necessary.
    /// Use `get` for standard cell access.
    pub fn get_cell_data(&self) -> &Vec<BitVec> {
        &self.data
    }

    /// Returns a reference to the underlying data vector
    pub fn data(&self) -> &Vec<BitVec> {
        &self.data
    }

    /// Calculates the 1D index into the `data` vector for the given 3D coordinates.
    ///
    /// Returns `None` if the coordinates `(x, y, z)` are outside the grid boundaries.
    /// The indexing scheme is `z * width * height + y * width + x`.
    #[inline]
    fn index(&self, x: usize, y: usize, z: usize) -> Option<usize> {
        if x < self.width && y < self.height && z < self.depth {
            Some(self.get_index(x, y, z))
        } else {
            None
        }
    }

    /// Returns the 1D index into the data vector for the given 3D coordinates.
    /// This is a public version of the internal index method.
    /// Note: This does not check bounds - use with caution!
    #[inline]
    pub fn get_index(&self, x: usize, y: usize, z: usize) -> usize {
        z * self.width * self.height + y * self.width + x
    }

    /// Sets a chunk of possibilities for a cell using a packed u32 representation.
    /// Each bit in the chunk_data represents whether a tile is possible (1) or not (0).
    /// The chunk_index determines which group of 32 tiles this chunk represents.
    /// For example, chunk_index 0 represents tiles 0-31, chunk_index 1 represents tiles 32-63, etc.
    ///
    /// # Arguments
    /// * `x`, `y`, `z` - The coordinates of the cell to modify
    /// * `chunk_index` - Which group of 32 tiles this chunk represents
    /// * `chunk_data` - A u32 where each bit represents a tile possibility
    ///
    /// # Returns
    /// `Ok(())` if successful, `Err(String)` if coordinates are invalid
    pub fn set_possibility_chunk(
        &mut self,
        x: usize,
        y: usize,
        z: usize,
        chunk_index: usize,
        chunk_data: u32,
    ) -> Result<(), String> {
        let num_tiles = self.num_tiles; // Get num_tiles before mutable borrow
        if let Some(cell) = self.get_mut(x, y, z) {
            let start_bit = chunk_index * 32;
            for bit_offset in 0..32 {
                let tile_id = start_bit + bit_offset;
                if tile_id < num_tiles {
                    cell.set(tile_id, (chunk_data & (1 << bit_offset)) != 0);
                }
            }
            Ok(())
        } else {
            Err(format!(
                "Invalid coordinates ({}, {}, {}) for setting possibility chunk",
                x, y, z
            ))
        }
    }

    /// Returns an immutable reference to the possibility `BitVec` at the given 3D coordinates `(x, y, z)`.
    ///
    /// Returns `None` if the coordinates are out of bounds.
    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> Option<&BitVec> {
        self.index(x, y, z).and_then(|idx| self.data.get(idx))
    }

    /// Returns a mutable reference to the possibility `BitVec` at the given 3D coordinates `(x, y, z)`.
    ///
    /// Returns `None` if the coordinates are out of bounds.
    #[inline]
    pub fn get_mut(&mut self, x: usize, y: usize, z: usize) -> Option<&mut BitVec> {
        self.index(x, y, z)
            .and_then(move |idx| self.data.get_mut(idx))
    }

    /// Collapses a cell to a single specified tile.
    ///
    /// Sets the possibility `BitVec` for the cell at `(x, y, z)` to contain only
    /// the `chosen_tile_id`.
    ///
    /// # Arguments
    /// * `x`, `y`, `z` - Coordinates of the cell to collapse.
    /// * `chosen_tile_id` - The ID (index) of the tile to collapse to.
    ///
    /// # Returns
    /// * `Ok(())` if collapse was successful.
    /// * `Err(String)` if coordinates are out of bounds or the chosen tile ID is invalid.
    pub fn collapse(
        &mut self,
        x: usize,
        y: usize,
        z: usize,
        chosen_tile_id: usize,
    ) -> Result<(), String> {
        if chosen_tile_id >= self.num_tiles {
            return Err(format!(
                "Chosen tile ID {} is out of bounds (num_tiles: {})",
                chosen_tile_id, self.num_tiles
            ));
        }
        if let Some(cell) = self.get_mut(x, y, z) {
            cell.fill(false); // Clear all possibilities
            cell.set(chosen_tile_id, true); // Set only the chosen one
            Ok(())
        } else {
            Err(format!(
                "Collapse coordinates ({}, {}, {}) out of bounds",
                x, y, z
            ))
        }
    }

    /// Checks if the grid is fully collapsed (every cell has exactly one possibility).
    /// Returns `Ok(true)` if fully collapsed, `Ok(false)` otherwise.
    /// Returns `Err(String)` if a contradiction (zero possibilities) or access error occurs.
    pub fn is_fully_collapsed(&self) -> Result<bool, String> {
        let mut fully_collapsed = true;
        for z in 0..self.depth {
            for y in 0..self.height {
                for x in 0..self.width {
                    if let Some(cell) = self.get(x, y, z) {
                        let count = cell.count_ones();
                        if count == 0 {
                            return Err(format!("Contradiction found at ({}, {}, {})", x, y, z));
                        }
                        if count > 1 {
                            fully_collapsed = false;
                            // Optimization: can return early if we only care if *any* cell is not collapsed
                            // return Ok(false);
                        }
                    } else {
                        // Should not happen with valid indices
                        return Err(format!(
                            "Grid access out of bounds at ({}, {}, {})",
                            x, y, z
                        ));
                    }
                }
            }
        }
        Ok(fully_collapsed)
    }
}

// --- EntropyGrid Definition ---

/// Type alias for a 3D grid storing floating-point entropy values for each cell.
/// Typically uses `f32` for entropy representation.
// Type alias for the entropy grid (can still use the generic Grid)
pub type EntropyGrid = Grid<f32>;

#[cfg(test)]
mod tests {
    use super::*;

    // --- Tests for generic Grid<T> ---

    #[test]
    fn test_grid_new() {
        let grid: Grid<i32> = Grid::new(3, 4, 5);
        assert_eq!(grid.width, 3);
        assert_eq!(grid.height, 4);
        assert_eq!(grid.depth, 5);
        assert_eq!(grid.data.len(), 3 * 4 * 5);
        // Check default initialization (i32 defaults to 0)
        assert!(grid.data.iter().all(|&v| v == 0));

        // Test zero dimension
        let grid_zero: Grid<f64> = Grid::new(0, 10, 10);
        assert_eq!(grid_zero.width, 0);
        assert_eq!(grid_zero.height, 10);
        assert_eq!(grid_zero.depth, 10);
        assert!(grid_zero.data.is_empty());

        let grid_zero2: Grid<f64> = Grid::new(10, 0, 10);
        assert!(grid_zero2.data.is_empty());

        let grid_zero3: Grid<f64> = Grid::new(10, 10, 0);
        assert!(grid_zero3.data.is_empty());
    }

    #[test]
    fn test_grid_index() {
        let grid: Grid<u8> = Grid::new(2, 3, 4); // size = 24

        // Test valid indices
        assert_eq!(grid.index(0, 0, 0), Some(0));
        assert_eq!(grid.index(1, 0, 0), Some(1));
        assert_eq!(grid.index(0, 1, 0), Some(2)); // y=1, x=0 -> 0*6 + 1*2 + 0 = 2
        assert_eq!(grid.index(1, 1, 0), Some(3)); // y=1, x=1 -> 0*6 + 1*2 + 1 = 3
        assert_eq!(grid.index(0, 0, 1), Some(6)); // z=1 -> 1*6 + 0*2 + 0 = 6
        assert_eq!(grid.index(1, 2, 3), Some(23)); // z=3, y=2, x=1 -> 3*6 + 2*2 + 1 = 18+4+1=23

        // Test out-of-bounds indices
        assert_eq!(grid.index(2, 0, 0), None); // x out of bounds
        assert_eq!(grid.index(0, 3, 0), None); // y out of bounds
        assert_eq!(grid.index(0, 0, 4), None); // z out of bounds
        assert_eq!(grid.index(2, 3, 4), None); // all out of bounds
    }

    #[test]
    fn test_grid_get() {
        let mut grid: Grid<String> = Grid::new(2, 2, 2);
        grid.data[0] = "Hello".to_string();
        grid.data[7] = "World".to_string();

        assert_eq!(grid.get(0, 0, 0), Some(&"Hello".to_string()));
        assert_eq!(grid.get(1, 1, 1), Some(&"World".to_string())); // Index 7
        assert_eq!(grid.get(1, 0, 0), Some(&String::default())); // Index 1, default value

        // Out of bounds
        assert_eq!(grid.get(2, 0, 0), None);
        assert_eq!(grid.get(0, 2, 0), None);
        assert_eq!(grid.get(0, 0, 2), None);
    }

    #[test]
    fn test_grid_get_mut() {
        let mut grid: Grid<i32> = Grid::new(2, 2, 2);

        // Get mutable reference and modify
        if let Some(val) = grid.get_mut(0, 1, 0) {
            // Index 2
            *val = 42;
        }
        assert_eq!(grid.data[2], 42);

        if let Some(val) = grid.get_mut(1, 1, 1) {
            // Index 7
            *val = -10;
        }
        assert_eq!(grid.data[7], -10);

        // Check original default values weren't changed elsewhere
        assert_eq!(grid.data[0], 0);
        assert_eq!(grid.data[1], 0);

        // Attempt to get out of bounds mutable reference
        assert!(grid.get_mut(2, 0, 0).is_none());
        assert!(grid.get_mut(0, 2, 0).is_none());
        assert!(grid.get_mut(0, 0, 2).is_none());
    }

    // --- Tests for PossibilityGrid ---

    #[test]
    fn test_possibility_grid_new() {
        let grid = PossibilityGrid::new(2, 3, 1, 4); // w=2, h=3, d=1, tiles=4
        assert_eq!(grid.width, 2);
        assert_eq!(grid.height, 3);
        assert_eq!(grid.depth, 1);
        assert_eq!(grid.num_tiles(), 4);
        assert_eq!(grid.data.len(), (2 * 3));

        // Check all bits are set initially
        for cell_bits in &grid.data {
            assert_eq!(cell_bits.len(), 4);
            assert!(cell_bits.all());
        }

        // Test zero dimension
        let grid_zero = PossibilityGrid::new(0, 5, 5, 10);
        assert!(grid_zero.data.is_empty());
    }

    #[test]
    #[should_panic]
    fn test_possibility_grid_new_zero_tiles() {
        let _ = PossibilityGrid::new(2, 2, 2, 0); // Should panic
    }

    #[test]
    fn test_possibility_grid_get() {
        let grid = PossibilityGrid::new(2, 2, 1, 3);
        assert!(grid.get(0, 0, 0).is_some());
        assert!(grid.get(1, 1, 0).is_some());
        assert!(grid.get(2, 0, 0).is_none()); // x out of bounds
        assert!(grid.get(0, 2, 0).is_none()); // y out of bounds
        assert!(grid.get(0, 0, 1).is_none()); // z out of bounds
    }

    #[test]
    fn test_possibility_grid_get_mut() {
        let mut grid = PossibilityGrid::new(2, 1, 1, 5);
        if let Some(cell) = grid.get_mut(0, 0, 0) {
            assert!(cell.all()); // Initially all possible
            cell.set(2, false); // Remove possibility for tile 2
        }
        assert!(!grid.get(0, 0, 0).unwrap()[2]);
        assert!(grid.get_mut(1, 0, 0).is_some());
        assert!(grid.get_mut(2, 0, 0).is_none());
    }

    #[test]
    #[cfg(feature = "serde")] // Guard test with feature flag
    fn test_possibility_grid_serialize_deserialize() {
        let mut grid = PossibilityGrid::new(2, 1, 1, 3);
        *grid.get_mut(0, 0, 0).unwrap() = bitvec![usize, Lsb0; 1, 0, 1];
        *grid.get_mut(1, 0, 0).unwrap() = bitvec![usize, Lsb0; 0, 1, 0];

        let serialized = serde_json::to_string(&grid).expect("Serialization failed");
        let deserialized: PossibilityGrid =
            serde_json::from_str(&serialized).expect("Deserialization failed");

        assert_eq!(grid.width, deserialized.width);
        assert_eq!(grid.height, deserialized.height);
        assert_eq!(grid.depth, deserialized.depth);
        assert_eq!(grid.num_tiles, deserialized.num_tiles);
        assert_eq!(grid.data.len(), deserialized.data.len());
        assert_eq!(grid.get(0, 0, 0), deserialized.get(0, 0, 0));
        assert_eq!(grid.get(1, 0, 0), deserialized.get(1, 0, 0));
        // Check specific bits
        assert!(deserialized.get(0, 0, 0).unwrap()[0]);
        assert!(!deserialized.get(0, 0, 0).unwrap()[1]);
        assert!(deserialized.get(0, 0, 0).unwrap()[2]);
        assert!(!deserialized.get(1, 0, 0).unwrap()[0]);
        assert!(deserialized.get(1, 0, 0).unwrap()[1]);
        assert!(!deserialized.get(1, 0, 0).unwrap()[2]);
    }

    #[test]
    fn test_possibility_grid_is_fully_collapsed() {
        let mut grid = PossibilityGrid::new(2, 1, 1, 3); // 2 cells, 3 tiles

        // Initially not collapsed
        assert_eq!(grid.is_fully_collapsed(), Ok(false));

        // Collapse cell 0
        grid.get_mut(0, 0, 0).unwrap().fill(false);
        grid.get_mut(0, 0, 0).unwrap().set(1, true); // Set to tile 1
        assert_eq!(grid.is_fully_collapsed(), Ok(false)); // Still not fully collapsed

        // Collapse cell 1
        grid.get_mut(1, 0, 0).unwrap().fill(false);
        grid.get_mut(1, 0, 0).unwrap().set(2, true); // Set to tile 2
        assert_eq!(grid.is_fully_collapsed(), Ok(true)); // Now fully collapsed

        // Introduce contradiction
        grid.get_mut(0, 0, 0).unwrap().fill(false);
        assert!(grid.is_fully_collapsed().is_err());
    }
}
