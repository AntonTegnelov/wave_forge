use bitvec::prelude::*;

/// Generic 3D Grid structure.
#[derive(Debug, Clone)] // Keep Clone for tests, consider removing if large grids are cloned often
pub struct Grid<T> {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub(crate) data: Vec<T>,
}

impl<T: Clone + Default> Grid<T> {
    /// Creates a new generic grid with the given dimensions, initialized with default values.
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

    /// Calculates the 1D index for the given 3D coordinates.
    /// Returns None if the coordinates are out of bounds.
    #[inline] // Make indexing inline for performance
    fn index(&self, x: usize, y: usize, z: usize) -> Option<usize> {
        if x < self.width && y < self.height && z < self.depth {
            Some(z * self.width * self.height + y * self.width + x)
        } else {
            None
        }
    }

    /// Returns an immutable reference to the element at the given coordinates,
    /// or None if the coordinates are out of bounds.
    #[inline] // Make access inline
    pub fn get(&self, x: usize, y: usize, z: usize) -> Option<&T> {
        self.index(x, y, z).and_then(|idx| self.data.get(idx))
    }

    /// Returns a mutable reference to the element at the given coordinates,
    /// or None if the coordinates are out of bounds.
    #[inline] // Make access inline
    pub fn get_mut(&mut self, x: usize, y: usize, z: usize) -> Option<&mut T> {
        self.index(x, y, z)
            .and_then(move |idx| self.data.get_mut(idx))
    }
}

// --- PossibilityGrid Definition ---

/// Specific Grid implementation for storing tile possibilities using BitVec.
#[derive(Debug, Clone)]
pub struct PossibilityGrid {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    num_tiles: usize, // Store num_tiles for validation/consistency
    data: Vec<BitVec>,
}

impl PossibilityGrid {
    /// Creates a new PossibilityGrid initialized with all tiles possible for every cell.
    ///
    /// # Arguments
    /// * `width` - Grid width (X dimension).
    /// * `height` - Grid height (Y dimension).
    /// * `depth` - Grid depth (Z dimension).
    /// * `num_tiles` - The total number of unique tile types.
    ///
    /// # Panics
    /// Panics if `num_tiles` is 0.
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
        let all_possible = bitvec![1; num_tiles];
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
    pub fn num_tiles(&self) -> usize {
        self.num_tiles
    }

    /// Provides read-only access to the internal data vector containing BitVecs for each cell.
    /// Intended for scenarios like GPU buffer packing where direct access is needed.
    pub fn get_cell_data(&self) -> &Vec<BitVec> {
        &self.data
    }

    /// Calculates the 1D index for the given 3D coordinates.
    /// Returns None if the coordinates are out of bounds.
    #[inline]
    fn index(&self, x: usize, y: usize, z: usize) -> Option<usize> {
        if x < self.width && y < self.height && z < self.depth {
            Some(z * self.width * self.height + y * self.width + x)
        } else {
            None
        }
    }

    /// Returns an immutable reference to the possibility BitVec at the given coordinates,
    /// or None if the coordinates are out of bounds.
    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> Option<&BitVec> {
        self.index(x, y, z).and_then(|idx| self.data.get(idx))
    }

    /// Returns a mutable reference to the possibility BitVec at the given coordinates,
    /// or None if the coordinates are out of bounds.
    #[inline]
    pub fn get_mut(&mut self, x: usize, y: usize, z: usize) -> Option<&mut BitVec> {
        self.index(x, y, z)
            .and_then(move |idx| self.data.get_mut(idx))
    }
}

// --- EntropyGrid Definition ---

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
        let grid = PossibilityGrid::new(2, 3, 4, 5); // 24 cells, 5 tiles
        assert_eq!(grid.width, 2);
        assert_eq!(grid.height, 3);
        assert_eq!(grid.depth, 4);
        assert_eq!(grid.num_tiles(), 5);
        assert_eq!(grid.data.len(), 2 * 3 * 4);

        // Check that all cells are initialized with all bits set
        let expected_bv = bitvec![1; 5];
        assert!(grid.data.iter().all(|bv| *bv == expected_bv));

        // Test zero dimension
        let grid_zero = PossibilityGrid::new(0, 10, 10, 3);
        assert!(grid_zero.data.is_empty());
        assert_eq!(grid_zero.num_tiles(), 3);
    }

    #[test]
    #[should_panic(expected = "num_tiles must be greater than 0")]
    fn test_possibility_grid_new_zero_tiles() {
        let _ = PossibilityGrid::new(2, 2, 2, 0);
    }

    #[test]
    fn test_possibility_grid_get() {
        let grid = PossibilityGrid::new(2, 2, 2, 3);
        let expected_bv = bitvec![1; 3];

        assert_eq!(grid.get(0, 0, 0), Some(&expected_bv));
        assert_eq!(grid.get(1, 1, 1), Some(&expected_bv));

        // Out of bounds
        assert_eq!(grid.get(2, 0, 0), None);
    }

    #[test]
    fn test_possibility_grid_get_mut() {
        let mut grid = PossibilityGrid::new(2, 2, 2, 4);
        let expected_initial = bitvec![1; 4];
        let modified_bv = bitvec![0, 1, 0, 1];

        // Check initial state
        assert_eq!(grid.get(0, 1, 0).unwrap(), &expected_initial);

        // Get mutable reference and modify
        if let Some(bv) = grid.get_mut(0, 1, 0) {
            // Index 2
            *bv = modified_bv.clone();
        }
        assert_eq!(grid.get(0, 1, 0).unwrap(), &modified_bv);

        // Check other cells weren't affected
        assert_eq!(grid.get(0, 0, 0).unwrap(), &expected_initial);
        assert_eq!(grid.get(1, 1, 1).unwrap(), &expected_initial);

        // Attempt to get out of bounds mutable reference
        assert!(grid.get_mut(2, 0, 0).is_none());
    }

    #[test]
    fn test_possibility_grid_num_tiles() {
        let grid = PossibilityGrid::new(1, 1, 1, 10);
        assert_eq!(grid.num_tiles(), 10);
    }
}
