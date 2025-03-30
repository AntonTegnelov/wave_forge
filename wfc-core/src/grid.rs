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
