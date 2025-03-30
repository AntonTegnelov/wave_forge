use bitvec::prelude::*;

#[derive(Debug, Clone)]
pub struct Grid<T> {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub(crate) data: Vec<T>,
}

impl<T: Clone + Default> Grid<T> {
    /// Creates a new grid with the given dimensions, initialized with default values.
    pub fn new(width: usize, height: usize, depth: usize) -> Self {
        let size = width * height * depth;
        let data = vec![T::default(); size];
        Self {
            width,
            height,
            depth,
            data,
        }
    }

    /// Returns an immutable reference to the element at the given coordinates,
    /// or None if the coordinates are out of bounds.
    pub fn get(&self, x: usize, y: usize, z: usize) -> Option<&T> {
        self.index(x, y, z).and_then(|idx| self.data.get(idx))
    }

    /// Returns a mutable reference to the element at the given coordinates,
    /// or None if the coordinates are out of bounds.
    pub fn get_mut(&mut self, x: usize, y: usize, z: usize) -> Option<&mut T> {
        self.index(x, y, z)
            .and_then(move |idx| self.data.get_mut(idx))
    }

    /// Calculates the 1D index for the given 3D coordinates.
    /// Returns None if the coordinates are out of bounds.
    fn index(&self, x: usize, y: usize, z: usize) -> Option<usize> {
        if x < self.width && y < self.height && z < self.depth {
            // Using usize ensures no overflow for reasonable grid sizes
            Some(z * self.width * self.height + y * self.width + x)
        } else {
            None
        }
    }
}

// Type alias for the possibilities grid
pub type PossibilityGrid = Grid<BitVec>;

// Type alias for the entropy grid
pub type EntropyGrid = Grid<f32>;
