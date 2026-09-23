//! Where the lattice sits in a Y-up engine's world space.
//!
//! The lattice is Z-up: x and y run along the ground and z points up. Godot and Bevy are Y-up, so
//! an engine's x is the lattice's x, its y is the lattice's z, and its z is the lattice's y. Every
//! integration maps positions and rotations through [`YUpSpace`], so the two engines cannot come to
//! disagree about where a cell is.
//!
//! Swapping two axes mirrors the world. That is harmless as long as everything goes through the
//! same mapping, models included, and it has one consequence worth knowing: a tile rotated
//! counter-clockwise about the lattice's +z (from +x towards +y) turns from +x towards +z in the
//! engine, which is a *negative* turn about the engine's +Y. [`YUpSpace::yaw`] gives that angle.

use wfc_core::{ChunkCoord, ChunkShape};

/// The lattice in a Y-up world: chunks of one shape, cells of one size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YUpSpace {
    chunk: ChunkShape,
    /// One cell's size along the engine's x, y and z.
    cell_size: [f32; 3],
}

impl YUpSpace {
    /// Chunks of `chunk` cells, each cell `cell_size` along the engine's own x, y and z.
    #[must_use]
    pub const fn new(chunk: ChunkShape, cell_size: [f32; 3]) -> Self {
        Self { chunk, cell_size }
    }

    /// One cell's size along the engine's x, y and z.
    #[must_use]
    pub const fn cell_size(&self) -> [f32; 3] {
        self.cell_size
    }

    /// One chunk's size along the engine's x, y and z.
    #[must_use]
    pub fn chunk_size(&self) -> [f32; 3] {
        [
            self.cell_size[0] * self.chunk.x as f32,
            self.cell_size[1] * self.chunk.z as f32,
            self.cell_size[2] * self.chunk.y as f32,
        ]
    }

    /// The chunk a point in the engine's world space falls in.
    #[must_use]
    pub fn chunk_at(&self, point: [f32; 3]) -> ChunkCoord {
        let size = self.chunk_size();
        ChunkCoord::new(
            (point[0] / size[0]).floor() as i32,
            (point[2] / size[2]).floor() as i32,
            (point[1] / size[1]).floor() as i32,
        )
    }

    /// Where a chunk's lowest corner sits in the engine's world space.
    #[must_use]
    pub fn chunk_origin(&self, chunk: ChunkCoord) -> [f32; 3] {
        let size = self.chunk_size();
        [
            chunk.x as f32 * size[0],
            chunk.z as f32 * size[1],
            chunk.y as f32 * size[2],
        ]
    }

    /// The centre of a cell in the engine's world space. `cell` indexes a chunk's cells the way
    /// its tiles are stored: x fastest, then y, then z.
    ///
    /// # Panics
    /// If `cell` is not a cell of the chunk.
    #[must_use]
    pub fn cell_center(&self, chunk: ChunkCoord, cell: u32) -> [f32; 3] {
        let shape = self.chunk;
        assert!(
            cell < shape.cells(),
            "cell {cell} of a chunk of {} cells",
            shape.cells()
        );
        let x = cell % shape.x;
        let y = (cell / shape.x) % shape.y;
        let z = cell / (shape.x * shape.y);
        let origin = self.chunk_origin(chunk);
        [
            origin[0] + (x as f32 + 0.5) * self.cell_size[0],
            origin[1] + (z as f32 + 0.5) * self.cell_size[1],
            origin[2] + (y as f32 + 0.5) * self.cell_size[2],
        ]
    }

    /// The turn about the engine's +Y, in radians, that matches `quarter_turns` counter-clockwise
    /// about the lattice's +z: the rotation a tile's model is placed with.
    #[must_use]
    pub fn yaw(quarter_turns: u8) -> f32 {
        -f32::from(quarter_turns % 4) * std::f32::consts::FRAC_PI_2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHAPE: ChunkShape = ChunkShape { x: 4, y: 2, z: 3 };

    fn space() -> YUpSpace {
        YUpSpace::new(SHAPE, [2.0, 1.0, 0.5])
    }

    #[test]
    fn the_lattice_z_is_the_engine_y() {
        let size = space().chunk_size();

        assert_eq!(
            size,
            [8.0, 3.0, 1.0],
            "x 4 cells of 2, up 3 cells of 1, z 2 cells of 0.5"
        );
    }

    #[test]
    fn a_chunk_contains_its_own_corner_and_cell_centres() {
        let space = space();
        let chunk = ChunkCoord::new(-2, 5, 1);

        assert_eq!(space.chunk_at(space.chunk_origin(chunk)), chunk);
        for cell in 0..SHAPE.cells() {
            assert_eq!(
                space.chunk_at(space.cell_center(chunk, cell)),
                chunk,
                "cell {cell}"
            );
        }
    }

    #[test]
    fn cells_are_indexed_x_first_then_y_then_z() {
        let space = space();
        let chunk = ChunkCoord::new(0, 0, 0);

        assert_eq!(space.cell_center(chunk, 0), [1.0, 0.5, 0.25]);
        assert_eq!(
            space.cell_center(chunk, 1),
            [3.0, 0.5, 0.25],
            "next along x"
        );
        assert_eq!(
            space.cell_center(chunk, 4),
            [1.0, 0.5, 0.75],
            "next along y, the engine's z"
        );
        assert_eq!(
            space.cell_center(chunk, 8),
            [1.0, 1.5, 0.25],
            "next along z, the engine's y"
        );
    }

    #[test]
    fn a_quarter_turn_takes_the_lattice_x_to_the_lattice_y() {
        // The lattice's +y is the engine's +z. A turn by `yaw` about +Y takes (1, 0, 0) to
        // (cos, 0, -sin).
        let yaw = YUpSpace::yaw(1);

        let turned = [yaw.cos(), 0.0, -yaw.sin()];

        assert!(
            (turned[0]).abs() < 1e-6 && (turned[2] - 1.0).abs() < 1e-6,
            "{turned:?}"
        );
        assert_eq!(YUpSpace::yaw(4), YUpSpace::yaw(0));
    }
}
