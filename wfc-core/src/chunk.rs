//! Chunk and region geometry: what a solver is handed, and where its cells sit in the world.
//!
//! A world is a lattice of chunks of one fixed shape. A chunk is solved as a *region*: the chunk
//! itself widened by a halo on the axes that have neighbouring chunks. The halo is solved and
//! thrown away; it exists because a chunk solved with free faces can leave border tiles that no
//! row of neighbours can complete (docs/solver-fit.md).

use crate::rules::axis_offset;

/// A cell in world coordinates.
pub type WorldCell = [i32; 3];

/// The cells of one chunk.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ChunkShape {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl ChunkShape {
    /// A cubic chunk.
    #[must_use]
    pub const fn cube(side: u32) -> Self {
        Self {
            x: side,
            y: side,
            z: side,
        }
    }

    /// How many cells a chunk holds.
    #[must_use]
    pub const fn cells(self) -> u32 {
        self.x * self.y * self.z
    }

    /// The region a chunk of this shape occupies when widened by `halo` on the given axes.
    #[must_use]
    pub const fn region(self, halo: [u32; 3]) -> RegionShape {
        RegionShape {
            x: self.x + 2 * halo[0],
            y: self.y + 2 * halo[1],
            z: self.z + 2 * halo[2],
            halo,
        }
    }
}

/// Which chunk, in chunk coordinates.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct ChunkCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl ChunkCoord {
    /// A coordinate.
    #[must_use]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Which half of the checkerboard the chunk is on. Chunks of one parity never share a face, so
    /// they can be solved in the same dispatch.
    #[must_use]
    pub const fn parity(self) -> u8 {
        (self.x.rem_euclid(2) + self.y.rem_euclid(2) + self.z.rem_euclid(2)).rem_euclid(2) as u8
    }

    /// The six chunks that share a face with this one, in axis order.
    #[must_use]
    pub fn face_neighbours(self) -> [Self; 6] {
        std::array::from_fn(|axis| {
            let [dx, dy, dz] = axis_offset(axis);
            Self {
                x: self.x + dx,
                y: self.y + dy,
                z: self.z + dz,
            }
        })
    }

    /// The world cell at the chunk's origin corner.
    #[must_use]
    pub const fn origin(self, shape: ChunkShape) -> WorldCell {
        [
            self.x * shape.x as i32,
            self.y * shape.y as i32,
            self.z * shape.z as i32,
        ]
    }

    /// Which chunk a world cell belongs to.
    #[must_use]
    pub const fn of_cell(at: WorldCell, shape: ChunkShape) -> Self {
        Self {
            x: at[0].div_euclid(shape.x as i32),
            y: at[1].div_euclid(shape.y as i32),
            z: at[2].div_euclid(shape.z as i32),
        }
    }

    /// The chunk's identity for the solver's hash, so its choices depend on where it is rather than
    /// on its place in a batch: an evicted chunk regenerates identically.
    #[must_use]
    pub const fn id(self) -> u32 {
        // A hash, not an index: the world is unbounded, so coordinates must fold into 32 bits
        // without a lattice pattern that would correlate neighbours' random streams.
        let mut hash = self.x as u32;
        hash = hash.wrapping_mul(0x9E37_79B1) ^ (self.y as u32).wrapping_mul(0x85EB_CA77);
        hash = hash.rotate_left(13) ^ (self.z as u32).wrapping_mul(0xC2B2_AE3D);
        hash = hash ^ (hash >> 15);
        hash.wrapping_mul(0x2545_F491)
    }
}

/// The cells of a region: a chunk plus its halo.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RegionShape {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    halo: [u32; 3],
}

impl RegionShape {
    /// How many cells a region holds.
    #[must_use]
    pub const fn cells(self) -> u32 {
        self.x * self.y * self.z
    }

    /// The halo widths this region carries.
    #[must_use]
    pub const fn halo(self) -> [u32; 3] {
        self.halo
    }

    /// The chunk shape inside the halo.
    #[must_use]
    pub const fn inner(self) -> ChunkShape {
        ChunkShape {
            x: self.x - 2 * self.halo[0],
            y: self.y - 2 * self.halo[1],
            z: self.z - 2 * self.halo[2],
        }
    }
}

/// One chunk's region, placed in the world.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Region {
    chunk: ChunkCoord,
    shape: RegionShape,
    origin: WorldCell,
}

impl Region {
    /// The region of `chunk`, widened by the region shape's halo.
    #[must_use]
    pub fn new(chunk: ChunkCoord, shape: RegionShape) -> Self {
        let inner = shape.inner();
        let corner = chunk.origin(inner);
        let halo = shape.halo();
        Self {
            chunk,
            shape,
            origin: [
                corner[0] - halo[0] as i32,
                corner[1] - halo[1] as i32,
                corner[2] - halo[2] as i32,
            ],
        }
    }

    /// Which chunk this region solves.
    #[must_use]
    pub const fn chunk(&self) -> ChunkCoord {
        self.chunk
    }

    /// The region's shape.
    #[must_use]
    pub const fn shape(&self) -> RegionShape {
        self.shape
    }

    /// The world cell of the region's first cell.
    #[must_use]
    pub const fn origin(&self) -> WorldCell {
        self.origin
    }

    /// Every cell of the region in row-major order, with whether it belongs to the chunk itself
    /// rather than to the halo.
    pub fn cells(&self) -> impl Iterator<Item = (WorldCell, bool)> + use<> {
        let (shape, origin, halo) = (self.shape, self.origin, self.shape.halo());
        let inner = shape.inner();
        (0..shape.z).flat_map(move |z| {
            (0..shape.y).flat_map(move |y| {
                (0..shape.x).map(move |x| {
                    let inside =
                        |value: u32, halo: u32, extent: u32| value >= halo && value < halo + extent;
                    let is_inner = inside(x, halo[0], inner.x)
                        && inside(y, halo[1], inner.y)
                        && inside(z, halo[2], inner.z);
                    let cell = [
                        origin[0] + x as i32,
                        origin[1] + y as i32,
                        origin[2] + z as i32,
                    ];
                    (cell, is_inner)
                })
            })
        })
    }

    /// Whether the region covers `at`.
    #[must_use]
    pub fn contains(&self, at: WorldCell) -> bool {
        self.index(at).is_some()
    }

    /// Where `at` sits in the region's cell order.
    #[must_use]
    pub fn index(&self, at: WorldCell) -> Option<u32> {
        let (x, y, z) = (
            at[0] - self.origin[0],
            at[1] - self.origin[1],
            at[2] - self.origin[2],
        );
        let within = |value: i32, extent: u32| value >= 0 && (value as u32) < extent;
        (within(x, self.shape.x) && within(y, self.shape.y) && within(z, self.shape.z))
            .then(|| (z as u32 * self.shape.y + y as u32) * self.shape.x + x as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHUNK: ChunkShape = ChunkShape::cube(8);

    #[test]
    fn a_region_widens_only_the_axes_with_a_halo() {
        let shape = CHUNK.region([1, 1, 0]);

        assert_eq!((shape.x, shape.y, shape.z), (10, 10, 8));
        assert_eq!(shape.cells(), 800);
        assert_eq!(shape.inner(), CHUNK);
    }

    #[test]
    fn a_regions_cells_start_at_its_halo_corner_and_mark_the_chunk() {
        let region = Region::new(ChunkCoord::new(1, 0, 0), CHUNK.region([1, 1, 0]));

        assert_eq!(region.origin(), [7, -1, 0]);
        let cells: Vec<(WorldCell, bool)> = region.cells().collect();
        assert_eq!(cells.len(), 800);
        assert_eq!(cells[0], ([7, -1, 0], false));
        assert_eq!(region.index([8, 0, 0]), Some(11));
        assert_eq!(cells[11], ([8, 0, 0], true), "the chunk's own first cell");
        assert_eq!(
            cells.iter().filter(|(_, inner)| *inner).count(),
            CHUNK.cells() as usize
        );
        assert!(!region.contains([17, 0, 0]));
    }

    #[test]
    fn chunks_of_one_parity_never_share_a_face() {
        let chunk = ChunkCoord::new(3, -2, 0);

        for neighbour in chunk.face_neighbours() {
            assert_ne!(chunk.parity(), neighbour.parity());
        }
    }

    #[test]
    fn a_chunk_id_is_stable_and_distinguishes_neighbours() {
        let coord = ChunkCoord::new(-4, 9, 1);

        assert_eq!(coord.id(), ChunkCoord::new(-4, 9, 1).id());
        let neighbours: std::collections::BTreeSet<u32> =
            coord.face_neighbours().iter().map(|c| c.id()).collect();
        assert_eq!(neighbours.len(), 6, "the six neighbours hash apart");
        assert!(!neighbours.contains(&coord.id()));
    }

    #[test]
    fn cells_map_back_to_their_chunk() {
        assert_eq!(
            ChunkCoord::of_cell([8, 0, 0], CHUNK),
            ChunkCoord::new(1, 0, 0)
        );
        assert_eq!(
            ChunkCoord::of_cell([-1, 0, 0], CHUNK),
            ChunkCoord::new(-1, 0, 0)
        );
        assert_eq!(ChunkCoord::new(-1, 0, 0).origin(CHUNK), [-8, 0, 0]);
    }
}
