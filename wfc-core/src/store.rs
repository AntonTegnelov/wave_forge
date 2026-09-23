//! The world's decided tiles, and how a region's starting domains are derived from them.

use crate::ModelError;
use crate::chunk::{ChunkCoord, ChunkShape, Region, WorldCell};
use crate::domains::Domains;
use crate::prior::Prior;
use crate::rules::{AXES, Ruleset, TileMask, axis_offset, opposite};
use std::collections::HashMap;
use std::ops::Range;

/// How far the world reaches, in chunks. An axis without a range is unbounded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorldExtent {
    shape: ChunkShape,
    chunks: [Option<Range<i32>>; 3],
}

impl WorldExtent {
    /// A world of `shape` chunks, unbounded on every axis.
    #[must_use]
    pub fn new(shape: ChunkShape) -> Self {
        Self {
            shape,
            chunks: [None, None, None],
        }
    }

    /// Bounds the world to these chunks along x.
    #[must_use]
    pub fn with_x(mut self, chunks: Range<i32>) -> Self {
        self.chunks[0] = Some(chunks);
        self
    }

    /// Bounds the world to these chunks along y.
    #[must_use]
    pub fn with_y(mut self, chunks: Range<i32>) -> Self {
        self.chunks[1] = Some(chunks);
        self
    }

    /// Bounds the world to these chunks along z. A city is one chunk tall.
    #[must_use]
    pub fn with_z(mut self, chunks: Range<i32>) -> Self {
        self.chunks[2] = Some(chunks);
        self
    }

    /// The shape of every chunk.
    #[must_use]
    pub const fn shape(&self) -> ChunkShape {
        self.shape
    }

    /// Whether the world holds this chunk.
    #[must_use]
    pub fn contains_chunk(&self, coord: ChunkCoord) -> bool {
        let coords = [coord.x, coord.y, coord.z];
        self.chunks
            .iter()
            .zip(coords)
            .all(|(bounds, coord)| bounds.as_ref().is_none_or(|range| range.contains(&coord)))
    }

    /// Whether the world holds this cell.
    #[must_use]
    pub fn contains_cell(&self, at: WorldCell) -> bool {
        self.contains_chunk(ChunkCoord::of_cell(at, self.shape))
    }

    /// The world z of its lowest cell; layer masks are indexed from here.
    #[must_use]
    pub fn lowest_cell_z(&self) -> i32 {
        self.chunks[2]
            .as_ref()
            .map_or(0, |range| range.start * self.shape.z as i32)
    }

    /// The chunks the world spans along `axis` (0 for x, 1 for y, 2 for z), or `None` if it is
    /// unbounded along it.
    ///
    /// # Panics
    /// If `axis` is not 0, 1 or 2.
    #[must_use]
    pub fn chunks_along(&self, axis: usize) -> Option<Range<i32>> {
        self.chunks[axis].clone()
    }

    /// Halo widths for a region, `halo` cells on every axis that has neighbouring chunks. An axis
    /// only one chunk wide gets none: there is nothing on the other side to agree with.
    #[must_use]
    pub fn halo(&self, halo: u32) -> [u32; 3] {
        std::array::from_fn(|axis| {
            let single = self.chunks[axis]
                .as_ref()
                .is_some_and(|range| range.len() <= 1);
            if single { 0 } else { halo }
        })
    }

    /// Every chunk of a bounded world, in row-major order.
    ///
    /// # Panics
    /// If any axis is unbounded.
    #[must_use]
    pub fn chunks(&self) -> Vec<ChunkCoord> {
        let bounds: Vec<Range<i32>> = self
            .chunks
            .iter()
            .map(|axis| axis.clone().expect("a bounded world"))
            .collect();
        bounds[2]
            .clone()
            .flat_map(|z| {
                let (x, y) = (bounds[0].clone(), bounds[1].clone());
                y.flat_map(move |y| x.clone().map(move |x| ChunkCoord::new(x, y, z)))
            })
            .collect()
    }
}

/// One solved chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chunk {
    pub coord: ChunkCoord,
    /// One tile per cell, in row-major order.
    pub tiles: Box<[u16]>,
    /// How often the chunk has been written. A repair rewrites neighbours, which bumps theirs.
    pub version: u32,
}

impl Chunk {
    /// The tile at a cell of the chunk.
    ///
    /// # Panics
    /// If the cell is out of range.
    #[must_use]
    pub fn tile(&self, shape: ChunkShape, at: WorldCell) -> u16 {
        let origin = self.coord.origin(shape);
        let (x, y, z) = (at[0] - origin[0], at[1] - origin[1], at[2] - origin[2]);
        let index = (z as u32 * shape.y + y as u32) * shape.x + x as u32;
        self.tiles[index as usize]
    }
}

/// Every chunk the world has decided so far.
#[derive(Clone, Debug)]
pub struct ChunkStore {
    extent: WorldExtent,
    chunks: HashMap<ChunkCoord, Chunk>,
}

impl ChunkStore {
    /// An empty world.
    #[must_use]
    pub fn new(extent: WorldExtent) -> Self {
        Self {
            extent,
            chunks: HashMap::new(),
        }
    }

    /// How far the world reaches.
    #[must_use]
    pub const fn extent(&self) -> &WorldExtent {
        &self.extent
    }

    /// The shape of every chunk.
    #[must_use]
    pub const fn shape(&self) -> ChunkShape {
        self.extent.shape()
    }

    /// A solved chunk.
    #[must_use]
    pub fn get(&self, coord: ChunkCoord) -> Option<&Chunk> {
        self.chunks.get(&coord)
    }

    /// Whether the chunk has been solved.
    #[must_use]
    pub fn contains(&self, coord: ChunkCoord) -> bool {
        self.chunks.contains_key(&coord)
    }

    /// How many chunks are solved.
    #[must_use]
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Whether nothing is solved yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Every solved chunk.
    pub fn iter(&self) -> impl Iterator<Item = &Chunk> {
        self.chunks.values()
    }

    /// The decided tile at a world cell.
    #[must_use]
    pub fn tile(&self, at: WorldCell) -> Option<u16> {
        let shape = self.shape();
        self.get(ChunkCoord::of_cell(at, shape))
            .map(|chunk| chunk.tile(shape, at))
    }

    /// Adds a chunk, for example one a game had persisted.
    ///
    /// # Errors
    /// If the chunk is outside the world or has the wrong number of cells.
    pub fn insert(&mut self, chunk: Chunk) -> Result<(), ModelError> {
        if !self.extent.contains_chunk(chunk.coord) {
            return Err(ModelError::OutsideWorld { chunk: chunk.coord });
        }
        let cells = self.shape().cells() as usize;
        if chunk.tiles.len() != cells {
            return Err(ModelError::ChunkCells {
                expected: cells,
                got: chunk.tiles.len(),
            });
        }
        self.chunks.insert(chunk.coord, chunk);
        Ok(())
    }

    /// Drops a chunk. A world regenerates it identically as long as no repair has rewritten its
    /// neighbours since (docs/architecture/world.md, "What determinism means here").
    pub fn remove(&mut self, coord: ChunkCoord) -> Option<Chunk> {
        self.chunks.remove(&coord)
    }

    /// Writes a solved region: the chunk's own cells, plus, when `release` is set, the halo cells
    /// that were already decided and have therefore been solved again.
    ///
    /// # Errors
    /// If a cell that should be written is not decided.
    pub fn commit(
        &mut self,
        region: &Region,
        domains: &Domains,
        release: bool,
    ) -> Result<Vec<ChunkCoord>, ModelError> {
        let shape = self.shape();
        let mut touched: Vec<ChunkCoord> = Vec::new();
        for (index, (at, inner)) in region.cells().enumerate() {
            if !self.extent.contains_cell(at) {
                continue;
            }
            let rewriting = release && self.tile(at).is_some();
            if !inner && !rewriting {
                continue;
            }
            let Some(tile) = domains.decided(index as u32) else {
                return Err(ModelError::Undecided { cell: at });
            };
            let coord = ChunkCoord::of_cell(at, shape);
            let chunk = self.chunks.entry(coord).or_insert_with(|| Chunk {
                coord,
                tiles: vec![0u16; shape.cells() as usize].into_boxed_slice(),
                version: 0,
            });
            let origin = coord.origin(shape);
            let (x, y, z) = (at[0] - origin[0], at[1] - origin[1], at[2] - origin[2]);
            let cell = (z as u32 * shape.y + y as u32) * shape.x + x as u32;
            chunk.tiles[cell as usize] = u16::try_from(tile).expect("a tile index fits u16");
            if !touched.contains(&coord) {
                touched.push(coord);
            }
        }
        for coord in &touched {
            if let Some(chunk) = self.chunks.get_mut(coord) {
                chunk.version += 1;
            }
        }
        Ok(touched)
    }
}

/// The starting domains of a region.
///
/// Cells already decided inside it are pinned to their tile, and a decided cell just outside
/// restricts the region cell next to it, which is the same as keeping a fixed halo. `release` frees
/// the decided cells so a repair can solve them again.
///
/// Decided cells are only read from chunks of the *other* parity, the ones a batch cannot contain.
/// A chunk therefore depends on its face neighbours and nothing else, so its tiles do not depend on
/// the order in which the world was generated. The cells this skips are diagonal halo cells, which
/// never touch a face of the chunk itself and are discarded with the rest of the halo.
#[must_use]
pub fn region_init(
    store: &ChunkStore,
    prior: &Prior,
    ruleset: &Ruleset,
    region: &Region,
    release: bool,
) -> Domains {
    let extent = store.extent();
    let shape = store.shape();
    let parity = region.chunk().parity();
    let readable = |at: WorldCell| -> Option<u16> {
        let same_parity = ChunkCoord::of_cell(at, shape).parity() == parity;
        if same_parity && !release {
            return None;
        }
        store.tile(at)
    };
    let masks = region.cells().map(|(at, _)| {
        if !extent.contains_cell(at) {
            return prior.open_domain(at, extent);
        }
        if let Some(tile) = readable(at).filter(|_| !release) {
            return TileMask::single(u32::from(tile));
        }
        let mut mask = prior.domain(at, extent);
        for axis in 0..AXES {
            let [dx, dy, dz] = axis_offset(axis);
            let neighbour = [at[0] + dx, at[1] + dy, at[2] + dz];
            if region.contains(neighbour) {
                continue;
            }
            if let Some(tile) = readable(neighbour) {
                // The neighbour lies along `axis` from this cell, so this cell lies along the
                // opposite axis from the neighbour.
                mask = mask.intersect(
                    ruleset
                        .table()
                        .allowed_next_to(u32::from(tile), opposite(axis)),
                );
            }
        }
        mask
    });
    Domains::from_masks(ruleset.words_per_cell(), masks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wfc_rules::AdjacencyRules;

    const CHUNK: ChunkShape = ChunkShape { x: 2, y: 2, z: 1 };

    /// Four tiles that may sit next to anything, so only pinning and folding show up.
    fn permissive() -> Ruleset {
        let tuples: Vec<(usize, usize, usize)> = (0..AXES)
            .flat_map(|axis| (0..4).flat_map(move |a| (0..4).map(move |b| (axis, a, b))))
            .collect();
        Ruleset::new(
            &AdjacencyRules::from_allowed_tuples(4, AXES, tuples),
            &[1.0; 4],
        )
        .expect("permissive rules")
    }

    /// Tile `t` may only sit east of tile `t`, and anything may sit elsewhere.
    fn matching() -> Ruleset {
        let tuples: Vec<(usize, usize, usize)> = (0..AXES)
            .flat_map(|axis| {
                (0..4).flat_map(move |a| {
                    (0..4).filter_map(move |b| ((axis > 1) || a == b).then_some((axis, a, b)))
                })
            })
            .collect();
        Ruleset::new(
            &AdjacencyRules::from_allowed_tuples(4, AXES, tuples),
            &[1.0; 4],
        )
        .expect("matching rules")
    }

    fn store() -> ChunkStore {
        ChunkStore::new(WorldExtent::new(CHUNK).with_z(0..1))
    }

    fn solid_chunk(coord: ChunkCoord, tile: u16) -> Chunk {
        Chunk {
            coord,
            tiles: vec![tile; CHUNK.cells() as usize].into_boxed_slice(),
            version: 1,
        }
    }

    #[test]
    fn a_region_pins_decided_cells_of_the_other_parity() {
        let mut store = store();
        store
            .insert(solid_chunk(ChunkCoord::new(0, 0, 0), 2))
            .expect("in the world");
        let region = Region::new(ChunkCoord::new(1, 0, 0), CHUNK.region([1, 1, 0]));

        let init = region_init(&store, &Prior::open(4), &permissive(), &region, false);

        let pinned = region
            .index([1, 0, 0])
            .expect("the halo covers the neighbour's cell");
        assert_eq!(init.decided(pinned), Some(2));
        let own = region.index([2, 0, 0]).expect("the chunk's own cell");
        assert_eq!(init.count(own), 4);
    }

    #[test]
    fn a_decided_cell_outside_the_region_restricts_the_cell_next_to_it() {
        let mut store = store();
        store
            .insert(solid_chunk(ChunkCoord::new(0, 0, 0), 3))
            .expect("in the world");
        // No halo: the neighbour's cells are outside the region, so they can only fold in.
        let region = Region::new(ChunkCoord::new(1, 0, 0), CHUNK.region([0, 0, 0]));

        let init = region_init(&store, &Prior::open(4), &matching(), &region, false);

        let border = region
            .index([2, 0, 0])
            .expect("the chunk's own border cell");
        assert_eq!(
            init.decided(border),
            Some(3),
            "matching rules force the same tile"
        );
    }

    #[test]
    fn a_same_parity_halo_cell_is_left_free() {
        let mut store = store();
        // (1, 0, 0) shares a face with the chunk being solved, so it is of the other parity.
        store
            .insert(solid_chunk(ChunkCoord::new(1, 0, 0), 1))
            .expect("in the world");
        // (1, 1, 0) lies diagonally across the corner and has the same parity: it could be in the
        // same batch, so its tiles are not read. No cell of it touches a face of this chunk.
        store
            .insert(solid_chunk(ChunkCoord::new(1, 1, 0), 2))
            .expect("in the world");
        let region = Region::new(ChunkCoord::new(0, 0, 0), CHUNK.region([1, 1, 0]));

        let init = region_init(&store, &Prior::open(4), &permissive(), &region, false);

        let across_the_face = region.index([2, 0, 0]).expect("in the halo");
        let across_the_corner = region.index([2, 2, 0]).expect("in the halo");
        assert_eq!(init.decided(across_the_face), Some(1));
        assert_eq!(
            init.count(across_the_corner),
            4,
            "solved, but not readable from here"
        );
    }

    #[test]
    fn a_release_frees_the_cells_a_repair_may_rewrite() {
        let mut store = store();
        store
            .insert(solid_chunk(ChunkCoord::new(1, 0, 0), 1))
            .expect("in the world");
        let region = Region::new(ChunkCoord::new(0, 0, 0), CHUNK.region([1, 1, 0]));

        let init = region_init(&store, &Prior::open(4), &permissive(), &region, true);

        let across_the_face = region.index([2, 0, 0]).expect("in the halo");
        assert_eq!(
            init.count(across_the_face),
            4,
            "a repair may choose these cells again"
        );
    }

    #[test]
    fn committing_writes_the_chunk_and_reports_it() {
        let mut store = store();
        let region = Region::new(ChunkCoord::new(0, 0, 0), CHUNK.region([1, 1, 0]));
        let solved = Domains::from_masks(
            1,
            region
                .cells()
                .map(|(at, _)| TileMask::single(if at[0] == 0 { 1 } else { 2 })),
        );

        let touched = store
            .commit(&region, &solved, false)
            .expect("every cell decided");

        assert_eq!(touched, vec![ChunkCoord::new(0, 0, 0)]);
        assert_eq!(store.tile([0, 0, 0]), Some(1));
        assert_eq!(store.tile([1, 0, 0]), Some(2));
        assert_eq!(store.tile([-1, 0, 0]), None, "the halo is not committed");
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn a_release_rewrites_the_halo_cells_that_were_decided() {
        let mut store = store();
        store
            .insert(solid_chunk(ChunkCoord::new(-1, 0, 0), 0))
            .expect("in the world");
        let region = Region::new(ChunkCoord::new(0, 0, 0), CHUNK.region([1, 1, 0]));
        let solved = Domains::from_masks(1, region.cells().map(|_| TileMask::single(3)));

        let touched = store
            .commit(&region, &solved, true)
            .expect("every cell decided");

        assert!(
            touched.contains(&ChunkCoord::new(-1, 0, 0)),
            "the rewritten neighbour is reported"
        );
        assert_eq!(store.tile([-1, 0, 0]), Some(3));
        assert_eq!(
            store
                .get(ChunkCoord::new(-1, 0, 0))
                .expect("still there")
                .version,
            2
        );
    }

    #[test]
    fn an_undecided_cell_cannot_be_committed() {
        let mut store = store();
        let region = Region::new(ChunkCoord::new(0, 0, 0), CHUNK.region([0, 0, 0]));
        let unsolved = Domains::filled(region.shape().cells(), 4);

        assert!(matches!(
            store.commit(&region, &unsolved, false),
            Err(ModelError::Undecided { .. })
        ));
    }

    #[test]
    fn a_single_chunk_axis_gets_no_halo() {
        let extent = WorldExtent::new(CHUNK).with_x(0..1).with_z(0..1);

        assert_eq!(
            extent.halo(1),
            [0, 1, 0],
            "x and z hold one chunk each, y is unbounded"
        );
    }

    #[test]
    fn a_bounded_world_lists_its_chunks_in_row_major_order() {
        let extent = WorldExtent::new(CHUNK)
            .with_x(0..2)
            .with_y(0..2)
            .with_z(0..1);

        let chunks = extent.chunks();

        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0], ChunkCoord::new(0, 0, 0));
        assert_eq!(chunks[1], ChunkCoord::new(1, 0, 0));
        assert_eq!(chunks[3], ChunkCoord::new(1, 1, 0));
    }
}
