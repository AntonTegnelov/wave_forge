//! Checks that generated grids obey the rules they were generated from.
//!
//! A result that looks plausible can still break adjacency rules in a handful of cells, and that
//! sparse kind of corruption is exactly what GPU races or host/shader layout mismatches produce.
//! Tests therefore assert invariants mechanically instead of relying on someone looking at an
//! image.

use wfc_core::{Chunk, ChunkCoord, ChunkShape, Domains};
use wfc_rules::AdjacencyRules;

/// How a grid's edges behave when the checker looks for a neighbour.
///
/// A generated world has no edges to speak of: it reaches as far as its extent, and what lies
/// beyond a chunk's faces is either a solved neighbour or the prior's own masks. The distinction
/// survives here because a checker still has to decide what the cell past the last one is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BoundaryCondition {
    /// Edges wrap around, so the grid is a torus.
    Periodic,
    /// The grid ends; a cell outside it has no tile to disagree with.
    #[default]
    Finite,
}

/// Neighbour offsets in the axis order used by [`AdjacencyRules`]: +x, -x, +y, -y, +z, -z.
pub const AXIS_OFFSETS: [(isize, isize, isize); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

/// A fully collapsed grid: exactly one tile index per cell.
///
/// The solver works on domains and the store on chunks; this is the flat, decided grid that checks
/// and renderers want, and what the CLI's text output reads back into.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileGrid {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    tiles: Vec<usize>,
}

impl TileGrid {
    /// Creates a grid from tiles in `z`, then `y`, then `x` order.
    pub fn new(
        width: usize,
        height: usize,
        depth: usize,
        tiles: Vec<usize>,
    ) -> Result<Self, String> {
        if tiles.len() != width * height * depth {
            return Err(format!(
                "expected {} tiles for a {width}x{height}x{depth} grid, got {}",
                width * height * depth,
                tiles.len()
            ));
        }
        Ok(Self {
            width,
            height,
            depth,
            tiles,
        })
    }

    /// Converts a solved region, failing on any cell that is not decided.
    ///
    /// # Errors
    /// If the domains do not describe a `width` x `height` x `depth` grid, or a cell is undecided.
    pub fn from_domains(
        domains: &Domains,
        width: usize,
        height: usize,
        depth: usize,
    ) -> Result<Self, String> {
        let tiles = (0..domains.cells())
            .map(|cell| {
                domains
                    .decided(cell)
                    .map(|tile| tile as usize)
                    .ok_or_else(|| {
                        format!("cell {cell} has {} possible tiles", domains.count(cell))
                    })
            })
            .collect::<Result<Vec<usize>, String>>()?;
        Self::new(width, height, depth, tiles)
    }

    /// Lays chunks of one `shape` out as a single grid covering their bounding box, with `fill` in
    /// every cell no chunk covers, and returns the grid with the chunk at its lowest corner.
    ///
    /// # Errors
    /// If there are no chunks, or a chunk does not hold one tile per cell of `shape`.
    pub fn from_chunks<'a>(
        shape: ChunkShape,
        chunks: impl IntoIterator<Item = &'a Chunk>,
        fill: usize,
    ) -> Result<(Self, ChunkCoord), String> {
        let chunks: Vec<&Chunk> = chunks.into_iter().collect();
        let Some(first) = chunks.first() else {
            return Err("no chunks to lay out".to_owned());
        };
        let (mut low, mut high) = (first.coord, first.coord);
        for chunk in &chunks {
            if chunk.tiles.len() != shape.cells() as usize {
                return Err(format!(
                    "chunk {:?} has {} tiles, a {shape:?} chunk has {}",
                    chunk.coord,
                    chunk.tiles.len(),
                    shape.cells()
                ));
            }
            low = ChunkCoord::new(
                low.x.min(chunk.coord.x),
                low.y.min(chunk.coord.y),
                low.z.min(chunk.coord.z),
            );
            high = ChunkCoord::new(
                high.x.max(chunk.coord.x),
                high.y.max(chunk.coord.y),
                high.z.max(chunk.coord.z),
            );
        }
        let (sx, sy, sz) = (shape.x as usize, shape.y as usize, shape.z as usize);
        let width = (high.x - low.x + 1) as usize * sx;
        let height = (high.y - low.y + 1) as usize * sy;
        let depth = (high.z - low.z + 1) as usize * sz;
        let mut tiles = vec![fill; width * height * depth];
        for chunk in &chunks {
            let (ox, oy, oz) = (
                (chunk.coord.x - low.x) as usize * sx,
                (chunk.coord.y - low.y) as usize * sy,
                (chunk.coord.z - low.z) as usize * sz,
            );
            for (index, &tile) in chunk.tiles.iter().enumerate() {
                let (x, y, z) = (index % sx, index / sx % sy, index / (sx * sy));
                tiles[((oz + z) * height + oy + y) * width + ox + x] = usize::from(tile);
            }
        }
        Ok((Self::new(width, height, depth, tiles)?, low))
    }

    /// Parses the text format the `wave-forge` CLI writes: space-separated tile indices along
    /// `x`, one line per `y` row, and a blank line between `z` layers.
    pub fn parse_text(text: &str) -> Result<Self, String> {
        let mut layers: Vec<Vec<Vec<usize>>> = vec![Vec::new()];
        for (line_no, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                if layers.last().is_some_and(|layer| !layer.is_empty()) {
                    layers.push(Vec::new());
                }
                continue;
            }
            let row = line
                .split_whitespace()
                .map(|token| {
                    token
                        .parse()
                        .map_err(|_| format!("line {}: '{token}' is not a tile index", line_no + 1))
                })
                .collect::<Result<Vec<usize>, _>>()?;
            layers.last_mut().expect("at least one layer").push(row);
        }
        if layers.last().is_some_and(Vec::is_empty) {
            layers.pop();
        }

        let depth = layers.len();
        let height = layers.first().map_or(0, Vec::len);
        let width = layers
            .first()
            .and_then(|layer| layer.first())
            .map_or(0, Vec::len);
        let mut tiles = Vec::with_capacity(width * height * depth);
        for (z, layer) in layers.iter().enumerate() {
            if layer.len() != height {
                return Err(format!(
                    "layer {z} has {} rows, expected {height}",
                    layer.len()
                ));
            }
            for (y, row) in layer.iter().enumerate() {
                if row.len() != width {
                    return Err(format!(
                        "layer {z}, row {y} has {} tiles, expected {width}",
                        row.len()
                    ));
                }
                tiles.extend_from_slice(row);
            }
        }
        Self::new(width, height, depth, tiles)
    }

    /// Writes the text format [`TileGrid::parse_text`] reads.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut text = String::new();
        for z in 0..self.depth {
            if z > 0 {
                text.push('\n');
            }
            for y in 0..self.height {
                for x in 0..self.width {
                    if x > 0 {
                        text.push(' ');
                    }
                    text.push_str(&self.get(x, y, z).to_string());
                }
                text.push('\n');
            }
        }
        text
    }

    const fn index(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.height + y) * self.width + x
    }

    /// Tile at a cell. Panics when out of bounds, like slice indexing.
    pub fn get(&self, x: usize, y: usize, z: usize) -> usize {
        assert!(
            x < self.width && y < self.height && z < self.depth,
            "cell ({x}, {y}, {z}) out of bounds"
        );
        self.tiles[self.index(x, y, z)]
    }

    /// Number of cells holding `tile`.
    pub fn count(&self, tile: usize) -> usize {
        self.tiles.iter().filter(|&&t| t == tile).count()
    }

    /// Neighbour of a cell along `axis`, or `None` when it lies outside a finite grid.
    pub fn neighbor(
        &self,
        (x, y, z): (usize, usize, usize),
        axis: usize,
        boundary: BoundaryCondition,
    ) -> Option<(usize, usize, usize)> {
        let (dx, dy, dz) = AXIS_OFFSETS[axis];
        let step = |value: usize, delta: isize, size: usize| -> Option<usize> {
            let moved = value as isize + delta;
            match boundary {
                BoundaryCondition::Periodic => Some(moved.rem_euclid(size as isize) as usize),
                BoundaryCondition::Finite => (0..size as isize)
                    .contains(&moved)
                    .then_some(moved as usize),
            }
        };
        Some((
            step(x, dx, self.width)?,
            step(y, dy, self.height)?,
            step(z, dz, self.depth)?,
        ))
    }
}

/// One broken adjacency: `tile` at `cell` does not allow `neighbor_tile` at `neighbor` along `axis`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    pub cell: (usize, usize, usize),
    pub tile: usize,
    pub axis: usize,
    pub neighbor: (usize, usize, usize),
    pub neighbor_tile: usize,
}

/// Every adjacency in `grid` that `rules` do not allow, checked in all six directions.
pub fn adjacency_violations(
    grid: &TileGrid,
    rules: &AdjacencyRules,
    boundary: BoundaryCondition,
) -> Vec<Violation> {
    let mut violations = Vec::new();
    for z in 0..grid.depth {
        for y in 0..grid.height {
            for x in 0..grid.width {
                let tile = grid.get(x, y, z);
                for axis in 0..AXIS_OFFSETS.len() {
                    let Some(neighbor) = grid.neighbor((x, y, z), axis, boundary) else {
                        continue;
                    };
                    let neighbor_tile = grid.get(neighbor.0, neighbor.1, neighbor.2);
                    if !rules.check(tile, neighbor_tile, axis) {
                        violations.push(Violation {
                            cell: (x, y, z),
                            tile,
                            axis,
                            neighbor,
                            neighbor_tile,
                        });
                    }
                }
            }
        }
    }
    violations
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiles 0 and 1 may only touch themselves.
    fn self_only_rules() -> AdjacencyRules {
        let tuples: Vec<(usize, usize, usize)> = (0..6)
            .flat_map(|axis| (0..2).map(move |t| (axis, t, t)))
            .collect();
        AdjacencyRules::from_allowed_tuples(2, 6, tuples)
    }

    #[test]
    fn parses_cli_text_output() {
        let grid = TileGrid::parse_text("0 1\n2 3\n\n4 5\n6 7\n").unwrap();
        assert_eq!((grid.width, grid.height, grid.depth), (2, 2, 2));
        assert_eq!(grid.get(1, 0, 0), 1);
        assert_eq!(grid.get(0, 1, 1), 6);
    }

    const SHAPE: ChunkShape = ChunkShape { x: 2, y: 2, z: 1 };

    /// A chunk whose cells hold `first`, `first + 1`, ... in the store's row-major order.
    fn numbered_chunk(coord: ChunkCoord, first: u16) -> Chunk {
        Chunk {
            coord,
            tiles: (first..first + SHAPE.cells() as u16).collect(),
            version: 1,
        }
    }

    #[test]
    fn chunks_side_by_side_land_where_their_cells_are() {
        let left = numbered_chunk(ChunkCoord::new(-1, 3, 0), 10);
        let right = numbered_chunk(ChunkCoord::new(0, 3, 0), 20);

        let (grid, lowest) = TileGrid::from_chunks(SHAPE, [&right, &left], 0).unwrap();

        assert_eq!(lowest, ChunkCoord::new(-1, 3, 0));
        assert_eq!((grid.width, grid.height, grid.depth), (4, 2, 1));
        for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let at = [x as i32 - 2, y as i32 + 6, 0];
            assert_eq!(grid.get(x, y, 0), usize::from(left.tile(SHAPE, at)));
            let at = [x as i32, y as i32 + 6, 0];
            assert_eq!(grid.get(x + 2, y, 0), usize::from(right.tile(SHAPE, at)));
        }
    }

    #[test]
    fn cells_no_chunk_covers_hold_the_fill() {
        let corner = numbered_chunk(ChunkCoord::new(0, 0, 0), 10);
        let opposite = numbered_chunk(ChunkCoord::new(1, 1, 0), 20);

        let (grid, _) = TileGrid::from_chunks(SHAPE, [&corner, &opposite], 99).unwrap();

        assert_eq!((grid.width, grid.height), (4, 4));
        assert_eq!(grid.count(99), 8);
        assert_eq!(grid.get(3, 0, 0), 99);
        assert_eq!(grid.get(0, 3, 0), 99);
    }

    #[test]
    fn no_chunks_make_no_grid() {
        assert!(TileGrid::from_chunks(SHAPE, [], 0).is_err());
    }

    #[test]
    fn rejects_ragged_text_output() {
        assert!(TileGrid::parse_text("0 1\n2\n").is_err());
    }

    #[test]
    fn finite_neighbors_stop_at_the_edge_and_periodic_ones_wrap() {
        let grid = TileGrid::new(3, 1, 1, vec![0; 3]).unwrap();
        assert_eq!(grid.neighbor((0, 0, 0), 1, BoundaryCondition::Finite), None);
        assert_eq!(
            grid.neighbor((0, 0, 0), 1, BoundaryCondition::Periodic),
            Some((2, 0, 0))
        );
    }

    #[test]
    fn uniform_grid_has_no_violations() {
        let grid = TileGrid::new(2, 2, 2, vec![1; 8]).unwrap();
        assert!(
            adjacency_violations(&grid, &self_only_rules(), BoundaryCondition::Periodic).is_empty()
        );
    }

    #[test]
    fn reports_each_broken_direction() {
        let grid = TileGrid::new(2, 1, 1, vec![0, 1]).unwrap();
        let violations = adjacency_violations(&grid, &self_only_rules(), BoundaryCondition::Finite);
        // 0 -> 1 along +x and 1 -> 0 along -x.
        assert_eq!(violations.len(), 2);
        assert_eq!(
            violations[0],
            Violation {
                cell: (0, 0, 0),
                tile: 0,
                axis: 0,
                neighbor: (1, 0, 0),
                neighbor_tile: 1
            }
        );
    }
}
