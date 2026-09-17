//! Checks that generated grids obey the rules they were generated from.
//!
//! A result that looks plausible can still break adjacency rules in a handful of cells, and that
//! sparse kind of corruption is exactly what GPU races or host/shader layout mismatches produce.
//! Tests therefore assert invariants mechanically instead of relying on someone looking at an
//! image.

use wfc_core::BoundaryCondition;
use wfc_core::Domains;
use wfc_core::grid::PossibilityGrid;
use wfc_rules::AdjacencyRules;

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
/// Kept separate from `PossibilityGrid` so checks and renderers work the same on solver output
/// and on grids read back from the CLI's text output.
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

    /// Converts solver output, failing on any cell that is not collapsed to exactly one tile.
    pub fn from_possibilities(grid: &PossibilityGrid) -> Result<Self, String> {
        let mut tiles = Vec::with_capacity(grid.width * grid.height * grid.depth);
        for z in 0..grid.depth {
            for y in 0..grid.height {
                for x in 0..grid.width {
                    let cell = grid
                        .get(x, y, z)
                        .ok_or_else(|| format!("cell ({x}, {y}, {z}) is out of bounds"))?;
                    match cell.count_ones() {
                        1 => tiles.push(cell.first_one().expect("one bit is set")),
                        n => return Err(format!("cell ({x}, {y}, {z}) has {n} possible tiles")),
                    }
                }
            }
        }
        Self::new(grid.width, grid.height, grid.depth, tiles)
    }

    /// Parses the text format written by the `wave_forge` CLI: space-separated tile indices along
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

    #[test]
    fn converts_only_fully_collapsed_possibility_grids() {
        let mut grid = PossibilityGrid::new(2, 1, 1, 2);
        assert!(TileGrid::from_possibilities(&grid).is_err());
        grid.collapse(0, 0, 0, 1).unwrap();
        grid.collapse(1, 0, 0, 0).unwrap();
        let tiles = TileGrid::from_possibilities(&grid).unwrap();
        assert_eq!((tiles.get(0, 0, 0), tiles.get(1, 0, 0)), (1, 0));
    }
}
