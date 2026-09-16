//! Global constraints: rules about the whole grid that adjacency between neighbours cannot express.
//!
//! Adjacency rules only see pairs of cells. They can stop a walkway from ending in mid-air, but not
//! demand that every walkable cell of a city belongs to one connected network: a closed ring of
//! walkways is locally valid everywhere. A [`GlobalConstraint`] runs on the CPU between propagation
//! steps, removes possibilities that would break it, and reports when it can no longer be satisfied
//! so the solver can restart.

use crate::grid::PossibilityGrid;

/// Cell coordinates `(x, y, z)`.
pub type Cell = (usize, usize, usize);

/// A constraint over the whole grid, enforced between propagation steps.
pub trait GlobalConstraint: Send + Sync {
    /// Removes possibilities that would violate the constraint and returns the cells it changed, so
    /// the solver can propagate them. Returns `Err` with a cell involved in the violation when the
    /// constraint can no longer be satisfied.
    fn apply(&self, grid: &mut PossibilityGrid) -> Result<Vec<Cell>, Cell>;
}

/// Neighbour offsets in axis order: `+x, -x, +y, -y, +z, -z`, as in `AdjacencyRules`.
const OFFSETS: [(isize, isize, isize); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

/// Keeps every cell that must belong to a network (for example every walkable cell) connected in one
/// piece, following the path constraint Boris the Brave describes for DeBroglie.
///
/// Tiles are *traversable* (part of the network, perhaps only as a passage such as a building
/// interior) or also *required* (a cell holding one must be connected). Two neighbouring cells can
/// connect when some remaining traversable tile in one *links* to some remaining traversable tile in
/// the other along that axis. Each application:
///
/// 1. builds that "could connect" graph over cells that can still hold a traversable tile;
/// 2. fails if cells that can only hold required tiles lie in different components, since no later
///    choice can join them;
/// 3. bans required tiles from cells outside that component, which could only become islands;
/// 4. bans non-traversable tiles from articulation points whose loss would separate required cells.
///
/// The graph only loses edges as possibilities shrink, so a fully collapsed grid that passes is one
/// connected network. Grid borders are finite.
pub struct ConnectivityConstraint {
    num_tiles: usize,
    words: usize,
    traversable: Vec<u64>,
    required: Vec<u64>,
    /// `links[(axis * num_tiles + tile) * words..][..words]`: tiles a neighbour along `axis` may hold
    /// to connect to `tile`.
    links: Vec<u64>,
}

fn set_bit(words: &mut [u64], bit: usize) {
    words[bit / 64] |= 1 << (bit % 64);
}

impl ConnectivityConstraint {
    /// Creates the constraint. Required tiles are traversable too. Links are `(axis, from, to)` and
    /// are made symmetric: `to` along `axis` from `from` also links `from` along the opposite axis.
    ///
    /// # Panics
    ///
    /// Panics if a tile index is not below `num_tiles` or an axis is not below 6.
    pub fn new(
        num_tiles: usize,
        traversable: impl IntoIterator<Item = usize>,
        required: impl IntoIterator<Item = usize>,
        links: impl IntoIterator<Item = (usize, usize, usize)>,
    ) -> Self {
        let words = num_tiles.div_ceil(64).max(1);
        let mut constraint = Self {
            num_tiles,
            words,
            traversable: vec![0; words],
            required: vec![0; words],
            links: vec![0; OFFSETS.len() * num_tiles * words],
        };
        for tile in traversable {
            assert!(tile < num_tiles, "traversable tile {tile} out of range");
            set_bit(&mut constraint.traversable, tile);
        }
        for tile in required {
            assert!(tile < num_tiles, "required tile {tile} out of range");
            set_bit(&mut constraint.required, tile);
            set_bit(&mut constraint.traversable, tile);
        }
        for (axis, from, to) in links {
            assert!(
                axis < OFFSETS.len() && from < num_tiles && to < num_tiles,
                "invalid link ({axis}, {from}, {to})"
            );
            let forward = (axis * num_tiles + from) * words;
            set_bit(&mut constraint.links[forward..forward + words], to);
            let backward = ((axis ^ 1) * num_tiles + to) * words;
            set_bit(&mut constraint.links[backward..backward + words], from);
        }
        constraint
    }

    fn link_words(&self, axis: usize, tile: usize) -> &[u64] {
        let start = (axis * self.num_tiles + tile) * self.words;
        &self.links[start..start + self.words]
    }
}

impl GlobalConstraint for ConnectivityConstraint {
    fn apply(&self, grid: &mut PossibilityGrid) -> Result<Vec<Cell>, Cell> {
        let (width, height, depth) = (grid.width, grid.height, grid.depth);
        let cells = width * height * depth;
        let words = self.words;
        let coord = |i: usize| (i % width, (i / width) % height, i / (width * height));
        let step = |i: usize, axis: usize| -> Option<usize> {
            let (x, y, z) = coord(i);
            let (dx, dy, dz) = OFFSETS[axis];
            let moved =
                |v: usize, d: isize, size: usize| v.checked_add_signed(d).filter(|&v| v < size);
            Some(
                (moved(z, dz, depth)? * height + moved(y, dy, height)?) * width
                    + moved(x, dx, width)?,
            )
        };

        // Possibilities as 64-bit words, cell by cell.
        let mut possible = vec![0u64; cells * words];
        for i in 0..cells {
            let (x, y, z) = coord(i);
            if let Some(cell) = grid.get(x, y, z) {
                for tile in cell.iter_ones().filter(|&t| t < self.num_tiles) {
                    set_bit(&mut possible[i * words..(i + 1) * words], tile);
                }
            }
        }
        let cell_words = |i: usize| &possible[i * words..(i + 1) * words];
        let node: Vec<bool> = (0..cells)
            .map(|i| {
                cell_words(i)
                    .iter()
                    .zip(&self.traversable)
                    .any(|(p, t)| p & t != 0)
            })
            .collect();
        let relevant: Vec<bool> = (0..cells)
            .map(|i| {
                let p = cell_words(i);
                p.iter().any(|&w| w != 0) && p.iter().zip(&self.required).all(|(p, r)| p & !r == 0)
            })
            .collect();
        let total = relevant.iter().filter(|&&r| r).count();
        let Some(root) = relevant.iter().position(|&r| r) else {
            return Ok(Vec::new());
        };

        // Which of the six directions could connect, per cell.
        let mut edges = vec![0u8; cells];
        let mut reach = vec![0u64; words];
        for i in (0..cells).filter(|&i| node[i]) {
            for axis in 0..OFFSETS.len() {
                let Some(j) = step(i, axis).filter(|&j| node[j]) else {
                    continue;
                };
                reach.fill(0);
                for (word, &bits) in cell_words(i).iter().enumerate() {
                    let mut bits = bits & self.traversable[word];
                    while bits != 0 {
                        let tile = word * 64 + bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        for (r, l) in reach.iter_mut().zip(self.link_words(axis, tile)) {
                            *r |= l;
                        }
                    }
                }
                if reach
                    .iter()
                    .zip(cell_words(j))
                    .zip(&self.traversable)
                    .any(|((r, p), t)| r & p & t != 0)
                {
                    edges[i] |= 1 << axis;
                }
            }
        }

        // Iterative Tarjan DFS from a required cell: discovery order, low links, and the number of
        // required cells in each DFS subtree.
        const UNSEEN: u32 = u32::MAX;
        let mut disc = vec![UNSEEN; cells];
        let mut low = vec![0u32; cells];
        let mut below = vec![0usize; cells];
        let mut parent = vec![usize::MAX; cells];
        let mut articulation = vec![false; cells];
        disc[root] = 0;
        below[root] = 1;
        let mut timer = 1u32;
        let mut stack: Vec<(usize, usize)> = vec![(root, 0)];
        while let Some(&(u, axis)) = stack.last() {
            if axis < OFFSETS.len() {
                stack.last_mut().expect("stack is not empty").1 += 1;
                if edges[u] & (1 << axis) == 0 {
                    continue;
                }
                let v = step(u, axis).expect("edges only point inside the grid");
                if disc[v] == UNSEEN {
                    parent[v] = u;
                    disc[v] = timer;
                    low[v] = timer;
                    timer += 1;
                    below[v] = usize::from(relevant[v]);
                    stack.push((v, 0));
                } else if v != parent[u] {
                    low[u] = low[u].min(disc[v]);
                }
            } else {
                stack.pop();
                if let Some(&(p, _)) = stack.last() {
                    low[p] = low[p].min(low[u]);
                    below[p] += below[u];
                    // Removing `p` would cut `u`'s subtree off from the rest of the graph; that matters
                    // when required cells lie on both sides. The root is itself required.
                    if p != root && low[u] >= disc[p] && below[u] > 0 && total > below[u] {
                        articulation[p] = true;
                    }
                }
            }
        }

        if let Some(cut_off) = (0..cells).find(|&i| relevant[i] && disc[i] == UNSEEN) {
            return Err(coord(cut_off));
        }

        let mut changed = Vec::new();
        for i in 0..cells {
            let banned: Vec<u64> = if node[i] && disc[i] == UNSEEN {
                cell_words(i)
                    .iter()
                    .zip(&self.required)
                    .map(|(p, r)| p & r)
                    .collect()
            } else if articulation[i] {
                cell_words(i)
                    .iter()
                    .zip(&self.traversable)
                    .map(|(p, t)| p & !t)
                    .collect()
            } else {
                continue;
            };
            if banned.iter().all(|&w| w == 0) {
                continue;
            }
            let (x, y, z) = coord(i);
            let cell = grid.get_mut(x, y, z).expect("cell in bounds");
            for (word, &bits) in banned.iter().enumerate() {
                let mut bits = bits;
                while bits != 0 {
                    cell.set(word * 64 + bits.trailing_zeros() as usize, false);
                    bits &= bits - 1;
                }
            }
            changed.push((x, y, z));
        }
        Ok(changed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WALL: usize = 0;
    const FLOOR: usize = 1;

    /// Floor connects to floor in every direction; walls connect to nothing.
    fn floors_connect() -> ConnectivityConstraint {
        ConnectivityConstraint::new(2, [], [FLOOR], (0..6).map(|axis| (axis, FLOOR, FLOOR)))
    }

    /// A row of cells along `x`, each limited to the given tiles.
    fn row(cells: &[&[usize]]) -> PossibilityGrid {
        let mut grid = PossibilityGrid::new(cells.len(), 1, 1, 2);
        for (x, tiles) in cells.iter().enumerate() {
            let cell = grid.get_mut(x, 0, 0).unwrap();
            cell.fill(false);
            for &tile in *tiles {
                cell.set(tile, true);
            }
        }
        grid
    }

    fn tiles(grid: &PossibilityGrid, x: usize) -> Vec<usize> {
        grid.get(x, 0, 0).unwrap().iter_ones().collect()
    }

    #[test]
    fn a_cell_joining_two_floors_is_forced_to_floor() {
        let mut grid = row(&[&[FLOOR], &[FLOOR, WALL], &[FLOOR]]);
        assert_eq!(floors_connect().apply(&mut grid), Ok(vec![(1, 0, 0)]));
        assert_eq!(tiles(&grid, 1), vec![FLOOR]);
    }

    #[test]
    fn floors_separated_by_a_wall_are_reported() {
        let mut grid = row(&[&[FLOOR], &[WALL], &[FLOOR]]);
        assert!(floors_connect().apply(&mut grid).is_err());
    }

    #[test]
    fn cells_cut_off_from_the_network_cannot_become_floor() {
        let mut grid = row(&[&[FLOOR], &[FLOOR, WALL], &[WALL], &[FLOOR, WALL]]);
        assert_eq!(floors_connect().apply(&mut grid), Ok(vec![(3, 0, 0)]));
        assert_eq!(tiles(&grid, 3), vec![WALL]);
        assert_eq!(
            tiles(&grid, 1),
            vec![WALL, FLOOR],
            "a dead-end branch may still become a wall"
        );
    }

    #[test]
    fn nothing_is_forced_before_any_cell_must_be_floor() {
        let mut grid = row(&[&[FLOOR, WALL], &[FLOOR, WALL], &[FLOOR, WALL]]);
        assert_eq!(floors_connect().apply(&mut grid), Ok(vec![]));
    }

    #[test]
    fn passages_can_be_forced_without_becoming_required() {
        // Tile 2 is a passage: traversable, links to floor, but not required itself.
        const PASSAGE: usize = 2;
        let links = (0..6).flat_map(|axis| [(axis, FLOOR, FLOOR), (axis, FLOOR, PASSAGE)]);
        let constraint = ConnectivityConstraint::new(3, [PASSAGE], [FLOOR], links);
        let mut grid = PossibilityGrid::new(3, 1, 1, 3);
        for (x, allowed) in [vec![FLOOR], vec![WALL, PASSAGE], vec![FLOOR]]
            .into_iter()
            .enumerate()
        {
            let cell = grid.get_mut(x, 0, 0).unwrap();
            cell.fill(false);
            for tile in allowed {
                cell.set(tile, true);
            }
        }
        assert_eq!(constraint.apply(&mut grid), Ok(vec![(1, 0, 0)]));
        assert_eq!(
            grid.get(1, 0, 0).unwrap().iter_ones().collect::<Vec<_>>(),
            vec![PASSAGE]
        );
    }

    #[test]
    fn connections_follow_the_links_axis_by_axis() {
        // Floors only link along x, so two floors stacked along y are disconnected.
        let constraint = ConnectivityConstraint::new(2, [], [FLOOR], [(0, FLOOR, FLOOR)]);
        let mut grid = PossibilityGrid::new(1, 2, 1, 2);
        for y in 0..2 {
            grid.get_mut(0, y, 0).unwrap().set(WALL, false);
        }
        assert!(constraint.apply(&mut grid).is_err());
    }
}
