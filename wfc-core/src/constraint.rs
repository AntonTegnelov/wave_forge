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

/// Forbids a tile from appearing within a bounded offset of another, for example "no balcony within
/// two cells directly above a road".
///
/// The cheapest kind of non-adjacent rule: directional, bounded, and decided by looking at one cell
/// and one offset at a time. Unlike [`ConnectivityConstraint`] it needs no graph and no global
/// analysis, which is why it is useful for separating *what kind* of rule causes thrashing from *how
/// expensive* the rule is to check.
///
/// Pruning only fires once the source cell is **decided**. While a cell might still hold something
/// other than `source`, it implies no exclusion at all, and banning early would remove solutions.
/// This is the same conservative discipline [`ConnectivityConstraint`] uses.
pub struct RangeExclusionConstraint {
    source: usize,
    forbidden: usize,
    /// Offsets from the source cell at which `forbidden` may not appear.
    offsets: Vec<(isize, isize, isize)>,
}

impl RangeExclusionConstraint {
    /// Forbids `forbidden` at each of `offsets` relative to any cell holding `source`.
    ///
    /// For "no `forbidden` within `n` cells directly above", pass `(0, 0, 1)..=(0, 0, n)`.
    pub fn new(
        source: usize,
        forbidden: usize,
        offsets: impl IntoIterator<Item = (isize, isize, isize)>,
    ) -> Self {
        Self {
            source,
            forbidden,
            offsets: offsets.into_iter().collect(),
        }
    }
}

impl GlobalConstraint for RangeExclusionConstraint {
    fn apply(&self, grid: &mut PossibilityGrid) -> Result<Vec<Cell>, Cell> {
        let (width, height, depth) = (grid.width, grid.height, grid.depth);
        // Cells already decided to be the source tile. Only these imply an exclusion.
        let mut sources = Vec::new();
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    let Some(cell) = grid.get(x, y, z) else {
                        continue;
                    };
                    if cell.count_ones() == 1 && cell[self.source] {
                        sources.push((x, y, z));
                    }
                }
            }
        }

        let mut changed = Vec::new();
        for (x, y, z) in sources {
            for &(dx, dy, dz) in &self.offsets {
                let moved =
                    |v: usize, d: isize, size: usize| v.checked_add_signed(d).filter(|&v| v < size);
                let (Some(tx), Some(ty), Some(tz)) = (
                    moved(x, dx, width),
                    moved(y, dy, height),
                    moved(z, dz, depth),
                ) else {
                    continue;
                };
                let Some(target) = grid.get_mut(tx, ty, tz) else {
                    continue;
                };
                // Report a cell only when a bit was actually cleared. The solver re-applies until
                // nothing changes, with no iteration cap of its own, so reporting an untouched cell
                // hangs the run rather than slowing it.
                if !target[self.forbidden] {
                    continue;
                }
                target.set(self.forbidden, false);
                if target.not_any() {
                    // Name the emptied cell: `apply`'s error becomes the backjump culprit, and where
                    // the search aims is what decides whether it escapes (docs/thrashing.md).
                    return Err((tx, ty, tz));
                }
                changed.push((tx, ty, tz));
            }
        }
        Ok(changed)
    }
}

/// Constrains a cell's whole neighbourhood rather than its faces: every cell within `radius` of one
/// holding `subject` must hold one of `allowed`.
///
/// This is the kind adjacency rules cannot express, and not merely a convenience over them.
/// `AdjacencyRules` relates a cell to its six axis neighbours one pair at a time, so a radius-1 ball —
/// 26 neighbours — leaves 20 diagonals that no adjacency rule can reach at all.
///
/// Like the others it waits until the subject cell is *decided*. While a cell might still hold
/// something else it implies nothing about its surroundings, and banning early would remove solutions.
pub struct SurroundingConstraint {
    subject: usize,
    allowed: Vec<usize>,
    radius: usize,
}

impl SurroundingConstraint {
    /// Requires every cell within Chebyshev `radius` of a `subject` cell to hold one of `allowed`.
    pub fn new(subject: usize, allowed: impl IntoIterator<Item = usize>, radius: usize) -> Self {
        Self {
            subject,
            allowed: allowed.into_iter().collect(),
            radius,
        }
    }
}

impl GlobalConstraint for SurroundingConstraint {
    fn apply(&self, grid: &mut PossibilityGrid) -> Result<Vec<Cell>, Cell> {
        let (width, height, depth) = (grid.width, grid.height, grid.depth);
        let mut subjects = Vec::new();
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    let decided_subject = grid
                        .get(x, y, z)
                        .is_some_and(|cell| cell.count_ones() == 1 && cell[self.subject]);
                    if decided_subject {
                        subjects.push((x, y, z));
                    }
                }
            }
        }

        let r = self.radius as isize;
        let mut changed = Vec::new();
        for (x, y, z) in subjects {
            for dz in -r..=r {
                for dy in -r..=r {
                    for dx in -r..=r {
                        if dx == 0 && dy == 0 && dz == 0 {
                            continue;
                        }
                        let moved = |v: usize, d: isize, size: usize| {
                            v.checked_add_signed(d).filter(|&v| v < size)
                        };
                        let (Some(nx), Some(ny), Some(nz)) = (
                            moved(x, dx, width),
                            moved(y, dy, height),
                            moved(z, dz, depth),
                        ) else {
                            continue;
                        };
                        let Some(cell) = grid.get_mut(nx, ny, nz) else {
                            continue;
                        };
                        // Report a cell only when a bit was actually cleared: the solver re-applies
                        // until nothing changes, with no iteration cap of its own.
                        let mut cleared = false;
                        for tile in 0..cell.len() {
                            if cell[tile] && !self.allowed.contains(&tile) {
                                cell.set(tile, false);
                                cleared = true;
                            }
                        }
                        if !cleared {
                            continue;
                        }
                        if cell.not_any() {
                            return Err((nx, ny, nz));
                        }
                        changed.push((nx, ny, nz));
                    }
                }
            }
        }
        Ok(changed)
    }
}

/// Requires at least `count` cells holding one of `wanted` within a radius of every cell that must
/// satisfy it, for example "at least three of either shop or office within three cells".
///
/// Non-local like connectivity, but with a *bounded* radius, which makes it the useful middle case:
/// failures can surface away from their cause without the whole-grid analysis connectivity needs.
///
/// Both bounds matter. The constraint fails when the **maximum possible** count in a ball falls below
/// `count`, since no later choice can raise it; and it bans a tile only when choosing it would push
/// that maximum below `count`. Counting only decided cells instead would prune states that are still
/// completable.
///
/// Two things make this stricter than it reads, and both bit us while writing its tests:
///
/// - **Balls are truncated at the borders.** A radius-1 ball holds 27 cells in the interior but only
///   8 in a corner, so a count that leaves slack in the middle of the grid can be tight or already
///   unsatisfiable at an edge.
/// - **An empty `subject` binds every cell**, including cells that cannot possibly help themselves.
///   "Every cell must have three shops nearby" fails the moment any cell is decided to something with
///   no shops in reach. Naming a subject is usually what is meant.
pub struct CountingConstraint {
    wanted: Vec<usize>,
    /// Cells holding one of these must satisfy the count; empty means every cell must.
    subject: Vec<usize>,
    radius: usize,
    count: usize,
}

impl CountingConstraint {
    /// Requires `count` of `wanted` within Chebyshev `radius` of cells holding a `subject` tile.
    /// An empty `subject` applies the requirement to every cell.
    pub fn new(
        wanted: impl IntoIterator<Item = usize>,
        subject: impl IntoIterator<Item = usize>,
        radius: usize,
        count: usize,
    ) -> Self {
        Self {
            wanted: wanted.into_iter().collect(),
            subject: subject.into_iter().collect(),
            radius,
            count,
        }
    }

    /// Whether this cell could still hold one of the wanted tiles.
    fn could_be_wanted(&self, grid: &PossibilityGrid, x: usize, y: usize, z: usize) -> bool {
        grid.get(x, y, z)
            .is_some_and(|cell| self.wanted.iter().any(|&tile| cell[tile]))
    }

    /// Whether this cell is still subject to the requirement.
    fn is_subject(&self, grid: &PossibilityGrid, x: usize, y: usize, z: usize) -> bool {
        let Some(cell) = grid.get(x, y, z) else {
            return false;
        };
        // No subject list means every cell counts. Otherwise only a cell already decided to a subject
        // tile is bound by the rule; while it could still be something else, nothing is implied.
        self.subject.is_empty()
            || (cell.count_ones() == 1 && self.subject.iter().any(|&tile| cell[tile]))
    }

    /// Cells within Chebyshev `radius`, the centre included.
    fn ball(&self, grid: &PossibilityGrid, x: usize, y: usize, z: usize) -> Vec<Cell> {
        let r = self.radius as isize;
        let mut cells = Vec::new();
        for dz in -r..=r {
            for dy in -r..=r {
                for dx in -r..=r {
                    let moved = |v: usize, d: isize, size: usize| {
                        v.checked_add_signed(d).filter(|&v| v < size)
                    };
                    if let (Some(nx), Some(ny), Some(nz)) = (
                        moved(x, dx, grid.width),
                        moved(y, dy, grid.height),
                        moved(z, dz, grid.depth),
                    ) {
                        cells.push((nx, ny, nz));
                    }
                }
            }
        }
        cells
    }
}

impl GlobalConstraint for CountingConstraint {
    fn apply(&self, grid: &mut PossibilityGrid) -> Result<Vec<Cell>, Cell> {
        let (width, height, depth) = (grid.width, grid.height, grid.depth);
        let subjects: Vec<Cell> = (0..depth)
            .flat_map(|z| (0..height).flat_map(move |y| (0..width).map(move |x| (x, y, z))))
            .filter(|&(x, y, z)| self.is_subject(grid, x, y, z))
            .collect();

        let mut changed = Vec::new();
        for (x, y, z) in subjects {
            let ball = self.ball(grid, x, y, z);
            let possible: Vec<Cell> = ball
                .iter()
                .copied()
                .filter(|&(nx, ny, nz)| self.could_be_wanted(grid, nx, ny, nz))
                .collect();
            if possible.len() < self.count {
                // Even giving every undecided cell in the ball a wanted tile falls short.
                return Err((x, y, z));
            }
            if possible.len() > self.count {
                // Slack remains, so no single cell is forced yet.
                continue;
            }
            // Exactly enough candidates: every one of them is now load-bearing, so none may drop its
            // wanted tiles. Ban everything else there.
            for (nx, ny, nz) in possible {
                let Some(cell) = grid.get_mut(nx, ny, nz) else {
                    continue;
                };
                let mut cleared = false;
                for tile in 0..cell.len() {
                    if cell[tile] && !self.wanted.contains(&tile) {
                        cell.set(tile, false);
                        cleared = true;
                    }
                }
                if !cleared {
                    continue;
                }
                if cell.not_any() {
                    return Err((nx, ny, nz));
                }
                changed.push((nx, ny, nz));
            }
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

    /// A column of cells along `z`, each limited to the given tiles.
    fn column(cells: &[&[usize]], num_tiles: usize) -> PossibilityGrid {
        let mut grid = PossibilityGrid::new(1, 1, cells.len(), num_tiles);
        for (z, allowed) in cells.iter().enumerate() {
            let cell = grid.get_mut(0, 0, z).unwrap();
            cell.fill(false);
            for &tile in *allowed {
                cell.set(tile, true);
            }
        }
        grid
    }

    /// "No FLOOR within two cells directly above a WALL."
    fn no_floor_above_wall() -> RangeExclusionConstraint {
        RangeExclusionConstraint::new(WALL, FLOOR, [(0, 0, 1), (0, 0, 2)])
    }

    #[test]
    fn range_exclusion_bans_the_tile_at_every_listed_offset() {
        let mut grid = column(&[&[WALL], &[WALL, FLOOR], &[WALL, FLOOR]], 2);
        assert_eq!(
            no_floor_above_wall().apply(&mut grid),
            Ok(vec![(0, 0, 1), (0, 0, 2)])
        );
        assert_eq!(
            grid.get(0, 0, 1).unwrap().iter_ones().collect::<Vec<_>>(),
            vec![WALL]
        );
        assert_eq!(
            grid.get(0, 0, 2).unwrap().iter_ones().collect::<Vec<_>>(),
            vec![WALL]
        );
    }

    #[test]
    fn range_exclusion_stays_silent_while_the_source_is_undecided() {
        // The bottom cell might still be FLOOR, so it implies no exclusion at all. Pruning here would
        // remove solutions rather than merely slow the search down.
        let mut grid = column(&[&[WALL, FLOOR], &[WALL, FLOOR]], 2);
        assert_eq!(no_floor_above_wall().apply(&mut grid), Ok(vec![]));
        assert_eq!(grid.get(0, 0, 1).unwrap().count_ones(), 2);
    }

    #[test]
    fn range_exclusion_reports_the_cell_it_empties() {
        let mut grid = column(&[&[WALL], &[FLOOR]], 2);
        assert_eq!(no_floor_above_wall().apply(&mut grid), Err((0, 0, 1)));
    }

    #[test]
    fn range_exclusion_reaches_no_further_than_its_offsets() {
        // Only z = 0 is decided, so it is the only source. Leaving the cells between undecided
        // matters: any cell decided to WALL is a source in its own right, and a column of decided
        // walls would put z = 3 within reach of the ones at z = 1 and z = 2.
        let mut grid = column(
            &[&[WALL], &[WALL, FLOOR], &[WALL, FLOOR], &[WALL, FLOOR]],
            2,
        );
        assert_eq!(
            no_floor_above_wall().apply(&mut grid),
            Ok(vec![(0, 0, 1), (0, 0, 2)])
        );
        assert_eq!(
            grid.get(0, 0, 3).unwrap().count_ones(),
            2,
            "three above the source is past the listed offsets"
        );
    }

    /// Everything around a WALL must be FLOOR.
    fn wall_surrounded_by_floor() -> SurroundingConstraint {
        SurroundingConstraint::new(WALL, [FLOOR], 1)
    }

    #[test]
    fn surrounding_bans_everything_not_allowed_in_the_ball() {
        let mut grid = row(&[&[WALL], &[WALL, FLOOR]]);
        assert_eq!(
            wall_surrounded_by_floor().apply(&mut grid),
            Ok(vec![(1, 0, 0)])
        );
        assert_eq!(tiles(&grid, 1), vec![FLOOR]);
    }

    #[test]
    fn surrounding_stays_silent_while_the_subject_is_undecided() {
        let mut grid = row(&[&[WALL, FLOOR], &[WALL, FLOOR]]);
        assert_eq!(wall_surrounded_by_floor().apply(&mut grid), Ok(vec![]));
        assert_eq!(grid.get(1, 0, 0).unwrap().count_ones(), 2);
    }

    #[test]
    fn surrounding_reports_the_cell_it_empties() {
        let mut grid = row(&[&[WALL], &[WALL]]);
        assert_eq!(wall_surrounded_by_floor().apply(&mut grid), Err((1, 0, 0)));
    }

    #[test]
    fn counting_stays_silent_while_slack_remains() {
        // One FLOOR needed within one cell, and every cell can still be FLOOR. Note the count has to
        // stay below the *border* ball size: at x = 0 the ball is truncated to two cells, so a count
        // of two would already be tight there rather than slack.
        let constraint = CountingConstraint::new([FLOOR], [], 1, 1);
        let mut grid = row(&[&[FLOOR, WALL], &[FLOOR, WALL], &[FLOOR, WALL]]);
        assert_eq!(constraint.apply(&mut grid), Ok(vec![]));
    }

    #[test]
    fn counting_forces_every_candidate_once_slack_runs_out() {
        // The WALL at x = 0 needs one FLOOR within a cell. It cannot supply one itself, so x = 1 is
        // the only candidate left and is therefore load-bearing: it may no longer become WALL.
        let constraint = CountingConstraint::new([FLOOR], [WALL], 1, 1);
        let mut grid = row(&[&[WALL], &[FLOOR, WALL]]);
        assert_eq!(constraint.apply(&mut grid), Ok(vec![(1, 0, 0)]));
        assert_eq!(tiles(&grid, 1), vec![FLOOR]);
    }

    #[test]
    fn counting_fails_when_the_ball_can_no_longer_reach_the_count() {
        let constraint = CountingConstraint::new([FLOOR], [], 1, 2);
        let mut grid = row(&[&[WALL], &[WALL], &[FLOOR]]);
        // Around x = 0 only one cell in reach can be FLOOR, and no later choice can add another.
        assert_eq!(constraint.apply(&mut grid), Err((0, 0, 0)));
    }

    #[test]
    fn counting_binds_only_cells_decided_to_a_subject_tile() {
        // WALL is the subject, but no cell is decided to WALL yet, so nothing is required of anyone
        // even though no cell could satisfy the count.
        let constraint = CountingConstraint::new([FLOOR], [WALL], 0, 1);
        let mut grid = row(&[&[FLOOR, WALL], &[FLOOR, WALL]]);
        assert_eq!(constraint.apply(&mut grid), Ok(vec![]));
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
