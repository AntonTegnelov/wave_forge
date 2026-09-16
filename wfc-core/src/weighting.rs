//! Rules that change how *likely* a tile is rather than whether it is *legal*.
//!
//! Every other rule kind in this solver removes possibilities: adjacency rules, and the
//! [`GlobalConstraint`](crate::constraint::GlobalConstraint) family. A statistical rule — "a shop
//! becomes more likely the more shops are nearby, and nearer ones count for more" — removes none. It
//! shifts probability mass inside the set of tiles that were already legal.
//!
//! That is why it cannot be a `GlobalConstraint`. `apply` may only clear bits and report what it
//! cleared, so a likelihood rule implemented there could only ever return `Ok(vec![])`: a no-op
//! dressed as a rule. Worse, making it prune to "express" a preference would change the set of valid
//! outputs, which is precisely what a rule about *likelihood* must not do.
//!
//! So it hooks the collapse choice instead. The solver picks a cell, then picks one of that cell's
//! remaining tiles in proportion to weight; [`TileWeighting`] makes that weight a function of the cell
//! and the grid around it rather than of the tile alone.

use crate::grid::PossibilityGrid;

/// Cell coordinates `(x, y, z)`, as in [`crate::constraint`].
pub type Cell = (usize, usize, usize);

/// Supplies the weight of a tile at a particular cell, given the grid as it currently stands.
///
/// Called once per remaining tile each time a cell is collapsed, so implementations should be cheap in
/// the size of the neighbourhood they inspect, not in the size of the grid.
///
/// Weights must be finite and non-negative, and at least one tile in a cell must weigh more than zero;
/// the caller turns a bad set into an error rather than panicking, so a buggy implementation surfaces
/// as a failed run rather than a silently skewed one.
pub trait TileWeighting: Send + Sync {
    /// The weight of `tile` at `cell`. Larger means more likely.
    fn weight(&self, grid: &PossibilityGrid, cell: Cell, tile: usize) -> f32;
}

/// A flat weight per tile, ignoring position — what `with_tile_weights` has always done, expressed
/// through the same interface so the two paths do not diverge.
pub struct UniformWeighting {
    weights: Vec<f32>,
}

impl UniformWeighting {
    /// Weights indexed by tile id. Tiles beyond the end weigh 1.0.
    pub fn new(weights: impl IntoIterator<Item = f32>) -> Self {
        Self {
            weights: weights.into_iter().collect(),
        }
    }
}

impl TileWeighting for UniformWeighting {
    fn weight(&self, _grid: &PossibilityGrid, _cell: Cell, tile: usize) -> f32 {
        self.weights.get(tile).copied().unwrap_or(1.0)
    }
}

/// Makes a tile more likely the more *attractor* tiles sit near the cell, with nearer ones counting
/// for more — the statistical rule kind described in `docs/thrashing.md`.
///
/// For each cell within `radius` that is **decided** to an attractor, the weight gains
/// `strength / distance`, using Chebyshev distance so the falloff matches the ball being scanned.
/// Only decided cells count: an undecided neighbour that *might* become an attractor is not evidence
/// that it will, and counting possibilities would make the weight jump around as propagation narrows
/// cells that were never going to be attractors anyway.
///
/// The result is a preference, never a requirement. A subject tile with no attractors nearby keeps
/// `base`, so it stays choosable — which is what keeps this a statistical rule rather than a
/// constraint wearing a different hat.
pub struct DistanceWeighting {
    /// Tiles whose weight this rule modifies.
    subject: Vec<usize>,
    /// Tiles that pull the subject's weight up.
    attractors: Vec<usize>,
    radius: usize,
    base: f32,
    strength: f32,
}

impl DistanceWeighting {
    /// Weights `subject` tiles as `base + strength / distance` summed over decided `attractors` within
    /// Chebyshev `radius`. Every other tile keeps weight `base`.
    pub fn new(
        subject: impl IntoIterator<Item = usize>,
        attractors: impl IntoIterator<Item = usize>,
        radius: usize,
        base: f32,
        strength: f32,
    ) -> Self {
        Self {
            subject: subject.into_iter().collect(),
            attractors: attractors.into_iter().collect(),
            radius,
            base,
            strength,
        }
    }

    /// Whether this cell is decided to one of the attractor tiles.
    fn is_attractor(&self, grid: &PossibilityGrid, x: usize, y: usize, z: usize) -> bool {
        grid.get(x, y, z).is_some_and(|cell| {
            cell.count_ones() == 1 && self.attractors.iter().any(|&tile| cell[tile])
        })
    }
}

impl TileWeighting for DistanceWeighting {
    fn weight(&self, grid: &PossibilityGrid, (x, y, z): Cell, tile: usize) -> f32 {
        if !self.subject.contains(&tile) {
            return self.base;
        }
        let r = self.radius as isize;
        let mut weight = self.base;
        for dz in -r..=r {
            for dy in -r..=r {
                for dx in -r..=r {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let moved =
                        |v: usize, d: isize, size: usize| v.checked_add_signed(d).filter(|&v| v < size);
                    let (Some(nx), Some(ny), Some(nz)) = (
                        moved(x, dx, grid.width),
                        moved(y, dy, grid.height),
                        moved(z, dz, grid.depth),
                    ) else {
                        continue;
                    };
                    if self.is_attractor(grid, nx, ny, nz) {
                        // Chebyshev distance, matching the shape of the ball being scanned.
                        let distance = dx.abs().max(dy.abs()).max(dz.abs()) as f32;
                        weight += self.strength / distance;
                    }
                }
            }
        }
        weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PLAIN: usize = 0;
    const SHOP: usize = 1;

    /// A row of cells along `x`, each limited to the given tiles.
    fn row(cells: &[&[usize]]) -> PossibilityGrid {
        let mut grid = PossibilityGrid::new(cells.len(), 1, 1, 2);
        for (x, allowed) in cells.iter().enumerate() {
            let cell = grid.get_mut(x, 0, 0).unwrap();
            cell.fill(false);
            for &tile in *allowed {
                cell.set(tile, true);
            }
        }
        grid
    }

    /// Shops attract shops, out to two cells.
    fn shops_cluster() -> DistanceWeighting {
        DistanceWeighting::new([SHOP], [SHOP], 2, 1.0, 4.0)
    }

    #[test]
    fn a_tile_with_no_attractors_nearby_keeps_its_base_weight() {
        let grid = row(&[&[PLAIN, SHOP], &[PLAIN], &[PLAIN]]);
        assert_eq!(shops_cluster().weight(&grid, (0, 0, 0), SHOP), 1.0);
    }

    #[test]
    fn a_nearer_attractor_counts_for_more_than_a_farther_one() {
        let near = row(&[&[PLAIN, SHOP], &[SHOP], &[PLAIN]]);
        let far = row(&[&[PLAIN, SHOP], &[PLAIN], &[SHOP]]);
        let rule = shops_cluster();
        let near_weight = rule.weight(&near, (0, 0, 0), SHOP);
        let far_weight = rule.weight(&far, (0, 0, 0), SHOP);
        assert_eq!(near_weight, 5.0, "adjacent attractor adds strength / 1");
        assert_eq!(far_weight, 3.0, "attractor two cells away adds strength / 2");
        assert!(near_weight > far_weight);
    }

    #[test]
    fn attractors_accumulate() {
        let grid = row(&[&[PLAIN, SHOP], &[SHOP], &[SHOP]]);
        // 1.0 base + 4.0/1 from x=1 + 4.0/2 from x=2.
        assert_eq!(shops_cluster().weight(&grid, (0, 0, 0), SHOP), 7.0);
    }

    #[test]
    fn only_decided_neighbours_count_as_attractors() {
        // x = 1 might still become a shop, which is not evidence that it will.
        let grid = row(&[&[PLAIN, SHOP], &[PLAIN, SHOP], &[PLAIN]]);
        assert_eq!(shops_cluster().weight(&grid, (0, 0, 0), SHOP), 1.0);
    }

    #[test]
    fn tiles_outside_the_subject_are_left_alone() {
        let grid = row(&[&[PLAIN, SHOP], &[SHOP], &[SHOP]]);
        assert_eq!(shops_cluster().weight(&grid, (0, 0, 0), PLAIN), 1.0);
    }

    #[test]
    fn attractors_beyond_the_radius_are_ignored() {
        let grid = row(&[&[PLAIN, SHOP], &[PLAIN], &[PLAIN], &[SHOP]]);
        assert_eq!(shops_cluster().weight(&grid, (0, 0, 0), SHOP), 1.0);
    }

    #[test]
    fn uniform_weighting_is_positional_only_in_name() {
        let grid = row(&[&[PLAIN, SHOP], &[SHOP]]);
        let rule = UniformWeighting::new([2.0, 3.0]);
        assert_eq!(rule.weight(&grid, (0, 0, 0), PLAIN), 2.0);
        assert_eq!(rule.weight(&grid, (0, 0, 0), SHOP), 3.0);
        assert_eq!(rule.weight(&grid, (0, 0, 0), 99), 1.0, "unknown tiles weigh 1");
    }
}
