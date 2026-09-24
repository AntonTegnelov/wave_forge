//! Paths between sites, the region job a pack names with a Network stage.
//!
//! In every region, the sites whose centre lies in it are joined by a minimum spanning tree over
//! the distances between their centres, and each edge becomes the cheapest path over the height
//! field that stays in the region: a step costs its length plus a price for the height it climbs
//! or falls, so a road goes round a ridge through a gap rather than over it. A path never leaves
//! its region, so no region reads another's, and every region comes out the same in any order.

use super::regions::{Attempt, Curve, CurveId, RegionInput, RegionJob};
use super::runtime::{Site, StageError};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// The Network stage's job: paths over the field `height` between the region's sites, `width`
/// wide, a step costing its length plus `climb` times the height it climbs or falls, never
/// through a column below `dry`.
pub(crate) struct SitePaths<'a> {
    pub(crate) height: &'a str,
    pub(crate) width: f32,
    pub(crate) climb: f32,
    pub(crate) dry: Option<f32>,
}

/// The eight steps from a column, with their lengths.
const STEPS: [(i64, i64, f32); 8] = [
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, std::f32::consts::SQRT_2),
    (1, -1, std::f32::consts::SQRT_2),
    (-1, 1, std::f32::consts::SQRT_2),
    (-1, -1, std::f32::consts::SQRT_2),
];

impl RegionJob for SitePaths<'_> {
    fn run(&self, input: &RegionInput<'_>) -> Result<Attempt, StageError> {
        let ([x0, y0], [x1, y1]) = input.columns();
        let (width, depth) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);
        let mut heights = Vec::with_capacity(width * depth);
        for y in y0..=y1 {
            for x in x0..=x1 {
                heights.push(input.field(self.height, x, y)?);
            }
        }
        let grid = Grid {
            origin: [x0, y0],
            width,
            depth,
            heights,
        };
        let chunk = input.chunk();
        let footprints: Vec<([i64; 2], [i64; 2])> = input
            .sites()
            .iter()
            .map(|site| footprint(site, chunk))
            .collect();
        let centres: Vec<[i64; 2]> = footprints
            .iter()
            .map(|(min, max)| [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2])
            .collect();
        let mut curves = Vec::new();
        for (a, b) in spanning_tree(&centres) {
            let Some(path) = grid.cheapest(centres[a], centres[b], self.climb, self.dry) else {
                continue;
            };
            let outside = |at: &[i64; 2]| {
                [footprints[a], footprints[b]].iter().all(|(min, max)| {
                    !((min[0]..max[0]).contains(&at[0]) && (min[1]..max[1]).contains(&at[1]))
                })
            };
            let points: Vec<[f32; 2]> = path
                .iter()
                .filter(|at| outside(at))
                .map(|at| [at[0] as f32 + 0.5, at[1] as f32 + 0.5])
                .collect();
            if points.len() < 2 {
                continue;
            }
            curves.push(Curve {
                id: CurveId::Region {
                    region: input.region(),
                    index: curves.len() as u32,
                },
                values: vec![self.width; points.len()],
                points,
            });
        }
        Ok(Attempt::Accepted(curves))
    }
}

/// A site's footprint in columns, from `min` up to but not including `max`.
fn footprint(site: &Site, chunk: [u32; 2]) -> ([i64; 2], [i64; 2]) {
    let [cx, cy] = chunk.map(i64::from);
    (
        [i64::from(site.min.0) * cx, i64::from(site.min.1) * cy],
        [i64::from(site.max.0) * cx, i64::from(site.max.1) * cy],
    )
}

/// The edges of a minimum spanning tree over `points` by straight distance, as pairs of indices,
/// grown from the first point; of equal distances, the lower indices win.
fn spanning_tree(points: &[[i64; 2]]) -> Vec<(usize, usize)> {
    let distance = |a: [i64; 2], b: [i64; 2]| (a[0] - b[0]).pow(2) + (a[1] - b[1]).pow(2);
    let mut joined = vec![false; points.len()];
    let mut edges = Vec::new();
    if let Some(first) = joined.first_mut() {
        *first = true;
    }
    for _ in 1..points.len() {
        let (from, to) = (0..points.len())
            .filter(|&from| joined[from])
            .flat_map(|from| {
                (0..points.len())
                    .filter(|&to| !joined[to])
                    .map(move |to| (from, to))
            })
            .min_by_key(|&(from, to)| (distance(points[from], points[to]), from, to))
            .expect("a point is left to join while fewer edges than points are chosen");
        joined[to] = true;
        edges.push((from, to));
    }
    edges
}

/// A region's heights, row by row with x fastest from `origin`.
struct Grid {
    origin: [i64; 2],
    width: usize,
    depth: usize,
    heights: Vec<f32>,
}

impl Grid {
    fn index(&self, at: [i64; 2]) -> Option<usize> {
        let (x, y) = (at[0] - self.origin[0], at[1] - self.origin[1]);
        ((0..self.width as i64).contains(&x) && (0..self.depth as i64).contains(&y))
            .then(|| y as usize * self.width + x as usize)
    }

    fn column(&self, index: usize) -> [i64; 2] {
        [
            self.origin[0] + (index % self.width) as i64,
            self.origin[1] + (index / self.width) as i64,
        ]
    }

    /// The cheapest path from `from` to `to` in the grid by A*, both ends included, or `None` if
    /// none avoids the columns below `dry`. The ends may lie below it.
    fn cheapest(
        &self,
        from: [i64; 2],
        to: [i64; 2],
        climb: f32,
        dry: Option<f32>,
    ) -> Option<Vec<[i64; 2]>> {
        let (start, goal) = (self.index(from)?, self.index(to)?);
        let remaining = |index: usize| {
            let at = self.column(index);
            (((at[0] - to[0]).pow(2) + (at[1] - to[1]).pow(2)) as f32).sqrt()
        };
        let mut cost = vec![f32::INFINITY; self.heights.len()];
        let mut came_from = vec![usize::MAX; self.heights.len()];
        // Costs are finite and never negative, so their bits order as the costs do.
        let mut open = BinaryHeap::new();
        cost[start] = 0.0;
        open.push(Reverse((remaining(start).to_bits(), start)));
        while let Some(Reverse((_, index))) = open.pop() {
            if index == goal {
                let mut path = vec![self.column(goal)];
                let mut at = goal;
                while at != start {
                    at = came_from[at];
                    path.push(self.column(at));
                }
                path.reverse();
                return Some(path);
            }
            let here = self.column(index);
            for (dx, dy, length) in STEPS {
                let Some(next) = self.index([here[0] + dx, here[1] + dy]) else {
                    continue;
                };
                if next != goal && dry.is_some_and(|dry| self.heights[next] < dry) {
                    continue;
                }
                let step = length + climb * (self.heights[next] - self.heights[index]).abs();
                let reached = cost[index] + step;
                if reached < cost[next] {
                    cost[next] = reached;
                    came_from[next] = index;
                    open.push(Reverse(((reached + remaining(next)).to_bits(), next)));
                }
            }
        }
        None
    }
}
