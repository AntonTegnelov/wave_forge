//! Cave levels: a plan of rooms linked by a pattern, and tunnels between linked rooms.
//!
//! A Cave stage plans one level per region, and a Tunnels stage joins its linked rooms. Both are
//! pure: a plan depends on the stage's parameters, the region and its hash stream alone, and a
//! tunnel on the plan, the noise and its own parameters, so a level is the same whatever order its
//! chunks come in, and neighbouring regions never need to agree on anything.

use super::pack::{Pattern, Room};
use super::runtime::unit;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use wfc_core::hash::pcg3d;

/// A region's cave level: its rooms in the order they were placed, and the pairs of rooms the
/// pattern links, by their place in `rooms`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CavePlan {
    pub rooms: Vec<PlacedRoom>,
    pub links: Vec<(usize, usize)>,
}

/// A room of a plan: which of the stage's rooms it is, its footprint's lowest column, and its
/// floor's height in cells.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PlacedRoom {
    pub room: usize,
    pub min: [i64; 2],
    pub floor: f32,
}

impl PlacedRoom {
    /// The centre of its box, in cells.
    pub(crate) fn centre(&self, rooms: &[Room]) -> [f32; 3] {
        let (sx, sy, sz) = rooms[self.room].size;
        [
            self.min[0] as f32 + sx as f32 / 2.0,
            self.min[1] as f32 + sy as f32 / 2.0,
            self.floor + sz as f32 / 2.0,
        ]
    }
}

/// What a Cave stage asks of a plan ([`super::pack::StageKind::Cave`]).
pub(crate) struct CaveRules<'a> {
    pub depth: (f32, f32),
    pub patterns: &'a [Pattern],
    pub count: (u32, u32),
    pub apart: f32,
    pub rooms: &'a [Room],
    pub tries: u32,
    pub rerolls: u32,
}

/// The room a room of `pattern` links to, by its place in the plan; none for the first.
const fn parent(pattern: Pattern, index: usize) -> Option<usize> {
    match (index, pattern) {
        (0, _) => None,
        (_, Pattern::Linear) => Some(index - 1),
        (_, Pattern::Star) => Some(0),
        (1..=3, Pattern::Hub) => Some(0),
        (_, Pattern::Hub) => Some(index - 3),
    }
}

/// The plan of a level whose rooms lie wholly inside the columns from `low` to `high`, both
/// included, from the hash stream `stream`; or, when no attempt placed every room, why each
/// attempt failed.
pub(crate) fn plan(
    rules: &CaveRules<'_>,
    low: [i64; 2],
    high: [i64; 2],
    stream: u32,
) -> Result<CavePlan, Vec<String>> {
    let total_weight: u32 = rules.rooms.iter().map(|room| room.weight).sum();
    let mut log = Vec::new();
    for attempt in 0..=rules.rerolls {
        let stream = stream ^ attempt.wrapping_mul(0x9E37_79B9);
        let [a, b, _] = pcg3d([stream, 0xCA7E, 0]);
        let pattern = rules.patterns[a as usize % rules.patterns.len()];
        let count = rules.count.0 + b % (rules.count.1 - rules.count.0 + 1);
        let mut placed: Vec<PlacedRoom> = Vec::with_capacity(count as usize);
        for index in 0..count as usize {
            let [pick, _, _] = pcg3d([stream, 0x5EED, index as u32]);
            let mut left = pick % total_weight;
            let room = rules
                .rooms
                .iter()
                .position(|room| {
                    let here = left < room.weight;
                    left = left.saturating_sub(room.weight);
                    here
                })
                .expect("a draw below the total weight falls on a room");
            let (sx, sy, _) = rules.rooms[room].size;
            let spot = (0..rules.tries).find_map(|try_| {
                let [x, y, z] = pcg3d([stream, index as u32, try_]);
                let centre = match parent(pattern, index) {
                    // The middle half of the area.
                    None => {
                        let span = |axis: usize| (high[axis] - low[axis] + 1) as f32;
                        [
                            low[0] as f32 + span(0) * (0.25 + 0.5 * unit(x)),
                            low[1] as f32 + span(1) * (0.25 + 0.5 * unit(y)),
                        ]
                    }
                    Some(parent) => {
                        let from = placed[parent].centre(rules.rooms);
                        let angle = unit(x) * std::f32::consts::TAU;
                        let reach = rules.apart * (1.0 + unit(y));
                        [
                            from[0] + reach * libm::cosf(angle),
                            from[1] + reach * libm::sinf(angle),
                        ]
                    }
                };
                let min = [
                    (centre[0] - sx as f32 / 2.0).round() as i64,
                    (centre[1] - sy as f32 / 2.0).round() as i64,
                ];
                let inside = min[0] >= low[0]
                    && min[1] >= low[1]
                    && min[0] + i64::from(sx) - 1 <= high[0]
                    && min[1] + i64::from(sy) - 1 <= high[1];
                let floor = rules.depth.0 + (rules.depth.1 - rules.depth.0) * unit(z);
                let candidate = PlacedRoom { room, min, floor };
                let here = candidate.centre(rules.rooms);
                let clear = placed.iter().all(|other| {
                    let there = other.centre(rules.rooms);
                    libm::hypotf(here[0] - there[0], here[1] - there[1]) >= rules.apart
                });
                (inside && clear).then_some(candidate)
            });
            match spot {
                Some(spot) => placed.push(spot),
                None => break,
            }
        }
        if placed.len() == count as usize {
            let links = (0..placed.len())
                .filter_map(|index| parent(pattern, index).map(|parent| (parent, index)))
                .collect();
            return Ok(CavePlan {
                rooms: placed,
                links,
            });
        }
        log.push(format!(
            "attempt {attempt}: a {pattern:?} level of {count} rooms placed {} of them in {} \
             tries each",
            placed.len(),
            rules.tries
        ));
    }
    Err(log)
}

/// The path of a tunnel from `from` to `to`, both inside the box from `low` to `high` in cells,
/// found by A* over a grid of `step` cells filling that box: from each grid point to any of its 26
/// neighbours, at the cost of the step's length times `cost` at the step's middle, which is at
/// least 1. The path starts at `from`, ends at `to`, and passes through the grid points between.
///
/// # Panics
/// If `from` or `to` lies outside the box, or `cost` is below 1 somewhere, which would let the
/// search return a path longer than the cheapest.
pub(crate) fn tunnel(
    from: [f32; 3],
    to: [f32; 3],
    low: [f32; 3],
    high: [f32; 3],
    step: f32,
    cost: impl Fn([f32; 3]) -> f32,
) -> Vec<[f32; 3]> {
    let size: [i64; 3] =
        std::array::from_fn(|axis| (((high[axis] - low[axis]) / step).floor() as i64).max(1));
    let point = |at: [i64; 3]| -> [f32; 3] {
        std::array::from_fn(|axis| low[axis] + (at[axis] as f32 + 0.5) * step)
    };
    let node = |at: [f32; 3]| -> [i64; 3] {
        std::array::from_fn(|axis| {
            assert!(
                (low[axis]..=high[axis]).contains(&at[axis]),
                "a tunnel's end lies inside its box"
            );
            (((at[axis] - low[axis]) / step).floor() as i64).clamp(0, size[axis] - 1)
        })
    };
    let index = |at: [i64; 3]| ((at[2] * size[1] + at[1]) * size[0] + at[0]) as usize;
    let distance = |a: [f32; 3], b: [f32; 3]| {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    };
    let (start, goal) = (node(from), node(to));
    let cells = (size[0] * size[1] * size[2]) as usize;
    let mut spent = vec![f32::INFINITY; cells];
    let mut came = vec![usize::MAX; cells];
    // Costs are never negative, so their bits order as the costs do; ties go to the lower index.
    let mut open = BinaryHeap::new();
    spent[index(start)] = 0.0;
    open.push(Reverse((
        distance(point(start), point(goal)).to_bits(),
        index(start),
        start,
    )));
    while let Some(Reverse((_, at_index, at))) = open.pop() {
        if at == goal {
            break;
        }
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let next = [at[0] + dx, at[1] + dy, at[2] + dz];
                    if (dx, dy, dz) == (0, 0, 0)
                        || (0..3).any(|axis| !(0..size[axis]).contains(&next[axis]))
                    {
                        continue;
                    }
                    let (a, b) = (point(at), point(next));
                    let middle = std::array::from_fn(|axis| (a[axis] + b[axis]) / 2.0);
                    let factor = cost(middle);
                    assert!(factor >= 1.0, "a step costs at least its length");
                    let through = spent[at_index] + distance(a, b) * factor;
                    let next_index = index(next);
                    if through < spent[next_index] {
                        spent[next_index] = through;
                        came[next_index] = at_index;
                        let estimate = through + distance(b, point(goal));
                        open.push(Reverse((estimate.to_bits(), next_index, next)));
                    }
                }
            }
        }
    }
    let unindex = |mut i: usize| -> [i64; 3] {
        let x = i as i64 % size[0];
        i /= size[0] as usize;
        [x, i as i64 % size[1], i as i64 / size[1]]
    };
    let mut nodes = vec![goal];
    let mut at = index(goal);
    while at != index(start) {
        at = came[at];
        nodes.push(unindex(at));
    }
    nodes.reverse();
    let mut path = vec![from];
    for at in nodes.into_iter().map(point).chain([to]) {
        if path.last() != Some(&at) {
            path.push(at);
        }
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tunnel_over_even_ground_runs_straight_between_its_ends() {
        let path = tunnel(
            [1.0, 5.0, 5.0],
            [19.0, 5.0, 5.0],
            [0.0; 3],
            [20.0, 10.0, 10.0],
            2.0,
            |_| 1.0,
        );

        assert!(
            path.iter()
                .all(|p| (p[1] - 5.0).abs() <= 1.0 && (p[2] - 5.0).abs() <= 1.0)
        );
        assert_eq!(
            (path[0], path[path.len() - 1]),
            ([1.0, 5.0, 5.0], [19.0, 5.0, 5.0])
        );
    }

    #[test]
    fn a_tunnel_goes_round_a_costly_wall() {
        // A wall across the middle, costly everywhere but its top.
        let wall = |p: [f32; 3]| {
            if (8.0..12.0).contains(&p[0]) && p[2] < 8.0 {
                50.0
            } else {
                1.0
            }
        };

        let path = tunnel(
            [1.0, 5.0, 3.0],
            [19.0, 5.0, 3.0],
            [0.0; 3],
            [20.0, 10.0, 10.0],
            2.0,
            wall,
        );

        let middles = path
            .windows(2)
            .map(|pair| std::array::from_fn(|axis| (pair[0][axis] + pair[1][axis]) / 2.0));
        assert!(middles.map(wall).all(|cost| cost == 1.0), "{path:?}");
    }
}
