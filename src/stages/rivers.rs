//! Rivers that run downhill, the region job a pack names with a Rivers stage.
//!
//! Every region's water leaves it through the sea and through crossings: on each side of the
//! region, the lowest pair of columns facing each other across the border, which the regions on
//! both sides find alike from the border's heights alone. Water crosses from the higher column of
//! the pair to the lower, so a crossing lets water out of one region and into the other.
//!
//! A priority flood from the sea and the crossings water leaves by points every column at the one
//! its water runs to next, through any hollow over the lowest point it spills over and across flats
//! toward their way out. A river runs up from every crossing water leaves by, along the columns that
//! gather the most water, and on from every crossing water comes in by, so a river that leaves one
//! region carries on in the next without either reading the other's rivers; a few more start on
//! high ground. Each follows the flood's columns until it reaches the sea, a crossing, a river traced
//! before it, or a lake. Every region comes out the same in any order.

use super::lakes::flood;
use super::regions::{Attempt, Curve, CurveId, RegionInput, RegionJob};
use super::runtime::StageError;

/// The Rivers stage's job: `sources` rivers per region down the field `height` to `sea`, the
/// pack's water level, or into one of its `lakes`, with a point every `step` cells along it,
/// widening from `width.0` at the source to `width.1` at the mouth.
pub(crate) struct DownhillRivers<'a> {
    pub(crate) height: &'a str,
    pub(crate) sources: u32,
    pub(crate) sea: f32,
    /// The pack's Lakes stage, whose water a river ends in, and the ground its lakes were filled
    /// on: a column is under a lake where the water stands above that ground, whatever height the
    /// river runs down.
    pub(crate) lakes: Option<(&'a str, &'a str)>,
    pub(crate) width: (f32, f32),
    pub(crate) step: u32,
}

/// How many columns a source is chosen among: the highest wins, so rivers start on high ground.
const SOURCE_TRIES: u32 = 8;
/// How many columns' water a river running up from a crossing still gathers where it starts.
const STEM_GATHERS: u32 = 64;

/// A side of a region, named by the direction it faces.
#[derive(Clone, Copy)]
enum Side {
    West,
    East,
    South,
    North,
}

impl RegionJob for DownhillRivers<'_> {
    fn run(&self, input: &RegionInput<'_>) -> Result<Attempt, StageError> {
        let ([x0, y0], [x1, y1]) = input.columns();
        let (width, depth) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);
        if width < 5 || depth < 5 {
            // So small a region has no column off its edge and away from its corners.
            return Ok(Attempt::Accepted(Vec::new()));
        }
        let mut heights = Vec::with_capacity(width * depth);
        for y in y0..=y1 {
            for x in x0..=x1 {
                heights.push(input.field(self.height, x, y)?);
            }
        }
        let column = |at: usize| (x0 + (at % width) as i64, y0 + (at / width) as i64);
        let index = |(x, y): (i64, i64)| (y - y0) as usize * width + (x - x0) as usize;

        // Each side's crossing: the pair of columns facing each other across the border whose
        // lower column is lowest, the first along the side on a tie, which the region beyond finds
        // alike. Water leaves by it where the column inside stands higher; on a level pair it runs
        // toward +x and +y. Each crossing water leaves by is kept with the column inside it, which
        // is where the region's water drains to it.
        let (mut leaving, mut entering): (Vec<(usize, usize)>, Vec<usize>) =
            (Vec::new(), Vec::new());
        for side in [Side::West, Side::East, Side::South, Side::North] {
            let along: Vec<((i64, i64), (i64, i64))> = match side {
                Side::West => (y0..=y1).map(|y| ((x0, y), (x0 - 1, y))).collect(),
                Side::East => (y0..=y1).map(|y| ((x1, y), (x1 + 1, y))).collect(),
                Side::South => (x0..=x1).map(|x| ((x, y0), (x, y0 - 1))).collect(),
                Side::North => (x0..=x1).map(|x| ((x, y1), (x, y1 + 1))).collect(),
            };
            let mut lowest: Option<(f32, usize, f32)> = None;
            // The two columns at each end are left out: a corner lies on two sides, whose regions
            // across would each take it for a crossing, and the columns beside it share the column
            // inside them with the next side's.
            for &(inside, outside) in &along[2..along.len() - 2] {
                let (here, there) = (
                    heights[index(inside)],
                    input.field(self.height, outside.0, outside.1)?,
                );
                if lowest.is_none_or(|(low, ..)| here.min(there) < low) {
                    lowest = Some((here.min(there), index(inside), there));
                }
            }
            let (_, at, there) = lowest.expect("a side has columns");
            let toward_plus = matches!(side, Side::East | Side::North);
            if heights[at] > there || (heights[at] == there && toward_plus) {
                let inward = match side {
                    Side::West => at + 1,
                    Side::East => at - 1,
                    Side::South => at + width,
                    Side::North => at - width,
                };
                leaving.push((at, inward));
            } else {
                entering.push(at);
            }
        }
        let sea = |at: usize| heights[at] <= self.sea;
        if leaving.is_empty() && !(0..heights.len()).any(sea) {
            // A region lower than all around it at every crossing drains off its lowest edge column.
            let edge = (0..heights.len()).filter(|&at| {
                let (x, y) = (at % width, at / width);
                x == 0 || y == 0 || x == width - 1 || y == depth - 1
            });
            let at = edge
                .min_by(|&a, &b| heights[a].total_cmp(&heights[b]))
                .expect("an edge");
            let (x, y) = (at % width, at / width);
            let inward = y.clamp(1, depth - 2) * width + x.clamp(1, width - 2);
            leaving.push((at, inward));
        }
        let flood = flood(&heights, [width, depth], [x0, y0], |at| {
            sea(at) || leaving.iter().any(|&(_, inward)| inward == at)
        });
        let toward = flood.toward;

        // How many columns' water each column gathers, and the column above it that gives most.
        let mut gathers = vec![1u32; heights.len()];
        for &at in flood.reached.iter().rev() {
            if let Some(next) = toward[at] {
                gathers[next] += gathers[at];
            }
        }
        let mut most: Vec<Option<usize>> = vec![None; heights.len()];
        for &at in &flood.reached {
            if let Some(next) = toward[at]
                && most[next].is_none_or(|best| gathers[at] > gathers[best])
            {
                most[next] = Some(at);
            }
        }

        // The columns earlier rivers run through: a later one ends where it joins one.
        let mut taken = vec![false; heights.len()];
        let mut curves = Vec::new();
        let mut add = |taken: &mut Vec<bool>, cells: Vec<usize>, id: u32, from: f32| {
            for &cell in &cells {
                taken[cell] = true;
            }
            if let Some(curve) = self.curve(&cells, column, input.region(), id, from) {
                curves.push(curve);
            }
        };
        let mut id = self.sources;
        // Up from each crossing water leaves by, along the columns that gather most.
        for &(mouth, inward) in &leaving {
            // The column inside the crossing is the flood's outlet, so the water about it leaves
            // by the crossing and the stem runs up from there.
            let mut cells = vec![mouth, inward];
            let mut at = inward;
            // The region across starts a river at every crossing water leaves by, so one always
            // arrives: from where it gathers enough water, or, from a smaller basin, a quarter of
            // what the crossing gathers.
            let enough = STEM_GATHERS.min((gathers[inward] / 4).max(1));
            while let Some(above) = most[at].filter(|&above| gathers[above] >= enough) {
                // A river that drains a lake starts where it leaves it.
                if self.in_lake(input, column(above))? {
                    break;
                }
                at = above;
                cells.push(at);
            }
            cells.reverse();
            add(&mut taken, cells, id, self.width.0);
            id += 1;
        }
        // On from each crossing water comes in by, already a river.
        for &start in &entering {
            let cells = self.downstream(input, start, &toward, &taken, column)?;
            add(&mut taken, cells, id, self.width.1);
            id += 1;
        }
        for source in 0..self.sources {
            let mut start = None;
            for attempt in 0..SOURCE_TRIES {
                let hash = input.hash(source * SOURCE_TRIES + attempt);
                // A source within the region's edge, where the crossings are.
                let at = 1
                    + (hash & 0xFFFF) as usize % (width - 2)
                    + (1 + (hash >> 16) as usize % (depth - 2)) * width;
                if start.is_none_or(|best: usize| heights[at] > heights[best]) {
                    start = Some(at);
                }
            }
            let start = start.expect("a source is chosen among several columns");
            let cells = self.downstream(input, start, &toward, &taken, column)?;
            add(&mut taken, cells, source, self.width.0);
        }
        Ok(Attempt::Accepted(curves))
    }
}

impl DownhillRivers<'_> {
    /// The columns a river starting at `start` runs through, following `toward` until it reaches
    /// an outlet, a column in `taken`, which it joins, or a lake of the pack's water.
    fn downstream(
        &self,
        input: &RegionInput<'_>,
        start: usize,
        toward: &[Option<usize>],
        taken: &[bool],
        column: impl Fn(usize) -> (i64, i64),
    ) -> Result<Vec<usize>, StageError> {
        let mut at = start;
        let mut cells = vec![at];
        // The flood's columns lead to an outlet without a loop.
        while let Some(next) = toward[at] {
            if taken[at] {
                break;
            }
            if self.in_lake(input, column(at))? {
                break;
            }
            at = next;
            cells.push(at);
        }
        Ok(cells)
    }

    /// Whether the column at `(x, y)` is under a lake a river ends in: one of the pack's lakes,
    /// unless rivers run through them.
    fn in_lake(&self, input: &RegionInput<'_>, (x, y): (i64, i64)) -> Result<bool, StageError> {
        Ok(match self.lakes {
            Some((lakes, ground)) => input.field(lakes, x, y)? > input.field(ground, x, y)?,
            None => false,
        })
    }

    /// The curve of a river through `cells`, a point every `step` cells and its mouth, named `id`
    /// in `region`, widening from `from` at its first point to `width.1` at its mouth; `None` for a
    /// river of a single point.
    fn curve(
        &self,
        cells: &[usize],
        column: impl Fn(usize) -> (i64, i64),
        region: (i32, i32),
        id: u32,
        from: f32,
    ) -> Option<Curve> {
        let step = self.step as usize;
        let mut path: Vec<(i64, i64)> = cells.iter().step_by(step).map(|&at| column(at)).collect();
        if !(cells.len() - 1).is_multiple_of(step) {
            path.push(column(*cells.last().expect("a river has its source")));
        }
        if path.len() < 2 {
            return None;
        }
        let last = (path.len() - 1) as f32;
        Some(Curve {
            id: CurveId::Region { region, index: id },
            points: path
                .iter()
                .map(|&(x, y)| [x as f32 + 0.5, y as f32 + 0.5])
                .collect(),
            values: (0..path.len())
                .map(|i| from + (self.width.1 - from) * i as f32 / last)
                .collect(),
            heights: Vec::new(),
        })
    }
}
