//! Rivers that run downhill, the region job a pack names with a Rivers stage.
//!
//! Every region sends a few rivers from high ground down its height field. A priority flood over
//! the region points every column at the one its water runs to next, toward the region's edge or
//! the sea; from each source a river follows those columns, through any hollow over the lowest point
//! it spills over and across flats toward their way out, until it reaches the sea, a lake or the
//! edge of its region. A river never leaves its region, so no region reads another's rivers, and
//! every region comes out the same in any order.

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

impl RegionJob for DownhillRivers<'_> {
    fn run(&self, input: &RegionInput<'_>) -> Result<Attempt, StageError> {
        let ([x0, y0], [x1, y1]) = input.columns();
        let (width, depth) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);
        if width < 3 || depth < 3 {
            // Every column of so small a region is on its edge, where no river starts.
            return Ok(Attempt::Accepted(Vec::new()));
        }
        let mut heights = Vec::with_capacity(width * depth);
        for y in y0..=y1 {
            for x in x0..=x1 {
                heights.push(input.field(self.height, x, y)?);
            }
        }
        let toward = flood(&heights, [width, depth], [x0, y0], self.sea).toward;
        let column = |at: usize| (x0 + (at % width) as i64, y0 + (at / width) as i64);
        let step = self.step as usize;
        // The columns earlier rivers run through: a later one ends where it joins one.
        let mut taken = vec![false; heights.len()];
        let mut curves = Vec::new();
        for source in 0..self.sources {
            let mut start = None;
            for attempt in 0..SOURCE_TRIES {
                let hash = input.hash(source * SOURCE_TRIES + attempt);
                // A source within the region's edge, whose columns drain off it.
                let at = 1
                    + (hash & 0xFFFF) as usize % (width - 2)
                    + (1 + (hash >> 16) as usize % (depth - 2)) * width;
                if start.is_none_or(|best: usize| heights[at] > heights[best]) {
                    start = Some(at);
                }
            }
            let mut at = start.expect("a source is chosen among several columns");
            let mut cells = vec![at];
            // The flood's columns lead to the edge or the sea without a loop.
            while let Some(next) = toward[at] {
                if taken[at] {
                    break;
                }
                let (x, y) = column(at);
                if let Some((lakes, ground)) = self.lakes
                    && input.field(lakes, x, y)? > input.field(ground, x, y)?
                {
                    break;
                }
                at = next;
                cells.push(at);
            }
            for &cell in &cells {
                taken[cell] = true;
            }
            // A point every `step` cells, and the mouth.
            let mut path: Vec<(i64, i64)> =
                cells.iter().step_by(step).map(|&at| column(at)).collect();
            if (cells.len() - 1) % step != 0 {
                path.push(column(*cells.last().expect("a river has its source")));
            }
            if path.len() < 2 {
                continue;
            }
            let last = (path.len() - 1) as f32;
            curves.push(Curve {
                id: CurveId::Region {
                    region: input.region(),
                    index: source,
                },
                points: path
                    .iter()
                    .map(|&(x, y)| [x as f32 + 0.5, y as f32 + 0.5])
                    .collect(),
                values: (0..path.len())
                    .map(|i| self.width.0 + (self.width.1 - self.width.0) * i as f32 / last)
                    .collect(),
                heights: Vec::new(),
            });
        }
        Ok(Attempt::Accepted(curves))
    }
}
