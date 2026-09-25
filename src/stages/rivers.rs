//! Rivers that run downhill, the region job a pack names with a Rivers stage.
//!
//! Every region sends a few rivers from high ground down its height field: from each source a
//! river steps to the lowest of the eight columns `step` cells around it until it reaches the sea,
//! a lake, a hollow it cannot leave, or the edge of its region. A river never leaves its region, so no
//! region reads another's rivers, and every region comes out the same in any order.

use super::regions::{Attempt, Curve, CurveId, RegionInput, RegionJob};
use super::runtime::StageError;

/// The Rivers stage's job: `sources` rivers per region down the field `height` to `sea`, the
/// pack's water level, or into one of its `lakes`, stepping `step` cells at a time, widening from
/// `width.0` at the source to `width.1` at the mouth.
pub(crate) struct DownhillRivers<'a> {
    pub(crate) height: &'a str,
    pub(crate) sources: u32,
    pub(crate) sea: f32,
    /// The pack's Lakes stage, whose water a river ends in.
    pub(crate) lakes: Option<&'a str>,
    pub(crate) width: (f32, f32),
    pub(crate) step: u32,
}

/// How many columns a source is chosen among: the highest wins, so rivers start on high ground.
const SOURCE_TRIES: u32 = 8;

impl RegionJob for DownhillRivers<'_> {
    fn run(&self, input: &RegionInput<'_>) -> Result<Attempt, StageError> {
        let ([x0, y0], [x1, y1]) = input.columns();
        let step = i64::from(self.step);
        let inside = |(x, y): (i64, i64)| (x0..=x1).contains(&x) && (y0..=y1).contains(&y);
        let mut curves = Vec::new();
        for source in 0..self.sources {
            let mut start = None;
            for attempt in 0..SOURCE_TRIES {
                let hash = input.hash(source * SOURCE_TRIES + attempt);
                let column = (
                    x0 + i64::from(hash & 0xFFFF) % (x1 - x0 + 1),
                    y0 + i64::from(hash >> 16) % (y1 - y0 + 1),
                );
                let height = input.field(self.height, column.0, column.1)?;
                if start.is_none_or(|(_, best)| height > best) {
                    start = Some((column, height));
                }
            }
            let (mut at, mut height) = start.expect("a source is chosen among several columns");
            let mut path = vec![at];
            // A river can take at most as many steps as its region has columns along a side.
            for _ in 0..(x1 - x0 + 1).max(y1 - y0 + 1) {
                if height < self.sea {
                    break;
                }
                if let Some(lakes) = self.lakes
                    && input.field(lakes, at.0, at.1)? > height
                {
                    break;
                }
                let mut lowest = (at, height);
                for (dx, dy) in [
                    (-1, -1),
                    (0, -1),
                    (1, -1),
                    (-1, 0),
                    (1, 0),
                    (-1, 1),
                    (0, 1),
                    (1, 1),
                ] {
                    let next = (at.0 + dx * step, at.1 + dy * step);
                    if !inside(next) {
                        continue;
                    }
                    let there = input.field(self.height, next.0, next.1)?;
                    if there < lowest.1 {
                        lowest = (next, there);
                    }
                }
                if lowest.0 == at {
                    break;
                }
                (at, height) = lowest;
                path.push(at);
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
            });
        }
        Ok(Attempt::Accepted(curves))
    }
}
