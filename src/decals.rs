//! Decals along curves (docs/reference/godot.md, "Road decals"): the pieces an engine projects a
//! decal onto the ground with, along a road or any other curve a stage makes.
//!
//! A curve's pieces stand at distances measured from its start, each as long as the curve is wide
//! where it begins, so a square texture stays square and every chunk the curve passes through cuts
//! the same pieces. A chunk keeps the pieces whose middle stands in it: neighbouring chunks meet
//! without a gap or a doubled piece, in any order.

use crate::stages::regions::Curve;
use wfc_core::ChunkCoord;

/// The shortest piece, in cells, so a curve with no width still gets pieces of a sensible count.
const MIN_LENGTH: f32 = 1.0;

/// One piece of a curve to project a decal onto, in an engine's world space.
#[derive(Clone, Debug, PartialEq)]
pub struct DecalPiece {
    /// The middle of the piece, on the ground.
    pub at: [f32; 3],
    /// The unit direction along the curve on the ground plane, as x and z.
    pub along: [f32; 2],
    /// How long the piece is along `along`.
    pub length: f32,
    /// How wide the piece is across `along`: the curve's width at the wider of its two ends.
    pub width: f32,
}

/// The decal pieces of `curves` whose middle stands in `chunk` of `size` columns, on the ground
/// `height` gives a column in cells, in an engine's world space with cells `cell_size`. A curve's
/// values are its radius in cells, as an Apply stage reads them.
///
/// Returns `None` if `height` lacks the column under a piece's middle, which is always one of
/// `chunk`'s own.
#[must_use]
pub fn curve_decals(
    curves: &[Curve],
    chunk: ChunkCoord,
    size: [u32; 2],
    height: impl Fn(i64, i64) -> Option<f32>,
    cell_size: [f32; 3],
) -> Option<Vec<DecalPiece>> {
    let (low_x, low_y) = (
        i64::from(chunk.x) * i64::from(size[0]),
        i64::from(chunk.y) * i64::from(size[1]),
    );
    let inside = |[x, y]: [f32; 2]| {
        let (column_x, column_y) = (x.floor() as i64, y.floor() as i64);
        (low_x..low_x + i64::from(size[0])).contains(&column_x)
            && (low_y..low_y + i64::from(size[1])).contains(&column_y)
    };
    let mut pieces = Vec::new();
    for curve in curves {
        let walk = Walk::new(curve);
        let mut start = 0.0;
        while start < walk.total {
            let length = (2.0 * walk.radius(start)).max(MIN_LENGTH);
            let end = (start + length).min(walk.total);
            let middle = walk.point((start + end) / 2.0);
            if inside(middle) {
                let (from, to) = (walk.point(start), walk.point(end));
                let chord = [
                    (to[0] - from[0]) * cell_size[0],
                    (to[1] - from[1]) * cell_size[2],
                ];
                let chord_length = libm::hypotf(chord[0], chord[1]);
                let along = [chord[0] / chord_length, chord[1] / chord_length];
                // Across the piece, in cells, to measure its width in the world's units.
                let across = [-along[1] / cell_size[0], along[0] / cell_size[2]];
                let across_cells = libm::hypotf(across[0], across[1]);
                let radius = walk.radius(start).max(walk.radius(end));
                let ground = height(middle[0].floor() as i64, middle[1].floor() as i64)?;
                pieces.push(DecalPiece {
                    at: [
                        middle[0] * cell_size[0],
                        ground * cell_size[1],
                        middle[1] * cell_size[2],
                    ],
                    along,
                    length: chord_length,
                    width: 2.0 * radius / across_cells,
                });
            }
            start = end;
        }
    }
    Some(pieces)
}

/// A curve measured along its length from its first point.
struct Walk<'a> {
    curve: &'a Curve,
    /// The distance from the first point to the start of each segment, and to the end of the last.
    starts: Vec<f32>,
    total: f32,
}

impl<'a> Walk<'a> {
    fn new(curve: &'a Curve) -> Self {
        let mut starts = vec![0.0];
        for pair in curve.points.windows(2) {
            let length = libm::hypotf(pair[1][0] - pair[0][0], pair[1][1] - pair[0][1]);
            starts.push(starts[starts.len() - 1] + length);
        }
        let total = starts[starts.len() - 1];
        Self {
            curve,
            starts,
            total,
        }
    }

    /// The segment `distance` falls in and how far along it, from 0 to 1.
    fn locate(&self, distance: f32) -> (usize, f32) {
        let segment = self
            .starts
            .partition_point(|&start| start <= distance)
            .saturating_sub(1)
            .min(self.starts.len().saturating_sub(2));
        let length = self.starts[segment + 1] - self.starts[segment];
        let t = if length > 0.0 {
            ((distance - self.starts[segment]) / length).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (segment, t)
    }

    fn point(&self, distance: f32) -> [f32; 2] {
        let (segment, t) = self.locate(distance);
        let (from, to) = (self.curve.points[segment], self.curve.points[segment + 1]);
        [
            from[0] + (to[0] - from[0]) * t,
            from[1] + (to[1] - from[1]) * t,
        ]
    }

    fn radius(&self, distance: f32) -> f32 {
        let (segment, t) = self.locate(distance);
        let values = &self.curve.values;
        values[segment] + (values[segment + 1] - values[segment]) * t
    }
}
