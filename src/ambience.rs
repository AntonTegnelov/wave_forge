//! The sounds the terrain makes (docs/reference/packs.md, "Ambience"): emitters along rivers, louder
//! where a river runs wider and falls faster, and at lake shores.
//!
//! A river's emitters stand along its curve at distances measured from the curve's start, so every
//! chunk the curve passes through places the same ones and keeps those inside it: neighbouring
//! chunks meet without a gap or a doubled emitter, in any order.

use crate::region_tags::Emitter;
use crate::stages::regions::Curve;
use crate::stages::{AmbienceDef, AmbienceKind, Field};
use wfc_core::ChunkCoord;

/// Cells between a river's emitters for each cell of its radius there.
const SPACING_PER_RADIUS: f32 = 6.0;
/// The fewest cells between two of a river's emitters.
const MIN_SPACING: f32 = 4.0;
/// Cells either side of an emitter over which its river's fall is measured.
const FALL_SPAN: f32 = 2.0;
/// A river's flow, its width in cells times its bed's fall per cell, at which it plays at full
/// volume; a river with less plays as much quieter, down to [`QUIETEST`].
pub const FULL_FLOW: f32 = 0.5;
/// The volume of a river that hardly flows, so it still murmurs.
pub const QUIETEST: f32 = 0.1;
/// The volume of a lake's shore.
pub const SHORE_VOLUME: f32 = 0.5;
/// How far, in cells of height, a lake's surface stands above the ground where its water counts.
const WET: f32 = 0.05;

/// The sounds the terrain makes in `chunk` of `size` columns, as a pack declares them in `defs`
/// ([`crate::stages::PackFile::ambience`]), in an engine's world space with cells `cell_size`:
/// [`river_emitters`] along each river and [`shore_emitters`] at each lake's shore, reading the
/// curves `curves` gives and the fields `field` gives, by stage and chunk.
///
/// Returns `None` until what they read has arrived: the curves of `chunk`, and the fields of
/// `chunk` and of the chunks around it.
#[must_use]
pub fn chunk_ambience<'a>(
    defs: &[AmbienceDef],
    chunk: ChunkCoord,
    size: [u32; 2],
    curves: impl Fn(&str, ChunkCoord) -> Option<&'a [Curve]>,
    field: impl Fn(&str, ChunkCoord) -> Option<&'a Field>,
    cell_size: [f32; 3],
) -> Option<Vec<Emitter>> {
    let column = |stage: &str, x: i64, y: i64| {
        let [sx, sy] = size.map(i64::from);
        let at = ChunkCoord::new(x.div_euclid(sx) as i32, y.div_euclid(sy) as i32, chunk.z);
        Some(field(stage, at)?.get(x.rem_euclid(sx) as u32, y.rem_euclid(sy) as u32))
    };
    let mut emitters = Vec::new();
    for def in defs {
        emitters.extend(match &def.kind {
            AmbienceKind::River {
                curves: stage,
                height,
            } => river_emitters(
                &def.key,
                curves(stage, chunk)?,
                chunk,
                size,
                |x, y| column(height, x, y),
                cell_size,
            )?,
            AmbienceKind::Lake { lakes, height } => shore_emitters(
                &def.key,
                chunk,
                size,
                |x, y| column(lakes, x, y),
                |x, y| column(height, x, y),
                cell_size,
            )?,
        });
    }
    Some(emitters)
}

/// The emitters, playing `key`, of `curves` that stand in `chunk` of `size` columns: along each
/// curve, spaced by its radius, at the height `height` gives a column in cells, in an engine's world
/// space with cells `cell_size`. Each plays as loud as its river's flow there, its width times the
/// fall of `height` along it per cell, against [`FULL_FLOW`].
///
/// Returns `None` if `height` lacks a column it reads: one under an emitter, or a few cells up or
/// down the river from it, which may lie in a neighbouring chunk.
#[must_use]
pub fn river_emitters(
    key: &str,
    curves: &[Curve],
    chunk: ChunkCoord,
    size: [u32; 2],
    height: impl Fn(i64, i64) -> Option<f32>,
    cell_size: [f32; 3],
) -> Option<Vec<Emitter>> {
    let (low_x, low_y) = (
        i64::from(chunk.x) * i64::from(size[0]),
        i64::from(chunk.y) * i64::from(size[1]),
    );
    let inside = |x: f32, y: f32| {
        let (column_x, column_y) = (x.floor() as i64, y.floor() as i64);
        (low_x..low_x + i64::from(size[0])).contains(&column_x)
            && (low_y..low_y + i64::from(size[1])).contains(&column_y)
    };
    let at = |x: f32, y: f32| height(x.floor() as i64, y.floor() as i64);
    let mut emitters = Vec::new();
    for curve in curves {
        let lengths: Vec<f32> = curve
            .points
            .windows(2)
            .map(|pair| (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]))
            .collect();
        let total: f32 = lengths.iter().sum();
        let radius_at = |segment: usize, t: f32| {
            curve.values[segment] + (curve.values[segment + 1] - curve.values[segment]) * t
        };
        let spacing = |radius: f32| (radius * SPACING_PER_RADIUS).max(MIN_SPACING);
        let mut distance = spacing(curve.values[0]) / 2.0;
        let (mut segment, mut start) = (0, 0.0);
        while distance < total {
            while start + lengths[segment] < distance {
                start += lengths[segment];
                segment += 1;
            }
            let t = (distance - start) / lengths[segment];
            let (from, to) = (curve.points[segment], curve.points[segment + 1]);
            let (x, y) = (
                from[0] + (to[0] - from[0]) * t,
                from[1] + (to[1] - from[1]) * t,
            );
            let radius = radius_at(segment, t);
            if inside(x, y) {
                let along = [
                    (to[0] - from[0]) / lengths[segment],
                    (to[1] - from[1]) / lengths[segment],
                ];
                let up = at(x - along[0] * FALL_SPAN, y - along[1] * FALL_SPAN)?;
                let down = at(x + along[0] * FALL_SPAN, y + along[1] * FALL_SPAN)?;
                let fall = ((up - down) / (2.0 * FALL_SPAN)).max(0.0);
                let flow = 2.0 * radius * fall;
                emitters.push(Emitter {
                    at: [x * cell_size[0], at(x, y)? * cell_size[1], y * cell_size[2]],
                    key: key.to_owned(),
                    volume: (flow / FULL_FLOW).clamp(QUIETEST, 1.0),
                });
            }
            distance += spacing(radius);
        }
    }
    Some(emitters)
}

/// The emitter, playing `key`, of the lakes in `chunk` of `size` columns: at the shore column
/// nearest the chunk's middle, a lake column with dry ground beside it, where `lakes` gives the
/// water's surface and `height` the ground, both in cells, in an engine's world space with cells
/// `cell_size`, at [`SHORE_VOLUME`]. None where the chunk has no shore.
///
/// Returns `None` if `lakes` or `height` lacks a column it reads: the chunk's own and those beside
/// its edges.
#[must_use]
pub fn shore_emitters(
    key: &str,
    chunk: ChunkCoord,
    size: [u32; 2],
    lakes: impl Fn(i64, i64) -> Option<f32>,
    height: impl Fn(i64, i64) -> Option<f32>,
    cell_size: [f32; 3],
) -> Option<Vec<Emitter>> {
    let wet = |x: i64, y: i64| Some(lakes(x, y)? - height(x, y)? > WET);
    let (low_x, low_y) = (
        i64::from(chunk.x) * i64::from(size[0]),
        i64::from(chunk.y) * i64::from(size[1]),
    );
    let middle = (
        low_x as f32 + size[0] as f32 / 2.0,
        low_y as f32 + size[1] as f32 / 2.0,
    );
    let mut best: Option<(f32, i64, i64)> = None;
    for y in low_y..low_y + i64::from(size[1]) {
        for x in low_x..low_x + i64::from(size[0]) {
            if !wet(x, y)? {
                continue;
            }
            let mut shore = false;
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                shore |= !wet(x + dx, y + dy)?;
            }
            if !shore {
                continue;
            }
            let near = (x as f32 + 0.5 - middle.0).hypot(y as f32 + 0.5 - middle.1);
            if best.is_none_or(|(nearest, ..)| near < nearest) {
                best = Some((near, x, y));
            }
        }
    }
    Some(
        best.map(|(_, x, y)| Emitter {
            at: [
                (x as f32 + 0.5) * cell_size[0],
                lakes(x, y).expect("read above") * cell_size[1],
                (y as f32 + 0.5) * cell_size[2],
            ],
            key: key.to_owned(),
            volume: SHORE_VOLUME,
        })
        .into_iter()
        .collect(),
    )
}
