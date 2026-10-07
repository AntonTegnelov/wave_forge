//! The Erode stage's filter: gullies cut into a height field, running downhill and branching, from
//! the field's slope around each column alone, so chunks generate in any order and meet without
//! seams.
//!
//! Every octave lays stripes across the slope through points jittered one to a lattice cell, each
//! point's stripes weighted by how near it is, so the stripes run downhill and break where the
//! points hand over. Below the slope that cuts fully, the stripes widen as well as fade, so where
//! the slope turns on flat ground they change smoothly instead of turning on the spot. Each octave runs along the slope the octaves before it left, ground and
//! gullies together, so finer gullies turn off coarser ones and branch. Each fades out where that
//! slope flattens, since there no slope says which way water runs.

use super::runtime::{StageError, lattice};
use std::f32::consts::TAU;

/// What an Erode stage cuts, as its pack declares it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Erosion {
    /// Cells between gullies in the first octave.
    pub spacing: f32,
    pub octaves: u32,
    /// Cells of height from a gully's floor to the ridge beside it in the first octave, halved.
    pub depth: f32,
    /// Each octave's depth against the one before it.
    pub gain: f32,
    /// The input's slope, in cells of height per cell, at which gullies are cut at full depth;
    /// below half of it none are.
    pub slope: f32,
}

/// The slope of `read`'s field at `column`, in its height per column along x and y: the plane
/// fitted by least squares over the square of `radius` columns around it, which smooths the
/// field's own creases.
pub(crate) fn fitted_slope(
    column: [i64; 2],
    radius: u32,
    read: impl Fn(i64, i64) -> Result<f32, StageError>,
) -> Result<[f32; 2], StageError> {
    let r = i64::from(radius);
    let (mut along_x, mut along_y) = (0.0_f32, 0.0_f32);
    for dy in -r..=r {
        for dx in -r..=r {
            let value = read(column[0] + dx, column[1] + dy)?;
            along_x += dx as f32 * value;
            along_y += dy as f32 * value;
        }
    }
    // Each offset's square summed over the window: its rows times the sum of squares along one.
    let side = (2 * r + 1) as f32;
    let squares = side * (r * (r + 1) * (2 * r + 1)) as f32 / 3.0;
    Ok([along_x / squares, along_y / squares])
}

/// How far `erosion` moves the ground at `at`, in cells, where the input slopes by `slope` cells of
/// height per cell: its gullies below zero and the ridges between them above.
pub(crate) fn erode(seed: u64, salt: u32, erosion: &Erosion, at: [f32; 2], slope: [f32; 2]) -> f32 {
    let mut downhill = slope;
    let mut moved = 0.0;
    let mut depth = erosion.depth;
    let mut frequency = 1.0 / erosion.spacing;
    for octave in 0..erosion.octaves {
        // Each octave fades out where the ground it runs along, gullies included, flattens to half
        // the slope that cuts fully: gentle ground keeps its shape, and where the slope is too
        // slight to settle a direction, which turns fast there, the octave has no depth left.
        let length = downhill[0].hypot(downhill[1]);
        let t = ((length / erosion.slope - 0.5) * 2.0).clamp(0.0, 1.0);
        let fade = t * t * (3.0 - 2.0 * t);
        if fade == 0.0 {
            break;
        }
        // Across the slope, at full length where it is steep enough to cut fully and shorter
        // below, so the stripes widen into the flat rather than turning on the spot.
        let reach = length.max(erosion.slope);
        let across = [-downhill[1] / reach, downhill[0] / reach];
        let (value, change) = stripes(
            seed,
            salt,
            octave,
            [at[0] * frequency, at[1] * frequency],
            across,
        );
        let cut = depth * fade;
        moved += value * cut;
        // The stripes' own slope, in cells of height per cell, steers the next octave.
        downhill[0] += change[0] * cut * frequency;
        downhill[1] += change[1] * cut * frequency;
        depth *= erosion.gain;
        frequency *= 2.0;
    }
    moved
}

/// The stripes of one octave at `p`, in lattice units, and their slope there: a cosine across the
/// slope through each jittered lattice point within two cells, weighted by the point's nearness,
/// from -1 in a gully to 1 on a ridge.
fn stripes(seed: u64, salt: u32, octave: u32, p: [f32; 2], across: [f32; 2]) -> (f32, [f32; 2]) {
    let (cx, cy) = (p[0].floor() as i64, p[1].floor() as i64);
    let (mut sum, mut total) = (0.0_f32, 0.0_f32);
    let mut change = [0.0_f32; 2];
    for y in cy - 2..=cy + 2 {
        for x in cx - 2..=cx + 2 {
            let point = [
                x as f32 + lattice(seed, salt, 2 * octave, x, y),
                y as f32 + lattice(seed, salt, 2 * octave + 1, x, y),
            ];
            let offset = [p[0] - point[0], p[1] - point[1]];
            let weight = (-3.0 * (offset[0] * offset[0] + offset[1] * offset[1])).exp();
            let phase = TAU * (offset[0] * across[0] + offset[1] * across[1]);
            sum += weight * phase.cos();
            let rate = -weight * phase.sin() * TAU;
            change[0] += rate * across[0];
            change[1] += rate * across[1];
            total += weight;
        }
    }
    (sum / total, [change[0] / total, change[1] / total])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn erosion() -> Erosion {
        Erosion {
            spacing: 16.0,
            octaves: 4,
            depth: 3.0,
            gain: 0.5,
            slope: 0.5,
        }
    }

    #[test]
    fn flat_ground_is_left_alone() {
        let moved = erode(7, 1, &erosion(), [12.5, 40.5], [0.0, 0.0]);

        assert_eq!(moved, 0.0);
    }

    #[test]
    fn a_fitted_slope_is_a_planes_own() {
        let plane = |x: i64, y: i64| Ok(0.3 * x as f32 - 0.7 * y as f32 + 2.0);

        let slope = fitted_slope([5, -3], 2, plane).expect("a slope");

        assert!(
            (slope[0] - 0.3).abs() < 1e-5 && (slope[1] + 0.7).abs() < 1e-5,
            "{slope:?}"
        );
    }

    #[test]
    fn the_cut_changes_smoothly_along_the_ground() {
        // Along a line where the slope turns and steepens, a step of a twentieth of a cell moves
        // the cut by a small share of its depth.
        let step = 0.05;
        let mut steepest = 0.0_f32;
        let mut last: Option<f32> = None;
        for i in 0..4000 {
            let x = i as f32 * step;
            let turn = x / 30.0;
            let slope = [0.6 * turn.cos(), 0.6 * turn.sin() * (x / 70.0)];
            let cut = erode(11, 2, &erosion(), [x, 7.3], slope);
            if let Some(previous) = last {
                steepest = steepest.max((cut - previous).abs());
            }
            last = Some(cut);
        }

        assert!(steepest < 0.3, "the cut moved {steepest} in {step} cells");
    }

    #[test]
    fn gullies_run_downhill() {
        // Ground falling along +x: a gully runs along x, so moving along x changes the cut far
        // less than moving across, along y.
        let slope = [1.0, 0.0];
        let first_octave = Erosion {
            octaves: 1,
            ..erosion()
        };
        let cut = |x: f32, y: f32| erode(3, 9, &first_octave, [x, y], slope);
        let (mut along, mut across) = (0.0, 0.0);
        for i in 0..400 {
            let (x, y) = (i as f32 * 1.7, (i % 37) as f32 * 3.1);
            along += (cut(x + 1.0, y) - cut(x, y)).abs();
            across += (cut(x, y + 1.0) - cut(x, y)).abs();
        }

        assert!(across > 3.0 * along, "along {along}, across {across}");
    }
}
