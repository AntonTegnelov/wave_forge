//! Brushes: a stroke painted along a path becomes the edits it makes.
//!
//! An editor's brush, or a player's tool, drags along a path; [`stroke`] turns the brush and the
//! path into the [`Edit`]s to add to the log, in the library so every engine paints the same way
//! (docs/architecture/engine-integration.md, "One model, three workflows"). A stroke reads the
//! world as the runtime holds it, edits included, and changes nothing itself: the caller appends the
//! edits and gives the runtime the log, and an editor undoes a stroke by taking them out again.

use super::edits::{Edit, PointId};
use super::pack::{Output, StageKind};
use super::runtime::{Runtime, StageError};
use serde::{Deserialize, Serialize};
use wfc_core::ChunkCoord;

/// What a stroke does along its path. Distances are in cells, along the lattice's x and y for the
/// field brushes and in 3D for the volume brushes.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub enum Brush {
    /// Raises a field stage along the stroke, or lowers it with a negative `strength`: each column
    /// of the stage whose centre lies within `radius` of the path gains `strength` times how near
    /// it lies, 1 on the path and falling smoothly to 0 at the radius.
    Raise {
        stage: String,
        radius: f32,
        strength: f32,
    },
    /// Moves each column of a field stage within `radius` of the path toward the average of the
    /// nine columns around it, by `strength`, from 0 to 1, times how near it lies. The stage has to
    /// be one a sample reads without chunks ([`Runtime::sample`]).
    Smooth {
        stage: String,
        radius: f32,
        strength: f32,
    },
    /// Digs balls of `radius` out of a Volume or Carve stage along the path, half a radius apart,
    /// so the stroke leaves a tunnel as wide as the brush.
    Dig { stage: String, radius: f32 },
    /// Fills balls into a volume along the path, as [`Brush::Dig`] digs them.
    Fill { stage: String, radius: f32 },
    /// Removes every point of the point stages `stages` standing within `radius` of the path on
    /// the ground plane, of the chunks the runtime holds.
    Remove { stages: Vec<String>, radius: f32 },
}

/// The edits a stroke of `brush` along `path` makes in the world `runtime` holds: points in cells,
/// along the lattice's x and y and up, in the order they were painted. A path of one point is a
/// dab.
///
/// # Errors
/// [`StageError::Edit`] for a path without points, a radius that is not positive, a strength out
/// of range, or a stage that is not of the kind the brush paints; and what [`Runtime::sample`]
/// gives for a smoothed stage it cannot sample.
pub fn stroke(
    runtime: &Runtime,
    brush: &Brush,
    path: &[[f32; 3]],
) -> Result<Vec<Edit>, StageError> {
    let wrong = |message: String| Err(StageError::Edit(message));
    if path.is_empty() {
        return wrong("a stroke without a path".to_owned());
    }
    let radius = match brush {
        Brush::Raise { radius, .. }
        | Brush::Smooth { radius, .. }
        | Brush::Dig { radius, .. }
        | Brush::Fill { radius, .. }
        | Brush::Remove { radius, .. } => *radius,
    };
    if !(radius.is_finite() && radius > 0.0) {
        return wrong(format!("a brush of radius {radius}"));
    }
    match brush {
        Brush::Raise {
            stage, strength, ..
        } => {
            if !strength.is_finite() {
                return wrong(format!("a strength of {strength}"));
            }
            let scale = field_scale(runtime, stage)?;
            Ok(columns_near(path, radius, scale)
                .map(|(column, weight)| Edit::Raise {
                    stage: stage.clone(),
                    column,
                    by: strength * weight,
                })
                .collect())
        }
        Brush::Smooth {
            stage, strength, ..
        } => {
            if !(0.0..=1.0).contains(strength) {
                return wrong(format!("a strength of {strength}; 0 to 1 are allowed"));
            }
            let scale = field_scale(runtime, stage)?;
            let value = |(x, y): (i64, i64)| {
                let centre = |c: i64| (c as f32 + 0.5) * scale as f32;
                runtime.sample(stage, [centre(x), centre(y)])
            };
            let mut edits = Vec::new();
            for (column, weight) in columns_near(path, radius, scale) {
                let mut sum = 0.0;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        sum += value((column.0 + dx, column.1 + dy))?;
                    }
                }
                let by = (sum / 9.0 - value(column)?) * strength * weight;
                edits.push(Edit::Raise {
                    stage: stage.clone(),
                    column,
                    by,
                });
            }
            Ok(edits)
        }
        Brush::Dig { stage, .. } | Brush::Fill { stage, .. } => {
            if runtime.pack().kind(stage).map(StageKind::output) != Some(Output::Volume) {
                return wrong(format!("{stage:?} is no volume to dig or fill"));
            }
            Ok(along(path, radius / 2.0)
                .into_iter()
                .map(|at| match brush {
                    Brush::Dig { .. } => Edit::Dig {
                        stage: stage.clone(),
                        at,
                        radius,
                    },
                    _ => Edit::Fill {
                        stage: stage.clone(),
                        at,
                        radius,
                    },
                })
                .collect())
        }
        Brush::Remove { stages, .. } => {
            let mut edits = Vec::new();
            for stage in stages {
                if runtime.pack().kind(stage).map(StageKind::output) != Some(Output::Points) {
                    return wrong(format!("{stage:?} places no points to remove"));
                }
                for chunk in chunks_near(runtime.chunk_size(), path, radius) {
                    for point in runtime.points(stage, chunk).unwrap_or_default() {
                        let at = [point.position[0], point.position[1]];
                        if distance_to_path(path, at) < radius {
                            edits.push(Edit::Remove {
                                point: PointId::from(point.id),
                                at,
                            });
                        }
                    }
                }
            }
            Ok(edits)
        }
    }
}

/// The scale of the field stage `stage`, which a field brush paints.
fn field_scale(runtime: &Runtime, stage: &str) -> Result<u32, StageError> {
    match runtime.pack().kind(stage).map(StageKind::output) {
        Some(Output::Field) => Ok(runtime.pack().scale(stage).expect("a stage of the pack")),
        _ => Err(StageError::Edit(format!("{stage:?} is no field to paint"))),
    }
}

/// Every column of a stage at `scale` whose centre lies within `radius` cells of `path` on the
/// ground plane, with how near it lies: 1 on the path, falling smoothly to 0 at the radius.
fn columns_near(
    path: &[[f32; 3]],
    radius: f32,
    scale: u32,
) -> impl Iterator<Item = ((i64, i64), f32)> + '_ {
    let scale = scale as f32;
    let span = |axis: usize| {
        let low = path.iter().map(|p| p[axis]).fold(f32::INFINITY, f32::min) - radius;
        let high = path
            .iter()
            .map(|p| p[axis])
            .fold(f32::NEG_INFINITY, f32::max)
            + radius;
        (low / scale).floor() as i64..=(high / scale).floor() as i64
    };
    let (xs, ys) = (span(0), span(1));
    ys.flat_map(move |y| xs.clone().map(move |x| (x, y)))
        .filter_map(move |(x, y)| {
            let centre = [(x as f32 + 0.5) * scale, (y as f32 + 0.5) * scale];
            let t = distance_to_path(path, centre) / radius;
            (t < 1.0).then_some(((x, y), 1.0 - t * t * (3.0 - 2.0 * t)))
        })
}

/// How far `at` lies from `path` on the ground plane.
fn distance_to_path(path: &[[f32; 3]], at: [f32; 2]) -> f32 {
    let to_point = |p: [f32; 3]| libm::hypotf(at[0] - p[0], at[1] - p[1]);
    let mut nearest = to_point(path[0]);
    for pair in path.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let along = [b[0] - a[0], b[1] - a[1]];
        let length = along[0] * along[0] + along[1] * along[1];
        if length == 0.0 {
            continue;
        }
        let t = (((at[0] - a[0]) * along[0] + (at[1] - a[1]) * along[1]) / length).clamp(0.0, 1.0);
        nearest = nearest.min(libm::hypotf(
            at[0] - a[0] - along[0] * t,
            at[1] - a[1] - along[1] * t,
        ));
    }
    nearest
}

/// Points along `path` in 3D no more than `step` apart: its first point, then as many evenly
/// spaced points along each segment as that takes, its last point included.
fn along(path: &[[f32; 3]], step: f32) -> Vec<[f32; 3]> {
    let mut points = vec![path[0]];
    for pair in path.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let length = (0..3).map(|i| (b[i] - a[i]).powi(2)).sum::<f32>().sqrt();
        let steps = (length / step).ceil().max(1.0) as u32;
        for i in 1..=steps {
            let t = i as f32 / steps as f32;
            points.push(std::array::from_fn(|axis| {
                a[axis] + (b[axis] - a[axis]) * t
            }));
        }
    }
    points
}

/// The chunks, of `size` columns, holding any column within `radius` of `path`'s bounding box on
/// the ground plane.
fn chunks_near(size: [u32; 2], path: &[[f32; 3]], radius: f32) -> Vec<ChunkCoord> {
    let span = |axis: usize| {
        let low = path.iter().map(|p| p[axis]).fold(f32::INFINITY, f32::min) - radius;
        let high = path
            .iter()
            .map(|p| p[axis])
            .fold(f32::NEG_INFINITY, f32::max)
            + radius;
        let side = size[axis] as f32;
        (low / side).floor() as i32..=(high / side).floor() as i32
    };
    let (xs, ys) = (span(0), span(1));
    ys.flat_map(|y| xs.clone().map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn points_along_a_path_are_no_further_apart_than_the_step_and_reach_its_end() {
        let path = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 3.0, 4.0]];

        let points = along(&path, 1.5);

        assert!(points.windows(2).all(|pair| {
            (0..3)
                .map(|i| (pair[1][i] - pair[0][i]).powi(2))
                .sum::<f32>()
                .sqrt()
                <= 1.5 + 1e-5
        }));
        assert_eq!(points.last(), Some(&[10.0, 3.0, 4.0]));
    }
}
