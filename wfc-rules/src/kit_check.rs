//! A model drawn for a module, checked against what its faces promise its neighbours: the
//! conventions authored kits are made to (the continent's town pieces, say), in a model's own
//! frame, a cell from -0.5 to 0.5 on every axis with Y up and its sides `+X`, `-X`, `+Z` and `-Z`
//! in the module file's order.
//!
//! The checks are the ones a bad model breaks a whole town with:
//! - it stays inside its cell, but for a stair, whose flight rises into the empty cell above;
//! - every walkable side (ground, a road, a walkway deck, a balcony) has a surface at
//!   [`WALK_HEIGHT`] just inside it, where its neighbour's surface meets it;
//! - nothing touches a side that faces air, but for a roof's foot, which covers the cell's
//!   footprint at the bottom of a roof standing on a building.

pub use crate::formats::module_format::{ModuleFaces, module_faces};
use std::fmt;

/// The height walkers walk at, at every walkable side.
pub const WALK_HEIGHT: f32 = -0.25;
/// How far a walking surface may lie from [`WALK_HEIGHT`].
const WALK_TOLERANCE: f32 = 0.03;
/// How far inside a walkable side its surface is looked for.
const WALK_INSET: f32 = 0.05;
/// How far a vertex may lie past its cell's bounds and still count as inside.
const BOUNDS_TOLERANCE: f32 = 0.01;
/// How high a stair's flight may rise: through the top of its cell to the landing in the cell
/// above.
const STAIR_TOP: f32 = 1.0;
/// How far inside a side that faces air a model keeps.
const AIR_CLEARANCE: f32 = 0.02;
/// Up to how high a roof standing on a building may reach its sides: its foot.
const ROOF_FOOT: f32 = -0.44;

/// Side faces a walker crosses, whose surface lies at [`WALK_HEIGHT`].
const WALKABLE: [&str; 6] = [
    "ground",
    "ground_path",
    "blocked",
    "road",
    "walkway",
    "balcony",
];

/// The sides of a model's cell, in the module file's order.
const SIDES: [(&str, usize, f32); 4] = [
    ("+X", 0, 1.0),
    ("-X", 0, -1.0),
    ("+Z", 2, 1.0),
    ("-Z", 2, -1.0),
];

/// Why a model does not meet its module's faces.
#[derive(Clone, Debug, PartialEq)]
pub enum ModelProblem {
    /// The model has no triangles.
    Empty,
    /// A vertex lies outside the cell (or above a stair's landing).
    OutsideCell { vertex: [f32; 3] },
    /// A walkable side has no surface at [`WALK_HEIGHT`] just inside it; `heights` are the
    /// surfaces found there, highest first.
    NoWalkingSurface { side: String, heights: Vec<f32> },
    /// A vertex touches a side that faces air.
    TouchesAir { side: String, vertex: [f32; 3] },
}

impl fmt::Display for ModelProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let point = |[x, y, z]: [f32; 3]| format!("({x:.3}, {y:.3}, {z:.3})");
        match self {
            Self::Empty => write!(f, "the model has no triangles"),
            Self::OutsideCell { vertex } => write!(
                f,
                "a vertex at {} lies outside the cell from -0.5 to 0.5",
                point(*vertex)
            ),
            Self::NoWalkingSurface { side, heights } => write!(
                f,
                "its {side} side is walkable, but just inside it no surface lies at Y = {WALK_HEIGHT} (surfaces there: {heights:?})"
            ),
            Self::TouchesAir { side, vertex } => write!(
                f,
                "its {side} side faces air, but a vertex at {} touches it",
                point(*vertex)
            ),
        }
    }
}

/// What `triangles`, three vertices each in the model's frame, break of `faces`' contracts: empty
/// for a model that meets them. Each side is reported once, with the first vertex that breaks it.
#[must_use]
pub fn check_model(faces: &ModuleFaces, triangles: &[[f32; 3]]) -> Vec<ModelProblem> {
    if triangles.is_empty() {
        return vec![ModelProblem::Empty];
    }
    let mut problems = Vec::new();
    let top = if faces.up == "stair" { STAIR_TOP } else { 0.5 };
    let inside = |value: f32, high: f32| {
        (-0.5 - BOUNDS_TOLERANCE..=high + BOUNDS_TOLERANCE).contains(&value)
    };
    if let Some(&vertex) = triangles
        .iter()
        .find(|v| !(inside(v[0], 0.5) && inside(v[1], top) && inside(v[2], 0.5)))
    {
        problems.push(ModelProblem::OutsideCell { vertex });
    }
    for ((side, axis, sign), face) in SIDES.iter().zip(&faces.sides) {
        if WALKABLE.contains(&face.as_str()) {
            let mut at = [0.0, 0.0];
            at[usize::from(*axis == 2)] = sign * (0.5 - WALK_INSET);
            let mut heights = heights_at(triangles, at);
            if !heights
                .iter()
                .any(|height| (height - WALK_HEIGHT).abs() <= WALK_TOLERANCE)
            {
                heights.sort_by(|a, b| b.total_cmp(a));
                problems.push(ModelProblem::NoWalkingSurface {
                    side: (*side).to_owned(),
                    heights,
                });
            }
        }
        if face == "air" {
            let roof = faces.down == "solid";
            if let Some(&vertex) = triangles
                .iter()
                .find(|v| v[*axis] * sign > 0.5 - AIR_CLEARANCE && !(roof && v[1] <= ROOF_FOOT))
            {
                problems.push(ModelProblem::TouchesAir {
                    side: (*side).to_owned(),
                    vertex,
                });
            }
        }
    }
    problems
}

/// The heights at which the vertical line through `[x, z]` meets `triangles`.
fn heights_at(triangles: &[[f32; 3]], [x, z]: [f32; 2]) -> Vec<f32> {
    triangles
        .as_chunks::<3>()
        .0
        .iter()
        .filter_map(|&[a, b, c]| {
            // Barycentric coordinates of the point in the triangle seen from above.
            let det = (b[2] - c[2]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[2] - c[2]);
            if det.abs() < 1e-9 {
                return None;
            }
            let u = ((b[2] - c[2]) * (x - c[0]) + (c[0] - b[0]) * (z - c[2])) / det;
            let v = ((c[2] - a[2]) * (x - c[0]) + (a[0] - c[0]) * (z - c[2])) / det;
            let w = 1.0 - u - v;
            (u >= 0.0 && v >= 0.0 && w >= 0.0).then(|| u * a[1] + v * b[1] + w * c[1])
        })
        .collect()
}
