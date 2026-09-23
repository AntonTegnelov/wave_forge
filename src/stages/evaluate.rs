//! Evaluating expressions: the nodes that combine values, whatever the leaves read.
//!
//! A stage's expression is evaluated at one of its columns, where its leaves read noise, inputs and
//! the column's position; a generated table's at one of its rows, where they read the parent row
//! and the row's hash stream. The caller says what the leaves read through [`Leaves`]; arithmetic,
//! shaping and conditions are evaluated here, once for every place an expression is evaluated.

use super::pack::{Condition, Expr};
use super::runtime::StageError;

/// What an expression's leaves read where it is evaluated.
pub(crate) trait Leaves {
    /// The value of an expression [`evaluate`] does not combine itself: a leaf, or a node like
    /// `Match` whose meaning depends on the place.
    fn leaf(&self, expr: &Expr) -> Result<f32, StageError>;
}

/// The value of `expr`, its leaves read through `leaves`.
pub(crate) fn evaluate(expr: &Expr, leaves: &impl Leaves) -> Result<f32, StageError> {
    let value = |expr: &Expr| evaluate(expr, leaves);
    Ok(match expr {
        Expr::Constant(value) => *value,
        Expr::Add(a, b) => value(a)? + value(b)?,
        Expr::Sub(a, b) => value(a)? - value(b)?,
        Expr::Mul(a, b) => value(a)? * value(b)?,
        Expr::Min(a, b) => value(a)?.min(value(b)?),
        Expr::Max(a, b) => value(a)?.max(value(b)?),
        Expr::Abs(a) => value(a)?.abs(),
        Expr::Floor(a) => value(a)?.floor(),
        Expr::Sin(a) => value(a)?.sin(),
        Expr::Clamp(a, low, high) => value(a)?.clamp(*low, *high),
        Expr::Smoothstep(low, high, a) => {
            let t = ((value(a)? - low) / (high - low)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        }
        Expr::Remap(a, (from_low, from_high), (to_low, to_high)) => {
            to_low + (value(a)? - from_low) / (from_high - from_low) * (to_high - to_low)
        }
        Expr::Curve(a, points) => curve(points, value(a)?),
        Expr::Select {
            when,
            then,
            otherwise,
        } => {
            let chosen = if holds(when, leaves)? {
                then
            } else {
                otherwise
            };
            value(chosen)?
        }
        Expr::Noise { .. }
        | Expr::Input(_)
        | Expr::X
        | Expr::Y
        | Expr::Distance(_)
        | Expr::Angle(_)
        | Expr::Is(..)
        | Expr::Match { .. }
        | Expr::Row(..)
        | Expr::Parent(_)
        | Expr::Random(..)
        | Expr::Index
        | Expr::Count
        | Expr::Share(_) => leaves.leaf(expr)?,
    })
}

/// Whether `condition` holds, its expressions' leaves read through `leaves`.
pub(crate) fn holds(condition: &Condition, leaves: &impl Leaves) -> Result<bool, StageError> {
    let value = |expr: &Expr| evaluate(expr, leaves);
    Ok(match condition {
        Condition::Less(a, b) => value(a)? < value(b)?,
        Condition::Greater(a, b) => value(a)? > value(b)?,
        Condition::Between(a, low, high) => (*low..=*high).contains(&value(a)?),
    })
}

/// A piecewise-linear curve at `x`: level beyond its first and last points, linear between.
/// The points are in increasing x and at least two, which loading checks.
fn curve(points: &[(f32, f32)], x: f32) -> f32 {
    let (first, last) = (points[0], points[points.len() - 1]);
    if x <= first.0 {
        return first.1;
    }
    if x >= last.0 {
        return last.1;
    }
    let after = points.partition_point(|point| point.0 <= x);
    let ((x0, y0), (x1, y1)) = (points[after - 1], points[after]);
    y0 + (x - x0) / (x1 - x0) * (y1 - y0)
}
