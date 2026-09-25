//! Levels of detail as Bevy draws them: one mesh per level, each on an entity of its own with the
//! range of distances from the camera it is drawn at, for the ground ([`crate::stages`]) and far
//! proxies ([`crate::WaveForgeWorld::proxy_levels`]).

use bevy_camera::visibility::VisibilityRange;
use bevy_math::Vec3;
use std::ops::Range;

/// How far a level may stray on screen before a finer level is drawn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelDetail {
    /// The most pixels a level's error may span.
    pub pixels: f32,
    /// The viewport's height in pixels.
    pub height: f32,
    /// The camera's vertical field of view, in radians.
    pub fov: f32,
}

impl LevelDetail {
    /// How many units from the camera one unit of error spans `pixels`.
    pub(crate) fn per_error(self) -> f32 {
        self.height / (2.0 * (self.fov / 2.0).tan() * self.pixels)
    }
}

/// Half the diagonal of the bounds of `positions`: how much nearer than a mesh's bounds' centre
/// its nearest point can be.
pub(crate) fn bounds_radius(positions: &[[f32; 3]]) -> f32 {
    let (low, high) = positions.iter().fold(
        (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
        |(low, high), &position| (low.min(position.into()), high.max(position.into())),
    );
    (high - low).length() / 2.0
}

/// An abrupt range measured from the centre of an entity's bounds.
pub(crate) fn visibility(range: &Range<f32>) -> VisibilityRange {
    VisibilityRange {
        use_aabb: true,
        ..VisibilityRange::abrupt(range.start, range.end)
    }
}

/// The distances each level is drawn at, finest first, from their `errors`: the finest from
/// `start`, a coarser one from where its error is `per_error` times closer than the camera, plus
/// the `radius` of the mesh's bounds, and never nearer than `start`, to where the next level
/// starts. A level's error counts as at least every finer level's, since a coarser level must not
/// be drawn nearer than a finer one.
pub(crate) fn level_ranges(
    errors: &[f32],
    radius: f32,
    per_error: f32,
    start: f32,
) -> Vec<Range<f32>> {
    let starts: Vec<f32> = errors
        .iter()
        .enumerate()
        .scan(0.0_f32, |key, (index, &error)| {
            *key = key.max(error);
            Some(if index == 0 {
                start
            } else {
                (*key * per_error + radius).max(start)
            })
        })
        .collect();
    (0..starts.len())
        .map(|index| starts[index]..starts.get(index + 1).copied().unwrap_or(f32::INFINITY))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_coarser_level_starts_where_its_error_is_small_enough_from_every_point_of_the_mesh() {
        let ranges = level_ranges(&[0.0, 0.5, 2.0], 10.0, 100.0, 0.0);

        assert_eq!(ranges, [0.0..60.0, 60.0..210.0, 210.0..f32::INFINITY]);
    }

    #[test]
    fn a_level_straying_less_than_a_finer_one_is_never_drawn_nearer() {
        let ranges = level_ranges(&[0.0, 2.0, 1.0, 3.0], 0.0, 1.0, 0.0);

        assert_eq!(ranges, [0.0..2.0, 2.0..2.0, 2.0..3.0, 3.0..f32::INFINITY]);
    }

    #[test]
    fn no_level_is_drawn_nearer_than_the_start() {
        let ranges = level_ranges(&[0.0, 0.5, 2.0], 10.0, 100.0, 100.0);

        assert_eq!(ranges, [100.0..100.0, 100.0..210.0, 210.0..f32::INFINITY]);
    }
}
