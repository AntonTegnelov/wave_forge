//! How many of a column's grass blades a chunk needs, where the grass thins with distance
//! (docs/reference/godot.md, "Grass"; docs/reference/bevy.md, "Grass").
//!
//! The reference grass shaders show a column's blades in order of rank, as many as the thinning
//! keeps at a blade's distance from the camera, so a chunk far away needs only the first few. An
//! engine draws such a chunk from a smaller set of blades, a column's first, and the grass looks
//! the same while costing less.

use crate::ChunkCoord;

/// How grass blades thin with distance from the camera: from `from` to `to` world units the share
/// of a column's blades that show falls from all of them to `keep`, along a smoothstep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GrassThinning {
    pub from: f32,
    pub to: f32,
    pub keep: f32,
}

impl GrassThinning {
    /// The reference grass shaders' thinning: from 4 to 14 world units from the camera, down to
    /// 0.12 of the blades.
    pub const REFERENCE: Self = Self {
        from: 4.0,
        to: 14.0,
        keep: 0.12,
    };

    /// The share of a column's blades that show at `distance` world units from the camera.
    #[must_use]
    pub fn keep_at(&self, distance: f32) -> f32 {
        let t = ((distance - self.from) / (self.to - self.from).max(1e-6)).clamp(0.0, 1.0);
        1.0 + (self.keep - 1.0) * t * t * (3.0 - 2.0 * t)
    }
}

/// How many of a column's `per_column` blades the reference grass shaders show anywhere on `chunk`,
/// chunks `span` world units wide along x and z with their corners at multiples of it, while the
/// camera stands within `room` world units of `camera` along the ground: as many as `thinning`
/// keeps at the chunk's nearest point to `camera`, less `room`. An engine chooses again once the
/// camera has moved farther than `room`.
#[must_use]
pub fn grass_blades_needed(
    chunk: ChunkCoord,
    camera: [f32; 2],
    room: f32,
    span: [f32; 2],
    thinning: GrassThinning,
    per_column: u32,
) -> u32 {
    let low = [chunk.x as f32 * span[0], chunk.y as f32 * span[1]];
    let nearest = [
        camera[0].clamp(low[0], low[0] + span[0]),
        camera[1].clamp(low[1], low[1] + span[1]),
    ];
    let away = (libm::hypotf(nearest[0] - camera[0], nearest[1] - camera[1]) - room).max(0.0);
    ((thinning.keep_at(away) * per_column as f32).ceil() as u32).clamp(1, per_column)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPAN: [f32; 2] = [8.0, 8.0];
    const CAMERA: [f32; 2] = [4.0, 4.0];
    const ROOM: f32 = 2.0;
    const THINNING: GrassThinning = GrassThinning {
        from: 8.0,
        to: 30.0,
        keep: 0.3,
    };

    fn needed(x: i32) -> u32 {
        grass_blades_needed(ChunkCoord::new(x, 0, 0), CAMERA, ROOM, SPAN, THINNING, 8)
    }

    #[test]
    fn near_chunks_need_every_blade() {
        assert_eq!(needed(0), 8);
        assert_eq!(needed(1), 8);
        assert_eq!(needed(-2), 8);
    }

    #[test]
    fn a_chunk_past_the_thinning_needs_the_share_it_keeps() {
        // Five chunks out the nearest point is 36 units away, 34 less the camera's room: the
        // thinning keeps 0.3 of a column's 8, 2.4, so 3 blades.
        assert_eq!(needed(5), 3);
        assert_eq!(needed(20), 3);
    }

    #[test]
    fn a_chunk_within_the_thinning_needs_what_it_keeps_at_its_nearest() {
        // Three out it is 20 less 2 away, 18, where the thinning keeps about 0.7 of 8, 5.6, so 6
        // blades.
        assert_eq!(needed(3), 6);
    }

    #[test]
    fn a_camera_with_more_room_needs_more_blades_on_the_same_chunk() {
        let roomy = grass_blades_needed(ChunkCoord::new(3, 0, 0), CAMERA, 8.0, SPAN, THINNING, 8);

        assert!(roomy > needed(3), "{roomy}");
    }

    #[test]
    fn the_thinning_keeps_every_blade_before_it_and_its_share_after() {
        assert_eq!(THINNING.keep_at(0.0), 1.0);
        assert_eq!(THINNING.keep_at(8.0), 1.0);
        assert!((THINNING.keep_at(30.0) - 0.3).abs() < 1e-6);
        assert!((THINNING.keep_at(100.0) - 0.3).abs() < 1e-6);
    }
}
