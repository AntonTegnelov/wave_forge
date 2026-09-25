//! Which chunks near the player to give something this frame and which to take it from: the
//! bookkeeping shared by what the world node builds per chunk within a radius (sound, occluders).

use std::collections::{BTreeSet, HashSet};
use wave_forge::ChunkCoord;

/// How many chunks apart two chunks are along the axis where they are furthest apart: the
/// distance the node's radii are measured in.
pub(crate) fn chunk_distance(a: ChunkCoord, b: ChunkCoord) -> i32 {
    (a.x - b.x)
        .abs()
        .max((a.y - b.y).abs())
        .max((a.z - b.z).abs())
}

/// What to do this frame for chunks kept within a radius.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct RadiusPlan {
    /// Built chunks that left the radius or are no longer generated.
    pub(crate) gone: Vec<ChunkCoord>,
    /// Chunks to build, nearest first, at most the budget: generated chunks in the radius not
    /// built yet, or whose tiles changed since they were.
    pub(crate) build: Vec<ChunkCoord>,
}

/// Plans a frame for the chunks within `radius` of `focus` (none when it is below zero):
/// `generated` lists the generated chunks, `built` those that have what is being kept, `due`
/// those whose tiles changed since, which `updated` adds to and the plan's build removes from.
pub(crate) fn plan(
    radius: i32,
    focus: ChunkCoord,
    generated: &HashSet<ChunkCoord>,
    built: &HashSet<ChunkCoord>,
    due: &mut BTreeSet<ChunkCoord>,
    updated: &[ChunkCoord],
    budget: usize,
) -> RadiusPlan {
    let within = |chunk: ChunkCoord| radius >= 0 && chunk_distance(chunk, focus) <= radius;
    let mut gone: Vec<ChunkCoord> = built
        .iter()
        .copied()
        .filter(|&chunk| !within(chunk) || !generated.contains(&chunk))
        .collect();
    gone.sort_unstable();
    due.extend(updated.iter().copied().filter(|&chunk| within(chunk)));
    due.retain(|&chunk| within(chunk));
    let mut build: Vec<ChunkCoord> = generated
        .iter()
        .copied()
        .filter(|&chunk| within(chunk))
        .filter(|chunk| !built.contains(chunk) || due.contains(chunk))
        .collect();
    build.sort_by_key(|&chunk| (chunk_distance(chunk, focus), chunk));
    build.truncate(budget);
    for chunk in &build {
        due.remove(chunk);
    }
    RadiusPlan { gone, build }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(chunks: &[(i32, i32)]) -> HashSet<ChunkCoord> {
        chunks
            .iter()
            .map(|&(x, y)| ChunkCoord::new(x, y, 0))
            .collect()
    }

    #[test]
    fn the_nearest_unbuilt_chunks_are_built_within_the_budget() {
        let generated = set(&[(0, 0), (1, 0), (2, 0), (0, 1), (5, 5)]);
        let mut due = BTreeSet::new();

        let plan = plan(
            1,
            ChunkCoord::new(0, 0, 0),
            &generated,
            &set(&[]),
            &mut due,
            &[],
            2,
        );

        assert_eq!(
            plan.build,
            [ChunkCoord::new(0, 0, 0), ChunkCoord::new(0, 1, 0)]
        );
        assert!(plan.gone.is_empty());
    }

    #[test]
    fn a_built_chunk_is_built_again_once_its_tiles_change() {
        let generated = set(&[(0, 0), (1, 0)]);
        let built = set(&[(0, 0), (1, 0)]);
        let mut due = BTreeSet::new();
        let focus = ChunkCoord::new(0, 0, 0);

        let unchanged = plan(1, focus, &generated, &built, &mut due, &[], 3);
        let changed = plan(
            1,
            focus,
            &generated,
            &built,
            &mut due,
            &[ChunkCoord::new(1, 0, 0)],
            3,
        );

        assert!(unchanged.build.is_empty());
        assert_eq!(changed.build, [ChunkCoord::new(1, 0, 0)]);
        assert!(due.is_empty());
    }

    #[test]
    fn a_chunk_out_of_range_or_no_longer_generated_is_gone() {
        let generated = set(&[(0, 0), (3, 0)]);
        let built = set(&[(0, 0), (3, 0), (1, 0)]);
        let mut due = BTreeSet::new();

        let plan = plan(
            1,
            ChunkCoord::new(0, 0, 0),
            &generated,
            &built,
            &mut due,
            &[],
            3,
        );

        assert_eq!(
            plan.gone,
            [ChunkCoord::new(1, 0, 0), ChunkCoord::new(3, 0, 0)]
        );
    }

    #[test]
    fn below_zero_nothing_is_kept() {
        let generated = set(&[(0, 0)]);
        let built = set(&[(0, 0)]);
        let mut due = BTreeSet::new();

        let plan = plan(
            -1,
            ChunkCoord::new(0, 0, 0),
            &generated,
            &built,
            &mut due,
            &[],
            3,
        );

        assert_eq!(plan.gone, [ChunkCoord::new(0, 0, 0)]);
        assert!(plan.build.is_empty());
    }
}
