//! A world bound: a finite world generates no target chunk wholly outside it, computes its region
//! jobs before play when asked for its whole bound, and keeps them.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use wave_forge::stages::regions::{Attempt, RegionInput, RegionJob};
use wave_forge::stages::{Pack, PackError, Runtime, StageError};
use wave_forge::{ChunkCoord, FocusPoint};

const SIZE: [u32; 2] = [8, 8];

const DISK: &str = r#"(
    version: 1,
    bound: Some(Disk(centre: (0.0, 0.0), radius: 40.0)),
    stages: [
        (name: "height", kind: Field(Noise(frequency: 0.1, octaves: 1))),
        (name: "smooth", kind: Blur(input: "height", radius: 4)),
    ],
)"#;

/// Whether any cell of `chunk` lies within 40 cells of the origin.
fn meets_disk(chunk: ChunkCoord) -> bool {
    let near = |at: i32| {
        let (low, high) = (at as f32 * 8.0, at as f32 * 8.0 + 8.0);
        0.0_f32.clamp(low, high)
    };
    near(chunk.x).hypot(near(chunk.y)) <= 40.0
}

#[test]
fn no_target_chunk_wholly_outside_the_bound_is_generated() {
    let mut runtime = Runtime::new(Arc::new(Pack::parse(DISK).expect("a valid pack")), 3, SIZE);
    let focus = FocusPoint::new(ChunkCoord::new(5, 0, 0), 3);

    runtime.request(&[focus], &["smooth"]).expect("a stage");
    runtime.run_until_idle().expect("the stages run");

    let asked: Vec<ChunkCoord> = (-3..=3)
        .flat_map(|y| (2..=8).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect();
    assert!(asked.iter().any(|&chunk| !meets_disk(chunk)));
    for chunk in asked {
        assert_eq!(
            runtime.field("smooth", chunk).is_some(),
            meets_disk(chunk),
            "{chunk:?}"
        );
    }
    // A target inside the bound still reads its inputs beyond it.
    assert!(runtime.field("height", ChunkCoord::new(6, 0, 0)).is_some());
}

#[test]
fn a_rectangle_bounds_a_world_too() {
    let pack = Pack::parse(
        r#"(version: 1, bound: Some(Rect(min: (0.0, 0.0), max: (24.0, 16.0))),
            stages: [(name: "height", kind: Field(Constant(1.0)))])"#,
    )
    .expect("a valid pack");
    let mut runtime = Runtime::new(Arc::new(pack), 3, SIZE);

    runtime
        .request(&[FocusPoint::new(ChunkCoord::new(1, 1, 0), 3)], &["height"])
        .expect("a stage");
    runtime.run_until_idle().expect("the stages run");

    let held: Vec<(i32, i32)> = (-2..=4)
        .flat_map(|y| (-2..=4).map(move |x| (x, y)))
        .filter(|&(x, y)| runtime.field("height", ChunkCoord::new(x, y, 0)).is_some())
        .collect();
    // Chunks 0 to 3 along x and 0 to 2 along y meet the rectangle, which includes its far edges;
    // chunk -1 ends where the rectangle begins, so it does not.
    let expected: Vec<(i32, i32)> = (0..=2).flat_map(|y| (0..=3).map(move |x| (x, y))).collect();
    assert_eq!(held, expected);
}

/// A region job that counts its runs and makes no curves.
struct Counted(Arc<AtomicUsize>);

impl RegionJob for Counted {
    fn run(&self, _input: &RegionInput<'_>) -> Result<Attempt, StageError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(Attempt::Accepted(Vec::new()))
    }
}

fn regions(bound: &str) -> (Runtime, Arc<AtomicUsize>) {
    let pack = Pack::parse(&format!(
        r#"(version: 1, bound: {bound}, stages: [
            (name: "height", kind: Field(Constant(1.0))),
            (name: "rivers", kind: Region(job: "count", region: 4, inputs: ["height"])),
        ])"#
    ))
    .expect("a valid pack");
    let runs = Arc::new(AtomicUsize::new(0));
    let runtime =
        Runtime::new(Arc::new(pack), 3, SIZE).with_region_job("count", Counted(Arc::clone(&runs)));
    (runtime, runs)
}

#[test]
fn a_finite_world_computes_its_region_jobs_before_play_and_keeps_them() {
    let (mut runtime, runs) = regions("Some(Disk(centre: (0.0, 0.0), radius: 40.0))");

    runtime.request_bound(&["rivers"]).expect("a bound");
    runtime.run_until_idle().expect("the stages run");
    let before_play = runs.load(Ordering::SeqCst);
    for centre in [
        ChunkCoord::new(-4, -4, 0),
        ChunkCoord::new(3, 3, 0),
        ChunkCoord::new(-4, -4, 0),
    ] {
        runtime
            .request(&[FocusPoint::new(centre, 1)], &["rivers"])
            .expect("a stage");
        runtime.run_until_idle().expect("the stages run");
    }

    // Chunks -5 to 4 meet the disk along each axis, so regions -2 to 1 of four chunks, less the
    // four corners, which no chunk of the disk reaches.
    assert_eq!(before_play, 12);
    assert_eq!(
        runs.load(Ordering::SeqCst),
        before_play,
        "no region is computed twice"
    );
}

#[test]
fn an_unbounded_world_computes_a_region_again_after_leaving_it() {
    let (mut runtime, runs) = regions("None");

    for centre in [
        ChunkCoord::new(-4, -4, 0),
        ChunkCoord::new(12, 12, 0),
        ChunkCoord::new(-4, -4, 0),
    ] {
        runtime
            .request(&[FocusPoint::new(centre, 0)], &["rivers"])
            .expect("a stage");
        runtime.run_until_idle().expect("the stages run");
    }

    assert_eq!(runs.load(Ordering::SeqCst), 3);
    assert_eq!(
        runtime.request_bound(&["rivers"]),
        Err(StageError::Unbounded)
    );
}

#[test]
fn a_bound_that_holds_nothing_is_refused() {
    for bound in [
        "Disk(centre: (0.0, 0.0), radius: 0.0)",
        "Rect(min: (5.0, 0.0), max: (1.0, 4.0))",
        "Disk(centre: (0.0, 0.0), radius: inf)",
    ] {
        let result = Pack::parse(&format!(
            r#"(version: 1, bound: Some({bound}), stages: [])"#
        ));
        assert!(
            matches!(result, Err(PackError::Bound(_) | PackError::Syntax(_))),
            "{bound}: {result:?}"
        );
    }
}
