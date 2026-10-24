//! A world bound: a finite world generates no target chunk wholly outside it, computes its region
//! jobs before play when asked for its whole bound, and keeps them, computing a kept region again
//! when an edit changes what it read.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use wave_forge::stages::regions::{Attempt, Curve, CurveId, RegionInput, RegionJob};
use wave_forge::stages::{Edit, Edits, Pack, PackError, Runtime, StageError};
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

/// Whether any of `chunk`, from its corner up to but not including the next chunk's, lies within 40
/// cells of the origin.
fn meets_disk(chunk: ChunkCoord) -> bool {
    // Along one axis: the nearest coordinate to 0 the chunk holds, or the edge it comes as near to
    // as one likes without holding, and whether it holds it.
    let near = |at: i32| {
        let (low, high) = (at as f32 * 8.0, at as f32 * 8.0 + 8.0);
        if high <= 0.0 {
            (high, false)
        } else {
            (low.max(0.0), true)
        }
    };
    let ((x, held_x), (y, held_y)) = (near(chunk.x), near(chunk.y));
    let distance = libm::hypotf(x, y);
    if held_x && held_y {
        distance <= 40.0
    } else {
        distance < 40.0
    }
}

#[test]
fn no_target_chunk_wholly_outside_the_bound_is_generated() {
    let mut runtime = Runtime::new(Arc::new(Pack::parse(DISK).expect("a valid pack")), 3, SIZE);
    // East and west of the disk: chunk 5 holds x = 40 on the circle, and chunk -6 ends there
    // without holding it.
    let foci = [
        FocusPoint::new(ChunkCoord::new(5, 0, 0), 3),
        FocusPoint::new(ChunkCoord::new(-5, 0, 0), 3),
    ];

    runtime.request(&foci, &["smooth"]).expect("a stage");
    runtime.run_until_idle().expect("the stages run");

    let asked: Vec<ChunkCoord> = (-3..=3)
        .flat_map(|y| {
            (-8..=-2)
                .chain(2..=8)
                .map(move |x| ChunkCoord::new(x, y, 0))
        })
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

    // Chunks -5 to 5 meet the disk along each axis, so regions -2 to 1 of four chunks, less the
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

/// A curve across its region whose value is the height at the region's lowest corner.
struct Corner;

impl RegionJob for Corner {
    fn run(&self, input: &RegionInput<'_>) -> Result<Attempt, StageError> {
        let ([x0, y0], [x1, y1]) = input.columns();
        let height = input.field("height", x0, y0)?;
        Ok(Attempt::Accepted(vec![Curve {
            id: CurveId::Region {
                region: input.region(),
                index: 0,
            },
            points: vec![[x0 as f32, y0 as f32], [x1 as f32 + 1.0, y1 as f32 + 1.0]],
            values: vec![height, height],
            heights: Vec::new(),
        }]))
    }
}

#[test]
fn a_kept_region_is_computed_again_when_an_edit_changes_what_it_read() {
    let pack = Pack::parse(
        r#"(version: 1, bound: Some(Rect(min: (-40.0, -40.0), max: (39.0, 39.0))), stages: [
            (name: "height", kind: Field(Constant(1.0))),
            (name: "corners", kind: Region(job: "corner", region: 4, inputs: ["height"])),
        ])"#,
    )
    .expect("a valid pack");
    let mut runtime = Runtime::new(Arc::new(pack), 3, SIZE).with_region_job("corner", Corner);
    let ask = |runtime: &mut Runtime, at: ChunkCoord| {
        runtime
            .request(&[FocusPoint::new(at, 0)], &["corners"])
            .expect("a stage");
        runtime.run_until_idle().expect("the stages run");
    };
    let near = ChunkCoord::new(1, 1, 0);
    ask(&mut runtime, near);
    ask(&mut runtime, ChunkCoord::new(-4, -4, 0));
    let mut edits = Edits::default();
    edits.push(Edit::Raise {
        stage: "height".to_owned(),
        column: (0, 0),
        by: 5.0,
    });

    runtime.set_edits(&edits).expect("edits");
    ask(&mut runtime, near);

    let curves = runtime.curves("corners", near).expect("generated");
    assert_eq!(curves[0].values, vec![6.0, 6.0]);
}
