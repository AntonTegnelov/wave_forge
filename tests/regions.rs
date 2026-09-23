//! Region jobs: one computation per region, agreeing with its neighbours through edge hashes.
//!
//! A river job runs across a row of regions: each region's river enters at a point hashed from its
//! west edge and leaves at one hashed from its east edge, bending towards the lowest ground on the
//! way, so neighbouring regions meet at the same point without reading each other. The tests check
//! the curves meet at every border and come out the same in any generation order, that a job is
//! retried with new hashes until it is accepted and refused with its log when its budget runs out,
//! that a finite world is computed once, and that a job cannot read beyond its halo.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use wave_forge::stages::regions::{Attempt, Curve, CurveId, Edge, RegionInput, RegionJob};
use wave_forge::stages::{
    Expr, Pack, PackError, PackFile, Runtime, StageDef, StageError, StageKind,
};
use wave_forge::{ChunkCoord, FocusPoint};

const SIZE: [u32; 2] = [8, 8];
const REGION: u32 = 2;

/// A river across the region from west to east: in at the west edge's hashed row, out at the east
/// edge's, and through the lowest column of the region's middle column on the way.
struct Rivers;

impl RegionJob for Rivers {
    fn run(&self, input: &RegionInput<'_>) -> Result<Attempt, StageError> {
        let ([x0, y0], [x1, y1]) = input.columns();
        let rows = (y1 - y0 + 1) as u32;
        let row = |hash: u32| y0 as f32 + (hash % rows) as f32 + 0.5;
        let enter = [x0 as f32, row(input.edge_hash(Edge::West, 0))];
        let leave = [x1 as f32 + 1.0, row(input.edge_hash(Edge::East, 0))];
        let middle = (x0 + x1) / 2;
        let mut lowest = (f32::INFINITY, y0);
        for y in y0..=y1 {
            let height = input.field("height", middle, y)?;
            if height < lowest.0 {
                lowest = (height, y);
            }
        }
        Ok(Attempt::Accepted(vec![Curve {
            id: CurveId {
                region: input.region(),
                index: 0,
            },
            points: vec![enter, [middle as f32 + 0.5, lowest.1 as f32 + 0.5], leave],
            values: vec![1.0, 2.0, 3.0],
        }]))
    }
}

fn stage(name: &str, kind: StageKind) -> StageDef {
    StageDef {
        name: name.to_owned(),
        scale: 1,
        kind,
    }
}

fn region(job: &str, region: u32, halo: u32, budget: u32) -> StageDef {
    stage(
        "rivers",
        StageKind::Region {
            job: job.to_owned(),
            region,
            halo,
            inputs: vec!["height".to_owned()],
            budget,
        },
    )
}

fn height() -> StageDef {
    stage(
        "height",
        StageKind::Field(Expr::Noise {
            frequency: 0.1,
            octaves: 2,
            name: None,
        }),
    )
}

fn pack(stages: Vec<StageDef>) -> Result<Pack, PackError> {
    Pack::from_file(PackFile { version: 1, stages })
}

fn runtime(stages: Vec<StageDef>) -> Runtime {
    Runtime::new(Arc::new(pack(stages).expect("a valid pack")), 5, SIZE)
}

/// The rivers of the chunks `requests` asks for, one request at a time, by the region that made
/// each, in id order.
fn rivers(requests: &[Vec<ChunkCoord>]) -> Vec<Curve> {
    let mut runtime =
        runtime(vec![height(), region("rivers", REGION, 1, 1)]).with_region_job("rivers", Rivers);
    let mut curves: Vec<Curve> = Vec::new();
    for request in requests {
        let focus: Vec<FocusPoint> = request.iter().map(|&c| FocusPoint::new(c, 0)).collect();
        runtime.request(&focus, &["rivers"]).expect("a stage");
        runtime.run_until_idle().expect("the stages run");
        for &chunk in request {
            for curve in runtime.curves("rivers", chunk).expect("generated") {
                if !curves.iter().any(|known| known.id == curve.id) {
                    curves.push(curve.clone());
                }
            }
        }
    }
    curves.sort_by_key(|curve| curve.id);
    curves
}

/// A row of four regions of two chunks each, two chunks deep.
fn row() -> Vec<ChunkCoord> {
    (0..2)
        .flat_map(|y| (0..8).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

#[test]
fn a_river_meets_its_neighbour_exactly_at_every_region_border() {
    let curves = rivers(&[row()]);

    assert_eq!(curves.len(), 4, "one river per region");
    for pair in curves.windows(2) {
        let (west, east) = (&pair[0], &pair[1]);
        assert_eq!(east.id.region.0, west.id.region.0 + 1);
        assert_eq!(
            west.points.last(),
            east.points.first(),
            "{:?} leaves where {:?} enters",
            west.id,
            east.id
        );
    }
}

#[test]
fn rivers_are_the_same_in_any_generation_order() {
    let all_at_once = rivers(&[row()]);
    let forward = rivers(&row().into_iter().map(|c| vec![c]).collect::<Vec<_>>());
    let backward = rivers(&row().into_iter().rev().map(|c| vec![c]).collect::<Vec<_>>());

    assert_eq!(all_at_once, forward);
    assert_eq!(all_at_once, backward);
}

#[test]
fn a_chunk_holds_the_curves_that_pass_through_it() {
    let mut runtime =
        runtime(vec![height(), region("rivers", REGION, 1, 1)]).with_region_job("rivers", Rivers);
    runtime
        .request(&[FocusPoint::new(ChunkCoord::new(0, 0, 0), 2)], &["rivers"])
        .expect("a stage");
    runtime.run_until_idle().expect("the stages run");

    for y in -2..=2 {
        for x in -2..=2 {
            let chunk = ChunkCoord::new(x, y, 0);
            let (min, max) = (
                [x as f32 * 8.0, y as f32 * 8.0],
                [x as f32 * 8.0 + 8.0, y as f32 * 8.0 + 8.0],
            );
            for curve in runtime.curves("rivers", chunk).expect("generated") {
                let near = curve.points.windows(2).any(|pair| {
                    (0..=16).any(|step| {
                        let t = step as f32 / 16.0;
                        let p = [
                            pair[0][0] + (pair[1][0] - pair[0][0]) * t,
                            pair[0][1] + (pair[1][1] - pair[0][1]) * t,
                        ];
                        let slack = 8.0;
                        p[0] >= min[0] - slack
                            && p[0] <= max[0] + slack
                            && p[1] >= min[1] - slack
                            && p[1] <= max[1] + slack
                    })
                });
                assert!(
                    near,
                    "{:?} is listed in {chunk:?} but passes nowhere near it",
                    curve.id
                );
            }
        }
    }
}

/// Rejects every attempt whose hash is odd, and counts its runs.
struct Picky(Arc<AtomicUsize>);

impl RegionJob for Picky {
    fn run(&self, input: &RegionInput<'_>) -> Result<Attempt, StageError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        if input.hash(7) % 2 == 1 {
            return Ok(Attempt::Rejected(format!(
                "attempt {} drew an odd hash",
                input.retry()
            )));
        }
        Ok(Attempt::Accepted(Vec::new()))
    }
}

/// Never satisfied.
struct Never;

impl RegionJob for Never {
    fn run(&self, input: &RegionInput<'_>) -> Result<Attempt, StageError> {
        Ok(Attempt::Rejected(format!(
            "attempt {} is never enough",
            input.retry()
        )))
    }
}

#[test]
fn a_rejected_attempt_is_retried_with_new_hashes_within_the_budget() {
    let runs = Arc::new(AtomicUsize::new(0));
    let mut runtime = runtime(vec![height(), region("picky", 1, 0, 64)])
        .with_region_job("picky", Picky(Arc::clone(&runs)));
    let chunks: Vec<ChunkCoord> = (0..6).map(|x| ChunkCoord::new(x, 0, 0)).collect();
    let focus: Vec<FocusPoint> = chunks.iter().map(|&c| FocusPoint::new(c, 0)).collect();

    runtime.request(&focus, &["rivers"]).expect("a stage");
    runtime
        .run_until_idle()
        .expect("every region is accepted within 64 attempts");

    assert!(
        runs.load(Ordering::SeqCst) > chunks.len(),
        "some region needed more than one attempt: {} runs for {} regions",
        runs.load(Ordering::SeqCst),
        chunks.len()
    );
}

#[test]
fn a_region_is_given_up_on_with_every_reason_when_its_budget_runs_out() {
    let mut runtime =
        runtime(vec![height(), region("never", 1, 0, 3)]).with_region_job("never", Never);
    runtime
        .request(
            &[FocusPoint::new(ChunkCoord::new(4, -2, 0), 0)],
            &["rivers"],
        )
        .expect("a stage");

    let error = runtime
        .run_until_idle()
        .expect_err("no attempt is accepted");

    assert_eq!(
        error,
        StageError::RegionRejected {
            stage: "rivers".to_owned(),
            region: (4, -2),
            log: vec![
                "attempt 0 is never enough".to_owned(),
                "attempt 1 is never enough".to_owned(),
                "attempt 2 is never enough".to_owned(),
            ],
        }
    );
}

#[test]
fn a_finite_world_is_one_region_computed_once() {
    let runs = Arc::new(AtomicUsize::new(0));
    // Every attempt is accepted: an even number is always drawn by a budget this large.
    let mut runtime = runtime(vec![height(), region("picky", 4, 0, 1024)])
        .with_region_job("picky", Picky(Arc::clone(&runs)));

    for y in 0..4 {
        for x in 0..4 {
            runtime
                .request(&[FocusPoint::new(ChunkCoord::new(x, y, 0), 0)], &["rivers"])
                .expect("a stage");
            runtime.run_until_idle().expect("the stages run");
        }
    }
    let attempts = runs.load(Ordering::SeqCst);

    runtime
        .request(&[FocusPoint::new(ChunkCoord::new(1, 2, 0), 1)], &["rivers"])
        .expect("a stage");
    runtime.run_until_idle().expect("the stages run");
    assert_eq!(
        runs.load(Ordering::SeqCst),
        attempts,
        "the world was computed once"
    );
}

/// Reads one column beyond the region and its halo.
struct Greedy;

impl RegionJob for Greedy {
    fn run(&self, input: &RegionInput<'_>) -> Result<Attempt, StageError> {
        let ([x0, y0], _) = input.columns();
        input.field("height", x0 - 9, y0)?;
        Ok(Attempt::Accepted(Vec::new()))
    }
}

#[test]
fn a_job_cannot_read_beyond_its_halo() {
    let mut runtime =
        runtime(vec![height(), region("greedy", REGION, 1, 1)]).with_region_job("greedy", Greedy);
    runtime
        .request(&[FocusPoint::new(ChunkCoord::new(0, 0, 0), 0)], &["rivers"])
        .expect("a stage");

    let error = runtime
        .run_until_idle()
        .expect_err("a read beyond the halo");

    assert!(
        matches!(&error, StageError::OutOfReach { stage, input, .. } if stage == "rivers" && input == "height"),
        "{error}"
    );
}

#[test]
fn a_region_stage_names_its_job_and_loading_checks_its_numbers() {
    let mut missing = runtime(vec![height(), region("rivers", REGION, 1, 1)]);
    missing
        .request(&[FocusPoint::new(ChunkCoord::new(0, 0, 0), 0)], &["rivers"])
        .expect("a stage");
    assert_eq!(
        missing.run_until_idle().expect_err("no job"),
        StageError::NoRegionJob("rivers".to_owned())
    );

    for (what, stages) in [
        (
            "a region of 0 chunks",
            vec![height(), region("rivers", 0, 1, 1)],
        ),
        ("a budget of 0", vec![height(), region("rivers", 2, 1, 0)]),
        (
            "a budget over the limit",
            vec![height(), region("rivers", 2, 1, 1025)],
        ),
    ] {
        let error = pack(stages).expect_err(what);
        assert!(
            matches!(&error, PackError::Invalid { stage, .. } if stage == "rivers"),
            "{what}: {error}"
        );
    }
}

#[test]
fn a_region_stage_reads_its_inputs_as_far_as_its_region_and_halo() {
    let pack = pack(vec![height(), region("rivers", 3, 2, 1)]).expect("a valid pack");

    let reach = pack.reach("rivers", SIZE).expect("a stage");

    assert_eq!(reach.get("height"), Some(&((3 - 1 + 2) * 8)), "{reach:?}");
}
