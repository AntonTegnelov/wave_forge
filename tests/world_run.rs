//! A whole finite world as one run (docs/product/user-stories.md, M1): every chunk of the bound
//! written to a store as it is done, with the runtime holding only what one chunk reads, and a run
//! stopped part way and resumed writing the same bytes as one that went at once.

use std::collections::BTreeMap;
use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wave_forge::stages::{Edits, Pack, RunProgress, Runtime, StageError, StageWorker};
use wave_forge::{ChunkCoord, FrozenStore, StoreError};

const PACK: &str = r#"(
    version: 1,
    bound: Some(Rect(min: (0.0, 0.0), max: (47.0, 39.0))),
    stages: [
        (name: "height", kind: Field(Mul(Noise(frequency: 0.05, octaves: 3), Constant(12.0)))),
        (name: "surface", kind: Rules(rules: [
            (category: "rock", when: [Greater(Input("height"), Constant(8.0))]),
        ], otherwise: "grass")),
        (name: "shrines", kind: Locations(height: "height", region: 3, kinds: [
            (name: "shrine", quota: 1, tries: 40),
        ])),
        (name: "ground", kind: Flatten(height: "height", sites: "shrines", blend: 3)),
        (name: "trees", kind: Scatter(kind: "tree", height: "ground", spacing: 3, apart: 2,
            avoid: Some(("shrines", 1)))),
    ],
)"#;

const TARGETS: [&str; 3] = ["ground", "surface", "trees"];
const SIZE: [u32; 2] = [8, 8];

/// A store in memory, by layer and chunk.
#[derive(Clone, Default)]
struct Memory(BTreeMap<(String, (i32, i32)), Vec<u8>>);

impl FrozenStore for Memory {
    fn keep(&mut self, layer: &str, chunk: ChunkCoord, bytes: Vec<u8>) -> Result<(), StoreError> {
        self.0.insert((layer.to_owned(), (chunk.x, chunk.y)), bytes);
        Ok(())
    }

    fn fetch(&mut self, layer: &str, chunk: ChunkCoord) -> Result<Option<Vec<u8>>, StoreError> {
        Ok(self.0.get(&(layer.to_owned(), (chunk.x, chunk.y))).cloned())
    }
}

fn runtime(text: &str) -> Runtime {
    Runtime::new(Arc::new(Pack::parse(text).expect("a valid pack")), 17, SIZE)
}

/// Runs the pack over `bound` to the end, and gives its last progress and the most products the
/// runtime held at once.
fn run(bound: &str) -> (RunProgress, usize, usize) {
    let text = PACK.replace("Rect(min: (0.0, 0.0), max: (47.0, 39.0))", bound);
    let mut store = Memory::default();
    let mut most = 0;
    let mut reports = 0;
    let end = runtime(&text)
        .run_world(&TARGETS, &mut store, |progress| {
            reports += 1;
            assert_eq!(progress.done, reports);
            most = most.max(progress.held);
            ControlFlow::Continue(())
        })
        .expect("a bounded pack");
    assert_eq!(store.0.len(), end.total * TARGETS.len());
    (end, most, store.0.len())
}

#[test]
fn a_run_writes_every_target_of_every_chunk_holding_what_a_chunk_reads_whatever_the_worlds_size() {
    let (small, small_most, _) = run("Rect(min: (0.0, 0.0), max: (47.0, 39.0))");
    let (large, large_most, _) = run("Rect(min: (0.0, 0.0), max: (95.0, 79.0))");

    // 48 by 40 cells, 6 by 5 chunks; and 12 by 10.
    assert_eq!((small.done, small.total, small.skipped), (30, 30, 0));
    assert_eq!((large.done, large.total), (120, 120));
    // Four times the chunks, and no more held at once than a chunk's reads need.
    assert!(
        large_most <= small_most + small_most / 10,
        "held up to {small_most} products for 30 chunks and {large_most} for 120"
    );
}

#[test]
fn a_run_stopped_and_resumed_writes_the_same_bytes_as_one_at_once() {
    let mut at_once = Memory::default();
    runtime(PACK)
        .run_world(&TARGETS, &mut at_once, |_| ControlFlow::Continue(()))
        .expect("a bounded pack");

    let mut pieces = Memory::default();
    let stopped = runtime(PACK)
        .run_world(&TARGETS, &mut pieces, |progress| {
            if progress.done == 11 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })
        .expect("a bounded pack");
    // A fresh runtime, as after a crash, picks up from the store.
    let resumed = runtime(PACK)
        .run_world(&TARGETS, &mut pieces, |_| ControlFlow::Continue(()))
        .expect("a bounded pack");

    assert_eq!((stopped.done, stopped.total), (11, 30));
    assert_eq!((resumed.done, resumed.skipped), (30, 11));
    assert_eq!(pieces.0, at_once.0);
}

#[test]
fn a_world_without_a_bound_cannot_be_run_whole() {
    let unbounded = PACK.replace("bound: Some(Rect(min: (0.0, 0.0), max: (47.0, 39.0))),", "");
    let mut store = Memory::default();

    let result = runtime(&unbounded).run_world(&TARGETS, &mut store, |_| ControlFlow::Continue(()));

    assert!(matches!(result, Err(StageError::Unbounded)), "{result:?}");
}

/// Drains `worker` until `done` holds or it fails, for ten seconds at most.
fn drain_until(worker: &mut StageWorker, done: impl Fn(&StageWorker) -> bool) {
    let started = Instant::now();
    while !done(worker) && worker.failure().is_none() {
        worker.drain();
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "nothing arrived"
        );
        std::thread::yield_now();
    }
}

#[test]
fn a_played_world_serves_what_the_run_wrote_and_generates_nothing() {
    let mut store = Memory::default();
    runtime(PACK)
        .run_world(&TARGETS, &mut store, |_| ControlFlow::Continue(()))
        .expect("a bounded pack");
    let pack = Arc::new(Pack::parse(PACK).expect("a valid pack"));
    let mut worker = StageWorker::play(Arc::clone(&pack), SIZE, Box::new(store));
    let centre = ChunkCoord::new(2, 2, 0);
    let around: Vec<ChunkCoord> = (1..=3)
        .flat_map(|y| (1..=3).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect();

    worker.request(&[wave_forge::FocusPoint::new(centre, 1)], &TARGETS);
    drain_until(&mut worker, |worker| {
        around
            .iter()
            .all(|&chunk| worker.points("trees", chunk).is_some())
    });

    assert!(worker.failure().is_none(), "{:?}", worker.failure());
    let mut direct = runtime(PACK);
    let focus: Vec<wave_forge::FocusPoint> = around
        .iter()
        .map(|&chunk| wave_forge::FocusPoint::new(chunk, 0))
        .collect();
    direct.request(&focus, &TARGETS).expect("the stages");
    direct.run_until_idle().expect("the stages run");
    for &chunk in &around {
        for target in TARGETS {
            assert_eq!(
                worker.shared(target, chunk).as_deref(),
                direct.product(target, chunk),
                "{target} at {chunk:?}"
            );
        }
    }
    assert!(
        worker.timings().is_empty(),
        "a stage ran: {:?}",
        worker.timings()
    );
    // As a runtime asked the same holds them: each target where the request and its readers
    // need it, and nowhere else.
    let mut asked = runtime(PACK);
    asked
        .request(&[wave_forge::FocusPoint::new(centre, 1)], &TARGETS)
        .expect("the stages");
    asked.run_until_idle().expect("the stages run");
    for y in -2..8 {
        for x in -2..8 {
            let chunk = ChunkCoord::new(x, y, 0);
            for target in TARGETS {
                assert_eq!(
                    worker.shared(target, chunk).is_some(),
                    asked.product(target, chunk).is_some(),
                    "{target} at {chunk:?}"
                );
            }
        }
    }
}

#[test]
fn a_played_world_fails_on_a_stage_the_run_did_not_write_or_an_edit() {
    let mut store = Memory::default();
    runtime(PACK)
        .run_world(&TARGETS, &mut store, |_| ControlFlow::Continue(()))
        .expect("a bounded pack");
    let pack = Arc::new(Pack::parse(PACK).expect("a valid pack"));
    let focus = [wave_forge::FocusPoint::new(ChunkCoord::new(2, 2, 0), 0)];

    let mut unwritten = StageWorker::play(Arc::clone(&pack), SIZE, Box::new(store.clone()));
    unwritten.request(&focus, &["height"]);
    drain_until(&mut unwritten, |_| false);
    let mut edited = StageWorker::play(pack, SIZE, Box::new(store));
    edited.set_edits(Edits::default());
    drain_until(&mut edited, |_| false);

    assert!(
        unwritten
            .failure()
            .is_some_and(|reason| reason.contains("height")),
        "{:?}",
        unwritten.failure()
    );
    assert!(
        edited.failure().is_some(),
        "an edit of a played world was taken"
    );
}

#[test]
fn a_run_reports_what_each_stage_has_generated_after_each_chunk() {
    let mut store = Memory::default();
    let mut reports: Vec<RunProgress> = Vec::new();

    runtime(PACK)
        .run_world(&TARGETS, &mut store, |progress| {
            reports.push(progress);
            ControlFlow::Continue(())
        })
        .expect("a bounded pack");

    let names: Vec<&str> = reports[0]
        .stages
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert_eq!(names, ["height", "surface", "shrines", "ground", "trees"]);
    for pair in reports.windows(2) {
        for ((name, before), (_, after)) in pair[0].stages.iter().zip(&pair[1].stages) {
            assert!(after.products >= before.products, "{name} went back");
        }
    }
    // Every target has generated at least every chunk of the world by the end.
    let last = reports.last().expect("a report per chunk");
    for (name, timing) in &last.stages {
        if TARGETS.contains(&name.as_str()) {
            assert!(timing.products >= 30, "{name}: {timing:?}");
        }
    }
}

const RIVERS: &str = r#"(
    version: 1,
    bound: Some(Rect(min: (0.0, 0.0), max: (95.0, 95.0))),
    water: Some((level: 0.0)),
    stages: [
        (name: "height", kind: Field(Sub(Mul(Noise(frequency: 0.04, octaves: 3), Constant(20.0)), Constant(4.0)))),
        (name: "rivers", kind: Rivers(height: "height", region: 4, sources: 2, step: 2)),
        (name: "ground", kind: Apply(height: "height", curves: "rivers", max_radius: 3, blend: 1, profile: Carve(1.0))),
    ],
)"#;

#[test]
fn a_run_generates_a_regions_inputs_about_once() {
    let pack = Arc::new(Pack::parse(RIVERS).expect("a valid pack"));
    let heights = |runtime: &Runtime| {
        runtime
            .timings()
            .into_iter()
            .find(|(stage, _)| stage == "height")
            .map_or(0, |(_, timing)| timing.products)
    };
    let mut at_once = Runtime::new(Arc::clone(&pack), 5, SIZE);
    at_once.request_bound(&["ground"]).expect("a bounded pack");
    at_once.run_until_idle().expect("the stages run");

    let mut run = Runtime::new(pack, 5, SIZE);
    run.run_world(&["ground"], &mut Memory::default(), |_| {
        ControlFlow::Continue(())
    })
    .expect("a bounded pack");

    // Twelve chunks a side in regions of four: the run may generate again only what a block's
    // neighbours read across its edges.
    assert!(
        heights(&run) * 2 <= heights(&at_once) * 3,
        "{} chunks of height for {} at once",
        heights(&run),
        heights(&at_once)
    );
}
