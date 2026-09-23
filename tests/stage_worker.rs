//! A pack's stages on a thread of their own: what arrives is what a runtime on the same thread
//! would have generated, a new request drops what it no longer needs, failures are reported, and
//! each stage's cost is counted as a runtime counts it.

use std::sync::Arc;
use std::time::{Duration, Instant};
use wave_forge::stages::{Facts, Pack, RowId, Runtime, StageEvent, StageWorker};
use wave_forge::{ChunkCoord, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "height", kind: Field(Mul(Noise(frequency: 0.04, octaves: 3), Constant(20.0)))),
        (name: "trees", kind: Scatter(kind: "tree", height: "height", spacing: 3, apart: 3)),
    ],
)"#;

const SIZE: [u32; 2] = [8, 8];

fn runtime() -> Runtime {
    Runtime::new(Arc::new(Pack::parse(PACK).expect("a valid pack")), 4, SIZE)
}

/// Drains `worker` until `done` holds, or fails after ten seconds.
fn drain_until(worker: &mut StageWorker, done: impl Fn(&StageWorker) -> bool) -> Vec<StageEvent> {
    let started = Instant::now();
    let mut events = Vec::new();
    loop {
        events.extend(worker.drain());
        assert!(worker.failure().is_none(), "{:?}", worker.failure());
        if done(worker) {
            return events;
        }
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "nothing arrived in 10 s"
        );
        std::thread::yield_now();
    }
}

fn chunks_around(centre: ChunkCoord) -> Vec<ChunkCoord> {
    (-1..=1)
        .flat_map(|y| (-1..=1).map(move |x| ChunkCoord::new(centre.x + x, centre.y + y, 0)))
        .collect()
}

#[test]
fn what_arrives_is_what_a_runtime_generates() {
    let centre = ChunkCoord::new(3, 2, 0);
    let focus = [FocusPoint::new(centre, 1)];
    let mut direct = runtime();
    direct
        .request(&focus, &["height", "trees"])
        .expect("stages");
    direct.run_until_idle().expect("the stages run");
    let mut worker = StageWorker::spawn(|| Ok(runtime()));

    worker.request(&focus, &["height", "trees"]);
    drain_until(&mut worker, |worker| {
        chunks_around(centre)
            .iter()
            .all(|&c| worker.field("height", c).is_some() && worker.points("trees", c).is_some())
    });

    for chunk in chunks_around(centre) {
        assert_eq!(worker.field("height", chunk), direct.field("height", chunk));
        assert_eq!(worker.points("trees", chunk), direct.points("trees", chunk));
    }
}

#[test]
fn a_request_elsewhere_drops_what_is_no_longer_needed() {
    let mut worker = StageWorker::spawn(|| Ok(runtime()));
    let first = ChunkCoord::new(0, 0, 0);
    worker.request(&[FocusPoint::new(first, 0)], &["trees"]);
    drain_until(&mut worker, |worker| {
        worker.points("trees", first).is_some()
    });

    worker.request(
        &[FocusPoint::new(ChunkCoord::new(40, 40, 0), 0)],
        &["trees"],
    );
    let events = drain_until(&mut worker, |worker| {
        worker.points("trees", first).is_none()
    });

    assert!(events.contains(&StageEvent::Dropped {
        stage: "trees".to_owned(),
        chunk: first
    }));
}

#[test]
fn a_runtime_that_cannot_be_built_is_reported() {
    let mut worker = StageWorker::spawn(|| Err("no device".to_owned()));
    let started = Instant::now();

    while worker.failure().is_none() && started.elapsed() < Duration::from_secs(10) {
        worker.drain();
        std::thread::yield_now();
    }

    assert_eq!(worker.failure(), Some("no device"));
}

#[test]
fn an_unknown_stage_is_reported() {
    let mut worker = StageWorker::spawn(|| Ok(runtime()));
    worker.request(&[FocusPoint::new(ChunkCoord::new(0, 0, 0), 0)], &["rivers"]);
    let started = Instant::now();

    while worker.failure().is_none() && started.elapsed() < Duration::from_secs(10) {
        worker.drain();
        std::thread::yield_now();
    }

    assert!(
        worker
            .failure()
            .is_some_and(|reason| reason.contains("rivers")),
        "{:?}",
        worker.failure()
    );
}

#[test]
fn each_stage_counts_the_products_it_generated_and_the_worker_reports_the_same() {
    let centre = ChunkCoord::new(-2, 1, 0);
    let focus = [FocusPoint::new(centre, 1)];
    let mut direct = runtime();
    direct.request(&focus, &["trees"]).expect("stages");
    let generated = direct.run_until_idle().expect("the stages run");
    let mut worker = StageWorker::spawn(|| Ok(runtime()));

    worker.request(&focus, &["trees"]);
    drain_until(&mut worker, |worker| {
        worker
            .timings()
            .iter()
            .map(|(_, t)| t.products)
            .sum::<u64>()
            == generated.len() as u64
    });

    let timings = direct.timings();
    let names: Vec<&str> = timings.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(names, ["height", "trees"], "in the order of the pack");
    for (name, timing) in &timings {
        let count = generated.iter().filter(|(stage, _)| stage == name).count() as u64;
        assert_eq!(timing.products, count, "{name}");
        assert!(
            timing.ms >= timing.slowest_ms && timing.slowest_ms >= 0.0,
            "{name}: {timing:?}"
        );
    }
    let reported: Vec<(&str, u64)> = worker
        .timings()
        .iter()
        .map(|(name, timing)| (name.as_str(), timing.products))
        .collect();
    let counted: Vec<(&str, u64)> = timings
        .iter()
        .map(|(name, timing)| (name.as_str(), timing.products))
        .collect();
    assert_eq!(reported, counted);
}

#[test]
fn a_new_focus_drops_what_read_the_old_row_and_generates_it_again() {
    const FACTS: &str = r#"(
        version: 1,
        tables: [(name: "worlds", kind: Generated(count: Constant(4.0), columns: [("sea", Random(0.0, 10.0))]))],
        stages: [(name: "height", kind: Field(Sub(Mul(Noise(frequency: 0.04, octaves: 3), Constant(20.0)), Row("worlds", "sea"))))],
    )"#;
    let pack = Arc::new(Pack::parse(FACTS).expect("a valid pack"));
    let facts = Facts::new(Arc::clone(&pack), 4).expect("facts");
    let focused = |row: u64| {
        let mut runtime = Runtime::new(Arc::clone(&pack), 4, SIZE);
        runtime.set_facts(facts.clone()).expect("facts");
        runtime.focus("worlds", RowId(vec![row])).expect("a row");
        runtime
    };
    let chunk = ChunkCoord::new(0, 0, 0);
    let mut direct = focused(2);
    direct
        .request(&[FocusPoint::new(chunk, 0)], &["height"])
        .expect("a stage");
    direct.run_until_idle().expect("the stages run");
    let first = focused(1);
    let mut worker = StageWorker::spawn(move || Ok(first));
    worker.request(&[FocusPoint::new(chunk, 0)], &["height"]);
    drain_until(&mut worker, |worker| {
        worker.field("height", chunk).is_some()
    });

    worker.focus("worlds", RowId(vec![2]));
    let events = drain_until(&mut worker, |worker| {
        worker.field("height", chunk) == direct.field("height", chunk)
    });

    assert_eq!(
        events.first(),
        Some(&StageEvent::Dropped {
            stage: "height".to_owned(),
            chunk
        })
    );
}
