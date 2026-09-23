//! A pack's stages on a thread of their own: what arrives is what a runtime on the same thread
//! would have generated, a new request drops what it no longer needs, and failures are reported.

use std::sync::Arc;
use std::time::{Duration, Instant};
use wave_forge::stages::{Pack, Runtime, StageEvent, StageWorker};
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
