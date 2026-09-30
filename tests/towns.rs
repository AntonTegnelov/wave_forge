//! Solve stages: a town per site, solved whole, the same whatever order its chunks are asked for in,
//! on a thread of their own while other stages generate, whose solver is gone once the runtime is.
//!
//! These run on the CPU reference solver with a small module set of ground and air, so they need no
//! GPU; the city on a GPU is checked in `wfc-devtools`.

use std::collections::BTreeMap;
use std::ops::ControlFlow;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};
use wave_forge::loader::parse_rule_file;
use wave_forge::stages::{Pack, Runtime, StageError, StageWorker};
use wave_forge::towns::{Town, TownError, TownRequest, TownSolver, WfcTowns};
use wave_forge::{ChunkCoord, ChunkShape, FocusPoint, FrozenStore, StoreError};
use wfc_core::reference::ReferenceSolver;

const RULES: &str = r#"(
    faces: {
        "air": Side(connector: "air"),
        "ground": Side(connector: "ground", walkable: true),
        "open": Top(connector: "open"),
        "bedrock": Top(connector: "bedrock"),
    },
    modules: [
        (name: "air", sides: ["air", "air", "air", "air"], up: "open", down: "open"),
        (name: "ground", sides: ["ground", "ground", "ground", "ground"], up: "open", down: "bedrock", tags: ["street_level"]),
        (name: "plaza", sides: ["ground", "ground", "ground", "ground"], up: "open", down: "bedrock", tags: ["street_level"]),
    ],
)"#;

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "height", kind: Field(Mul(Noise(frequency: 0.05, octaves: 2), Constant(10.0)))),
        (name: "towns", kind: Sites(height: "height", region: 5, size: (1, 2), chance: 1.0)),
        (name: "buildings", kind: Solve(sites: "towns", rules: "blocks", bottom: Some(Tagged("street_level")), top: Some(Named("air")))),
    ],
)"#;

const CHUNK: ChunkShape = ChunkShape { x: 4, y: 4, z: 3 };

fn runtime() -> Runtime {
    let file = parse_rule_file(RULES).expect("a module set");
    let towns = WfcTowns::new(CHUNK)
        .with_rules("blocks", file, |ruleset| Ok(ReferenceSolver::new(ruleset)))
        .expect("the rules compile");
    Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        5,
        [CHUNK.x, CHUNK.y],
    )
    .with_towns(Box::new(towns))
    .expect("matching chunks")
}

fn area() -> Vec<ChunkCoord> {
    (0..10)
        .flat_map(|y| (0..10).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

/// Every chunk's town tiles over the area, or `None` outside the sites.
fn generate(order: &[Vec<ChunkCoord>]) -> BTreeMap<ChunkCoord, Option<Vec<u16>>> {
    let mut runtime = runtime();
    let mut out = BTreeMap::new();
    for request in order {
        let focus: Vec<FocusPoint> = request.iter().map(|&c| FocusPoint::new(c, 0)).collect();
        runtime.request(&focus, &["buildings"]).expect("a stage");
        runtime.run_until_idle().expect("the stages run");
        for &chunk in request {
            out.insert(
                chunk,
                runtime
                    .tiles("buildings", chunk)
                    .map(|town| town.tiles.to_vec()),
            );
        }
    }
    out
}

#[test]
fn a_town_stands_only_on_its_site_street_level_below_and_air_above() {
    let file = parse_rule_file(RULES).expect("a module set");
    let street: Vec<u16> = file
        .tiles_tagged("street_level")
        .iter()
        .map(|&t| t as u16)
        .collect();
    let air: Vec<u16> = file.tiles_named("air").iter().map(|&t| t as u16).collect();
    let layer = (CHUNK.x * CHUNK.y) as usize;

    let tiles = generate(&[area()]);

    let towns: Vec<&Vec<u16>> = tiles.values().flatten().collect();
    assert!(towns.len() >= 4, "only {} town chunks", towns.len());
    assert!(
        towns.len() < tiles.len(),
        "some chunks lie outside every site"
    );
    for chunk in towns {
        assert!(
            chunk[..layer].iter().all(|t| street.contains(t)),
            "{chunk:?}"
        );
        assert!(
            chunk[chunk.len() - layer..].iter().all(|t| air.contains(t)),
            "{chunk:?}"
        );
    }
}

#[test]
fn towns_come_out_the_same_in_any_order() {
    let all_at_once = generate(&[area()]);
    let one_by_one = generate(
        &area()
            .into_iter()
            .rev()
            .map(|c| vec![c])
            .collect::<Vec<_>>(),
    );

    assert_eq!(all_at_once, one_by_one);
}

#[test]
fn a_solve_stage_without_a_town_solver_says_so() {
    let mut runtime = Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        5,
        [CHUNK.x, CHUNK.y],
    );
    runtime
        .request(
            &[FocusPoint::new(ChunkCoord::new(2, 2, 0), 3)],
            &["buildings"],
        )
        .expect("a stage");

    let result = runtime.run_until_idle();

    assert_eq!(
        result,
        Err(StageError::NoTownSolver("buildings".to_owned()))
    );
}

#[test]
fn a_rule_set_the_solver_was_not_given_is_named() {
    let file = parse_rule_file(RULES).expect("a module set");
    let towns = WfcTowns::new(CHUNK)
        .with_rules("other", file, |ruleset| Ok(ReferenceSolver::new(ruleset)))
        .expect("the rules compile");
    let mut runtime = Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        5,
        [CHUNK.x, CHUNK.y],
    )
    .with_towns(Box::new(towns))
    .expect("matching chunks");
    runtime
        .request(
            &[FocusPoint::new(ChunkCoord::new(2, 2, 0), 3)],
            &["buildings"],
        )
        .expect("a stage");

    let result = runtime.run_until_idle();

    assert!(
        matches!(&result, Err(StageError::Town { message, .. }) if message.contains("blocks")),
        "{result:?}"
    );
}

/// A town solver that holds each town until the test lets it go.
struct Gated {
    gate: Receiver<()>,
}

impl TownSolver for Gated {
    fn chunk_shape(&self) -> ChunkShape {
        CHUNK
    }

    fn solve(&mut self, request: &TownRequest<'_>) -> Result<Town, TownError> {
        self.gate.recv().expect("the test lets the town go");
        let (w, h) = request.size;
        let cells = (CHUNK.x * CHUNK.y * CHUNK.z) as usize;
        Ok(Town {
            size: request.size,
            chunks: vec![Arc::from(vec![0u16; cells]); (w * h) as usize],
        })
    }
}

#[test]
fn every_other_stage_generates_while_a_town_is_being_solved() {
    let (open, gate) = channel::<()>();
    let mut runtime = Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        5,
        [CHUNK.x, CHUNK.y],
    )
    .with_towns(Box::new(Gated { gate }))
    .expect("matching chunks");
    let focus: Vec<FocusPoint> = area().into_iter().map(|c| FocusPoint::new(c, 0)).collect();
    runtime
        .request(&focus, &["height", "buildings"])
        .expect("stages");

    let first = runtime.step(usize::MAX).expect("the stages run");
    let waiting = !runtime.is_idle();
    for _ in 0..area().len() {
        open.send(()).expect("the town thread");
    }
    let rest = runtime.run_until_idle().expect("the stages run");

    let in_a_town = |(stage, chunk): &(String, ChunkCoord)| {
        stage == "buildings" && runtime.tiles("buildings", *chunk).is_some()
    };
    assert!(first.iter().filter(|(stage, _)| stage == "height").count() >= area().len());
    assert!(
        !first.iter().any(in_a_town),
        "a town chunk came before its town"
    );
    assert!(waiting);
    assert!(rest.iter().any(in_a_town));
    assert!(runtime.is_idle());
}

/// A town solver that is slow to drop, as a GPU solver's device is, and says when it has been.
struct SlowToDrop {
    dropped: Arc<AtomicBool>,
}

impl TownSolver for SlowToDrop {
    fn chunk_shape(&self) -> ChunkShape {
        CHUNK
    }

    fn solve(&mut self, _request: &TownRequest<'_>) -> Result<Town, TownError> {
        unreachable!("no town is asked for")
    }
}

impl Drop for SlowToDrop {
    fn drop(&mut self) {
        std::thread::sleep(Duration::from_millis(100));
        self.dropped.store(true, Ordering::SeqCst);
    }
}

// A process may end as soon as its runtime is dropped, and a device still being torn down on
// another thread then faults in the driver.
#[test]
fn a_dropped_runtime_has_dropped_its_town_solver() {
    let dropped = Arc::new(AtomicBool::new(false));
    let runtime = Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        5,
        [CHUNK.x, CHUNK.y],
    )
    .with_towns(Box::new(SlowToDrop {
        dropped: Arc::clone(&dropped),
    }))
    .expect("matching chunks");

    drop(runtime);

    assert!(dropped.load(Ordering::SeqCst));
}

#[test]
fn a_finished_workers_thread_has_dropped_its_town_solver() {
    let dropped = Arc::new(AtomicBool::new(false));
    let solver_dropped = Arc::clone(&dropped);
    let worker = StageWorker::spawn(move || {
        Runtime::new(
            Arc::new(Pack::parse(PACK).expect("a valid pack")),
            5,
            [CHUNK.x, CHUNK.y],
        )
        .with_towns(Box::new(SlowToDrop {
            dropped: solver_dropped,
        }))
        .map_err(|error| error.to_string())
    });

    worker.finish().join().expect("the thread ends");

    assert!(dropped.load(Ordering::SeqCst));
}

/// A town solver as slow as a large town on a GPU.
struct SlowTowns;

impl TownSolver for SlowTowns {
    fn chunk_shape(&self) -> ChunkShape {
        CHUNK
    }

    fn solve(&mut self, request: &TownRequest<'_>) -> Result<Town, TownError> {
        std::thread::sleep(Duration::from_secs(3));
        let (w, h) = request.size;
        let cells = (CHUNK.x * CHUNK.y * CHUNK.z) as usize;
        Ok(Town {
            size: request.size,
            chunks: vec![Arc::from(vec![0u16; cells]); (w * h) as usize],
        })
    }
}

/// A store that keeps nothing.
struct Nowhere;

impl FrozenStore for Nowhere {
    fn keep(
        &mut self,
        _layer: &str,
        _chunk: ChunkCoord,
        _bytes: Vec<u8>,
    ) -> Result<(), StoreError> {
        Ok(())
    }

    fn fetch(&mut self, _layer: &str, _chunk: ChunkCoord) -> Result<Option<Vec<u8>>, StoreError> {
        Ok(None)
    }
}

// A block of a large world takes minutes, and an engine quitting part way waits for its run.
#[test]
fn a_world_run_told_to_stop_while_a_block_generates_stops_before_the_block_is_done() {
    let bounded = PACK.replace(
        "version: 1,",
        "version: 1, bound: Some(Rect(min: (0.0, 0.0), max: (19.0, 19.0))),",
    );
    let mut runtime = Runtime::new(
        Arc::new(Pack::parse(&bounded).expect("a valid pack")),
        5,
        [CHUNK.x, CHUNK.y],
    )
    .with_towns(Box::new(SlowTowns))
    .expect("matching chunks");
    let started = Instant::now();

    let stopped = runtime
        .run_world(&["buildings"], &mut Nowhere, |_| ControlFlow::Break(()))
        .expect("a bounded pack");

    assert_eq!(stopped.done, 0);
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "{:?}",
        started.elapsed()
    );
}
