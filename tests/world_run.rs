//! A whole finite world as one run (docs/product/user-stories.md, M1): every chunk of the bound
//! written to a store as it is done, with the runtime holding only what one chunk reads, and a run
//! stopped part way and resumed writing the same bytes as one that went at once.

use std::collections::BTreeMap;
use std::ops::ControlFlow;
use std::sync::Arc;
use wave_forge::stages::{Pack, RunProgress, Runtime, StageError};
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
#[derive(Default)]
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
