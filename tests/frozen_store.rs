//! A frozen stage with a store: chunks no request needs leave memory for the store and come back
//! from it unchanged, so a walk across an infinite world holds a bounded number of them, and a
//! chunk walked back to, or read by a runtime of a changed pack, is as it was first generated.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wave_forge::stages::{Pack, Point, Runtime};
use wave_forge::{ChunkCoord, FocusPoint, FrozenStore, StoreError};

fn pack(tree_spacing: u32) -> Arc<Pack> {
    Arc::new(
        Pack::parse(&format!(
            r#"(version: 1, stages: [
                (name: "ground", kind: Field(Mul(Noise(frequency: 0.05, octaves: 2), Constant(10.0)))),
                (name: "trees", persist: Frozen, kind: Scatter(kind: "tree", height: "ground", spacing: {tree_spacing})),
            ])"#
        ))
        .expect("a valid pack"),
    )
}

/// The bytes kept, by layer and chunk.
type Kept = HashMap<(String, ChunkCoord), Vec<u8>>;

/// A store in memory, shared by every runtime given a clone of it, as a directory on disk is.
#[derive(Clone, Default)]
struct SharedStore(Arc<Mutex<Kept>>);

impl FrozenStore for SharedStore {
    fn keep(&mut self, layer: &str, chunk: ChunkCoord, bytes: Vec<u8>) -> Result<(), StoreError> {
        self.0
            .lock()
            .expect("not poisoned")
            .insert((layer.to_owned(), chunk), bytes);
        Ok(())
    }

    fn fetch(&mut self, layer: &str, chunk: ChunkCoord) -> Result<Option<Vec<u8>>, StoreError> {
        Ok(self
            .0
            .lock()
            .expect("not poisoned")
            .get(&(layer.to_owned(), chunk))
            .cloned())
    }
}

/// The trees of `chunk`, after asking for the trees around it.
fn trees(runtime: &mut Runtime, chunk: ChunkCoord) -> Vec<Point> {
    runtime
        .request(&[FocusPoint::new(chunk, 1)], &["trees"])
        .expect("stages");
    runtime.run_until_idle().expect("the stages run");
    runtime.points("trees", chunk).expect("generated").to_vec()
}

/// How many frozen chunks the runtime holds in memory: all a save carries when it has a store.
fn frozen_in_memory(runtime: &Runtime) -> usize {
    runtime.save().frozen.len()
}

#[test]
fn a_walk_across_the_world_holds_a_bounded_number_of_frozen_chunks() {
    let store = SharedStore::default();
    let mut runtime = Runtime::new(pack(3), 2, [8, 8]).with_store(Box::new(store.clone()));

    let mut most = 0;
    for x in 0..60 {
        trees(&mut runtime, ChunkCoord::new(x, 0, 0));
        most = most.max(frozen_in_memory(&runtime));
    }

    assert!(
        most <= 9,
        "{most} frozen chunks in memory for a focus of radius 1"
    );
    assert!(
        store.0.lock().expect("not poisoned").len() >= 50,
        "the chunks walked past are in the store"
    );
}

#[test]
fn a_frozen_chunk_walked_back_to_is_as_it_was_first_generated() {
    let origin = ChunkCoord::new(0, 0, 0);
    let mut runtime = Runtime::new(pack(3), 2, [8, 8]).with_store(Box::new(SharedStore::default()));
    let first = trees(&mut runtime, origin);

    trees(&mut runtime, ChunkCoord::new(40, 0, 0));
    let back = trees(&mut runtime, origin);

    assert!(!first.is_empty(), "no trees to compare");
    assert_eq!(back, first);
}

#[test]
fn a_runtime_of_a_changed_pack_reads_a_frozen_chunk_from_the_store_as_it_was() {
    let origin = ChunkCoord::new(0, 0, 0);
    let store = SharedStore::default();
    let mut before = Runtime::new(pack(3), 2, [8, 8]).with_store(Box::new(store.clone()));
    let frozen = trees(&mut before, origin);
    trees(&mut before, ChunkCoord::new(40, 0, 0));

    let mut after = Runtime::new(pack(5), 2, [8, 8]).with_store(Box::new(store));
    let changed = trees(&mut Runtime::new(pack(5), 2, [8, 8]), origin);

    assert_ne!(frozen, changed, "the pack's change moves the trees");
    assert_eq!(trees(&mut after, origin), frozen);
}
