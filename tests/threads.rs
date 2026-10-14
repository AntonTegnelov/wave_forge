//! Generating on several threads (`Runtime::with_threads`): every product of every stage of the
//! ring world and of the cave level is the same as on one thread, whatever the number of threads.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::sync::Arc;
use wave_forge::stages::{Pack, Runtime};
use wave_forge::{ChunkCoord, FocusPoint};

fn pack(path: &str) -> Arc<Pack> {
    let text = std::fs::read_to_string(format!("{}/{path}", env!("CARGO_MANIFEST_DIR")))
        .expect("the pack's file");
    Arc::new(Pack::parse(&text).expect("a valid pack"))
}

/// Every chunk's hash over every stage of the pack at `path`, the chunks of `area` asked for at
/// once, generated on `threads` threads.
fn hashes(
    path: &str,
    size: [u32; 2],
    area: &[ChunkCoord],
    threads: usize,
) -> BTreeMap<ChunkCoord, u64> {
    let pack = pack(path);
    let stages: Vec<&str> = pack.stage_names().collect();
    let mut runtime = Runtime::new(Arc::clone(&pack), 7, size)
        .with_threads(NonZeroUsize::new(threads).expect("a thread at least"));
    let focus: Vec<FocusPoint> = area.iter().map(|&c| FocusPoint::new(c, 0)).collect();
    runtime.request(&focus, &stages).expect("every stage");
    runtime.run_until_idle().expect("the stages run");
    area.iter()
        .map(|&chunk| {
            let hash = runtime
                .chunk_hash(chunk, &stages)
                .expect("every stage there");
            (chunk, hash)
        })
        .collect()
}

fn square(low: i32, high: i32) -> Vec<ChunkCoord> {
    (low..high)
        .flat_map(|y| (low..high).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

#[test]
fn every_stage_of_the_ring_world_is_the_same_on_one_thread_and_on_several() {
    let area = square(-3, 3);
    let one = hashes("examples/rings.world.ron", [8, 8], &area, 1);

    for threads in [2, 8] {
        assert_eq!(
            hashes("examples/rings.world.ron", [8, 8], &area, threads),
            one,
            "{threads} threads"
        );
    }
}

#[test]
fn every_stage_of_the_cave_level_is_the_same_on_one_thread_and_on_several() {
    let area = square(0, 3);
    let one = hashes("examples/cave_level.world.ron", [16, 16], &area, 1);

    for threads in [2, 8] {
        assert_eq!(
            hashes("examples/cave_level.world.ron", [16, 16], &area, threads),
            one,
            "{threads} threads"
        );
    }
}
