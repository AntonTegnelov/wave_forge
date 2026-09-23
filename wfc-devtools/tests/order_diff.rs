//! The same world whatever order its chunks are generated in.
//!
//! A world is meant to be a function of its configuration, not of the path a player took through it
//! (docs/generation-model.md §2). This generates a small city three ways: all at once, chunk by chunk
//! in raster order, and chunk by chunk in the reverse order, and compares every chunk's tiles and the
//! set of chunks given up on. A mismatch names the first chunk and cell that differ and how many
//! chunks differ in all.
//!
//! A chunk of the first parity is solved alone and one of the second against its first-parity
//! neighbours only, so its tiles are a function of its coordinate. A repair rewrites neighbours
//! that were already solved, and stays a function of coordinates because it waits for every
//! neighbour it can see and for the repairs of lower classes around it (docs/architecture.md §6.3).

use std::collections::{BTreeMap, BTreeSet};
use wave_forge::{
    Builder, ChunkCoord, ChunkEvent, ChunkShape, FocusPoint, RepairPolicy, Ruleset, WorldExtent,
};
use wfc_devtools::city::{self, city_prior};

const CHUNK: ChunkShape = ChunkShape::cube(8);
const CHUNKS: i32 = 4;

/// Everything a generated world holds: each chunk's tiles, and the chunks given up on.
#[derive(Debug, PartialEq)]
struct World {
    tiles: BTreeMap<ChunkCoord, Vec<u16>>,
    failed: BTreeSet<ChunkCoord>,
}

/// The order focus points are handed to the generator in: all chunks in one request, or one
/// request per chunk, each generated before the next is asked for.
enum Order {
    AllAtOnce,
    ChunkByChunk(Vec<ChunkCoord>),
}

fn raster() -> Vec<ChunkCoord> {
    (0..CHUNKS)
        .flat_map(|y| (0..CHUNKS).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

fn generate(seed: u64, repair: RepairPolicy, order: &Order) -> World {
    let city = city::city();
    let ruleset = Ruleset::from_modules(&city.modules).expect("the city compiles");
    let mut world = Builder::new(ruleset, city_prior(&city, CHUNK.z))
        .seed(seed)
        .extent(
            WorldExtent::new(CHUNK)
                .with_x(0..CHUNKS)
                .with_y(0..CHUNKS)
                .with_z(0..1),
        )
        .halo(1)
        .repair(repair)
        .build()
        .expect("a compute device");
    let mut failed = BTreeSet::new();
    let mut run = |focus: FocusPoint| {
        world.request(&[focus]);
        for event in world.run_until_idle().expect("the solver runs") {
            if let ChunkEvent::Failed { chunk, .. } = event {
                failed.insert(chunk);
            }
        }
    };
    match order {
        Order::AllAtOnce => run(FocusPoint::new(ChunkCoord::new(1, 1, 0), 3)),
        Order::ChunkByChunk(chunks) => {
            for &chunk in chunks {
                run(FocusPoint::new(chunk, 0));
            }
        }
    }
    let tiles = world
        .store()
        .iter()
        .map(|chunk| (chunk.coord, chunk.tiles.to_vec()))
        .collect();
    World { tiles, failed }
}

/// How two worlds differ, or `None` if they are the same: the chunks that differ, and the first
/// differing cell of the first of them.
fn difference(a: &World, b: &World) -> Option<String> {
    if a == b {
        return None;
    }
    let differing: Vec<ChunkCoord> = a
        .tiles
        .keys()
        .chain(b.tiles.keys())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter(|coord| a.tiles.get(coord) != b.tiles.get(coord))
        .copied()
        .collect();
    let first = differing.first().map(|coord| {
        let cell = match (a.tiles.get(coord), b.tiles.get(coord)) {
            (Some(x), Some(y)) => x.iter().zip(y).position(|(p, q)| p != q),
            _ => None,
        };
        format!("first {coord:?} at cell {cell:?}")
    });
    Some(format!(
        "{} of {} chunks differ ({}); failed {:?} against {:?}",
        differing.len(),
        a.tiles.len().max(b.tiles.len()),
        first.unwrap_or_default(),
        a.failed,
        b.failed
    ))
}

#[test]
fn without_repairs_generation_order_does_not_change_the_world() {
    let off = RepairPolicy {
        enabled: false,
        ..RepairPolicy::default()
    };
    let mut reverse = raster();
    reverse.reverse();

    for seed in [8, 11] {
        let all = generate(seed, off, &Order::AllAtOnce);
        let forward = generate(seed, off, &Order::ChunkByChunk(raster()));
        let backward = generate(seed, off, &Order::ChunkByChunk(reverse.clone()));

        assert_eq!(
            all.tiles.len() + all.failed.len(),
            (CHUNKS * CHUNKS) as usize,
            "seed {seed}: every chunk is either generated or given up on"
        );
        for (name, other) in [("raster", &forward), ("reverse raster", &backward)] {
            if let Some(difference) = difference(&all, other) {
                panic!("seed {seed}: all at once against {name}: {difference}");
            }
        }
    }
}

#[test]
fn with_repairs_generation_order_does_not_change_the_world() {
    let mut reverse = raster();
    reverse.reverse();

    for seed in [8, 11] {
        let all = generate(seed, RepairPolicy::default(), &Order::AllAtOnce);
        let forward = generate(
            seed,
            RepairPolicy::default(),
            &Order::ChunkByChunk(raster()),
        );
        let backward = generate(
            seed,
            RepairPolicy::default(),
            &Order::ChunkByChunk(reverse.clone()),
        );

        for (name, other) in [("raster", &forward), ("reverse raster", &backward)] {
            if let Some(difference) = difference(&all, other) {
                panic!("seed {seed}, repairs on: all at once against {name}: {difference}");
            }
        }
    }
}
