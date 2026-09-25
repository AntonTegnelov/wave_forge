//! A city evicted in part and generated again comes back tile for tile, repairs included.
//!
//! A repair rewrites the neighbours of the chunk it repairs (docs/architecture/world.md,
//! "Repairs"), and a chunk's first attempt reads its neighbours as they were at a fixed point of the
//! schedule. If regeneration read the neighbours that stayed as they are now, a chunk asked for
//! again would come back different from the first time. These tests generate a city, drop
//! everything beyond a cut, ask for the whole city again, and compare every chunk with a city
//! generated once, for every cut across it; and walk a focus along a strip and back, evicting
//! behind it, which puts chunks generated for the first time beside chunks generated again.

use std::collections::BTreeMap;
use wave_forge::{
    BlockSolver, Builder, ChunkCoord, ChunkEvent, ChunkShape, FocusPoint, Ruleset, WgpuBackend,
    WorldExtent, WorldGenerator,
};
use wfc_devtools::city::{self, city_prior};
use wfc_devtools::{BoundaryCondition, TileGrid, adjacency_violations};

const CHUNK: ChunkShape = ChunkShape::cube(8);
const WIDE: i32 = 8;
const DEEP: i32 = 4;

type Tiles = BTreeMap<ChunkCoord, Vec<u16>>;

fn tiles(world: &WorldGenerator<BlockSolver<WgpuBackend>>) -> Tiles {
    world
        .store()
        .iter()
        .map(|chunk| (chunk.coord, chunk.tiles.to_vec()))
        .collect()
}

/// A city of `seed`, `wide` by [`DEEP`] chunks, not yet generated.
fn city_world(seed: u64, wide: i32) -> WorldGenerator<BlockSolver<WgpuBackend>> {
    let city = city::city();
    let ruleset = Ruleset::from_modules(&city.modules).expect("the city compiles");
    Builder::new(ruleset, city_prior(&city, CHUNK.z))
        .seed(seed)
        .extent(
            WorldExtent::new(CHUNK)
                .with_x(0..wide)
                .with_y(0..DEEP)
                .with_z(0..1),
        )
        .halo(1)
        .build()
        .expect("a compute device")
}

/// The whole city of `seed`, `wide` chunks long, generated at once.
fn generated_at_once(seed: u64, wide: i32) -> WorldGenerator<BlockSolver<WgpuBackend>> {
    let mut world = city_world(seed, wide);
    world.request(&[FocusPoint::new(ChunkCoord::new(0, 0, 0), wide as u32)]);
    world.run_until_idle().expect("the solver runs");
    world
}

/// What one cut cost to generate again, against the first generation of the whole city.
#[derive(Debug)]
struct Cut {
    cut: i32,
    changed: usize,
    replayed: u32,
    batches: u32,
    solver_ms: f64,
}

/// Every cut across an 8 by 4 chunk city of `seed`: the city generated once, then for each cut a
/// city generated whole, evicted beyond the cut from its corner and asked for whole again, how many
/// of its chunks differ from the first and what generating them again cost. The first city has to
/// have no seam, no chunk given up on, and repairs that rewrote neighbours, or there is nothing to
/// test.
fn each_cut(seed: u64) -> Vec<Cut> {
    let city = city::city();
    let world = || city_world(seed, WIDE);
    let everything = [FocusPoint::new(ChunkCoord::new(0, 0, 0), WIDE as u32)];
    let mut first = world();
    first.request(&everything);
    let events = first.run_until_idle().expect("the solver runs");
    let expected = tiles(&first);
    assert_eq!(expected.len(), (WIDE * DEEP) as usize, "the whole city");
    let (grid, _) = TileGrid::from_chunks(CHUNK, first.store().iter(), 0).expect("chunks");
    assert!(
        adjacency_violations(&grid, &city.modules.rules, BoundaryCondition::Finite).is_empty(),
        "seed {seed}: a seam before any eviction"
    );
    assert!(
        first.stats().rewritten_by_repair > 0,
        "seed {seed} rewrites no neighbour, so nothing is tested"
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, ChunkEvent::Failed { .. })),
        "seed {seed}: a chunk failed"
    );

    let first_stats = *first.stats();
    eprintln!(
        "partial_eviction: seed {seed}, the whole city at once: {} batches, {:.0} ms",
        first_stats.batches, first_stats.solver_ms
    );
    (1..WIDE)
        .map(|cut| {
            let mut streamed = world();
            streamed.request(&everything);
            streamed.run_until_idle().expect("the solver runs");
            let before = *streamed.stats();
            // Everything further than `cut - 1` chunks from the corner, along either axis.
            let near = [FocusPoint::new(ChunkCoord::new(0, 0, 0), (cut - 1) as u32)];
            streamed.request(&near);
            let evicted = streamed.evict_outside(&near, 0).len();
            streamed.request(&everything);
            streamed.run_until_idle().expect("the solver runs");

            let got = tiles(&streamed);
            let after = *streamed.stats();
            let result = Cut {
                cut,
                changed: expected
                    .iter()
                    .filter(|(coord, tiles)| got.get(coord) != Some(tiles))
                    .count(),
                replayed: after.replayed - before.replayed,
                batches: after.batches - before.batches,
                solver_ms: after.solver_ms - before.solver_ms,
            };
            eprintln!(
                "partial_eviction: seed {seed}, cut {}: {evicted} chunks evicted and generated \
                 again in {} batches, {:.0} ms, {} repairs replayed, {} chunks changed",
                result.cut, result.batches, result.solver_ms, result.replayed, result.changed
            );
            result
        })
        .collect()
}

/// A focus of radius 2 walked along a 16 by 4 chunk city of `seed` and back, evicting everything a
/// chunk beyond its view after every step, then walked out again; and the chunks at its final view
/// whose neighbourhood has settled that differ from the city generated at once.
fn walked(seed: u64) -> usize {
    const LONG: i32 = 16;
    const RADIUS: u32 = 2;
    let expected = tiles(&generated_at_once(seed, LONG));
    let mut world = city_world(seed, LONG);
    let steps = (0..LONG - 1).chain((1..LONG - 1).rev()).chain(0..LONG / 2);
    let mut at = ChunkCoord::new(0, 1, 0);
    for x in steps {
        at = ChunkCoord::new(x, 1, 0);
        let focus = [FocusPoint::new(at, RADIUS)];
        world.request(&focus);
        world.run_until_idle().expect("the solver runs");
        world.evict_outside(&focus, 1);
    }
    assert!(
        world.stats().replayed > 0,
        "seed {seed}: nothing was replayed"
    );

    let got = tiles(&world);
    // A chunk this close to the focus has every chunk that can rewrite it generated.
    let settled: Vec<&ChunkCoord> = got
        .keys()
        .filter(|coord| (coord.x - at.x).abs().max((coord.y - at.y).abs()) < RADIUS as i32)
        .collect();
    assert!(!settled.is_empty(), "nothing settled to compare");
    settled
        .into_iter()
        .filter(|coord| got.get(coord) != expected.get(coord))
        .count()
}

#[test]
fn a_city_of_seed_8_evicted_beyond_any_cut_comes_back_tile_for_tile() {
    let cuts = each_cut(8);

    assert!(cuts.iter().all(|cut| cut.changed == 0), "{cuts:?}");
    assert!(
        cuts.iter().any(|cut| cut.replayed > 0),
        "nothing replayed: {cuts:?}"
    );
}

#[test]
fn a_city_of_seed_11_evicted_beyond_any_cut_comes_back_tile_for_tile() {
    let cuts = each_cut(11);

    assert!(cuts.iter().all(|cut| cut.changed == 0), "{cuts:?}");
    assert!(
        cuts.iter().any(|cut| cut.replayed > 0),
        "nothing replayed: {cuts:?}"
    );
}

#[test]
fn a_city_walked_there_and_back_is_the_city_generated_at_once() {
    for seed in [8, 11] {
        let changed = walked(seed);

        assert_eq!(changed, 0, "seed {seed}: {changed} settled chunks differ");
    }
}
