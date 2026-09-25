//! A city evicted in part and generated again has no seam, repairs included.
//!
//! A repair rewrites the neighbours of the chunk it repairs (docs/architecture/world.md,
//! "Repairs"). If streaming dropped only part of such a neighbourhood, the chunks asked for again
//! would come back as their first attempts, beside neighbours the repair had fitted to them, and
//! leave a seam. This generates a city, drops everything further than a cut from its corner, asks
//! for the whole city again, and checks every adjacency in it, for every cut across it.

use wave_forge::{Builder, ChunkCoord, ChunkEvent, ChunkShape, FocusPoint, Ruleset, WorldExtent};
use wfc_devtools::city::{self, city_prior};
use wfc_devtools::{BoundaryCondition, TileGrid, adjacency_violations};

const CHUNK: ChunkShape = ChunkShape::cube(8);
const WIDE: i32 = 8;
const DEEP: i32 = 4;
const SEED: u64 = 8;

/// How many adjacencies across the whole generated city the rules do not allow.
fn seams(
    world: &wave_forge::WorldGenerator<impl wave_forge::Solver>,
    rules: &wave_forge::AdjacencyRules,
) -> usize {
    assert_eq!(
        world.store().len(),
        (WIDE * DEEP) as usize,
        "the whole city"
    );
    let (grid, _) = TileGrid::from_chunks(CHUNK, world.store().iter(), 0).expect("chunks");
    adjacency_violations(&grid, rules, BoundaryCondition::Finite).len()
}

#[test]
fn a_city_evicted_beyond_any_cut_comes_back_without_a_seam() {
    let city = city::city();
    let ruleset = Ruleset::from_modules(&city.modules).expect("the city compiles");
    let world = || {
        Builder::new(ruleset.clone(), city_prior(&city, CHUNK.z))
            .seed(SEED)
            .extent(
                WorldExtent::new(CHUNK)
                    .with_x(0..WIDE)
                    .with_y(0..DEEP)
                    .with_z(0..1),
            )
            .halo(1)
            .build()
            .expect("a compute device")
    };
    let everything = [FocusPoint::new(ChunkCoord::new(0, 0, 0), WIDE as u32)];
    let mut first = world();
    first.request(&everything);
    let events = first.run_until_idle().expect("the solver runs");
    assert_eq!(
        seams(&first, &city.modules.rules),
        0,
        "a seam before any eviction"
    );
    assert!(
        first.stats().rewritten_by_repair > 0,
        "seed {SEED} rewrites no neighbour, so nothing is tested"
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, ChunkEvent::Failed { .. })),
        "a chunk failed"
    );

    for cut in 1..WIDE {
        let mut streamed = world();
        streamed.request(&everything);
        streamed.run_until_idle().expect("the solver runs");
        // Everything further than `cut - 1` chunks from the corner, along either axis.
        let near = [FocusPoint::new(ChunkCoord::new(0, 0, 0), (cut - 1) as u32)];
        streamed.request(&near);
        streamed.evict_outside(&near, 0);
        streamed.request(&everything);
        streamed.run_until_idle().expect("the solver runs");

        let seams = seams(&streamed, &city.modules.rules);
        assert_eq!(
            seams, 0,
            "evicted beyond {cut} chunks from the corner, {seams} seams"
        );
    }
}
