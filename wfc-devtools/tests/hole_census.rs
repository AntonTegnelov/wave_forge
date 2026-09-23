//! Why chunks of the city cannot be placed: a census of every chunk a streamed world gave up on.
//!
//! ```text
//! cargo test -p wfc-devtools --release --test hole_census -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `#[ignore]`d: it needs a compute device, and it is a measurement rather than a contract. It asks
//! for a large unbounded city at once, then takes every chunk the generator reported as failed and
//! solves that chunk's exact problem again, from the tiles its neighbours ended with, in ways the
//! generator did not try: with a larger budget, with more seeds, with its neighbours' cells fixed or
//! released, at every halo the device fits. What still fails with every seed and a large budget is
//! a border no arrangement satisfies; what solves is a solver limit. Where a failing solve found its
//! contradiction says which part of a chunk the rule set cannot complete.

mod kernels;

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::{
    BlockSolver, Builder, ChunkCoord, ChunkEvent, ChunkShape, Domains, FocusPoint, Region,
    RegionBatch, RegionStatus, Ruleset, SolveBudget, Solver, SolverConfig, WgpuBackend,
    WorldExtent,
};
use wfc_core::region_init;
use wfc_devtools::city::{self, city_prior};

const CHUNK: ChunkShape = ChunkShape::cube(8);

/// Seeds each problem is tried with. A border that fails for every one of them is very likely one
/// no arrangement satisfies.
const SEEDS: u32 = 32;

#[test]
#[ignore = "needs a compute device; run with --ignored in release mode"]
fn census_of_the_chunks_a_city_gives_up_on() {
    let city = city::city();
    let ruleset = Ruleset::from_modules(&city.modules).expect("the city compiles");
    let prior = city_prior(&city, CHUNK.z);
    let extent = WorldExtent::new(CHUNK).with_z(0..1);
    let mut world = Builder::new(ruleset.clone(), prior.clone())
        .seed(11)
        .extent(extent.clone())
        .halo(1)
        .build()
        .expect("a compute device");
    kernels::warm(&mut world, &[1, 64, 128]);

    world.request(&[FocusPoint::new(ChunkCoord::new(0, 0, 0), 8)]);
    let events = world.run_until_idle().expect("the solver runs");
    let failed: Vec<(ChunkCoord, RegionStatus)> = events
        .iter()
        .filter_map(|event| match event {
            ChunkEvent::Failed { chunk, status } => Some((*chunk, *status)),
            _ => None,
        })
        .collect();
    eprintln!(
        "census: {} chunks generated, {} given up on, {:?}",
        world.store().len(),
        failed.len(),
        world.stats()
    );
    let mut by_status: BTreeMap<String, u32> = BTreeMap::new();
    for (_, status) in &failed {
        *by_status.entry(format!("{status:?}")).or_default() += 1;
    }
    eprintln!("census: the generator's last word on them: {by_status:?}");

    let mut solver = BlockSolver::new(
        WgpuBackend::from_env().expect("a compute device"),
        Arc::new(ruleset.clone()),
        SolverConfig::default(),
    )
    .expect("a solver");
    let store = world.store();
    // What a repair could be: the failed chunk with its neighbours' cells released, SEEDS seeds side
    // by side in one dispatch, the lowest seed that solves winning. Per budget: how many chunks it
    // places, how many seeds that took, and how long the dispatch is, since the slowest seed sets it.
    let halo = extent.halo(1);
    let shape = CHUNK.region(halo);
    for (attempts, steps) in [(8, 5_000), (16, 10_000), (32, 20_000), (64, 50_000)] {
        let budget = SolveBudget {
            max_attempts: attempts,
            max_steps: steps,
        };
        let mut placed = 0;
        let mut seeds_needed: Vec<u32> = Vec::new();
        let mut dispatch_ms: Vec<f64> = Vec::new();
        for &(chunk, _) in &failed {
            let region = Region::new(chunk, shape);
            let one = region_init(store, &prior, &ruleset, &region, true);
            let mut init =
                Domains::from_words(0, one.words_per_cell(), Vec::new()).expect("an empty batch");
            for _ in 0..SEEDS {
                init.append(&one);
            }
            let started = std::time::Instant::now();
            let job = solver
                .start(RegionBatch {
                    region: shape,
                    ids: vec![chunk.id(); SEEDS as usize],
                    seeds: (0..SEEDS).map(|seed| seed * 7919 + 1).collect(),
                    init,
                    budget: Some(budget),
                })
                .expect("a well-formed batch");
            let result = solver.wait(job).expect("the dispatch finishes");
            dispatch_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            if let Some(first) = result
                .statuses
                .iter()
                .position(|s| *s == RegionStatus::Solved)
            {
                placed += 1;
                seeds_needed.push(first as u32 + 1);
            }
        }
        seeds_needed.sort_unstable();
        dispatch_ms.sort_by(f64::total_cmp);
        eprintln!(
            "census: portfolio of {SEEDS} at halo 1 released, budget {attempts} attempts / {steps} steps: \
             {placed} of {} placed; seeds needed {seeds_needed:?}; dispatch ms median {:.0}, max {:.0}",
            failed.len(),
            dispatch_ms[dispatch_ms.len() / 2],
            dispatch_ms[dispatch_ms.len() - 1],
        );
    }
}
