//! What one chunk costs the block kernel, against what it costs a CPU.
//!
//! ```text
//! cargo test -p wfc-gpu --release --test block_solver_bench -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Every test is `#[ignore]`d. A timing describes one build on one machine and driver stack; see
//! docs/solver-fit.md for what each number means. Correctness is asserted here too, because a
//! measurement of a wrong solver is worthless. Whole worlds are measured through the library
//! instead, in `wfc-devtools/tests/streaming.rs`.

use std::sync::Arc;
use std::time::Instant;
use wfc_core::reference::ReferenceSolver;
use wfc_core::{
    ChunkCoord, ChunkShape, ChunkStore, Domains, Region, RegionBatch, RegionStats, Ruleset, Solver,
    WorldExtent, region_init,
};
use wfc_devtools::city::{self, city_prior};
use wfc_gpu::Params;
use wfc_gpu::block_solver::BlockSolver;
use wfc_gpu::kernel::SolverConfig;
use wfc_gpu::wgpu_backend::WgpuBackend;

/// The chunk the city is generated in: 8 cells of 2 m is a 16 m block, eight storeys tall, which is
/// marian42's scale.
const CHUNK: ChunkShape = ChunkShape::cube(8);

/// Median of `samples`, which must not be empty.
fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

/// Sums a statistic over a batch's regions.
fn total(stats: &[RegionStats], of: fn(&RegionStats) -> u32) -> u32 {
    stats.iter().map(of).sum()
}

/// One chunk with free faces, the region every solver in the throughput test sees.
fn lone_chunk() -> (Arc<Ruleset>, Region, Domains) {
    let city = city::city();
    let ruleset = Arc::new(Ruleset::from_modules(&city.modules).expect("the city compiles"));
    let prior = city_prior(&city, CHUNK.z);
    let extent = WorldExtent::new(CHUNK)
        .with_x(0..1)
        .with_y(0..1)
        .with_z(0..1);
    let region = Region::new(ChunkCoord::new(0, 0, 0), CHUNK.region([0, 0, 0]));
    let init = region_init(&ChunkStore::new(extent), &prior, &ruleset, &region, false);
    (ruleset, region, init)
}

/// The same region repeated across a batch, with one id per region.
fn repeated(region: &Region, init: &Domains, regions: u32, seed: u32) -> RegionBatch {
    let mut domains = init.clone();
    for _ in 1..regions {
        let more = init.clone();
        domains.append(&more);
    }
    RegionBatch {
        region: region.shape(),
        ids: (0..regions).collect(),
        seeds: vec![seed; regions as usize],
        init: domains,
        budget: None,
        portfolio: false,
    }
}

/// How chunk throughput scales with the chunks in one dispatch, how the invocation count and the
/// selection radius change it, and what the same chunk costs one CPU thread and every CPU thread.
#[test]
#[ignore = "benchmark; run with --ignored in release mode"]
fn chunk_throughput_against_the_cpu_reference() {
    let (ruleset, region, init) = lone_chunk();
    let shape = region.shape();

    // The CPU yardstick: sixteen seeds of the same chunk on one thread, after a warm-up.
    let cpu = ReferenceSolver::new(Arc::clone(&ruleset));
    let _ = cpu.solve_region(shape, &init, 0, 0);
    let cpu_ms = median(
        (1..=16)
            .map(|seed| {
                let started = Instant::now();
                let _ = cpu.solve_region(shape, &init, 0, seed);
                started.elapsed().as_secs_f64() * 1000.0
            })
            .collect(),
    );
    eprintln!(
        "block_solver: cpu reference {cpu_ms:.3} ms per chunk (median of 16 seeds, one thread)"
    );
    // The same chunks over every hardware thread, which is what a dispatch competes with.
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let all_cores_ms = median(
        (0..3)
            .map(|_| {
                let started = Instant::now();
                std::thread::scope(|scope| {
                    for thread in 0..threads {
                        let (cpu, init) = (&cpu, &init);
                        scope.spawn(move || {
                            for seed in (0..256u32).filter(|s| *s as usize % threads == thread) {
                                let _ = cpu.solve_region(shape, init, 0, seed);
                            }
                        });
                    }
                });
                started.elapsed().as_secs_f64() * 1000.0
            })
            .collect(),
    );
    eprintln!(
        "block_solver: cpu reference on {threads} threads: 256 chunks in {all_cores_ms:.1} ms, \
         {:.3} ms per chunk (median of 3)",
        all_cores_ms / 256.0
    );

    for (invocations, radius) in [(64u32, 1u32), (256, 0), (256, 1), (256, 2)] {
        let config = SolverConfig {
            invocations,
            radius,
            ..SolverConfig::default()
        };
        let backend = WgpuBackend::from_env().expect("a compute device");
        let mut solver = BlockSolver::new(backend, Arc::clone(&ruleset), config).expect("a solver");
        // Windows resets the device when one dispatch runs for about two seconds, and a reset takes
        // the host's display driver with it. Chunk counts grow by four, so a dispatch is only
        // attempted while four times the last one stays well inside that.
        let mut previous_ms = 0.0;
        for chunks in [1u32, 4, 16, 64, 256] {
            if previous_ms * 4.0 > 600.0 {
                eprintln!(
                    "block_solver: invocations={invocations} radius={radius} chunks={chunks} \
                     skipped: {previous_ms:.0} ms at a quarter of the chunks risks the timeout"
                );
                break;
            }
            // Pipeline creation and a cold device are not what is being measured.
            for _ in 0..3 {
                let job = solver
                    .start(repeated(&region, &init, chunks, 7))
                    .expect("a well-formed batch");
                let _ = solver.wait(job).expect("the dispatch finishes");
            }
            let mut samples = Vec::new();
            let mut last = None;
            for _ in 0..5 {
                let started = Instant::now();
                let job = solver
                    .start(repeated(&region, &init, chunks, 7))
                    .expect("a well-formed batch");
                let result = solver.wait(job).expect("the dispatch finishes");
                samples.push(started.elapsed().as_secs_f64() * 1000.0);
                last = Some(result);
            }
            let wall_ms = median(samples);
            previous_ms = wall_ms;
            let result = last.expect("five samples");

            let failed = result.statuses.iter().filter(|s| !s.is_solved()).count();
            for (index, status) in result.statuses.iter().enumerate() {
                assert!(
                    !status.is_solved() || result.region(index, shape.cells()).all_decided(),
                    "a solved region must be decided"
                );
            }
            let collapses = total(&result.stats, |s| s.collapses);
            let sweeps = total(&result.stats, |s| s.sweeps);
            let mut restarts: Vec<u32> = result.stats.iter().map(|s| s.restarts).collect();
            restarts.sort_unstable();
            let max_steps = result
                .stats
                .iter()
                .map(|s| s.steps)
                .max()
                .expect("a region");
            let cells = f64::from(chunks * shape.cells());
            eprintln!(
                "block_solver: invocations={invocations} radius={radius} chunks={chunks} \
                 wall_ms={wall_ms:.2} ms_per_chunk={:.3} cells_per_s={:.0} vs_cpu_thread={:.2}x \
                 failed={failed} sweeps_per_collapse={:.2} restarts[median,max]=[{},{}] \
                 max_steps={max_steps} us_per_step_of_slowest={:.1}",
                wall_ms / f64::from(chunks),
                cells / (wall_ms / 1000.0),
                cpu_ms * f64::from(chunks) / wall_ms,
                f64::from(sweeps) / f64::from(collapses.max(1)),
                restarts[restarts.len() / 2],
                restarts[restarts.len() - 1],
                wall_ms * 1000.0 / f64::from(max_steps),
            );
        }
    }
}

/// Every chunk of a many-chunk dispatch that reports success is valid, in every selection and
/// recovery mode. One chunk can hide a rare failure; 64 different random streams rarely do.
#[test]
#[ignore = "benchmark; run with --ignored in release mode"]
fn every_reported_success_is_a_valid_chunk() {
    let (ruleset, region, init) = lone_chunk();
    let shape = region.shape();
    let chunks = 64u32;

    for (radius, undo) in [(0u32, 0u32), (2, 0), (0, 1), (2, 1)] {
        let config = SolverConfig {
            radius,
            ..SolverConfig::default()
        };
        let backend = WgpuBackend::from_env().expect("a compute device");
        let mut solver = BlockSolver::new(backend, Arc::clone(&ruleset), config).expect("a solver");
        let params = Params {
            undo,
            ..Params::solve(&config)
        };

        let job = solver
            .start_with(repeated(&region, &init, chunks, 7), params)
            .expect("a well-formed batch");
        let result = solver.wait(job).expect("the dispatch finishes");

        let invalid: Vec<(usize, u32, u32)> = result
            .statuses
            .iter()
            .enumerate()
            .filter(|(_, status)| status.is_solved())
            .filter_map(|(index, _)| {
                let region = result.region(index, shape.cells());
                let empty = (0..region.cells())
                    .filter(|c| region.count(*c) == 0)
                    .count() as u32;
                let open = (0..region.cells()).filter(|c| region.count(*c) > 1).count() as u32;
                (empty + open > 0).then_some((index, empty, open))
            })
            .collect();
        let solved = result.statuses.iter().filter(|s| s.is_solved()).count();
        eprintln!(
            "block_solver: radius={radius} undo={undo}: {solved} of {chunks} solved, \
             invalid among them {} {:?}",
            invalid.len(),
            invalid.first()
        );
        assert!(invalid.is_empty(), "radius {radius}, undo {undo}");
    }
}
