//! What the block solver costs: per chunk, per batch, and across a world stitched out of chunks.
//!
//! ```text
//! cargo test -p wfc-gpu --release --test block_solver_bench -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Every test is `#[ignore]`d. A timing describes one build on one machine and driver stack; see
//! docs/solver-fit.md for what each number means. Correctness is asserted here too, because a
//! measurement of a wrong solver is worthless.

use std::sync::Arc;
use std::time::Instant;
use wfc_core::reference::ReferenceSolver;
use wfc_core::{
    BoundaryCondition, ChunkCoord, ChunkShape, ChunkStore, Domains, Prior, Region, RegionBatch,
    RegionShape, RegionStats, RegionStatus, Ruleset, Solver, WorldExtent, region_init,
};
use wfc_devtools::city::{self, City, city_prior};
use wfc_gpu::Params;
use wfc_gpu::block_solver::BlockSolver;
use wfc_gpu::kernel::SolverConfig;
use wfc_gpu::wgpu_backend::WgpuBackend;

/// The chunk the city is generated in: 8 cells of 2 m is a 16 m block, eight storeys tall, which is
/// marian42's scale.
const CHUNK: ChunkShape = ChunkShape::cube(8);

/// A city world being generated chunk by chunk.
struct World {
    city: City,
    ruleset: Arc<Ruleset>,
    prior: Prior,
    store: ChunkStore,
    solver: BlockSolver<WgpuBackend>,
    /// How long the last dispatch took, excluding the host work around it.
    last_wall_ms: f64,
}

/// What stitching a world produced.
struct Stitched {
    undecided_chunks: usize,
    violations: usize,
    repaired: usize,
}

impl World {
    /// An empty world of `chunks_x` by `chunks_y` city chunks.
    fn new(chunks_x: i32, chunks_y: i32, config: SolverConfig) -> Self {
        let city = city::city();
        let ruleset = Arc::new(Ruleset::from_modules(&city.modules).expect("the city compiles"));
        let prior = city_prior(&city, CHUNK.z);
        let extent = WorldExtent::new(CHUNK)
            .with_x(0..chunks_x)
            .with_y(0..chunks_y)
            .with_z(0..1);
        let backend = WgpuBackend::from_env().expect("a compute device");
        eprintln!("block_solver: device with {}", backend.describe());
        let solver = BlockSolver::new(backend, Arc::clone(&ruleset), config).expect("a solver");
        Self {
            city,
            ruleset,
            prior,
            store: ChunkStore::new(extent),
            solver,
            last_wall_ms: 0.0,
        }
    }

    fn region(&self, chunk: ChunkCoord, halo: u32) -> Region {
        Region::new(chunk, CHUNK.region(self.store.extent().halo(halo)))
    }

    /// Solves `chunks` in one dispatch and commits what succeeded. Chunks of one dispatch must not
    /// share a face, which the caller arranges.
    fn solve(
        &mut self,
        label: &str,
        chunks: &[ChunkCoord],
        halo: u32,
        release: bool,
        seed: u32,
    ) -> Vec<RegionStatus> {
        let regions: Vec<Region> = chunks.iter().map(|&c| self.region(c, halo)).collect();
        let shape = regions[0].shape();
        let mut init = Domains::from_words(0, self.ruleset.words_per_cell(), Vec::new())
            .expect("an empty batch");
        for region in &regions {
            let domains = region_init(&self.store, &self.prior, &self.ruleset, region, release);
            init.append(&domains);
        }
        let batch = RegionBatch {
            region: shape,
            ids: chunks.iter().map(|c| c.id()).collect(),
            seeds: vec![seed; chunks.len()],
            init: init.clone(),
        };

        let params = if release {
            // A repair is a fallback, not a search: when it cannot place a chunk quickly, widening
            // its halo is the better move, so it gets a fraction of the budget.
            Params {
                max_attempts: 8,
                max_steps: 5_000,
                ..Params::solve(self.solver.config())
            }
        } else {
            Params::solve(self.solver.config())
        };
        let started = Instant::now();
        let job = self
            .solver
            .start_with(batch, params)
            .expect("a well-formed batch");
        let result = self.solver.wait(job).expect("the dispatch finishes");
        self.last_wall_ms = started.elapsed().as_secs_f64() * 1000.0;

        let mut counts = [0usize; 4];
        for (index, (&chunk, region)) in chunks.iter().zip(&regions).enumerate() {
            let status = result.statuses[index];
            counts[status_index(status)] += 1;
            if status == RegionStatus::BorderContradiction {
                self.check_border_contradiction(label, chunk, region, &init, index, counts[3] == 1);
            }
            if status == RegionStatus::Solved {
                let domains = result.region(index, shape.cells());
                self.store
                    .commit(region, &domains, release)
                    .expect("a solved region is decided");
            }
        }
        eprintln!(
            "block_solver: {label}: {} chunks in {:.1} ms, \
             [solved, exhausted, step cap, border contradiction] = {counts:?}",
            chunks.len(),
            self.last_wall_ms
        );
        result.statuses
    }

    /// A border contradiction must be real: the CPU reference, propagating the same starting
    /// domains, has to empty a cell too.
    fn check_border_contradiction(
        &self,
        label: &str,
        chunk: ChunkCoord,
        region: &Region,
        init: &Domains,
        index: usize,
        first: bool,
    ) {
        let cells = region.shape().cells();
        let mut domains = init.chunk(index as u32, cells);
        let mut stack: Vec<u32> = (0..cells).collect();
        let emptied = ReferenceSolver::new(Arc::clone(&self.ruleset))
            .propagate(region.shape(), &mut domains, &mut stack)
            .expect_err("the kernel reports a border contradiction the CPU does not find");
        if first {
            let (at, inner) = region.cells().nth(emptied as usize).expect("in the region");
            eprintln!(
                "block_solver: {label}: chunk ({}, {}) first border contradiction at world {at:?}, {}",
                chunk.x,
                chunk.y,
                if inner {
                    "inside the chunk"
                } else {
                    "in the halo"
                }
            );
        }
    }

    /// Solves `chunks` in as few dispatches as the schedule allows, repairing what fails, and
    /// returns the time the dispatches took and how many chunks a repair rewrote.
    ///
    /// Chunks sharing a face cannot be in one dispatch, so each batch takes one parity of the chunk
    /// grid. A chunk that still fails is solved alone with its halo released, which may rewrite the
    /// neighbouring cells it covers, as in modifying in blocks.
    fn solve_batch(
        &mut self,
        label: &str,
        chunks: &[ChunkCoord],
        halo: u32,
        seed: u32,
    ) -> (f64, usize, Vec<ChunkCoord>) {
        let mut wall_ms = 0.0;
        let mut repaired = 0;
        let mut failed = Vec::new();
        for parity in [0, 1] {
            let batch: Vec<ChunkCoord> = chunks
                .iter()
                .copied()
                .filter(|c| c.parity() == parity)
                .collect();
            if batch.is_empty() {
                continue;
            }
            let statuses = self.solve(
                &format!("{label} parity {parity}"),
                &batch,
                halo,
                false,
                seed,
            );
            wall_ms += self.last_wall_ms;
            for (&chunk, _) in batch.iter().zip(&statuses).filter(|(_, s)| !s.is_solved()) {
                let mut recovered = false;
                for widened in self.repair_halos() {
                    let label = format!("{label} repair ({}, {}) halo {widened}", chunk.x, chunk.y);
                    let solved = self.solve(&label, &[chunk], widened, true, seed + widened)[0];
                    wall_ms += self.last_wall_ms;
                    if solved.is_solved() {
                        repaired += 1;
                        recovered = true;
                        break;
                    }
                }
                // A solve is a function of the chunk and its neighbours, so asking again would
                // fail the same way. The caller is told once and the chunk is left alone.
                if !recovered {
                    failed.push(chunk);
                }
            }
        }
        (wall_ms, repaired, failed)
    }

    /// Compiles every kernel a run will need, which is what a game would do when it loads: the
    /// first dispatch of a new specialisation otherwise pays for translating it to the device's
    /// own language, and that is seconds, not milliseconds.
    fn warm(&mut self, capacities: &[u32], halos: &[u32]) {
        let shapes: Vec<(u32, RegionShape)> = halos
            .iter()
            .flat_map(|&halo| {
                let region = CHUNK.region(self.store.extent().halo(halo));
                capacities.iter().map(move |&capacity| (capacity, region))
            })
            .filter(|(_, region)| self.solver.fits(*region))
            .collect();
        let started = Instant::now();
        self.solver.warm(&shapes).expect("the kernels compile");
        eprintln!(
            "block_solver: compiled {} kernels in {:.1} s",
            shapes.len(),
            started.elapsed().as_secs_f64()
        );
    }

    /// The halos a repair may widen to, in order. A wider region needs more workgroup memory, and
    /// past some width the device has none left, which the solver knows.
    fn repair_halos(&self) -> Vec<u32> {
        (1..=3)
            .filter(|halo| {
                self.solver
                    .fits(CHUNK.region(self.store.extent().halo(*halo)))
            })
            .collect()
    }

    /// Checks and renders the world; only pairs of decided cells can violate a rule.
    fn report(&self, name: &str, repaired: usize) -> Stitched {
        let chunks = self.store.extent().chunks();
        let undecided_chunks = chunks.iter().filter(|c| !self.store.contains(**c)).count();
        let (width, height, depth) = world_cells(&chunks);
        let decided = |x: usize, y: usize, z: usize| {
            self.store.tile([x as i32, y as i32, z as i32]).is_some()
        };
        let tiles: Vec<usize> = (0..depth)
            .flat_map(|z| {
                (0..height).flat_map(move |y| {
                    (0..width).map(move |x| {
                        self.store
                            .tile([x as i32, y as i32, z as i32])
                            .map_or(self.city.air, usize::from)
                    })
                })
            })
            .collect();
        let grid = wfc_devtools::TileGrid::new(width, height, depth, tiles).expect("dimensions");
        let violations = wfc_devtools::adjacency_violations(
            &grid,
            &self.city.modules.rules,
            BoundaryCondition::Finite,
        )
        .into_iter()
        .filter(|v| {
            decided(v.cell.0, v.cell.1, v.cell.2)
                && decided(v.neighbor.0, v.neighbor.1, v.neighbor.2)
        })
        .count();
        eprintln!(
            "block_solver: {name} world {width}x{height}x{depth}: undecided chunks \
             {undecided_chunks}, violations between decided cells {violations}, repaired {repaired}"
        );
        let path =
            std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}_city.png"));
        wfc_devtools::render::render_voxel_isometric(&grid, &self.city.voxels, 2)
            .save(&path)
            .expect("write PNG");
        eprintln!("block_solver: rendered {}", path.display());
        Stitched {
            undecided_chunks,
            violations,
            repaired,
        }
    }
}

fn status_index(status: RegionStatus) -> usize {
    match status {
        RegionStatus::Solved => 0,
        RegionStatus::Exhausted => 1,
        RegionStatus::StepCap => 2,
        RegionStatus::BorderContradiction => 3,
    }
}

/// The cell dimensions a list of chunks covers, counting from the origin.
fn world_cells(chunks: &[ChunkCoord]) -> (usize, usize, usize) {
    let extent = |axis: fn(&ChunkCoord) -> i32, cells: u32| {
        (chunks.iter().map(axis).max().unwrap_or(0) + 1) as usize * cells as usize
    };
    (
        extent(|c| c.x, CHUNK.x),
        extent(|c| c.y, CHUNK.y),
        CHUNK.z as usize,
    )
}

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

/// Chunks solved pass by pass stitch into a world. `halo` cells around each chunk are solved and
/// discarded, which is what keeps a chunk from leaving border tiles no row of neighbours can
/// complete; `repair` solves a failed chunk again with its halo released.
fn stitch_world(
    name: &str,
    halo: u32,
    repair: bool,
    passes: usize,
    pass_of: fn(ChunkCoord) -> usize,
) -> Stitched {
    let mut world = World::new(8, 8, SolverConfig::default());
    let chunks = world.store.extent().chunks();
    let mut repaired = 0;
    for pass in 0..passes {
        let members: Vec<ChunkCoord> = chunks
            .iter()
            .copied()
            .filter(|c| pass_of(*c) == pass)
            .collect();
        if members.is_empty() {
            continue;
        }
        let label = format!("{name} pass {pass}");
        // Chunks of one pass never share a face, so each pass is one dispatch.
        let statuses = world.solve(&label, &members, halo, false, 7 + pass as u32);
        if !repair {
            continue;
        }
        for (&chunk, _) in members
            .iter()
            .zip(&statuses)
            .filter(|(_, s)| !s.is_solved())
        {
            for widened in world.repair_halos() {
                let label = format!("{label} repair ({}, {}) halo {widened}", chunk.x, chunk.y);
                if world.solve(&label, &[chunk], widened, true, 1000 + widened)[0].is_solved() {
                    repaired += 1;
                    break;
                }
            }
        }
    }
    world.report(name, repaired)
}

/// A checkerboard without a halo: the second pass has every side face fixed, which is where chunks
/// are left with borders no row of neighbours can complete. What is decided is still valid.
#[test]
#[ignore = "benchmark; run with --ignored in release mode"]
fn a_checkerboard_schedule_never_breaks_a_seam() {
    let bare = stitch_world("checkerboard", 0, false, 2, |c| usize::from(c.parity()));

    assert_eq!(bare.violations, 0, "decided cells never violate the rules");
    eprintln!(
        "block_solver: a checkerboard without a halo left {} of 64 chunks unsolved",
        bare.undecided_chunks
    );
}

/// The same with a one-cell halo and repair: the world must come out complete.
#[test]
#[ignore = "benchmark; run with --ignored in release mode"]
fn a_checkerboard_schedule_with_repair_completes_the_world() {
    let stitched = stitch_world("checkerboard_repair", 1, true, 2, |c| {
        usize::from(c.parity())
    });

    assert_eq!(stitched.violations, 0);
    assert_eq!(
        stitched.undecided_chunks, 0,
        "{} chunks needed a repair",
        stitched.repaired
    );
}

/// Diagonal waves, as in N-WFC: chunks with the same x + y share no face, and every chunk has at
/// most two fixed faces.
#[test]
#[ignore = "benchmark; run with --ignored in release mode"]
fn a_diagonal_schedule_with_repair_completes_the_world() {
    let stitched = stitch_world("diagonal_repair", 1, true, 16, |c| (c.x + c.y) as usize);

    assert_eq!(stitched.violations, 0);
    assert_eq!(
        stitched.undecided_chunks, 0,
        "{} chunks needed a repair",
        stitched.repaired
    );
}

/// Can the city be generated live, in front of a walking player?
///
/// The player walks along a 24x8-chunk world; every tick, the chunks that have come within the view
/// radius are generated. An 8-cell chunk of 2 m blocks is 16 m, so a walking pace of 1.4 m/s
/// crosses one chunk every 11 s. Generation keeps up if the work a tick asks for fits in the tick.
#[test]
#[ignore = "benchmark; run with --ignored in release mode"]
fn live_streaming_keeps_ahead_of_a_walking_player() {
    const CELL_M: f64 = 2.0;
    const WALK_M_S: f64 = 1.4;
    const TICK_S: f64 = 0.5;
    const VIEW: i32 = 4;

    let (chunks_x, chunks_y) = (24, 8);
    let mut world = World::new(chunks_x, chunks_y, SolverConfig::default());
    let chunk_m = CELL_M * f64::from(CHUNK.x);
    let focus_y = chunks_y / 2;
    world.warm(&[1, 4, 8, 16, 32], &[1, 2, 3]);
    let repair_halos = world.repair_halos();
    eprintln!("block_solver: repairs may widen to halos {repair_halos:?}");

    let mut ticks: Vec<(f64, usize)> = Vec::new();
    let (mut repaired, mut generated) = (0, 0);
    let mut failed: std::collections::BTreeSet<ChunkCoord> = std::collections::BTreeSet::new();
    let mut focus_m = 0.0;
    while focus_m < f64::from(chunks_x - VIEW) * chunk_m {
        let focus_x = (focus_m / chunk_m) as i32;
        // Nearest first, as a streaming scheduler would order them.
        let mut wanted: Vec<(i32, ChunkCoord)> = ((focus_x - VIEW)..=(focus_x + VIEW))
            .flat_map(|x| ((focus_y - VIEW)..=(focus_y + VIEW)).map(move |y| (x, y)))
            .map(|(x, y)| ChunkCoord::new(x, y, 0))
            .filter(|c| world.store.extent().contains_chunk(*c))
            .filter(|c| !world.store.contains(*c) && !failed.contains(c))
            .map(|c| ((c.x - focus_x).abs().max((c.y - focus_y).abs()), c))
            .collect();
        wanted.sort_unstable();
        let missing: Vec<ChunkCoord> = wanted.into_iter().map(|(_, c)| c).collect();
        if !missing.is_empty() {
            let label = format!("live tick at {focus_m:.0} m");
            let (wall_ms, fixed, gave_up) = world.solve_batch(&label, &missing, 1, 11);
            ticks.push((wall_ms, missing.len()));
            generated += missing.len();
            repaired += fixed;
            failed.extend(gave_up);
        }
        focus_m += WALK_M_S * TICK_S;
    }

    let mut walls: Vec<f64> = ticks.iter().map(|(wall, _)| *wall).collect();
    walls.sort_by(f64::total_cmp);
    let busiest = ticks
        .iter()
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .expect("a tick");
    let total_ms: f64 = walls.iter().sum();
    let cells = generated * CHUNK.cells() as usize;
    eprintln!(
        "block_solver: live streaming across {chunks_x}x{chunks_y} chunks: {generated} chunks \
         ({cells} cells) in {total_ms:.0} ms of dispatches, {repaired} repaired; ticks needing \
         work {}, median {:.1} ms, p90 {:.1} ms, busiest {:.1} ms for {} chunks; budget {:.0} ms \
         per tick; {:.0} cells/s while generating; {} chunks could not be placed",
        ticks.len(),
        walls[walls.len() / 2],
        walls[walls.len() * 9 / 10],
        busiest.0,
        busiest.1,
        TICK_S * 1000.0,
        cells as f64 / (total_ms / 1000.0),
        failed.len(),
    );

    // Filling the first view is a load, not a step of play; every later tick must fit its budget.
    let worst_in_play = ticks[1..].iter().map(|(wall, _)| *wall).fold(0.0, f64::max);
    assert!(
        worst_in_play < TICK_S * 1000.0,
        "a tick needed {worst_in_play:.0} ms of a {:.0} ms budget",
        TICK_S * 1000.0
    );
    let stitched = world.report("live", repaired);
    assert_eq!(stitched.violations, 0, "what is generated is always valid");
    // A chunk whose borders no arrangement satisfies is a property of the module set, not of the
    // solver: the city's is not streaming-clean (docs/solver-fit.md). It stays rare.
    assert!(
        failed.len() * 50 < generated,
        "{} of {generated} chunks could not be placed: {failed:?}",
        failed.len()
    );
}
