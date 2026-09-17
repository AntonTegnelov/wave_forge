//! Can a city be generated through the library while a player walks through it?
//!
//! ```text
//! cargo test -p wfc-devtools --release --test streaming -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Both tests are `#[ignore]`d: they need a compute device and they are measurements, not unit
//! tests. A timing describes one build on one machine and driver stack; docs/solver-fit.md records
//! what each number means. Validity is asserted here too, because a measurement of a wrong world
//! is worthless.

use std::collections::BTreeSet;
use std::time::Instant;
use wave_forge::{
    BlockSolver, Builder, ChunkCoord, ChunkEvent, ChunkShape, FocusPoint, RegionShape, Ruleset,
    WgpuBackend, WorldExtent, WorldGenerator,
};
use wfc_core::BoundaryCondition;
use wfc_devtools::city::{self, City, city_prior};
use wfc_devtools::{TileGrid, adjacency_violations};

/// The chunk the city is generated in: 8 cells of 2 m is a 16 m block, eight storeys tall, which is
/// marian42's scale.
const CHUNK: ChunkShape = ChunkShape::cube(8);

type CityWorld = WorldGenerator<BlockSolver<WgpuBackend>>;

/// A city world of `chunks_x` by `chunks_y` chunks on a device of its own.
fn city_world(chunks_x: i32, chunks_y: i32) -> (City, CityWorld) {
    let city = city::city();
    let ruleset = Ruleset::from_modules(&city.modules).expect("the city compiles");
    let world = Builder::new(ruleset, city_prior(&city, CHUNK.z))
        .seed(11)
        .extent(
            WorldExtent::new(CHUNK)
                .with_x(0..chunks_x)
                .with_y(0..chunks_y)
                .with_z(0..1),
        )
        .halo(1)
        .build()
        .expect("a compute device");
    (city, world)
}

/// Compiles every kernel a run will need, which is what a game would do when it loads: the first
/// dispatch of a new specialisation otherwise pays for translating it to the device's own language,
/// and that is seconds, not milliseconds.
fn warm(world: &mut CityWorld, capacities: &[u32]) {
    let config = world.config().clone();
    let solver = world.solver_mut();
    let shapes: Vec<(u32, RegionShape)> = (1..=config.repair.max_halo)
        .map(|halo| config.chunk.region(config.extent.halo(halo)))
        .filter(|region| solver.fits(*region))
        .flat_map(|region| capacities.iter().map(move |&capacity| (capacity, region)))
        .collect();
    let started = Instant::now();
    solver.warm(&shapes).expect("the kernels compile");
    eprintln!(
        "streaming: compiled {} kernels in {:.1} s",
        shapes.len(),
        started.elapsed().as_secs_f64()
    );
}

/// What a generated world looks like from the outside: undecided chunks, and rule violations
/// between cells that were decided. Renders the world as a voxel model as well.
fn report(name: &str, city: &City, world: &CityWorld) -> (usize, usize) {
    let chunks = world.config().extent.chunks();
    let undecided = chunks
        .iter()
        .filter(|chunk| world.chunk(**chunk).is_none())
        .count();
    let (width, height, depth) = (
        (chunks.iter().map(|c| c.x).max().unwrap_or(0) + 1) as usize * CHUNK.x as usize,
        (chunks.iter().map(|c| c.y).max().unwrap_or(0) + 1) as usize * CHUNK.y as usize,
        CHUNK.z as usize,
    );
    let store = world.store();
    let decided =
        |x: usize, y: usize, z: usize| store.tile([x as i32, y as i32, z as i32]).is_some();
    let tiles: Vec<usize> = (0..depth)
        .flat_map(|z| {
            (0..height).flat_map(move |y| {
                (0..width).map(move |x| {
                    store
                        .tile([x as i32, y as i32, z as i32])
                        .map_or(city.air, usize::from)
                })
            })
        })
        .collect();
    let grid = TileGrid::new(width, height, depth, tiles).expect("the world's dimensions");
    let violations = adjacency_violations(&grid, &city.modules.rules, BoundaryCondition::Finite)
        .into_iter()
        .filter(|violation| {
            decided(violation.cell.0, violation.cell.1, violation.cell.2)
                && decided(
                    violation.neighbor.0,
                    violation.neighbor.1,
                    violation.neighbor.2,
                )
        })
        .count();
    let stats = world.stats();
    eprintln!(
        "streaming: {name} world {width}x{height}x{depth}: undecided chunks {undecided}, \
         violations between decided cells {violations}, {stats:?}"
    );
    let path = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}.png"));
    wfc_devtools::render::render_voxel_isometric(&grid, &city.voxels, 2)
        .save(&path)
        .expect("write PNG");
    eprintln!("streaming: rendered {}", path.display());
    (undecided, violations)
}

/// Asked for all at once, the chunks of a world stitch into one city: the generator solves a parity
/// of the chunk lattice at a time, one dispatch per parity, and repairs what fixed borders left
/// unsolvable.
#[test]
#[ignore = "needs a compute device; run with --ignored in release mode"]
fn a_world_asked_for_at_once_comes_out_seamless() {
    let (city, mut world) = city_world(8, 8);
    warm(&mut world, &[1, 8, 32]);

    let started = Instant::now();
    world.request(&[FocusPoint::new(ChunkCoord::new(4, 4, 0), 4)]);
    let events = world.run_until_idle().expect("the solver runs");
    let wall_s = started.elapsed().as_secs_f64();

    let failed: Vec<&ChunkEvent> = events
        .iter()
        .filter(|event| matches!(event, ChunkEvent::Failed { .. }))
        .collect();
    eprintln!(
        "streaming: 64 chunks in {wall_s:.2} s, {} events, {} chunks given up on: {:?}",
        events.len(),
        failed.len(),
        failed.first()
    );
    let (undecided, violations) = report("stitched", &city, &world);

    assert_eq!(violations, 0, "decided cells never violate the rules");
    assert_eq!(undecided, failed.len(), "only what was reported is missing");
    // A chunk whose borders no arrangement satisfies is a property of the module set, not of the
    // solver: the city's is not streaming-clean (docs/solver-fit.md). It stays rare.
    assert!(
        failed.len() * 10 < 64,
        "{} of 64 chunks: {failed:?}",
        failed.len()
    );
}

/// Can the city be generated live, in front of a walking player?
///
/// The player walks along a 24x8-chunk world; every tick, the chunks that have come within the view
/// radius are asked for and generated. An 8-cell chunk of 2 m blocks is 16 m, so a walking pace of
/// 1.4 m/s crosses one chunk every 11 s. Generation keeps up if the work a tick asks for fits in
/// the tick.
#[test]
#[ignore = "needs a compute device; run with --ignored in release mode"]
fn live_streaming_keeps_ahead_of_a_walking_player() {
    const CELL_M: f64 = 2.0;
    const WALK_M_S: f64 = 1.4;
    const TICK_S: f64 = 0.5;
    const VIEW: u32 = 4;

    let (chunks_x, chunks_y) = (24, 8);
    let (city, mut world) = city_world(chunks_x, chunks_y);
    warm(&mut world, &[1, 4, 8, 16, 32]);
    let chunk_m = CELL_M * f64::from(CHUNK.x);
    let focus_y = chunks_y / 2;

    let mut ticks: Vec<(f64, usize)> = Vec::new();
    let mut failed: BTreeSet<ChunkCoord> = BTreeSet::new();
    let mut focus_m = 0.0;
    while focus_m < f64::from(chunks_x - VIEW as i32) * chunk_m {
        let focus = ChunkCoord::new((focus_m / chunk_m) as i32, focus_y, 0);
        world.request(&[FocusPoint::new(focus, VIEW)]);
        let wanted = world.pending_chunks();
        if wanted > 0 {
            let started = Instant::now();
            let events = world.run_until_idle().expect("the solver runs");
            ticks.push((started.elapsed().as_secs_f64() * 1000.0, wanted));
            failed.extend(events.iter().filter_map(|event| match event {
                ChunkEvent::Failed { chunk, .. } => Some(*chunk),
                ChunkEvent::Updated(_) => None,
            }));
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
    let stats = *world.stats();
    let cells = stats.solved as usize * CHUNK.cells() as usize;
    eprintln!(
        "streaming: live across {chunks_x}x{chunks_y} chunks: {stats:?}; ticks needing work {}, \
         median {:.1} ms, p90 {:.1} ms, busiest {:.1} ms for {} chunks; budget {:.0} ms per tick; \
         {:.0} cells/s while generating; {} chunks could not be placed",
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
    let (_, violations) = report("live", &city, &world);
    assert_eq!(violations, 0, "what is generated is always valid");
    // A chunk whose borders no arrangement satisfies is a property of the module set, not of the
    // solver: the city's is not streaming-clean, and every chunk of the second parity is solved
    // against four fixed faces so that its tiles do not depend on where the player came from. That
    // costs placements; 3.2% of chunks on this build (docs/solver-fit.md).
    assert!(
        failed.len() * 20 < stats.solved as usize,
        "{} of {} chunks could not be placed: {failed:?}",
        failed.len(),
        stats.solved
    );
}
