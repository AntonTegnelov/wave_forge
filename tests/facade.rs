//! What the facade promises, checked on the CPU reference solver so no device is needed.
//!
//! The determinism contract in the crate documentation is the subject here: the same requests give
//! the same world, the order they arrive in does not matter, and a repair says which chunks it
//! rewrote. The two solvers that are not the reference are test doubles at the [`Solver`] seam,
//! which is where a Godot or Bevy backend will sit too.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use wave_forge::{
    BatchResult, Builder, ChunkCoord, ChunkEvent, ChunkShape, Domains, FocusPoint, JobId, Prior,
    Region, RegionBatch, RegionShape, RegionStats, RegionStatus, RepairPolicy, Ruleset,
    SolveBudget, Solver, SolverError, TileMask, Worker, WorldExtent, WorldGenerator,
};
use wfc_core::reference::ReferenceSolver;

const TILES: u32 = 3;
const CHUNK: ChunkShape = ChunkShape::cube(4);

/// Three tiles where anything may sit beside anything except 0 beside 2. Loose enough that every
/// chunk solves, tight enough that propagation has something to do.
fn ruleset() -> Ruleset {
    let allowed = (0..6).flat_map(|axis| {
        (0..TILES as usize).flat_map(move |a| (0..TILES as usize).map(move |b| (axis, a, b)))
    });
    let rules = wfc_rules::AdjacencyRules::from_allowed_tuples(
        TILES as usize,
        6,
        allowed.filter(|&(_, a, b)| a.abs_diff(b) != 2),
    );
    Ruleset::new(&rules, &[3.0, 1.0, 2.0]).expect("a rule set of three tiles")
}

fn extent() -> WorldExtent {
    WorldExtent::new(CHUNK)
        .with_x(0..3)
        .with_y(0..3)
        .with_z(0..1)
}

fn world(repairs: bool) -> WorldGenerator<ReferenceSolver> {
    let ruleset = ruleset();
    let solver = ReferenceSolver::new(Arc::new(ruleset.clone()));
    generator(ruleset, repairs).build_with(solver)
}

fn generator(ruleset: Ruleset, repairs: bool) -> Builder {
    Builder::new(ruleset, Prior::open(TILES))
        .seed(7)
        .extent(extent())
        .halo(1)
        .repair(RepairPolicy {
            enabled: repairs,
            ..RepairPolicy::default()
        })
}

/// Every generated chunk's tiles, so two worlds can be compared cell for cell.
fn tiles<S: Solver>(world: &WorldGenerator<S>) -> BTreeMap<ChunkCoord, Vec<u16>> {
    world
        .store()
        .iter()
        .map(|chunk| (chunk.coord, chunk.tiles.to_vec()))
        .collect()
}

#[test]
fn the_same_requests_generate_the_same_world() {
    let focus = [FocusPoint::new(ChunkCoord::new(1, 1, 0), 1)];

    let mut first = world(false);
    first.request(&focus);
    let events = first.run_until_idle().expect("the reference solves");
    let mut second = world(false);
    second.request(&focus);
    second.run_until_idle().expect("the reference solves");

    assert_eq!(tiles(&first), tiles(&second));
    assert_eq!(first.store().len(), 9, "the whole 3x3 world");
    assert!(
        events
            .iter()
            .all(|event| matches!(event, ChunkEvent::Updated(_))),
        "{events:?}"
    );
}

#[test]
fn the_order_the_chunks_are_asked_for_does_not_change_them() {
    let mut at_once = world(false);
    at_once.request(&[FocusPoint::new(ChunkCoord::new(1, 1, 0), 1)]);
    at_once.run_until_idle().expect("the reference solves");

    // The same nine chunks, asked for a column at a time as a player would walk into them.
    let mut walked = world(false);
    for x in 0..3 {
        walked.request(&[FocusPoint::new(ChunkCoord::new(x, 1, 0), 1)]);
        walked.run_until_idle().expect("the reference solves");
    }

    assert_eq!(tiles(&at_once), tiles(&walked));
}

#[test]
fn a_chunk_comes_back_the_same_when_its_neighbours_went_with_it() {
    let focus = [FocusPoint::new(ChunkCoord::new(1, 1, 0), 1)];
    let mut world = world(false);
    world.request(&focus);
    world.run_until_idle().expect("the reference solves");
    let before = tiles(&world);

    let evicted = world.evict_outside(&[FocusPoint::new(ChunkCoord::new(0, 0, 0), 0)], 0);
    world.request(&focus);
    world.run_until_idle().expect("the reference solves");

    assert_eq!(evicted.len(), 8, "everything but the chunk left in focus");
    assert_eq!(tiles(&world), before);
}

#[test]
fn a_worker_generates_on_a_thread_of_its_own() {
    let mut direct = world(false);
    direct.request(&[FocusPoint::new(ChunkCoord::new(1, 1, 0), 1)]);
    direct.run_until_idle().expect("the reference solves");

    let mut worker = Worker::spawn(|| Ok(world(false)));
    worker.request(&[FocusPoint::new(ChunkCoord::new(1, 1, 0), 1)]);
    let mut updated = BTreeSet::new();
    let deadline = Instant::now() + Duration::from_secs(30);
    while updated.len() < 9 {
        assert!(
            Instant::now() < deadline,
            "only {} chunks after 30 s: {:?}",
            updated.len(),
            worker.failure()
        );
        for event in worker.drain() {
            match event {
                ChunkEvent::Updated(chunk) => {
                    updated.insert(chunk);
                }
                ChunkEvent::Evicted(chunk) => panic!("nothing was evicted: {chunk:?}"),
                ChunkEvent::Failed { chunk, status } => panic!("{chunk:?}: {status:?}"),
            }
        }
        std::thread::yield_now();
    }

    for (coord, tiles) in tiles(&direct) {
        let from_worker = worker.chunk(coord).expect("the worker reported it");
        assert_eq!(from_worker.tiles.to_vec(), tiles, "{coord:?}");
    }
}

/// The chunk the scripted solver refuses to place on its first attempt.
const STUBBORN: ChunkCoord = ChunkCoord::new(1, 0, 0);

/// A solver that reports a border contradiction the first time it sees [`STUBBORN`] and paints
/// every cell it is handed afterwards.
///
/// A rule set rigid enough to make a real chunk fail is also rigid enough that no halo of one can
/// repair it, so what a repair reports is pinned here at the seam instead. The city rule set
/// exercises the real thing (`wfc-devtools/tests/streaming.rs`).
struct Scripted {
    tile: u32,
    seen_stubborn: bool,
    next_job: u64,
    finished: Option<(JobId, BatchResult)>,
    /// The chunk ids of every batch, in order, with whether the batch was a repair.
    batches: Vec<(Vec<u32>, bool)>,
    /// How many regions of which shape every batch held.
    shapes: Vec<(u32, RegionShape)>,
}

impl Solver for Scripted {
    fn max_batch(&self) -> u32 {
        64
    }

    fn start(&mut self, batch: RegionBatch) -> Result<JobId, SolverError> {
        assert!(self.finished.is_none(), "one batch at a time");
        let repairing = batch.budget.is_some();
        self.batches.push((batch.ids.clone(), repairing));
        self.shapes.push((batch.len() as u32, batch.region));
        // A repair releases the halo, so this is what paints over a neighbour's cells.
        if repairing {
            self.tile += 1;
        }
        let mut statuses = Vec::new();
        let mut domains =
            Domains::from_words(0, batch.init.words_per_cell(), Vec::new()).expect("empty");
        for &id in &batch.ids {
            let stubborn = id == STUBBORN.id();
            let refuse = stubborn && !self.seen_stubborn;
            self.seen_stubborn |= stubborn;
            statuses.push(if refuse {
                RegionStatus::BorderContradiction
            } else {
                RegionStatus::Solved
            });
            let painted = (0..batch.region.cells()).map(|_| TileMask::single(self.tile));
            domains.append(&Domains::from_masks(batch.init.words_per_cell(), painted));
        }
        let job = JobId(self.next_job);
        self.next_job += 1;
        self.finished = Some((
            job,
            BatchResult {
                statuses,
                stats: vec![RegionStats::default(); batch.len()],
                domains,
            },
        ));
        Ok(job)
    }

    fn poll(&mut self, job: JobId) -> Result<Option<BatchResult>, SolverError> {
        match self.finished.take() {
            Some((finished, result)) if finished == job => Ok(Some(result)),
            other => {
                self.finished = other;
                Err(SolverError::UnknownJob(job))
            }
        }
    }

    fn wait(&mut self, job: JobId) -> Result<BatchResult, SolverError> {
        self.poll(job)?.ok_or(SolverError::UnknownJob(job))
    }
}

fn scripted() -> WorldGenerator<Scripted> {
    generator(ruleset(), true).build_with(Scripted {
        tile: 0,
        seen_stubborn: false,
        next_job: 1,
        finished: None,
        batches: Vec::new(),
        shapes: Vec::new(),
    })
}

#[test]
fn every_batch_a_run_dispatches_has_a_kernel_to_warm() {
    let mut world = scripted();
    let radius = 1;
    let warmed = world.kernel_shapes(radius);

    world.request(&[FocusPoint::new(ChunkCoord::new(1, 1, 0), radius)]);
    world.run_until_idle().expect("the scripted solver");

    // A kernel is specialised per shape and per batch size rounded up to a power of two.
    let capacity = |regions: u32| regions.next_power_of_two();
    for &(regions, shape) in &world.solver().shapes {
        assert!(
            warmed
                .iter()
                .any(|&(warm, warm_shape)| warm_shape == shape
                    && capacity(warm) == capacity(regions)),
            "a batch of {regions} regions of {shape:?} was not in {warmed:?}"
        );
    }
    assert!(
        world.solver().batches.iter().any(|(_, repair)| *repair),
        "the run included a repair"
    );
}

#[test]
fn a_repair_reports_every_chunk_it_rewrote() {
    let mut world = scripted();

    world.request(&[FocusPoint::new(ChunkCoord::new(1, 1, 0), 1)]);
    let events = world.run_until_idle().expect("the scripted solver");

    // The repair solves the region again with its halo released, so it rewrites cells in every
    // chunk the halo reaches, the diagonal ones included.
    let region = Region::new(STUBBORN, CHUNK.region(extent().halo(1)));
    let rewritten: BTreeSet<ChunkCoord> = region
        .cells()
        .map(|(at, _)| ChunkCoord::of_cell(at, CHUNK))
        .filter(|chunk| extent().contains_chunk(*chunk) && *chunk != STUBBORN)
        .collect();
    let mut reported: BTreeMap<ChunkCoord, usize> = BTreeMap::new();
    for event in &events {
        match event {
            ChunkEvent::Updated(chunk) => *reported.entry(*chunk).or_default() += 1,
            ChunkEvent::Evicted(chunk) => panic!("nothing was evicted: {chunk:?}"),
            ChunkEvent::Failed { chunk, status } => panic!("{chunk:?}: {status:?}"),
        }
    }

    assert_eq!(
        reported.get(&STUBBORN),
        Some(&1),
        "the repaired chunk, once: {reported:?}"
    );
    for chunk in &rewritten {
        assert_eq!(
            reported.get(chunk),
            Some(&2),
            "{chunk:?} was generated and then rewritten by the repair: {reported:?}"
        );
    }
    let stats = world.stats();
    assert_eq!(stats.repaired, 1);
    assert_eq!(stats.rewritten_by_repair, rewritten.len() as u32);
    assert!(world.failed().is_empty(), "the repair placed it");
    // Three batches: one per parity, then the repair. A chunk waiting for a repair is not
    // dispatched again in between, because solving it again would fail the same way.
    let batches = &world.solver().batches;
    assert_eq!(batches.len(), 3, "{batches:?}");
    let (repair_ids, repairing) = &batches[2];
    assert!(
        *repairing,
        "the third batch is the repair, with its halo released"
    );
    assert!(
        !repair_ids.is_empty() && repair_ids.iter().all(|&id| id == STUBBORN.id()),
        "the repair is the chunk alone, tried with several seeds: {repair_ids:?}"
    );
    // The repair painted its halo, so the cells its neighbours gave up carry its tile.
    let origin = STUBBORN.origin(CHUNK);
    let border = [origin[0] - 1, origin[1], origin[2]];
    assert_eq!(world.store().tile(border), Some(1));
}

/// A solver whose repairs only succeed for some seeds, and which paints each region of a batch with
/// the tile of its index, so the committed tiles say which seed won.
///
/// The first attempt at [`STUBBORN`] fails; in a repair batch, the regions whose index is in
/// `solving` solve and the others exhaust their attempts.
struct Picky {
    solving: Vec<usize>,
    next_job: u64,
    finished: Option<(JobId, BatchResult)>,
    /// The seeds of every repair batch.
    repair_seeds: Vec<Vec<u32>>,
}

impl Solver for Picky {
    fn max_batch(&self) -> u32 {
        64
    }

    fn start(&mut self, batch: RegionBatch) -> Result<JobId, SolverError> {
        let repairing = batch.budget.is_some();
        if repairing {
            self.repair_seeds.push(batch.seeds.clone());
        }
        let mut statuses = Vec::new();
        let mut domains =
            Domains::from_words(0, batch.init.words_per_cell(), Vec::new()).expect("empty");
        for (index, &id) in batch.ids.iter().enumerate() {
            let solves = if repairing {
                self.solving.contains(&index)
            } else {
                id != STUBBORN.id()
            };
            statuses.push(if solves {
                RegionStatus::Solved
            } else {
                RegionStatus::Exhausted
            });
            let tile = (index % TILES as usize) as u32;
            let painted = (0..batch.region.cells()).map(|_| TileMask::single(tile));
            domains.append(&Domains::from_masks(batch.init.words_per_cell(), painted));
        }
        let job = JobId(self.next_job);
        self.next_job += 1;
        self.finished = Some((
            job,
            BatchResult {
                statuses,
                stats: vec![RegionStats::default(); batch.len()],
                domains,
            },
        ));
        Ok(job)
    }

    fn poll(&mut self, job: JobId) -> Result<Option<BatchResult>, SolverError> {
        match self.finished.take() {
            Some((finished, result)) if finished == job => Ok(Some(result)),
            other => {
                self.finished = other;
                Err(SolverError::UnknownJob(job))
            }
        }
    }

    fn wait(&mut self, job: JobId) -> Result<BatchResult, SolverError> {
        self.poll(job)?.ok_or(SolverError::UnknownJob(job))
    }
}

fn picky(solving: Vec<usize>) -> WorldGenerator<Picky> {
    generator(ruleset(), true).build_with(Picky {
        solving,
        next_job: 1,
        finished: None,
        repair_seeds: Vec::new(),
    })
}

#[test]
fn a_repair_tries_several_seeds_and_keeps_the_lowest_that_solves() {
    // Only the sixth and the tenth seed of a repair solve, and the tenth paints a different tile.
    let mut world = picky(vec![5, 9]);

    world.request(&[FocusPoint::new(ChunkCoord::new(1, 1, 0), 1)]);
    world.run_until_idle().expect("the picky solver");

    assert!(world.failed().is_empty(), "a later seed placed the chunk");
    let tile = world
        .store()
        .tile(STUBBORN.origin(CHUNK))
        .expect("the chunk was placed");
    assert_eq!(
        u32::from(tile),
        5 % TILES,
        "the lowest seed that solved won"
    );
    let seeds = &world.solver().repair_seeds[0];
    let distinct: BTreeSet<u32> = seeds.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        seeds.len(),
        "every seed of a repair differs: {seeds:?}"
    );
}

#[test]
fn a_repair_tries_the_same_seeds_in_the_same_world() {
    let mut first = picky(vec![3]);
    let mut second = picky(vec![3]);

    for world in [&mut first, &mut second] {
        world.request(&[FocusPoint::new(ChunkCoord::new(1, 1, 0), 1)]);
        world.run_until_idle().expect("the picky solver");
    }

    assert_eq!(first.solver().repair_seeds, second.solver().repair_seeds);
    assert_eq!(tiles(&first), tiles(&second));
}

#[test]
fn a_repair_no_seed_solves_gives_the_chunk_up() {
    let mut world = picky(Vec::new());

    world.request(&[FocusPoint::new(ChunkCoord::new(1, 1, 0), 1)]);
    let events = world.run_until_idle().expect("the picky solver");

    assert_eq!(
        world.failed().iter().copied().collect::<Vec<_>>(),
        vec![STUBBORN]
    );
    assert!(events.contains(&ChunkEvent::Failed {
        chunk: STUBBORN,
        status: RegionStatus::Exhausted
    }));
}

#[test]
fn no_batch_holds_two_chunks_that_share_a_face() {
    let mut world = generator(ruleset(), false).build_with(Recording {
        inner: ReferenceSolver::new(Arc::new(ruleset())),
        batches: Vec::new(),
    });

    world.request(&[FocusPoint::new(ChunkCoord::new(1, 1, 0), 1)]);
    world.run_until_idle().expect("the reference solves");

    let by_id: BTreeMap<u32, ChunkCoord> = extent()
        .chunks()
        .into_iter()
        .map(|chunk| (chunk.id(), chunk))
        .collect();
    let batches = &world.solver().batches;
    assert!(batches.len() >= 2, "one batch per parity at least");
    for ids in batches {
        let chunks: Vec<ChunkCoord> = ids.iter().map(|id| by_id[id]).collect();
        for chunk in &chunks {
            for neighbour in chunk.face_neighbours() {
                assert!(
                    !chunks.contains(&neighbour),
                    "{chunk:?} and {neighbour:?} rode in one dispatch: {chunks:?}"
                );
            }
        }
    }
}

/// The reference solver, with the chunk ids of every batch kept for inspection.
struct Recording {
    inner: ReferenceSolver,
    batches: Vec<Vec<u32>>,
}

impl Solver for Recording {
    fn max_batch(&self) -> u32 {
        self.inner.max_batch()
    }

    fn accepts(&self, region: RegionShape) -> bool {
        self.inner.accepts(region)
    }

    fn start(&mut self, batch: RegionBatch) -> Result<JobId, SolverError> {
        self.batches.push(batch.ids.clone());
        self.inner.start(batch)
    }

    fn poll(&mut self, job: JobId) -> Result<Option<BatchResult>, SolverError> {
        self.inner.poll(job)
    }

    fn wait(&mut self, job: JobId) -> Result<BatchResult, SolverError> {
        self.inner.wait(job)
    }
}

#[test]
fn a_repair_budget_does_not_limit_a_first_attempt() {
    let ruleset = ruleset();
    let solver = ReferenceSolver::new(Arc::new(ruleset.clone()));
    let mut world = Builder::new(ruleset, Prior::open(TILES))
        .seed(7)
        .extent(extent())
        .halo(1)
        .repair(RepairPolicy {
            enabled: false,
            max_halo: 1,
            budget: SolveBudget {
                max_attempts: 1,
                max_steps: 1,
            },
            seeds: 1,
        })
        .build_with(solver);

    world.request(&[FocusPoint::new(ChunkCoord::new(1, 1, 0), 1)]);
    let events = world.run_until_idle().expect("the reference solves");

    assert!(
        events
            .iter()
            .all(|event| matches!(event, ChunkEvent::Updated(_))),
        "a first attempt is not budgeted like a repair: {events:?}"
    );
}
