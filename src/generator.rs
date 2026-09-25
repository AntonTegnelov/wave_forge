//! Generating a world around its focus points, one batch at a time.

use crate::layers::{Layers, Phase, cell_index};
use crate::scheduler::{self, FocusPoint};
use crate::{Error, WorldConfig};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;
use wfc_core::{
    Chunk, ChunkCoord, ChunkStore, Domains, JobId, ModelError, Prior, Region, RegionBatch,
    RegionShape, RegionStatus, Ruleset, Solver, region_init_by,
};

/// What happened to a chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkEvent {
    /// The chunk's tiles are new or have changed, so anything built from them is stale. A repair
    /// reports every chunk it rewrote, not only the one it was repairing.
    Updated(ChunkCoord),
    /// The chunk left every focus and was dropped, so anything built from it can go too.
    ///
    /// Only a [`crate::Worker`] reports this: [`WorldGenerator::evict_outside`] hands the chunks
    /// straight back to its caller instead.
    Evicted(ChunkCoord),
    /// The chunk could not be generated. Asking again would fail the same way, so the generator
    /// leaves it alone until its neighbourhood changes.
    Failed {
        chunk: ChunkCoord,
        status: RegionStatus,
    },
}

/// What generating a world has cost so far.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GeneratorStats {
    /// Batches dispatched, including repairs.
    pub batches: u32,
    /// Chunks generated.
    pub solved: u32,
    /// Chunks that needed a repair, and the chunks those repairs rewrote. A rule set is
    /// streaming-clean when the first of these stays zero.
    pub repaired: u32,
    /// Chunks a repair rewrote cells of, counted once per repair that touched them.
    pub rewritten_by_repair: u32,
    /// Chunks no repair could place.
    pub failed: u32,
    /// Time spent waiting for the solver.
    pub solver_ms: f64,
    /// Repair batches dispatched, whether they placed their chunk or not.
    pub repair_batches: u32,
    /// The part of `solver_ms` spent on repairs.
    pub repair_ms: f64,
    /// Repairs of chunks in memory run again, to rewrite a neighbour generated again as they
    /// rewrote it the first time.
    pub replayed: u32,
}

/// A batch the solver is working on.
struct Pending {
    job: JobId,
    chunks: Vec<ChunkCoord>,
    regions: Vec<Region>,
    /// Whether the batch may rewrite cells its chunks did not own, which a repair does.
    release: bool,
    /// Whether the batch is a repair run again ([`WorldGenerator::replays`]).
    replay: bool,
    halo: u32,
    started: Instant,
}

/// How many chunks beyond the ones a request asks for its repairs may need generated, in a world
/// one chunk tall: a second-parity chunk at the edge of the request waits for its diagonal
/// neighbours' first attempts and for their face neighbours, one of which may itself wait for a
/// repair that has to see its own diagonals. Those chunks are kept while they are needed, whatever
/// margin [`WorldGenerator::evict_outside`] is given.
pub const REPAIR_REACH: u32 = 3;

/// Generates a world around focus points, in chunks, on any [`Solver`].
///
/// Nothing here blocks unless you ask it to: [`WorldGenerator::tick`] starts work and
/// [`WorldGenerator::poll`] collects it, so a game can drive generation from a frame. See
/// [`crate::Worker`] for the shape a blocking engine wants.
pub struct WorldGenerator<S: Solver> {
    config: WorldConfig,
    ruleset: Arc<Ruleset>,
    prior: Prior,
    store: ChunkStore,
    solver: S,
    wanted: BTreeSet<ChunkCoord>,
    focus: Vec<FocusPoint>,
    /// Chunks no repair could place. They stay out of the way until something around them changes.
    failed: BTreeSet<ChunkCoord>,
    /// Chunks whose first attempt failed, and the halo their next repair uses.
    repairs: BTreeMap<ChunkCoord, u32>,
    /// Chunks in memory that a repair placed, and the halo it placed them with.
    repaired: BTreeMap<ChunkCoord, u32>,
    /// Repairs to run again: of chunks in memory whose repair rewrote a neighbour that has since
    /// been evicted and generated again, and the halo each used (docs/architecture/world.md,
    /// "Regenerating exactly").
    replays: BTreeMap<ChunkCoord, u32>,
    /// What each chunk in memory was after each phase that wrote it.
    layers: Layers,
    pending: Option<Pending>,
    events: Vec<ChunkEvent>,
    stats: GeneratorStats,
}

impl<S: Solver> WorldGenerator<S> {
    /// A generator over `solver`. Prefer [`crate::Builder`], which builds both.
    pub(crate) fn new(config: WorldConfig, ruleset: Arc<Ruleset>, prior: Prior, solver: S) -> Self {
        let store = ChunkStore::new(config.extent.clone());
        Self {
            config,
            ruleset,
            prior,
            store,
            solver,
            wanted: BTreeSet::new(),
            focus: Vec::new(),
            failed: BTreeSet::new(),
            repairs: BTreeMap::new(),
            repaired: BTreeMap::new(),
            replays: BTreeMap::new(),
            layers: Layers::default(),
            pending: None,
            events: Vec::new(),
            stats: GeneratorStats::default(),
        }
    }

    /// Asks for the chunks around these focus points, replacing the previous request.
    pub fn request(&mut self, focus: &[FocusPoint]) {
        self.wanted = scheduler::wanted(focus, &self.config.extent);
        self.focus = focus.to_vec();
    }

    /// Starts the next batch if the solver is free.
    ///
    /// # Errors
    /// If the solver refuses the batch.
    pub fn tick(&mut self) -> Result<(), Error> {
        if self.pending.is_some() {
            return Ok(());
        }
        // A second-parity chunk reads its face neighbours after the first-parity repairs, so it
        // also waits for those run again.
        let missing: Vec<ChunkCoord> =
            scheduler::missing(&self.needed(), &self.store, &self.deferred(), &self.focus)
                .into_iter()
                .filter(|chunk| {
                    chunk.parity() == 0
                        || !chunk
                            .face_neighbours()
                            .iter()
                            .any(|face| self.replays.contains_key(face))
                })
                .collect();
        let batch = scheduler::next_batch(
            &missing,
            &self.store,
            &self.failed,
            &self.config.extent,
            self.solver.max_batch(),
        );
        if !batch.is_empty() {
            let halo = self.first_attempt_halo(batch[0].parity());
            return self.start(&batch, halo, false);
        }
        // Nothing left to generate, so a chunk that failed gets its repair: the same region again
        // with its halo released, which lets it rewrite the neighbouring cells it covers. Only a
        // repair whose neighbourhood is complete may run (`repair_ready`), and among those the
        // lowest class first; with nothing ready, what `needed` added is still being generated.
        loop {
            let ready = self
                .active_repairs()
                .into_iter()
                .filter(|chunk| self.repair_ready(*chunk))
                .min_by_key(|chunk| (scheduler::repair_class(*chunk), *chunk));
            let Some(chunk) = ready else {
                break;
            };
            if let Some(halo) = self.replays.remove(&chunk) {
                // A replay with nothing left to write is dropped, and the next one looked at.
                if self.replay_writes(chunk) {
                    self.stats.replayed += 1;
                    return self.start_repair(chunk, halo, true);
                }
                continue;
            }
            let halo = self
                .repairs
                .remove(&chunk)
                .expect("chosen from the queued repairs");
            return self.start_repair(chunk, halo, false);
        }
        // A repair waits only for first attempts, and for repairs of lower classes, none of which
        // wait for it: a first-parity repair never waits for a second-parity chunk. So with active
        // repairs queued, a batch or a ready repair always exists.
        let active = self.active_repairs();
        assert!(
            active.is_empty(),
            "queued repairs wait for chunks nothing can generate: {active:?}"
        );
        Ok(())
    }

    /// Whether a queued repair of `chunk` may run: every chunk it can see has had its first
    /// attempt, and every failed one of a lower class has been repaired or given up on. Its result
    /// is then the same whatever order the world was generated in.
    fn repair_ready(&self, chunk: ChunkCoord) -> bool {
        let class = scheduler::repair_class(chunk);
        scheduler::repair_neighbourhood(chunk, &self.config.extent)
            .into_iter()
            .all(|neighbour| {
                if self.queued(neighbour) {
                    scheduler::repair_class(neighbour) > class
                } else {
                    self.store.contains(neighbour) || self.failed.contains(&neighbour)
                }
            })
    }

    /// The chunks to generate: those asked for, and those the active repairs have to see first,
    /// with the neighbours both are solved against.
    fn needed(&self) -> BTreeSet<ChunkCoord> {
        let mut needed = self.repair_needs(&self.active_repairs());
        needed.extend(self.wanted.iter().copied());
        needed
    }

    /// The chunks `repairs` have to see, with the neighbours those are solved against.
    fn repair_needs(&self, repairs: &BTreeSet<ChunkCoord>) -> BTreeSet<ChunkCoord> {
        let seen: BTreeSet<ChunkCoord> = repairs
            .iter()
            .flat_map(|chunk| scheduler::repair_neighbourhood(*chunk, &self.config.extent))
            .collect();
        scheduler::with_read_neighbours(&seen, &self.config.extent)
    }

    /// The queued repairs that matter to what is asked for now: those of wanted chunks, and those
    /// they wait for, which are the failed neighbours of a lower class and the failed first-parity
    /// chunks that a neighbour they have to see is solved against. In a world one chunk tall a
    /// wait goes at most one class down within each parity, so nothing an active repair needs lies
    /// more than [`REPAIR_REACH`] chunks beyond what was asked for.
    ///
    /// The other queued repairs stay queued, remembered as failed first attempts, rather than being
    /// dropped: attempting such a chunk again later could see neighbours a repair has rewritten
    /// since, and so depend on the order.
    fn active_repairs(&self) -> BTreeSet<ChunkCoord> {
        let extent = &self.config.extent;
        let mut active: BTreeSet<ChunkCoord> = self
            .repairs
            .keys()
            .copied()
            .filter(|chunk| self.wanted.contains(chunk))
            .chain(self.replays.keys().copied())
            .collect();
        let mut frontier: Vec<ChunkCoord> = active.iter().copied().collect();
        while let Some(chunk) = frontier.pop() {
            let class = scheduler::repair_class(chunk);
            for neighbour in scheduler::repair_neighbourhood(chunk, extent) {
                let lower = self.queued(neighbour) && scheduler::repair_class(neighbour) < class;
                let unattempted = !self.store.contains(neighbour)
                    && !self.failed.contains(&neighbour)
                    && !self.repairs.contains_key(&neighbour);
                let blockers = (unattempted && neighbour.parity() == 1)
                    .then(|| neighbour.face_neighbours())
                    .into_iter()
                    .flatten()
                    .filter(|face| self.queued(*face));
                for waited_for in lower.then_some(neighbour).into_iter().chain(blockers) {
                    if active.insert(waited_for) {
                        frontier.push(waited_for);
                    }
                }
            }
        }
        active
    }

    /// Whether a repair of `chunk` is queued, to run for the first time or again.
    fn queued(&self, chunk: ChunkCoord) -> bool {
        self.repairs.contains_key(&chunk) || self.replays.contains_key(&chunk)
    }

    /// Takes whatever the solver has finished, without blocking.
    ///
    /// # Errors
    /// If the solver or the store failed.
    pub fn poll(&mut self) -> Result<Vec<ChunkEvent>, Error> {
        if let Some(pending) = self.pending.take() {
            match self.solver.poll(pending.job)? {
                Some(result) => self.commit(pending, result)?,
                None => self.pending = Some(pending),
            }
        }
        Ok(std::mem::take(&mut self.events))
    }

    /// Waits for the solver and takes the result.
    ///
    /// # Errors
    /// If the solver or the store failed.
    pub fn wait(&mut self) -> Result<Vec<ChunkEvent>, Error> {
        if let Some(pending) = self.pending.take() {
            let result = self.solver.wait(pending.job)?;
            self.commit(pending, result)?;
        }
        Ok(std::mem::take(&mut self.events))
    }

    /// Generates everything asked for, blocking until there is nothing left to do.
    ///
    /// # Errors
    /// If the solver or the store failed.
    pub fn run_until_idle(&mut self) -> Result<Vec<ChunkEvent>, Error> {
        let mut events = Vec::new();
        loop {
            self.tick()?;
            if self.pending.is_none() {
                return Ok(events);
            }
            events.extend(self.wait()?);
        }
    }

    /// Whether there is nothing in flight and nothing left to start.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.pending.is_none() && self.pending_chunks() == 0
    }

    /// How many chunks are still to generate for what is asked for, including those its repairs
    /// have to see and the repairs themselves.
    #[must_use]
    pub fn pending_chunks(&self) -> usize {
        scheduler::missing(&self.needed(), &self.store, &self.deferred(), &self.focus).len()
            + self.active_repairs().len()
    }

    /// The chunks a batch must leave alone: those given up on, and those a repair is queued for.
    fn deferred(&self) -> BTreeSet<ChunkCoord> {
        self.failed
            .iter()
            .copied()
            .chain(self.repairs.keys().copied())
            .collect()
    }

    /// A generated chunk.
    #[must_use]
    pub fn chunk(&self, coord: ChunkCoord) -> Option<&Chunk> {
        self.store.get(coord)
    }

    /// Every chunk generated so far.
    #[must_use]
    pub const fn store(&self) -> &ChunkStore {
        &self.store
    }

    /// What the world is configured as.
    #[must_use]
    pub const fn config(&self) -> &WorldConfig {
        &self.config
    }

    /// The solver the world generates on.
    #[must_use]
    pub const fn solver(&self) -> &S {
        &self.solver
    }

    /// Ends the world and hands its solver back, to generate another world on it.
    #[must_use]
    pub fn into_solver(self) -> S {
        self.solver
    }

    /// The solver, to set up before generation starts: a GPU solver compiles a kernel per region
    /// shape, and [`wfc_gpu::BlockSolver::warm`] does that at load rather than at the first batch.
    pub const fn solver_mut(&mut self) -> &mut S {
        &mut self.solver
    }

    /// What generation has cost.
    #[must_use]
    pub const fn stats(&self) -> &GeneratorStats {
        &self.stats
    }

    /// The chunks no repair could place.
    #[must_use]
    pub const fn failed(&self) -> &BTreeSet<ChunkCoord> {
        &self.failed
    }

    /// Drops the chunks further than `margin` beyond every focus point and hands them back. A chunk
    /// asked for again comes back with the tiles it had, repairs included (see the determinism
    /// contract in the crate documentation), so a game need not keep them. Chunks a queued repair
    /// still has to see stay, or they would be generated again at once.
    pub fn evict_outside(&mut self, focus: &[FocusPoint], margin: u32) -> Vec<Chunk> {
        let needed = self.repair_needs(&self.active_repairs());
        let far: Vec<ChunkCoord> = self
            .store
            .iter()
            .map(|chunk| chunk.coord)
            .filter(|coord| !needed.contains(coord))
            .filter(|coord| {
                focus
                    .iter()
                    .all(|focus| focus.distance(*coord) > focus.radius + margin)
            })
            .collect();
        for coord in &far {
            self.layers.forget(*coord);
            self.repaired.remove(coord);
            self.replays.remove(coord);
        }
        far.into_iter()
            .filter_map(|coord| self.store.remove(coord))
            .collect()
    }

    /// Puts a chunk back, for example one a game had persisted. It is taken as it is: every
    /// operation around it reads its tiles, whatever phase of the schedule it is in.
    ///
    /// # Errors
    /// If the chunk is outside the world or the wrong size.
    pub fn import(&mut self, chunk: Chunk) -> Result<(), Error> {
        let coord = chunk.coord;
        self.store.insert(chunk)?;
        self.failed.remove(&coord);
        self.repaired.remove(&coord);
        self.replays.remove(&coord);
        self.layers.born(coord, None);
        Ok(())
    }

    /// Dispatches `chunks` as one batch.
    fn start(&mut self, chunks: &[ChunkCoord], halo: u32, release: bool) -> Result<(), Error> {
        let shape = self.region_shape(halo);
        let regions: Vec<Region> = chunks.iter().map(|&c| Region::new(c, shape)).collect();
        let mut init = Domains::from_words(0, self.ruleset.words_per_cell(), Vec::new())
            .expect("an empty batch");
        for region in &regions {
            let domains = self.region_init(region, release);
            init.append(&domains);
        }
        let batch = RegionBatch {
            region: shape,
            ids: chunks.iter().map(|chunk| chunk.id()).collect(),
            // A chunk's own identity salts the choice, so every region of a batch shares the seed.
            seeds: vec![batch_seed(self.config.seed); chunks.len()],
            init,
            budget: None,
            portfolio: false,
        };
        self.dispatch(batch, chunks.to_vec(), regions, halo, false, false)
    }

    /// Repairs one chunk: its region with the halo released, solved once per seed of the repair
    /// policy in one dispatch. A chunk that exhausted a first attempt usually has an arrangement
    /// that another seed finds (docs/architecture/world.md, "Repairs"), and the seeds run side by
    /// side, so trying many costs about as much as trying one.
    fn start_repair(&mut self, chunk: ChunkCoord, halo: u32, replay: bool) -> Result<(), Error> {
        let shape = self.region_shape(halo);
        let region = Region::new(chunk, shape);
        let domains = self.region_init(&region, true);
        let seeds = repair_seeds(self.config.seed, halo, self.repair_width());
        let mut init = Domains::from_words(0, self.ruleset.words_per_cell(), Vec::new())
            .expect("an empty batch");
        for _ in &seeds {
            init.append(&domains);
        }
        let batch = RegionBatch {
            region: shape,
            ids: vec![chunk.id(); seeds.len()],
            seeds,
            init,
            budget: Some(self.config.repair.budget),
            // One chunk tried with many seeds, of which only the lowest that solves is kept.
            portfolio: true,
        };
        self.dispatch(batch, vec![chunk], vec![region], halo, true, replay)
    }

    /// Whether running the repair of `chunk` again would write anything: a neighbour it rewrote in
    /// a fresh world is in memory and has not yet passed the repair's phase, because it was
    /// generated again. A repair is run again with the halo it placed the chunk with, and writes
    /// the neighbours as it wrote them the first time.
    fn replay_writes(&self, chunk: ChunkCoord) -> bool {
        let phase = Phase::repair(chunk);
        scheduler::repair_neighbourhood(chunk, &self.config.extent)
            .into_iter()
            .chain(chunk.face_neighbours())
            .filter(|neighbour| self.store.contains(*neighbour))
            .any(|neighbour| self.layers.latest(neighbour) < Some(phase))
    }

    /// The starting domains of `region` for the operation on its chunk: a repair when `release`
    /// holds, otherwise a first attempt. Its neighbours are read as they were before the
    /// operation's phase, which is how a fresh world had them however much has been evicted and
    /// generated again since; and a first attempt reads only chunks of the other parity, as the
    /// ones a batch cannot contain.
    fn region_init(&self, region: &Region, release: bool) -> Domains {
        let chunk = region.chunk();
        let phase = if release {
            Phase::repair(chunk)
        } else {
            Phase::first_attempt(chunk)
        };
        let shape = self.store.shape();
        region_init_by(
            &self.config.extent,
            &self.prior,
            &self.ruleset,
            region,
            release,
            |at| {
                if !release && ChunkCoord::of_cell(at, shape).parity() == chunk.parity() {
                    return None;
                }
                self.layers.tile_before(&self.store, at, phase)
            },
        )
    }

    /// Every (regions, region shape) pair a run can dispatch for focus points of up to `radius`:
    /// first attempts of every batch size up to the largest such a focus produces, and every repair
    /// the solver accepts. A solver that compiles per shape, as the GPU solver does, is warmed with
    /// these while a game loads, because compiling one mid-play is a stall of seconds.
    #[must_use]
    pub fn kernel_shapes(&self, radius: u32) -> Vec<(u32, RegionShape)> {
        let largest = scheduler::largest_batch(radius, &self.config.extent)
            .min(self.solver.max_batch())
            .next_power_of_two();
        let mut firsts = vec![
            self.region_shape(self.first_attempt_halo(0)),
            self.region_shape(self.first_attempt_halo(1)),
        ];
        firsts.dedup();
        let first_attempts = std::iter::successors(Some(1u32), |&n| (n < largest).then_some(n * 2))
            .flat_map(move |regions| {
                firsts
                    .clone()
                    .into_iter()
                    .map(move |shape| (regions, shape))
            });
        let width = self.repair_width();
        let repairs = self
            .repair_halos()
            .into_iter()
            .map(|halo| (width, self.region_shape(halo)));
        first_attempts.chain(repairs).collect()
    }

    /// The halo a first attempt at a chunk of `parity` is solved with.
    ///
    /// A chunk of the first parity is solved before its neighbours, and its halo is what leaves
    /// them room to complete its borders. A chunk of the second parity is solved after all its face
    /// neighbours, whose cells a halo could only pin as they are; what a halo adds is its diagonal
    /// corner cells, squeezed between two fixed neighbours, and those fail chunks that would
    /// otherwise solve. Measured on the city over five worlds: 176 repairs instead of 280 and a
    /// third less solver time (docs/architecture/world.md, "The schedule").
    const fn first_attempt_halo(&self, parity: u8) -> u32 {
        if parity == 1 { 0 } else { self.config.halo }
    }

    /// How many seeds one repair tries: as many as the policy asks for and the solver takes.
    fn repair_width(&self) -> u32 {
        self.config.repair.seeds.clamp(1, self.solver.max_batch())
    }

    fn dispatch(
        &mut self,
        batch: RegionBatch,
        chunks: Vec<ChunkCoord>,
        regions: Vec<Region>,
        halo: u32,
        release: bool,
        replay: bool,
    ) -> Result<(), Error> {
        let job = self.solver.start(batch)?;
        self.stats.batches += 1;
        self.pending = Some(Pending {
            job,
            chunks,
            regions,
            release,
            replay,
            halo,
            started: Instant::now(),
        });
        Ok(())
    }

    /// Writes what a finished batch solved and decides what to do about what it did not.
    fn commit(&mut self, pending: Pending, result: wfc_core::BatchResult) -> Result<(), Error> {
        let elapsed_ms = pending.started.elapsed().as_secs_f64() * 1000.0;
        self.stats.solver_ms += elapsed_ms;
        if pending.release {
            self.stats.repair_batches += 1;
            self.stats.repair_ms += elapsed_ms;
        }
        let cells = pending.regions[0].shape().cells();
        for (index, (&chunk, region)) in pending.chunks.iter().zip(&pending.regions).enumerate() {
            // A repair is one chunk tried with many seeds; the lowest seed that solved is the one
            // kept, so the outcome does not depend on anything but the configuration.
            let solved = if pending.release {
                result.statuses.iter().position(|status| status.is_solved())
            } else {
                Some(index).filter(|&index| result.statuses[index].is_solved())
            };
            let Some(solved) = solved else {
                assert!(
                    !pending.replay,
                    "a repair of {chunk:?} run again did not solve, though it read what it read \
                     the first time"
                );
                self.give_up_or_repair(
                    chunk,
                    result.statuses[index],
                    pending.halo,
                    pending.release,
                );
                continue;
            };
            let domains = result.region(solved, cells);
            let touched = self.write(region, &domains, pending.release)?;
            if pending.replay {
                self.events
                    .extend(touched.into_iter().map(ChunkEvent::Updated));
                continue;
            }
            self.stats.solved += 1;
            if pending.release {
                self.stats.repaired += 1;
                self.stats.rewritten_by_repair += touched.len() as u32 - 1;
                self.repaired.insert(chunk, pending.halo);
            }
            self.failed.remove(&chunk);
            self.events
                .extend(touched.into_iter().map(ChunkEvent::Updated));
        }
        Ok(())
    }

    /// Writes what the operation on `region`'s chunk decided, a repair when `release` holds, and
    /// returns the chunks it changed. It writes the chunk itself and, for a repair, the cells of
    /// the neighbours in its region that exist and have not yet passed its phase; a neighbour last
    /// written by another operation of the same phase takes these cells into that version. A
    /// neighbour that has passed the phase holds this write already, from the first time: what
    /// was decided now has to be what it holds, or the operation read something other than it read
    /// then.
    ///
    /// # Errors
    /// If a cell to write was left undecided.
    ///
    /// # Panics
    /// If a neighbour that has passed the phase holds anything else, which would mean the phases
    /// do not order the operations as the scheduler runs them.
    fn write(
        &mut self,
        region: &Region,
        domains: &Domains,
        release: bool,
    ) -> Result<Vec<ChunkCoord>, Error> {
        let own = region.chunk();
        let phase = if release {
            Phase::repair(own)
        } else {
            Phase::first_attempt(own)
        };
        let shape = self.store.shape();
        let mut by_chunk: BTreeMap<ChunkCoord, Vec<(usize, u16)>> = BTreeMap::new();
        for (index, (at, inner)) in region.cells().enumerate() {
            if !self.config.extent.contains_cell(at) {
                continue;
            }
            let coord = ChunkCoord::of_cell(at, shape);
            if !inner && !(release && self.store.contains(coord)) {
                continue;
            }
            let Some(tile) = domains.decided(index as u32) else {
                return Err(ModelError::Undecided { cell: at }.into());
            };
            let tile = u16::try_from(tile).expect("a tile index fits u16");
            by_chunk
                .entry(coord)
                .or_default()
                .push((cell_index(coord, shape, at), tile));
        }
        let mut touched = Vec::new();
        for (coord, cells) in by_chunk {
            let Some(chunk) = self.store.get(coord) else {
                let mut tiles = vec![0u16; shape.cells() as usize].into_boxed_slice();
                for &(cell, tile) in &cells {
                    tiles[cell] = tile;
                }
                self.store.insert(Chunk {
                    coord,
                    tiles,
                    version: 1,
                })?;
                self.layers.born(coord, Some(phase));
                self.replay_around(coord, phase);
                touched.push(coord);
                continue;
            };
            let latest = self.layers.latest(coord);
            if latest < Some(phase) {
                let before = chunk.tiles.clone();
                let mut tiles = before.clone();
                for &(cell, tile) in &cells {
                    tiles[cell] = tile;
                }
                let version = chunk.version + 1;
                self.store.insert(Chunk {
                    coord,
                    tiles,
                    version,
                })?;
                self.layers.rewritten(coord, phase, before);
                touched.push(coord);
                continue;
            }
            if latest == Some(phase) {
                // Another operation of this phase wrote the chunk last. Operations of one phase
                // are two chunks apart and write different cells of a chunk between them, so this
                // one's cells join the same version.
                if cells.iter().any(|&(cell, tile)| chunk.tiles[cell] != tile) {
                    let mut tiles = chunk.tiles.clone();
                    for &(cell, tile) in &cells {
                        tiles[cell] = tile;
                    }
                    let version = chunk.version + 1;
                    self.store.insert(Chunk {
                        coord,
                        tiles,
                        version,
                    })?;
                    touched.push(coord);
                }
                continue;
            }
            match self.layers.written_by(&self.store, coord, phase) {
                Some(held) => assert!(
                    cells.iter().all(|&(cell, tile)| held[cell] == tile),
                    "the operation on {own:?} at {phase:?}, run again, wrote {coord:?} otherwise \
                     than the first time"
                ),
                None => assert!(
                    !self.layers.existed_before(coord, phase),
                    "{coord:?} existed before the operation on {own:?} at {phase:?} and was not \
                     written by it"
                ),
            }
        }
        Ok(touched)
    }

    /// A chunk entered the store at `phase`: every chunk in memory within one of it whose repair
    /// comes after `phase` in a fresh world rewrote it then, so that repair runs again. In a fresh
    /// world no such repair has run yet, since each waits for the chunks around it.
    fn replay_around(&mut self, chunk: ChunkCoord, phase: Phase) {
        let around: Vec<ChunkCoord> = (-1..=1)
            .flat_map(|x| (-1..=1).flat_map(move |y| (-1..=1).map(move |z| (x, y, z))))
            .filter(|&offset| offset != (0, 0, 0))
            .map(|(x, y, z)| ChunkCoord::new(chunk.x + x, chunk.y + y, chunk.z + z))
            .collect();
        for neighbour in around {
            if let Some(&halo) = self.repaired.get(&neighbour)
                && Phase::repair(neighbour) > phase
            {
                self.replays.insert(neighbour, halo);
            }
        }
    }

    /// Queues a repair for a chunk that would not solve, or gives up on it.
    ///
    /// A first repair may use the halo the batch already had, because releasing that halo is itself
    /// the change: the repair may rewrite the neighbouring cells it covers. After that only a wider
    /// one is worth trying.
    fn give_up_or_repair(
        &mut self,
        chunk: ChunkCoord,
        status: RegionStatus,
        halo: u32,
        was_repair: bool,
    ) {
        let next = self
            .repair_halos()
            .into_iter()
            .find(|wider| if was_repair { *wider > halo } else { true });
        match next.filter(|_| self.config.repair.enabled) {
            Some(wider) => {
                self.repairs.insert(chunk, wider);
            }
            None => {
                self.failed.insert(chunk);
                self.stats.failed += 1;
                self.events.push(ChunkEvent::Failed { chunk, status });
            }
        }
    }

    /// The halos a repair may use, widest last: past some width a region no longer fits the device,
    /// and one reaching half a chunk would let two repairs of one class meet, which
    /// [`scheduler::repair_class`] relies on them never doing.
    fn repair_halos(&self) -> Vec<u32> {
        let chunk = self.config.chunk;
        (1..=self.config.repair.max_halo)
            .filter(|&halo| {
                let halos = self.config.extent.halo(halo);
                [chunk.x, chunk.y, chunk.z]
                    .iter()
                    .zip(halos)
                    .all(|(&size, reach)| reach == 0 || 2 * reach + 2 <= size)
            })
            .filter(|halo| self.solver.accepts(self.region_shape(*halo)))
            .collect()
    }

    fn region_shape(&self, halo: u32) -> RegionShape {
        self.config.chunk.region(self.config.extent.halo(halo))
    }
}

/// The seed a batch's choices derive from, as the solver's hash takes it. A chunk's own identity is
/// mixed in there, so folding the high half in here only has to keep both halves of the seed
/// meaningful.
fn batch_seed(seed: u64) -> u32 {
    (seed as u32) ^ ((seed >> 32) as u32)
}

/// The seeds one repair tries, `width` of them, all different from each other and from the first
/// attempt's: the chunk exhausted every restart the first attempt had, so trying that sequence of
/// choices again is the weakest thing a repair could do. They depend on nothing but the world seed,
/// the halo and their position, so the world stays a function of its configuration.
fn repair_seeds(seed: u64, halo: u32, width: u32) -> Vec<u32> {
    let world = batch_seed(seed);
    (0..width)
        .map(|index| {
            world ^ 0x9E37_79B9_u32.wrapping_mul(halo + 1) ^ 0x85EB_CA6B_u32.wrapping_mul(index)
        })
        .collect()
}
