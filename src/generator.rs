//! Generating a world around its focus points, one batch at a time.

use crate::scheduler::{self, FocusPoint};
use crate::{Error, WorldConfig};
use std::collections::{BTreeSet, VecDeque};
use std::sync::Arc;
use std::time::Instant;
use wfc_core::{
    Chunk, ChunkCoord, ChunkStore, Domains, JobId, Prior, Region, RegionBatch, RegionShape,
    RegionStatus, Ruleset, Solver, region_init,
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
}

/// A batch the solver is working on.
struct Pending {
    job: JobId,
    chunks: Vec<ChunkCoord>,
    regions: Vec<Region>,
    /// Whether the batch may rewrite cells its chunks did not own, which a repair does.
    release: bool,
    halo: u32,
    started: Instant,
}

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
    repairs: VecDeque<(ChunkCoord, u32)>,
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
            repairs: VecDeque::new(),
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
        let missing = scheduler::missing(&self.wanted, &self.store, &self.deferred(), &self.focus);
        let batch = scheduler::next_batch(
            &missing,
            &self.store,
            &self.failed,
            &self.config.extent,
            self.solver.max_batch(),
        );
        if !batch.is_empty() {
            return self.start(&batch, self.config.halo, false);
        }
        // Nothing left to generate, so a chunk that failed gets its repair: the same region again
        // with its halo released, which lets it rewrite the neighbouring cells it covers.
        if let Some((chunk, halo)) = self.repairs.pop_front() {
            return self.start(&[chunk], halo, true);
        }
        Ok(())
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

    /// How many wanted chunks are still to generate, including any being worked on.
    #[must_use]
    pub fn pending_chunks(&self) -> usize {
        scheduler::missing(&self.wanted, &self.store, &self.deferred(), &self.focus).len()
            + self.repairs.len()
    }

    /// The chunks a batch must leave alone: those given up on, and those a repair is queued for.
    fn deferred(&self) -> BTreeSet<ChunkCoord> {
        self.failed
            .iter()
            .copied()
            .chain(self.repairs.iter().map(|(chunk, _)| *chunk))
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

    /// Drops the chunks further than `margin` beyond every focus point and hands them back, so a
    /// game can persist them. Regenerating one gives the same tiles unless a repair has rewritten
    /// its neighbours since (see the determinism contract in the crate documentation).
    pub fn evict_outside(&mut self, focus: &[FocusPoint], margin: u32) -> Vec<Chunk> {
        let far: Vec<ChunkCoord> = self
            .store
            .iter()
            .map(|chunk| chunk.coord)
            .filter(|coord| {
                focus
                    .iter()
                    .all(|focus| focus.distance(*coord) > focus.radius + margin)
            })
            .collect();
        far.into_iter()
            .filter_map(|coord| self.store.remove(coord))
            .collect()
    }

    /// Puts a chunk back, for example one a game had persisted.
    ///
    /// # Errors
    /// If the chunk is outside the world or the wrong size.
    pub fn import(&mut self, chunk: Chunk) -> Result<(), Error> {
        self.failed.remove(&chunk.coord);
        Ok(self.store.insert(chunk)?)
    }

    /// Dispatches `chunks` as one batch.
    fn start(&mut self, chunks: &[ChunkCoord], halo: u32, release: bool) -> Result<(), Error> {
        let shape = self.region_shape(halo);
        let regions: Vec<Region> = chunks.iter().map(|&c| Region::new(c, shape)).collect();
        let mut init = Domains::from_words(0, self.ruleset.words_per_cell(), Vec::new())
            .expect("an empty batch");
        for region in &regions {
            let domains = region_init(&self.store, &self.prior, &self.ruleset, region, release);
            init.append(&domains);
        }
        let batch = RegionBatch {
            region: shape,
            ids: chunks.iter().map(|chunk| chunk.id()).collect(),
            // A chunk's own identity salts the choice, so every region of a batch shares the seed.
            seeds: vec![batch_seed(self.config.seed, halo, release); chunks.len()],
            init,
            budget: release.then_some(self.config.repair.budget),
        };
        let job = self.solver.start(batch)?;
        self.stats.batches += 1;
        self.pending = Some(Pending {
            job,
            chunks: chunks.to_vec(),
            regions,
            release,
            halo,
            started: Instant::now(),
        });
        Ok(())
    }

    /// Writes what a finished batch solved and decides what to do about what it did not.
    fn commit(&mut self, pending: Pending, result: wfc_core::BatchResult) -> Result<(), Error> {
        self.stats.solver_ms += pending.started.elapsed().as_secs_f64() * 1000.0;
        let cells = pending.regions[0].shape().cells();
        for (index, (&chunk, region)) in pending.chunks.iter().zip(&pending.regions).enumerate() {
            let status = result.statuses[index];
            if !status.is_solved() {
                self.give_up_or_repair(chunk, status, pending.halo, pending.release);
                continue;
            }
            let domains = result.region(index, cells);
            let touched = self.store.commit(region, &domains, pending.release)?;
            self.stats.solved += 1;
            if pending.release {
                self.stats.repaired += 1;
                self.stats.rewritten_by_repair += touched.len() as u32 - 1;
            }
            self.failed.remove(&chunk);
            self.events
                .extend(touched.into_iter().map(ChunkEvent::Updated));
        }
        Ok(())
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
            Some(wider) => self.repairs.push_back((chunk, wider)),
            None => {
                self.failed.insert(chunk);
                self.stats.failed += 1;
                self.events.push(ChunkEvent::Failed { chunk, status });
            }
        }
    }

    /// The halos a repair may use, widest last: past some width a region no longer fits the device.
    fn repair_halos(&self) -> Vec<u32> {
        (1..=self.config.repair.max_halo)
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
///
/// A repair gets a stream of its own, keyed by how wide it is: its chunk exhausted every restart the
/// first attempt had, so trying the same sequence of choices again with one more row of freedom is
/// the weakest thing a repair could do. The stream still depends on nothing but the world seed and
/// the halo, so the world stays a function of its configuration.
fn batch_seed(seed: u64, halo: u32, repair: bool) -> u32 {
    let world = (seed as u32) ^ ((seed >> 32) as u32);
    if repair {
        world ^ 0x9E37_79B9_u32.wrapping_mul(halo + 1)
    } else {
        world
    }
}
