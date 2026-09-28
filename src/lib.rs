//! Wave Forge: rule-based world generation, in chunks, while a game runs.
//!
//! A world is a lattice of chunks. You describe what the world allows with a [`Ruleset`] and a
//! [`Prior`], say where generation is wanted with [`FocusPoint`]s, and the generator solves the
//! chunks around them on a GPU, one batch per dispatch. Nothing here needs an async runtime: work
//! is started with [`WorldGenerator::tick`] and collected with [`WorldGenerator::poll`], so an
//! engine decides where blocking happens, or a [`Worker`] does it on a thread.
//!
//! ```no_run
//! use wave_forge::{Builder, ChunkCoord, FocusPoint, Prior, Ruleset};
//!
//! # fn main() -> Result<(), wave_forge::Error> {
//! # let rules = wfc_rules::AdjacencyRules::from_allowed_tuples(1, 6, []);
//! let ruleset = Ruleset::new(&rules, &[1.0])?;
//! let mut world = Builder::new(ruleset, Prior::open(1)).seed(7).build()?;
//!
//! world.request(&[FocusPoint::new(ChunkCoord::new(0, 0, 0), 4)]);
//! world.tick()?;
//! for event in world.poll()? {
//!     println!("{event:?}");
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Determinism
//!
//! For a fixed rule set, prior and configuration:
//!
//! - The same sequence of requests gives an identical world on any backend, on any number of
//!   threads, and whatever a solver's invocation count is. Every choice is a hash of the world
//!   seed, the chunk's coordinate, and where the solve had got to.
//! - A chunk is solved against its face neighbours and nothing else. The scheduler solves one
//!   parity of the chunk lattice first and holds a chunk of the other parity until the four or six
//!   chunks it reads are finished.
//! - A repair rewrites cells of the neighbours it covers, and still leaves the world a function of
//!   its configuration: it waits until every chunk it can see has had its first attempt and the
//!   failed ones of lower repair classes around it are repaired, so it sees the same neighbourhood
//!   whatever order the world was generated in. Every chunk a repair rewrote is reported as
//!   [`ChunkEvent::Updated`] and counted in [`GeneratorStats`].
//! - A chunk evicted and asked for again comes back tile for tile, repairs included, with nothing
//!   saved: every operation reads its neighbours as a fresh world had them, and the repairs of
//!   chunks that stayed run again for the neighbours generated again (docs/architecture/world.md,
//!   "Regenerating exactly").
//! - A frozen world, given a store ([`WorldGenerator::with_store`]), keeps what it evicts and
//!   brings it back as it was instead, even after the rule set changes.
//! - One limit remains: in a world more than one chunk tall a first-parity repair also reaches
//!   corner chunks it does not wait for. A rule set is *streaming-clean* when no repair is ever
//!   needed, which lifts that limit too.

pub mod cell_boxes;
pub mod far_ground;
pub mod frozen;
pub mod generator;
pub mod ground;
mod layers;
pub mod noise;
pub mod occluders;
pub mod products;
pub mod proxies;
pub mod region_tags;
pub mod scheduler;
pub mod space;
pub mod stages;
pub mod towns;
pub mod worker;

pub use cell_boxes::CellBox;
pub use far_ground::{FarGround, far_ground};
pub use frozen::{DirectoryStore, FrozenStore, StoreError};
pub use generator::{ChunkEvent, GeneratorStats, REPAIR_REACH, WorldGenerator};
pub use ground::{
    GroundLevel, GroundMesh, ground, ground_height, ground_materials, ground_readers,
};
pub use occluders::occluders;
pub use products::{InstanceId, InstanceSet, NavSource, NavSourceError, instance_sets, nav_source};
pub use proxies::{ProxyLevel, ProxyMesh, proxy_mesh};
pub use region_tags::{Emitter, RegionTags, region_tags, surface_at};
pub use scheduler::FocusPoint;
pub use space::YUpSpace;
pub use worker::Worker;

pub use wfc_core::{
    BatchResult, Chunk, ChunkCoord, ChunkShape, ChunkStore, Domains, JobId, ModelError, Prior,
    Region, RegionBatch, RegionShape, RegionStats, RegionStatus, RuleTable, Ruleset, SolveBudget,
    Solver, SolverError, TileMask, WorldCell, WorldExtent,
};
#[cfg(feature = "wgpu")]
pub use wfc_gpu::{BlockSolver, Compilation, SolverConfig, wgpu_backend::WgpuBackend};
pub use wfc_rules::{AdjacencyRules, LoadError, TileSet, loader, modules};

use std::sync::Arc;
use thiserror::Error as ErrorDerive;

/// What generating a world can fail at.
#[derive(ErrorDerive, Debug)]
pub enum Error {
    /// The rule set, the world or a chunk is not something the generator can work with.
    #[error(transparent)]
    Model(#[from] ModelError),
    /// The solver refused or failed.
    #[error(transparent)]
    Solver(#[from] SolverError),
    /// A GPU solver could not be built.
    #[cfg(feature = "wgpu")]
    #[error(transparent)]
    Gpu(#[from] wfc_gpu::error::GpuError),
    /// A frozen world's store failed, or held something that is not a chunk of the world.
    #[error(transparent)]
    Store(#[from] frozen::StoreError),
}

/// When a chunk will not solve, how hard to try again.
///
/// A repair solves the failed chunk alone with its halo released, so it may rewrite the
/// neighbouring cells the halo covers. That is what lets a world finish where fixed borders had
/// painted a chunk into a corner, and it is why a repair reports every chunk it touched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RepairPolicy {
    /// Whether a chunk that will not solve is repaired at all. With this off, it is reported as
    /// [`ChunkEvent::Failed`] and left out of the world.
    pub enabled: bool,
    /// The widest halo a repair may use. Wider needs more of the device's workgroup memory than it
    /// has at some point, and the solver is asked before each step.
    pub max_halo: u32,
    /// How hard each seed of a repair tries before a wider halo is the better move.
    pub budget: SolveBudget,
    /// How many seeds a repair tries side by side, in one dispatch. The lowest that solves is kept.
    pub seeds: u32,
}

impl Default for RepairPolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            max_halo: 3,
            // Measured on the city (docs/research/measurements.md): every chunk a streamed world
            // gave up on was placed by 32 seeds of this budget, in a dispatch of 50 ms at most.
            budget: SolveBudget {
                max_attempts: 32,
                max_steps: 20_000,
            },
            seeds: 32,
        }
    }
}

/// How a world is generated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorldConfig {
    /// Every choice in the world derives from this.
    pub seed: u64,
    /// The cells of one chunk.
    pub chunk: ChunkShape,
    /// How far the world reaches.
    pub extent: WorldExtent,
    /// Cells solved around a chunk and then discarded. Without one, a chunk with free faces can
    /// leave border tiles that no row of neighbours can complete (docs/architecture/world.md,
    /// "Chunks and the halo").
    pub halo: u32,
    pub repair: RepairPolicy,
}

/// Builds a [`WorldGenerator`].
pub struct Builder {
    ruleset: Ruleset,
    prior: Prior,
    config: WorldConfig,
    #[cfg(feature = "wgpu")]
    solver: SolverConfig,
}

impl Builder {
    /// Starts describing a world. The chunk shape defaults to 8 by 8 by 8 cells, one chunk tall,
    /// unbounded along x and y.
    #[must_use]
    pub fn new(ruleset: Ruleset, prior: Prior) -> Self {
        let chunk = ChunkShape::cube(8);
        Self {
            ruleset,
            prior,
            config: WorldConfig {
                seed: 0,
                chunk,
                extent: WorldExtent::new(chunk).with_z(0..1),
                halo: 1,
                repair: RepairPolicy::default(),
            },
            #[cfg(feature = "wgpu")]
            solver: SolverConfig::default(),
        }
    }

    /// The world seed every choice derives from.
    #[must_use]
    pub const fn seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// The cells of one chunk. The extent follows it, so set this first.
    #[must_use]
    pub fn chunk_shape(mut self, chunk: ChunkShape) -> Self {
        self.config.chunk = chunk;
        self.config.extent = WorldExtent::new(chunk).with_z(0..1);
        self
    }

    /// How far the world reaches.
    #[must_use]
    pub fn extent(mut self, extent: WorldExtent) -> Self {
        self.config.chunk = extent.shape();
        self.config.extent = extent;
        self
    }

    /// Cells solved around each chunk and then discarded.
    #[must_use]
    pub const fn halo(mut self, halo: u32) -> Self {
        self.config.halo = halo;
        self
    }

    /// What to do about a chunk that will not solve.
    #[must_use]
    pub const fn repair(mut self, repair: RepairPolicy) -> Self {
        self.config.repair = repair;
        self
    }

    /// How the GPU solver runs a region.
    #[cfg(feature = "wgpu")]
    #[must_use]
    pub const fn solver_config(mut self, solver: SolverConfig) -> Self {
        self.solver = solver;
        self
    }

    /// What the world is configured as, for a caller building its own solver.
    #[must_use]
    pub const fn config(&self) -> &WorldConfig {
        &self.config
    }

    /// The rule set the world is generated from.
    #[must_use]
    pub const fn ruleset(&self) -> &Ruleset {
        &self.ruleset
    }

    /// A generator on a GPU device of its own.
    ///
    /// # Errors
    /// If no device is available or the rule set does not fit one.
    #[cfg(feature = "wgpu")]
    pub fn build(self) -> Result<WorldGenerator<BlockSolver<WgpuBackend>>, Error> {
        let backend = WgpuBackend::from_env().map_err(wfc_gpu::error::GpuError::from)?;
        self.build_on_backend(backend)
    }

    /// A generator on a device an engine already owns, which is how a Bevy plugin shares Bevy's.
    ///
    /// # Errors
    /// If the rule set does not fit the device.
    #[cfg(feature = "wgpu")]
    pub fn build_on(
        self,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Result<WorldGenerator<BlockSolver<WgpuBackend>>, Error> {
        self.build_on_backend(WgpuBackend::from_device(device, queue))
    }

    #[cfg(feature = "wgpu")]
    fn build_on_backend(
        self,
        backend: WgpuBackend,
    ) -> Result<WorldGenerator<BlockSolver<WgpuBackend>>, Error> {
        let ruleset = Arc::new(self.ruleset);
        let solver = BlockSolver::new(backend, Arc::clone(&ruleset), self.solver)?;
        Ok(WorldGenerator::new(
            self.config,
            ruleset,
            self.prior,
            solver,
        ))
    }

    /// A generator on a solver of your own: a backend over another engine's compute API, or the CPU
    /// reference in a test.
    #[must_use]
    pub fn build_with<S: Solver>(self, solver: S) -> WorldGenerator<S> {
        WorldGenerator::new(self.config, Arc::new(self.ruleset), self.prior, solver)
    }
}
