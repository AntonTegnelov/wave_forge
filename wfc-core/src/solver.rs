//! The seam between the world and whatever solves its regions.
//!
//! A solver takes a batch of equally shaped regions with their starting domains and returns one
//! result per region. It is job-based rather than blocking so that an engine can start work and
//! come back for it on a later frame: Bevy polls on its own thread, Godot polls a task it handed to
//! a worker. Nothing here is async, because a library that needs a runtime forces one into every
//! game (docs/architecture/solver.md, "The solver seam").

use crate::chunk::RegionShape;
use crate::domains::Domains;
use thiserror::Error;

/// How hard a solver should try on a batch.
///
/// A first attempt at a chunk deserves the solver's full budget. A repair does not: when it cannot
/// place a chunk quickly, widening its halo is the better move, and a repair that searches for a
/// second holds up everything behind it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SolveBudget {
    /// Attempts at a region before it is reported as exhausted.
    pub max_attempts: u32,
    /// Steps a region may take, whatever a step means to the solver.
    pub max_steps: u32,
}

/// A batch of regions to solve, all of one shape.
#[derive(Clone, Debug)]
pub struct RegionBatch {
    /// The shape every region in the batch has.
    pub region: RegionShape,
    /// The world identity of each region's chunk, which seeds its choices.
    pub ids: Vec<u32>,
    /// Each region's seed, derived from the world seed and the chunk.
    pub seeds: Vec<u32>,
    /// Every region's starting domains, one region after another.
    pub init: Domains,
    /// How hard to try, or the solver's own default.
    pub budget: Option<SolveBudget>,
    /// Whether the regions are one problem tried with different seeds, of which only the lowest
    /// that solves is wanted. A solver may then stop a region as soon as a lower one has solved,
    /// and report it [`RegionStatus::Superseded`]; the lowest solving region is the same either
    /// way.
    pub portfolio: bool,
}

impl RegionBatch {
    /// How many regions the batch holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the batch is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Whether the batch is consistent: one id and seed per region, and domains for all of them.
    #[must_use]
    pub fn is_well_formed(&self) -> bool {
        self.ids.len() == self.seeds.len()
            && self.init.cells() == self.region.cells() * self.ids.len() as u32
    }
}

/// How a region's solve ended.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RegionStatus {
    /// Every cell of the region is decided.
    Solved,
    /// The solver ran out of attempts.
    Exhausted,
    /// The solver hit its step budget, which exists so a kernel cannot hang a device.
    StepCap,
    /// The region's borders cannot be satisfied at all: propagating the starting domains alone
    /// empties a cell. The caller's recourse is to solve it again with its borders released.
    BorderContradiction,
    /// A lower region of the same portfolio solved first, so this one stopped; its domains are
    /// not a result. See [`RegionBatch::portfolio`].
    Superseded,
}

impl RegionStatus {
    /// Whether the region's domains can be committed.
    #[must_use]
    pub fn is_solved(self) -> bool {
        self == Self::Solved
    }
}

/// What one region's solve cost.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RegionStats {
    pub sweeps: u32,
    pub collapses: u32,
    pub restarts: u32,
    pub backtracks: u32,
    pub steps: u32,
    /// Contradictions met, including those recovered from.
    pub tries: u32,
    /// Where the last contradiction emptied a cell, as an index into the region.
    pub contradiction_cell: Option<u32>,
}

/// One region result per region of the batch, in the batch's order.
#[derive(Clone, Debug)]
pub struct BatchResult {
    pub statuses: Vec<RegionStatus>,
    pub stats: Vec<RegionStats>,
    /// Every region's final domains, one region after another. Only a solved region's are decided.
    pub domains: Domains,
}

impl BatchResult {
    /// The domains of one region of the batch.
    ///
    /// # Panics
    /// If `region` is out of range.
    #[must_use]
    pub fn region(&self, region: usize, cells: u32) -> Domains {
        self.domains.chunk(region as u32, cells)
    }
}

/// A running batch.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct JobId(pub u64);

/// Why a batch could not be started or finished.
#[derive(Error, Debug)]
pub enum SolverError {
    #[error("a batch is already running")]
    Busy,
    #[error("job {0:?} is not the running one")]
    UnknownJob(JobId),
    #[error("the batch is malformed: {0}")]
    Malformed(String),
    #[error("{regions} regions exceed the solver's batch limit of {max}")]
    BatchTooLarge { regions: usize, max: u32 },
    #[error(
        "a region of {needed} B of workgroup storage does not fit the device's {available} B; \
         use a smaller chunk, a smaller halo or fewer tiles"
    )]
    WorkgroupStorage { needed: u32, available: u32 },
    #[error("the solver restored a checkpoint holding an empty cell in region {region}")]
    BadCheckpoint { region: u32 },
    /// The device said the batch finished, but region `region` never reported: the driver ended
    /// the work without running it to the end, and nothing that was read back is a result.
    #[error(
        "the device finished the batch but region {region} never reported; the compute driver \
         dropped the work"
    )]
    NoReport { region: u32 },
    #[error("compute backend: {0}")]
    Backend(String),
}

/// Solves batches of regions.
pub trait Solver {
    /// The most regions one batch may hold.
    fn max_batch(&self) -> u32;

    /// Whether the solver can take regions of this shape at all. A device bounds how large a region
    /// may be, so a caller widening a halo asks rather than guessing.
    fn accepts(&self, region: RegionShape) -> bool {
        let _ = region;
        true
    }

    /// Prepares for batches of these region counts and shapes, so the first batch of each does not
    /// pay for it: a solver that compiles per shape, as the GPU solver does, compiles them here. A
    /// solver with nothing to prepare does nothing.
    ///
    /// # Errors
    /// If the solver cannot prepare for one of them, a kernel that does not compile say.
    fn warm(&mut self, shapes: &[(u32, RegionShape)]) -> Result<(), SolverError> {
        let _ = shapes;
        Ok(())
    }

    /// Starts a batch.
    ///
    /// # Errors
    /// [`SolverError::Busy`] while another batch is running, or if the batch is malformed or too
    /// large for this solver.
    fn start(&mut self, batch: RegionBatch) -> Result<JobId, SolverError>;

    /// Takes the batch's result if it has finished, without blocking.
    ///
    /// # Errors
    /// If `job` is not the running batch, or the solver failed.
    fn poll(&mut self, job: JobId) -> Result<Option<BatchResult>, SolverError>;

    /// Waits for the batch and takes its result.
    ///
    /// # Errors
    /// If `job` is not the running batch, or the solver failed.
    fn wait(&mut self, job: JobId) -> Result<BatchResult, SolverError>;
}
