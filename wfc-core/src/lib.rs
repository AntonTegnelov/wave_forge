//! Core library for the Wave Function Collapse algorithm implementation.
//! Defines the fundamental data structures and platform-agnostic logic.

use propagator::PropagationError;
// REMOVED: use bitvec::prelude::{BitVec, Lsb0};
use rand::distr::weighted::Error as WeightedError;
#[cfg(feature = "serde")] // Guard serde imports
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;
use wfc_rules::TileSetError;
// use bitvec::prelude::{BitVec, Lsb0}; // Add necessary bitvec imports

// Module declarations (keep public if they contain public items)
/// Entropy calculation logic and traits.
pub mod constraint;
pub mod entropy;
/// Generic 3D grid structures and specialized WFC grids.
pub mod grid;
/// Constraint propagation logic and traits.
pub mod propagator;
/// Rules that change how likely a tile is rather than whether it is legal.
pub mod weighting;

/// The core WFC algorithm runner.
pub mod runner;

// The model the chunk solver works on (docs/architecture.md §3). These replace the grid and runner
// above, which the old per-collapse GPU loop still uses.
pub mod chunk;
pub mod domains;
pub mod hash;
pub mod prior;
#[cfg(feature = "reference")]
pub mod reference;
pub mod rules;
pub mod solver;
pub mod store;

pub use crate::chunk::{ChunkCoord, ChunkShape, Region, RegionShape, WorldCell};
pub use crate::domains::Domains;
pub use crate::prior::Prior;
pub use crate::rules::{MAX_TILES, RuleTable, Ruleset, TileMask};
pub use crate::solver::{
    BatchResult, JobId, RegionBatch, RegionStats, RegionStatus, SolveBudget, Solver, SolverError,
};
pub use crate::store::{Chunk, ChunkStore, WorldExtent, region_init};

/// What a rule set, a world or a region can be wrong about.
#[derive(Error, Debug)]
pub enum ModelError {
    /// A rule set the solver cannot represent.
    #[error("a rule set of {tiles} tiles is outside the supported range of 1 to {max}")]
    TileCount { tiles: usize, max: u32 },
    /// Rules defined over the wrong number of directions.
    #[error("rules over {axes} axes, but a solver needs the 6 of a cubic grid")]
    AxisCount { axes: usize },
    /// One weight per tile is required.
    #[error("{tiles} tiles but {weights} weights")]
    WeightCount { tiles: usize, weights: usize },
    /// A weight that cannot be used.
    #[error("tile {tile} has weight {weight}, which is not a finite, non-negative number")]
    Weight { tile: usize, weight: f32 },
    /// Every tile has weight zero, so nothing could ever be chosen.
    #[error("no tile has a positive weight")]
    NoPositiveWeight,
    /// Domain words that do not match the cells they describe.
    #[error("{expected} words describe these cells, but {got} were given")]
    WordCount { expected: usize, got: usize },
    /// A chunk outside the world's extent.
    #[error("chunk ({}, {}, {}) is outside the world", chunk.x, chunk.y, chunk.z)]
    OutsideWorld { chunk: crate::chunk::ChunkCoord },
    /// A chunk with the wrong number of cells.
    #[error("a chunk of this world holds {expected} cells, but {got} were given")]
    ChunkCells { expected: usize, got: usize },
    /// A cell that should have been decided by now.
    #[error("cell ({}, {}, {}) is not decided", cell[0], cell[1], cell[2])]
    Undecided { cell: crate::chunk::WorldCell },
}

// Re-export core public items

/// Trait defining the interface for entropy calculation strategies.
pub use crate::entropy::EntropyCalculator;
/// Error type for entropy calculation.
pub use crate::entropy::EntropyError;
/// Grid specifically storing entropy values (f32).
pub use crate::grid::EntropyGrid;
/// Generic 3D grid structure.
pub use crate::grid::Grid;
/// Grid specifically storing possibility bitsets for WFC.
pub use crate::grid::PossibilityGrid;
/// The main function to execute the Wave Function Collapse algorithm.
pub use crate::runner::run;

/// Errors that can occur during the Wave Function Collapse algorithm.
#[derive(Error, Debug)]
pub enum WfcError {
    /// Propagation failed due to finding a cell with no possible tiles remaining.
    /// Includes the (x, y, z) coordinates of the contradictory cell.
    #[error("Propagation failed: Contradiction found at ({0}, {1}, {2})")]
    Contradiction(usize, usize, usize),
    /// An error occurred during the constraint propagation phase.
    #[error("Propagation error: {0}")]
    PropagationError(#[from] PropagationError),
    /// An error related to grid dimensions or accessing grid data.
    #[error("Grid error: {0}")]
    GridError(String),
    /// An error related to invalid configuration (e.g., rules, tileset weights).
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    /// An unexpected internal error occurred.
    #[error("Internal error: {0}")]
    InternalError(String),
    /// WFC finished, but not all cells could be collapsed to a single state.
    #[error("WFC finished prematurely due to incomplete collapse")]
    IncompleteCollapse,
    /// WFC exceeded the maximum number of iterations, likely due to an infinite loop.
    #[error("WFC exceeded maximum iterations, potential infinite loop")]
    TimeoutOrInfiniteLoop,
    /// An error occurred validating the TileSet configuration.
    #[error("TileSet configuration error: {0}")]
    TileSetError(#[from] TileSetError),
    /// WFC run was interrupted by an external signal (e.g., Ctrl+C).
    #[error("WFC run interrupted by signal")]
    Interrupted,
    /// An unknown or unspecified error occurred.
    #[error("Unknown WFC error")]
    Unknown,
    /// Error related to loading or validating a checkpoint.
    #[error("Checkpoint error: {0}")]
    CheckpointError(String),
    /// Error occurred during weighted random selection.
    #[error("Weighted selection error: {0}")]
    WeightedChoiceError(#[from] WeightedError),
    /// An error occurred during entropy calculation.
    #[error("Entropy calculation error: {0}")]
    EntropyError(#[from] EntropyError),
    /// WFC exceeded the configured maximum number of iterations.
    #[error("Maximum iterations ({0}) reached")]
    MaxIterationsReached(u64),
    /// WFC run was interrupted by the external shutdown signal.
    #[error("Shutdown signal received")]
    ShutdownSignalReceived,
    /// An error occurred during propagation (wrapper for PropagationError).
    #[error("Propagation error: {0}")]
    Propagation(PropagationError),
}

/// Information about the current state of the WFC algorithm execution.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ProgressInfo {
    /// The total number of cells that have been collapsed.
    pub collapsed_cells: usize,
    /// The total number of cells in the grid.
    pub total_cells: usize,
    /// Time elapsed since the WFC run started.
    pub elapsed_time: Duration,
    /// The number of iterations completed so far.
    pub iterations: u64,
    /// A clone of the possibility grid state at the time of the callback.
    pub grid_state: PossibilityGrid,
}

/// Represents a saved state of the WFC algorithm for checkpointing.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WfcCheckpoint {
    /// The state of the possibility grid at the time of the checkpoint.
    pub grid: PossibilityGrid,
    /// The number of iterations completed when the checkpoint was saved.
    pub iterations: u64,
    // Note: RNG state is not saved currently.
    // Resuming will use a new RNG seed unless managed externally.
}

/// Defines different boundary handling strategies for the grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum BoundaryCondition {
    /// Edges wrap around (toroidal topology).
    Periodic,
    /// Grid boundaries act as hard walls; neighbors outside the grid are ignored.
    #[default]
    Finite,
}

/// Represents the execution mode (GPU) for WFC components.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionMode {
    Gpu,
}
