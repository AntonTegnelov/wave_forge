//! Core library for the Wave Function Collapse algorithm implementation.
//! Defines the fundamental data structures and platform-agnostic logic.

use propagator::propagator::PropagationError;
// REMOVED: use bitvec::prelude::{BitVec, Lsb0};
use rand::distributions::WeightedError;
#[cfg(feature = "serde")] // Guard serde imports
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;
use wfc_rules::TileSetError;
// use bitvec::prelude::{BitVec, Lsb0}; // Add necessary bitvec imports

// Module declarations (keep public if they contain public items)
/// Entropy calculation logic and traits.
pub mod entropy;
/// Generic 3D grid structures and specialized WFC grids.
pub mod grid;
/// Constraint propagation logic and traits.
pub mod propagator;

/// The core WFC algorithm runner.
pub mod runner;

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
