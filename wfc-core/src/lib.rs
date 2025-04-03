//! Core library for the Wave Function Collapse algorithm implementation.
//! Defines the fundamental data structures and platform-agnostic logic.

// Removed: use bitvec::prelude::*; (unused import)
// use crate::grid::PossibilityGrid;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;
use wfc_rules::TileSetError;

// Module declarations (keep public if they contain public items)
/// Entropy calculation logic and traits.
pub mod entropy;
/// Generic 3D grid structures and specialized WFC grids.
pub mod grid;
/// Constraint propagation logic and traits.
pub mod propagator;
/// Tile adjacency rule representation.
pub mod rules;
/// The core WFC algorithm runner.
pub mod runner;
/// Tile representation (ID, TileSet).
pub mod tile;

// Re-export core public items

/// Trait defining the interface for entropy calculation strategies.
pub use crate::entropy::EntropyCalculator;
/// Grid specifically storing entropy values (f32).
pub use crate::grid::EntropyGrid;
/// Generic 3D grid structure.
pub use crate::grid::Grid;
/// Grid specifically storing possibility bitsets for WFC.
pub use crate::grid::PossibilityGrid;
/// Trait defining the interface for constraint propagation strategies.
pub use crate::propagator::ConstraintPropagator;
/// Errors specific to the propagation phase.
pub use crate::propagator::PropagationError;
/// The main function to execute the Wave Function Collapse algorithm.
pub use crate::runner::run;

/// Errors that can occur during the Wave Function Collapse algorithm.
#[derive(Error, Debug, Clone)]
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
}

/// Information about the current state of the WFC algorithm execution.
#[derive(Debug, Clone)]
pub struct ProgressInfo {
    /// The total number of cells that have been collapsed.
    pub collapsed_cells: usize,
    /// The total number of cells in the grid.
    pub total_cells: usize,
    /// Time elapsed since the WFC run started.
    pub elapsed_time: Duration,
    /// The number of iterations completed so far.
    pub iterations: u64,
}

/// Represents a saved state of the WFC algorithm for checkpointing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WfcCheckpoint {
    /// The state of the possibility grid at the time of the checkpoint.
    pub grid: PossibilityGrid,
    /// The number of iterations completed when the checkpoint was saved.
    pub iterations: u64,
    // Note: RNG state is not saved currently.
    // Resuming will use a new RNG seed unless managed externally.
}
