//! Core library for the Wave Function Collapse algorithm implementation.
//! Defines the fundamental data structures and platform-agnostic logic.

// Removed: use bitvec::prelude::*; (unused import)
use std::time::Duration;
use thiserror::Error;

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
/// Represents the adjacency rules between tiles.
pub use crate::rules::AdjacencyRules;
/// The main function to execute the Wave Function Collapse algorithm.
pub use crate::runner::run;
/// Represents a unique identifier for a tile.
pub use crate::tile::TileId;
/// Contains information about the set of tiles, like weights.
pub use crate::tile::TileSet;
/// Errors related to TileSet configuration.
pub use crate::tile::TileSetError;

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
}

/// Information about the current state of the WFC algorithm execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgressInfo {
    /// The current iteration number.
    pub iteration: usize,
    /// The total number of cells that have been collapsed.
    pub collapsed_cells: usize,
    /// The total number of cells in the grid.
    pub total_cells: usize,
    /// Time elapsed since the WFC run started.
    pub elapsed_time: Duration,
}
