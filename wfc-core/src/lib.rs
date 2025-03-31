// Removed: use bitvec::prelude::*; (unused import)
use thiserror::Error;

// Module declarations (keep public if they contain public items)
pub mod entropy;
pub mod grid;
pub mod propagator;
pub mod rules;
pub mod runner;
pub mod tile;

// Re-export core public items

/// Provides CPU-specific implementation for calculating entropy.
pub use crate::entropy::CpuEntropyCalculator;
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
/// Provides CPU-specific implementation for constraint propagation.
pub use crate::propagator::CpuConstraintPropagator;
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
    #[error("Propagation failed: Contradiction found")]
    Contradiction,
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
    /// Optional: Number of contradictions encountered so far.
    pub contradictions: Option<usize>,
    // Add other relevant progress metrics as needed
}
