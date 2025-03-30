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
pub use crate::entropy::{CpuEntropyCalculator, EntropyCalculator};
pub use crate::grid::PossibilityGrid; // Renamed from Grid<BitVec> if applicable, or adjust
pub use crate::propagator::{ConstraintPropagator, CpuConstraintPropagator, PropagationError};
pub use crate::rules::AdjacencyRules;
pub use crate::runner::run;
pub use crate::tile::{TileId, TileSet};

// Public error type
#[derive(Error, Debug)]
pub enum WfcError {
    #[error("Propagation failed: Contradiction found")]
    Contradiction,
    #[error("Propagation error: {0}")]
    PropagationError(#[from] PropagationError), // Use the re-exported error
    #[error("Grid error: {0}")]
    GridError(String), // Placeholder
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    #[error("Internal error: {0}")]
    InternalError(String),
    #[error("WFC finished prematurely due to incomplete collapse")]
    IncompleteCollapse,
    #[error("WFC exceeded maximum iterations, potential infinite loop")]
    TimeoutOrInfiniteLoop,
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
