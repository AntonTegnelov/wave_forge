//! Crate responsible for loading and parsing WFC tile and rule definitions from external sources.

use thiserror::Error;

/// Contains format-specific parsing logic (e.g., RON).
pub mod formats;
/// Contains logic for generating expanded rules based on symmetry.
pub mod generator;
/// Contains the main rule loading functions.
pub mod loader;
/// Contains the core types (`TileId`, `TileSet`, `AdjacencyRules`).
pub mod types;

/// Re-export core types for easier access
pub use types::{AdjacencyRules, TileId, TileSet, TileSetError, Transformation};

/// Errors that can occur when loading or parsing adjacency rules.
#[derive(Error, Debug)]
pub enum LoadError {
    /// An underlying I/O error occurred while trying to read a rule file.
    #[error("I/O error reading file: {0}")]
    Io(#[from] std::io::Error),
    /// An error occurred while parsing the rule file format (e.g., RON, JSON).
    /// Contains a description of the parsing failure.
    #[error("Failed to parse rules format (e.g., RON/JSON): {0}")]
    ParseError(String),
    /// The rule data, although parsed correctly, was found to be invalid or inconsistent.
    /// Contains a description of the validation failure.
    #[error("Invalid rule data: {0}")]
    InvalidData(String),
    /// An unspecified error occurred during the loading process.
    #[error("Unknown loading error")]
    Unknown,
    /// Required feature not enabled
    #[error("Required feature not enabled: {0}")]
    FeatureNotEnabled(String),
}
