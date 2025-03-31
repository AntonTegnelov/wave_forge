use thiserror::Error;

pub mod formats;
pub mod loader;

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
}
