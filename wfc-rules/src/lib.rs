use thiserror::Error;

pub mod formats;
pub mod loader;

#[derive(Error, Debug)]
pub enum LoadError {
    #[error("I/O error reading file: {0}")]
    Io(#[from] std::io::Error),
    #[error("Failed to parse rules format (e.g., RON/JSON): {0}")]
    ParseError(String), // Placeholder, specific parse errors better
    #[error("Invalid rule data: {0}")]
    InvalidData(String),
    #[error("Unknown loading error")]
    Unknown,
}
