//! Module defining parsers for different rule file formats.

// Export the core parser trait
pub mod parser;
pub use parser::FormatParser;

// Format-specific implementations
/// Module sets written as RON.
#[cfg(feature = "serde")]
pub mod module_format;
pub mod ron_format; // Example
// pub mod json_format; // Example
