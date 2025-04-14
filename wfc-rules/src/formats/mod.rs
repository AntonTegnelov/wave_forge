//! Module defining parsers for different rule file formats.

// Export the core parser trait
pub mod parser;
pub use parser::FormatParser;

// Format-specific implementations
pub mod ron_format; // Example
                    // pub mod json_format; // Example
