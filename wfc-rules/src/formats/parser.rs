use crate::{AdjacencyRules, LoadError, TileSet};

/// Trait defining the interface for format-specific rule parsers.
///
/// Implementors of this trait can parse WFC rules from different file formats
/// (e.g., RON, JSON, YAML, or custom binary formats).
pub trait FormatParser {
    /// Parses rule content into a TileSet and corresponding AdjacencyRules.
    ///
    /// # Arguments
    ///
    /// * `content` - A string slice containing the rule content
    ///
    /// # Returns
    ///
    /// * `Ok((TileSet, AdjacencyRules))` - Successfully parsed rules
    /// * `Err(LoadError)` - Error encountered during parsing
    fn parse(&self, content: &str) -> Result<(TileSet, AdjacencyRules), LoadError>;

    /// Returns a descriptive name for this parser format.
    ///
    /// This can be used for debugging, logging, or user-facing error messages.
    fn format_name(&self) -> &'static str;
}
