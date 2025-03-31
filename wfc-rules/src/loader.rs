use crate::{formats, LoadError};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use wfc_core::{AdjacencyRules, TileSet};

/// Loads a `TileSet` and corresponding `AdjacencyRules` from a rule definition file.
///
/// This function acts as the main entry point for loading WFC rules.
/// It currently expects the file to be in RON (Rusty Object Notation) format,
/// but the underlying parsing is handled by the `formats` module.
///
/// # Arguments
///
/// * `path` - A reference to the `Path` of the rule file to load.
///
/// # Returns
///
/// * `Ok((TileSet, AdjacencyRules))` containing the loaded tile information and adjacency constraints
///   if the file is successfully read and parsed.
/// * `Err(LoadError)` if any error occurs during file reading, parsing, or data validation.
///   The specific error type (`Io`, `ParseError`, `InvalidData`) provides more details.
pub fn load_from_file(path: &Path) -> Result<(TileSet, AdjacencyRules), LoadError> {
    // 1. Read the file content
    let mut file = File::open(path).map_err(LoadError::Io)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).map_err(LoadError::Io)?;

    // 2. Parse the content (delegating to the format-specific parser)
    // Assuming RON format for now.
    formats::ron_format::parse_ron_rules(&contents)
}
