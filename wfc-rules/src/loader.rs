use crate::{formats, LoadError};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use wfc_core::{AdjacencyRules, TileSet};

/// Loads TileSet and AdjacencyRules from a specified file path.
///
/// The file format is determined by the implementation (currently expects RON).
pub fn load_from_file(path: &Path) -> Result<(TileSet, AdjacencyRules), LoadError> {
    // 1. Read the file content
    let mut file = File::open(path).map_err(LoadError::Io)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).map_err(LoadError::Io)?;

    // 2. Parse the content (delegating to the format-specific parser)
    // Assuming RON format for now.
    formats::ron_format::parse_ron_rules(&contents)
}
