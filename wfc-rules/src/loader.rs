use crate::formats::{ron_format::RonFormatParser, FormatParser};
use crate::{formats, AdjacencyRules, LoadError, TileSet};
use std::fs::File;
use std::io::Read as _;
use std::path::Path;

/// Loads a `TileSet` and corresponding `AdjacencyRules` from a rule definition file.
///
/// This function acts as the main entry point for loading WFC rules.
/// It attempts to determine the appropriate parser based on the file extension.
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

    // 2. Determine the parser based on file extension
    let parser = get_parser_for_path(path)?;

    // 3. Parse the content with the selected parser
    parser.parse(&contents)
}

/// Loads a `TileSet` and corresponding `AdjacencyRules` from a string in RON format.
///
/// This function is maintained for backward compatibility with the old API.
///
/// # Arguments
///
/// * `content` - A string containing the RON-formatted rule definition.
///
/// # Returns
///
/// * `Ok((TileSet, AdjacencyRules))` if parsing succeeds.
/// * `Err(LoadError)` if parsing fails.
pub fn load_from_ron_string(content: &str) -> Result<(TileSet, AdjacencyRules), LoadError> {
    formats::ron_format::parse_ron_rules(content)
}

/// Returns an appropriate parser implementation based on the file extension.
///
/// # Arguments
///
/// * `path` - The file path from which to determine the format.
///
/// # Returns
///
/// * `Ok(Box<dyn FormatParser>)` - A parser instance appropriate for the file extension.
/// * `Err(LoadError)` - If no appropriate parser can be determined.
fn get_parser_for_path(path: &Path) -> Result<Box<dyn FormatParser>, LoadError> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("ron") => Ok(Box::new(RonFormatParser::new())),
        // Add additional formats here as they are implemented
        // Some("json") => Ok(Box::new(JsonFormatParser::new())),
        // Some("yaml") | Some("yml") => Ok(Box::new(YamlFormatParser::new())),
        // Some("bin") => Ok(Box::new(BitcodeFormatParser::new())),
        Some(ext) => Err(LoadError::InvalidData(format!(
            "Unsupported file extension: .{ext} - supported formats are: .ron"
        ))),
        None => Err(LoadError::InvalidData(
            "File has no extension - unable to determine format".to_string(),
        )),
    }
}
