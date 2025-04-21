// Only use TileId and Transformation when serde feature is enabled
use crate::formats::FormatParser;
use crate::types::{AdjacencyRules, TileSet, TileSetError};
#[cfg(feature = "serde")]
use crate::types::{TileId, Transformation};
use crate::LoadError;
#[cfg(feature = "serde")]
use serde::Deserialize;
#[cfg(feature = "serde")] // Only need HashMap when deserializing
use std::collections::HashMap;

#[cfg(feature = "serde")] // Only need generator when parsing rules
use crate::generator::generate_transformed_rules;

/// A parser implementation for RON (Rusty Object Notation) format rules.
pub struct RonFormatParser;

impl Default for RonFormatParser {
    fn default() -> Self {
        Self::new()
    }
}

impl RonFormatParser {
    /// Creates a new RON format parser
    pub fn new() -> Self {
        Self
    }
}

// --- Structs mirroring the RON format (only needed with serde) ---

#[cfg(feature = "serde")]
#[cfg_attr(feature = "serde", derive(Deserialize))] // Keep cfg_attr for clarity
#[derive(Debug, Clone)]
struct RonTileData {
    /// The unique identifier name for the tile used in rule definitions.
    name: String, // Used for defining rules by name
    /// The weight associated with this tile, influencing its selection probability.
    weight: f32,
    // Add other tile properties like symmetry later if needed
}

/// Represents the top-level structure of the WFC rule file in RON format.
/// Used internally for deserialization.
#[cfg(feature = "serde")]
#[cfg_attr(feature = "serde", derive(Deserialize))]
#[derive(Debug, Clone)]
struct RonRuleFile {
    /// A list defining all available tiles.
    tiles: Vec<RonTileData>,
    /// A list defining the allowed adjacencies between tiles along specific axes.
    /// Each tuple represents `(Tile1 Name, Tile2 Name, Axis Name)`.
    adjacency: Vec<(String, String, String)>,
}

// --- Parsing Logic ---

// This function is only needed when parsing rules, which requires serde
#[cfg(feature = "serde")]
fn axis_name_to_index(axis_name: &str) -> Result<usize, LoadError> {
    match axis_name {
        "+x" => Ok(0),
        "-x" => Ok(1),
        "+y" => Ok(2),
        "-y" => Ok(3),
        "+z" => Ok(4),
        "-z" => Ok(5),
        _ => {
            Err(LoadError::InvalidData(format!(
                "Invalid axis name: {axis_name}"
            )))
        }
    }
}

// Implement the FormatParser trait
impl FormatParser for RonFormatParser {
    fn format_name(&self) -> &'static str {
        "Rusty Object Notation (RON)"
    }

    #[cfg(feature = "serde")]
    fn parse(&self, ron_content: &str) -> Result<(TileSet, AdjacencyRules), LoadError> {
        // 1. Deserialize the RON string
        let rule_file: RonRuleFile = ron::from_str(ron_content).map_err(|e| {
            LoadError::ParseError(format!("RON deserialization failed: {e}"))
        })?;

        // 2. Validate tiles
        if rule_file.tiles.is_empty() {
            return Err(LoadError::InvalidData("No tiles defined.".to_owned()));
        }
        let mut tile_name_to_id = HashMap::new();
        for (index, tile_data) in rule_file.tiles.iter().enumerate() {
            if tile_name_to_id
                .insert(tile_data.name.clone(), TileId(index))
                .is_some()
            {
                return Err(LoadError::InvalidData(format!(
                    "Duplicate tile name: {}",
                    tile_data.name
                )));
            }
        }

        // 3. Create TileSet
        let weights: Vec<f32> = rule_file.tiles.iter().map(|t| t.weight).collect();
        let default_transformations = vec![vec![Transformation::Identity]; weights.len()];
        let tileset = TileSet::new(weights, default_transformations).map_err(LoadError::from)?;

        // 4. Convert named rules to base rules
        let num_axes = 6;
        let mut base_rules = Vec::new();
        for (t1_name, t2_name, axis_name) in &rule_file.adjacency {
            let tile1_id = *tile_name_to_id
                .get(t1_name)
                .ok_or_else(|| LoadError::InvalidData(format!("Unknown tile: {t1_name}")))?;
            let tile2_id = *tile_name_to_id
                .get(t2_name)
                .ok_or_else(|| LoadError::InvalidData(format!("Unknown tile: {t2_name}")))?;
            let axis_index = axis_name_to_index(axis_name)?;
            base_rules.push((tile1_id, tile2_id, axis_index));
        }

        // 5. Generate transformed rules
        let allowed_transformed_tuples =
            generate_transformed_rules(&base_rules, &tileset, num_axes);

        // 6. Create AdjacencyRules
        let rules = AdjacencyRules::from_allowed_tuples(
            tileset.num_transformed_tiles(),
            num_axes,
            allowed_transformed_tuples,
        );

        Ok((tileset, rules))
    }

    /// Stub implementation when the `serde` feature is not enabled.
    #[cfg(not(feature = "serde"))]
    fn parse(&self, _ron_content: &str) -> Result<(TileSet, AdjacencyRules), LoadError> {
        Err(LoadError::FeatureNotEnabled(
            "serde (required for RON parsing)".to_string(),
        ))
    }
}

/// Parses WFC rules defined in a RON (Rusty Object Notation) string.
/// This implementation is maintained for backward compatibility.
pub fn parse_ron_rules(ron_content: &str) -> Result<(TileSet, AdjacencyRules), LoadError> {
    let parser = RonFormatParser::new();
    parser.parse(ron_content)
}

// Implement From<TileSetError> for LoadError to simplify error handling
// This impl doesn't depend on serde, so it stays outside cfg blocks
impl From<TileSetError> for LoadError {
    fn from(error: TileSetError) -> Self {
        Self::InvalidData(format!("TileSet Error: {error}"))
    }
}
