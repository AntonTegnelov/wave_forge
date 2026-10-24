//! Tile sets written as RON: tiles with weights, and every adjacency listed by hand.

use crate::LoadError;
use crate::types::{AdjacencyRules, TileSet, TileSetError};
#[cfg(feature = "serde")]
use serde::Deserialize;
#[cfg(feature = "serde")]
use std::collections::HashMap;

/// One tile of a tile set file.
#[cfg(feature = "serde")]
#[derive(Debug, Clone, Deserialize)]
struct RonTileData {
    /// The name rules refer to the tile by.
    name: String,
    /// The tile's relative weight in a choice.
    weight: f32,
}

/// A tile set file.
#[cfg(feature = "serde")]
#[derive(Debug, Clone, Deserialize)]
struct RonRuleFile {
    /// Every tile.
    tiles: Vec<RonTileData>,
    /// The allowed adjacencies: `(tile, its neighbour, the axis from the tile to it)`.
    adjacency: Vec<(String, String, String)>,
}

/// The axes' names, in index order.
#[cfg(feature = "serde")]
const AXIS_NAMES: [&str; 6] = ["+x", "-x", "+y", "-y", "+z", "-z"];

#[cfg(feature = "serde")]
fn axis_name_to_index(axis_name: &str) -> Result<usize, LoadError> {
    AXIS_NAMES
        .iter()
        .position(|name| *name == axis_name)
        .ok_or_else(|| LoadError::InvalidData(format!("Invalid axis name: {axis_name}")))
}

/// Parses a tile set written as RON.
///
/// # Errors
/// [`LoadError::ParseError`] if the text is not a tile set file, and [`LoadError::InvalidData`] for
/// no tiles, a duplicate or unknown tile name, an unknown axis, a weight that is not a positive
/// finite number, or an adjacency listed from one tile only.
#[cfg(feature = "serde")]
pub fn parse_ron_rules(ron_content: &str) -> Result<(TileSet, AdjacencyRules), LoadError> {
    let rule_file: RonRuleFile = ron::from_str(ron_content)
        .map_err(|e| LoadError::ParseError(format!("RON deserialization failed: {e}")))?;
    if rule_file.tiles.is_empty() {
        return Err(LoadError::InvalidData("No tiles defined.".to_owned()));
    }
    let mut tile_name_to_id = HashMap::new();
    for (index, tile_data) in rule_file.tiles.iter().enumerate() {
        if tile_name_to_id
            .insert(tile_data.name.clone(), index)
            .is_some()
        {
            return Err(LoadError::InvalidData(format!(
                "Duplicate tile name: {}",
                tile_data.name
            )));
        }
    }
    let tileset = TileSet::new(rule_file.tiles.iter().map(|t| t.weight).collect())?;

    let mut tuples = Vec::new();
    for (t1_name, t2_name, axis_name) in &rule_file.adjacency {
        let tile1 = *tile_name_to_id
            .get(t1_name)
            .ok_or_else(|| LoadError::InvalidData(format!("Unknown tile: {t1_name}")))?;
        let tile2 = *tile_name_to_id
            .get(t2_name)
            .ok_or_else(|| LoadError::InvalidData(format!("Unknown tile: {t2_name}")))?;
        tuples.push((axis_name_to_index(axis_name)?, tile1, tile2));
    }
    // The solver reads a pair from whichever tile was decided first, so a pair allowed from one
    // side only would be allowed or not by the order chunks are generated in. Axes come in pairs,
    // +x then -x and so on, so an axis's opposite differs in its lowest bit.
    for (t1_name, t2_name, axis_name) in &rule_file.adjacency {
        let (tile1, tile2) = (tile_name_to_id[t1_name], tile_name_to_id[t2_name]);
        let opposite = axis_name_to_index(axis_name)? ^ 1;
        if !tuples.contains(&(opposite, tile2, tile1)) {
            return Err(LoadError::InvalidData(format!(
                "({t1_name:?}, {t2_name:?}, {axis_name:?}) has no ({t2_name:?}, {t1_name:?}, {:?}): \
                 every adjacency is listed from both tiles",
                AXIS_NAMES[opposite]
            )));
        }
    }
    let rules = AdjacencyRules::from_allowed_tuples(tileset.num_tiles(), AXIS_NAMES.len(), tuples);
    Ok((tileset, rules))
}

/// Parsing a tile set needs the `serde` feature.
///
/// # Errors
/// Always [`LoadError::FeatureNotEnabled`].
#[cfg(not(feature = "serde"))]
pub fn parse_ron_rules(_ron_content: &str) -> Result<(TileSet, AdjacencyRules), LoadError> {
    Err(LoadError::FeatureNotEnabled(
        "serde (required for RON parsing)".to_string(),
    ))
}

impl From<TileSetError> for LoadError {
    fn from(error: TileSetError) -> Self {
        Self::InvalidData(format!("TileSet Error: {error}"))
    }
}
