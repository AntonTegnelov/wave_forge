use crate::LoadError;
use serde::Deserialize;
use std::collections::HashMap;
use wfc_core::{AdjacencyRules, TileId, TileSet};

// --- Structs mirroring the RON format ---

#[derive(Deserialize, Debug, Clone)]
struct RonTileData {
    name: String, // Used for defining rules by name
    weight: f32,
    // Add other tile properties like symmetry later if needed
}

#[derive(Deserialize, Debug, Clone)]
struct RonRuleFile {
    tiles: Vec<RonTileData>,
    // Rules defined as (Tile1 Name, Tile2 Name, Axis Name)
    adjacency: Vec<(String, String, String)>,
}

// --- Parsing Logic ---

/// Parses the RON file content into TileSet and AdjacencyRules.
pub fn parse_ron_rules(ron_content: &str) -> Result<(TileSet, AdjacencyRules), LoadError> {
    // 1. Deserialize the RON string into RonRuleFile struct
    let rule_file: RonRuleFile = ron::from_str(ron_content)
        .map_err(|e| LoadError::ParseError(format!("RON deserialization failed: {}", e)))?;

    // 2. Validate tiles (e.g., check for empty list, duplicate names)
    if rule_file.tiles.is_empty() {
        return Err(LoadError::InvalidData(
            "No tiles defined in rule file.".to_string(),
        ));
    }
    // Create a map for quick name-to-TileId lookup and check duplicates
    let mut tile_name_to_id = HashMap::new();
    for (index, tile_data) in rule_file.tiles.iter().enumerate() {
        if tile_name_to_id
            .insert(tile_data.name.clone(), TileId(index))
            .is_some()
        {
            return Err(LoadError::InvalidData(format!(
                "Duplicate tile name found: {}",
                tile_data.name
            )));
        }
        if tile_data.weight <= 0.0 {
            return Err(LoadError::InvalidData(format!(
                "Tile '{}' has non-positive weight: {}",
                tile_data.name, tile_data.weight
            )));
        }
    }

    // 3. Create TileSet from tile weights
    let weights: Vec<f32> = rule_file.tiles.iter().map(|t| t.weight).collect();
    let tileset = TileSet::new(weights); // Panics on validation handled by TileSet::new

    // 4. Convert named adjacency rules into the flattened boolean format
    // TODO: Implement the conversion logic
    //      - Map axis names ("+x", "-x", etc.) to indices (0-5)
    //      - Use tile_name_to_id map to get TileIds
    //      - Populate the `allowed` Vec<bool>
    //      - Create AdjacencyRules::new()
    let num_tiles = rule_file.tiles.len();
    let num_axes = 6; // Assuming 3D standard axes
    let allowed = vec![false; num_axes * num_tiles * num_tiles]; // Placeholder

    // TODO: Populate 'allowed' based on rule_file.adjacency

    let rules = AdjacencyRules::new(num_tiles, num_axes, allowed);

    Ok((tileset, rules))
}
