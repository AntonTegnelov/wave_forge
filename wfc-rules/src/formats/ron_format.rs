use crate::LoadError;
use serde::Deserialize;
use std::collections::HashMap;
use wfc_core::{AdjacencyRules, TileId, TileSet, TileSetError};

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

// Map axis names to indices (consistent with wfc-core::propagator)
fn axis_name_to_index(axis_name: &str) -> Result<usize, LoadError> {
    match axis_name {
        "+x" => Ok(0),
        "-x" => Ok(1),
        "+y" => Ok(2),
        "-y" => Ok(3),
        "+z" => Ok(4),
        "-z" => Ok(5),
        _ => Err(LoadError::InvalidData(format!(
            "Invalid axis name: {}",
            axis_name
        ))),
    }
}

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
    }

    // 3. Create TileSet from tile weights, handling the Result
    let weights: Vec<f32> = rule_file.tiles.iter().map(|t| t.weight).collect();
    let tileset = TileSet::new(weights).map_err(|e| match e {
        // Convert TileSetError to LoadError::InvalidData for consistency
        TileSetError::EmptyWeights => {
            LoadError::InvalidData("TileSet weights cannot be empty.".to_string())
        }
        TileSetError::NonPositiveWeight(idx, val) => LoadError::InvalidData(format!(
            "Tile '{}' (index {}) has non-positive weight: {}",
            rule_file.tiles.get(idx).map_or("unknown", |t| &t.name),
            idx,
            val
        )),
    })?;

    // 4. Convert named adjacency rules into the flattened boolean format
    let num_tiles = rule_file.tiles.len();
    let num_axes = 6; // Assuming 3D standard axes
    let mut allowed = vec![false; num_axes * num_tiles * num_tiles]; // Initialize to false

    for (t1_name, t2_name, axis_name) in &rule_file.adjacency {
        // Get TileIds from names
        let tile1_id = *tile_name_to_id.get(t1_name).ok_or_else(|| {
            LoadError::InvalidData(format!("Rule references unknown tile name: {}", t1_name))
        })?;
        let tile2_id = *tile_name_to_id.get(t2_name).ok_or_else(|| {
            LoadError::InvalidData(format!("Rule references unknown tile name: {}", t2_name))
        })?;

        // Get axis index from name
        let axis_index = axis_name_to_index(axis_name)?;

        // Calculate the flat index
        // Formula: axis * num_tiles * num_tiles + tile1.0 * num_tiles + tile2.0
        let flat_index = axis_index * num_tiles * num_tiles + tile1_id.0 * num_tiles + tile2_id.0;

        // Set the corresponding entry to true
        if flat_index >= allowed.len() {
            // Should not happen with valid TileIds and axis_index, but check defensively
            return Err(LoadError::InvalidData(format!(
                 "Internal error: Calculated rule index out of bounds for rule ({:?}, {:?}, axis {})",
                 tile1_id, tile2_id, axis_index
             )));
        }
        allowed[flat_index] = true;
    }

    let rules = AdjacencyRules::new(num_tiles, num_axes, allowed);

    Ok((tileset, rules))
}
