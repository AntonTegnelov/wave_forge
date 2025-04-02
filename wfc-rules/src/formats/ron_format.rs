use crate::{AdjacencyRules, LoadError, TileId, TileSet, TileSetError};
use serde::Deserialize;
use std::collections::HashMap;

// --- Structs mirroring the RON format ---

/// Represents the data for a single tile as defined in the RON file.
/// Used internally for deserialization.
#[derive(Deserialize, Debug, Clone)]
struct RonTileData {
    /// The unique identifier name for the tile used in rule definitions.
    name: String, // Used for defining rules by name
    /// The weight associated with this tile, influencing its selection probability.
    weight: f32,
    // Add other tile properties like symmetry later if needed
}

/// Represents the top-level structure of the WFC rule file in RON format.
/// Used internally for deserialization.
#[derive(Deserialize, Debug, Clone)]
struct RonRuleFile {
    /// A list defining all available tiles.
    tiles: Vec<RonTileData>,
    /// A list defining the allowed adjacencies between tiles along specific axes.
    /// Each tuple represents `(Tile1 Name, Tile2 Name, Axis Name)`.
    adjacency: Vec<(String, String, String)>,
}

// --- Parsing Logic ---

/// Maps standard axis names (e.g., "+x", "-y") to their corresponding numerical indices (0-5).
/// Used internally during rule parsing.
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

/// Parses WFC rules defined in a RON (Rusty Object Notation) string.
///
/// This function takes the content of a RON file as a string, deserializes it,
/// validates the data, and converts it into the core `TileSet` and `AdjacencyRules`
/// structures used by the `wfc-core` engine.
///
/// # Arguments
///
/// * `ron_content` - A string slice containing the WFC rules in RON format.
///
/// # Returns
///
/// * `Ok((TileSet, AdjacencyRules))` containing the parsed tile set and rules if successful.
/// * `Err(LoadError)` if deserialization fails, the data is invalid (e.g., duplicate tile names,
///   non-positive weights, unknown tile names in rules, invalid axis names), or any other
///   parsing/validation issue occurs.
///
/// # Expected RON Format Example:
///
/// ```ron
/// (
///     tiles: [
///         (name: "Air", weight: 1.0),
///         (name: "Ground", weight: 1.0),
///     ],
///     adjacency: [
///         // Air can be above Ground along +y
///         ("Ground", "Air", "+y"),
///         // Ground can be below Air along -y
///         ("Air", "Ground", "-y"),
///         // Air can be next to Air horizontally
///         ("Air", "Air", "+x"), ("Air", "Air", "-x"),
///         // Ground can be next to Ground horizontally
///         ("Ground", "Ground", "+x"), ("Ground", "Ground", "-x"),
///         // Assuming Z-axis rules are similar...
///     ],
/// )
/// ```
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
