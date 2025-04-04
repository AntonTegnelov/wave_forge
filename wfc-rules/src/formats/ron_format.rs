use crate::types::{AdjacencyRules, TileId, TileSet, TileSetError, Transformation};
use crate::LoadError;
use serde::Deserialize;
use std::collections::HashMap;

use crate::generator::generate_transformed_rules; // Import generator

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
/// validates the data, and converts it into the `TileSet` and `AdjacencyRules`
/// structures defined in this crate (`wfc-rules`).
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
    // Generate default transformations (Identity only for each tile)
    let default_transformations = vec![vec![Transformation::Identity]; weights.len()];

    let tileset = TileSet::new(weights, default_transformations).map_err(|e| {
        // Convert TileSetError to LoadError::InvalidData for consistency
        match e {
            TileSetError::EmptyWeights => {
                LoadError::InvalidData("TileSet weights cannot be empty.".to_string())
            }
            TileSetError::NonPositiveWeight(idx, val) => LoadError::InvalidData(format!(
                "Tile '{}' (index {}) has non-positive weight: {}",
                rule_file.tiles.get(idx).map_or("unknown", |t| &t.name),
                idx,
                val
            )),
            // Handle new TileSet errors related to transformations
            // Treat these as InvalidData as they stem from potentially bad input causing internal issues
            TileSetError::TransformationWeightMismatch(_, _) => LoadError::InvalidData(
                "Internal consistency error: Mismatch between weights and generated transformations."
                    .to_string(),
            ),
            TileSetError::EmptyTransformations(idx) => LoadError::InvalidData(format!(
                "Internal consistency error: Generated empty default transformations for tile index {}.",
                idx
            )),
            TileSetError::MissingIdentityTransformation(idx) => LoadError::InvalidData(format!(
                "Internal consistency error: Missing Identity in default transformations for tile index {}.",
                idx
            )),
        }
    })?;

    // 4. Convert named adjacency rules into a list of BASE rules
    let num_axes = 6; // Assuming 3D standard axes
    let mut base_rules = Vec::new();

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

        base_rules.push((tile1_id, tile2_id, axis_index));
    }

    // 5. Generate the full transformed rule set (now returns tuples)
    let allowed_transformed_tuples = generate_transformed_rules(&base_rules, &tileset, num_axes);

    // 6. Create AdjacencyRules using the transformed rule tuples
    let rules = AdjacencyRules::from_allowed_tuples(
        tileset.num_transformed_tiles(),
        num_axes,
        allowed_transformed_tuples, // Pass the tuples directly
    );

    Ok((tileset, rules))
}
