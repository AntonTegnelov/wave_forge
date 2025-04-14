// use wfc_core::TileId; // This line should already be removed or commented out
// Removed unused: use std::fs;
// Removed unused: use tempfile::tempdir;
use wfc_rules::loader::load_from_file;
use wfc_rules::{LoadError, TileId, Transformation}; // Removed unused: AdjacencyRules, TileSet, TileSetError

// Helper function to create the full path to test data
fn test_data_path(filename: &str) -> std::path::PathBuf {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests");
    path.push("rules_data");
    path.push(filename);
    return path
}

#[test]
fn test_load_valid_simple() {
    let path = test_data_path("valid_simple.ron");
    let result = load_from_file(&path);

    assert!(result.is_ok());
    let (tileset, rules) = result.unwrap();

    // Verify TileSet
    assert_eq!(tileset.weights.len(), 2);
    assert_eq!(tileset.allowed_transformations.len(), 2);
    assert!(tileset.allowed_transformations[0].contains(&Transformation::Identity));
    assert_eq!(tileset.allowed_transformations[0].len(), 1);
    assert!(tileset.allowed_transformations[1].contains(&Transformation::Identity));
    assert_eq!(tileset.allowed_transformations[1].len(), 1);
    assert!((tileset.weights[0] - 1.0).abs() < f32::EPSILON);
    assert!((tileset.weights[1] - 2.0).abs() < f32::EPSILON);

    // Get transformed IDs for base tiles (Identity transformation)
    let ttid_a = tileset
        .get_transformed_id(TileId(0), Transformation::Identity)
        .expect("Failed to get transformed ID for Tile A");
    let ttid_b = tileset
        .get_transformed_id(TileId(1), Transformation::Identity)
        .expect("Failed to get transformed ID for Tile B");

    // Verify AdjacencyRules
    // Since valid_simple.ron doesn't define transformations, TileSet creates Identity only.
    // The generator should only produce rules for Identity transforms.
    // num_tiles should reflect only the base tiles * their Identity transforms (so, still 2).
    assert_eq!(tileset.num_transformed_tiles(), 2);
    assert_eq!(rules.num_tiles(), tileset.num_transformed_tiles());
    assert_eq!(rules.num_axes(), 6);

    // Check some allowed rules using transformed IDs
    assert!(rules.check(ttid_a, ttid_a, 0)); // A-A +x
    assert!(rules.check(ttid_b, ttid_b, 3)); // B-B -y
    assert!(rules.check(ttid_a, ttid_b, 0)); // A-B +x
    assert!(rules.check(ttid_b, ttid_a, 1)); // B-A -x

    // Check some disallowed rules using transformed IDs
    assert!(!rules.check(ttid_a, ttid_b, 1)); // A-B -x (not explicitly defined)
    assert!(!rules.check(ttid_b, ttid_a, 0)); // B-A +x (not explicitly defined)
    assert!(!rules.check(ttid_a, ttid_b, 2)); // A-B +y
}

#[test]
fn test_load_invalid_dup_name() {
    let path = test_data_path("invalid_dup_name.ron");
    let result = load_from_file(&path);
    assert!(result.is_err());
    match result.err().unwrap() {
        LoadError::InvalidData(msg) => assert!(msg.contains("Duplicate tile name: A")),
        _ => panic!("Expected InvalidData error for duplicate name"),
    }
}

#[test]
fn test_load_invalid_neg_weight() {
    let path = test_data_path("invalid_neg_weight.ron");
    let result = load_from_file(&path);
    assert!(result.is_err());
    match result.err().unwrap() {
        LoadError::InvalidData(msg) => {
            // Make assertion less brittle, check for key phrases
            assert!(
                msg.contains("non-positive weight"),
                "Error message missing 'non-positive weight': {msg}"
            );
            assert!(msg.contains("-1"), "Error message missing '-1': {msg}");
        }
        _ => panic!("Expected InvalidData error for negative weight"),
    }
}

#[test]
fn test_load_invalid_bad_axis() {
    let path = test_data_path("invalid_bad_axis.ron");
    let result = load_from_file(&path);
    assert!(result.is_err());
    match result.err().unwrap() {
        LoadError::InvalidData(msg) => assert!(msg.contains("Invalid axis name: InvalidAxis")),
        _ => panic!("Expected InvalidData error for bad axis"),
    }
}

#[test]
fn test_load_invalid_unknown_tile() {
    let path = test_data_path("invalid_unknown_tile.ron");
    let result = load_from_file(&path);
    assert!(result.is_err());
    match result.err().unwrap() {
        LoadError::InvalidData(msg) => {
            assert!(msg.contains("Unknown tile: Unknown"));
        }
        _ => panic!("Expected InvalidData error for unknown tile"),
    }
}

#[test]
fn test_load_invalid_format() {
    let path = test_data_path("invalid_format.ron");
    let result = load_from_file(&path);
    assert!(result.is_err());
    match result.err().unwrap() {
        LoadError::ParseError(msg) => assert!(msg.contains("RON deserialization failed")),
        _ => panic!("Expected ParseError for invalid format"),
    }
}

#[test]
fn test_load_file_not_found() {
    let path = test_data_path("non_existent_file.ron");
    let result = load_from_file(&path);
    assert!(result.is_err());
    match result.err().unwrap() {
        LoadError::Io(_) => { /* Expected */ }
        _ => panic!("Expected Io error for non-existent file"),
    }
}
