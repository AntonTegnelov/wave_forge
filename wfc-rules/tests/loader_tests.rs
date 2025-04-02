// use wfc_core::TileId; // This line should already be removed or commented out
use std::path::PathBuf;
use wfc_rules::loader::load_from_file;
use wfc_rules::{AdjacencyRules, LoadError, TileId, TileSet, TileSetError};

// Helper function to create the full path to test data
fn test_data_path(filename: &str) -> std::path::PathBuf {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests");
    path.push("rules_data");
    path.push(filename);
    path
}

#[test]
fn test_load_valid_simple() {
    let path = test_data_path("valid_simple.ron");
    let result = load_from_file(&path);

    assert!(result.is_ok());
    let (tileset, rules) = result.unwrap();

    // Verify TileSet
    assert_eq!(tileset.weights.len(), 2);
    assert!((tileset.weights[0] - 1.0).abs() < f32::EPSILON);
    assert!((tileset.weights[1] - 2.0).abs() < f32::EPSILON);

    // Verify AdjacencyRules
    assert_eq!(rules.num_tiles(), 2);
    assert_eq!(rules.num_axes(), 6);

    let tile_a = TileId(0);
    let tile_b = TileId(1);

    // Check some allowed rules
    assert!(rules.check(tile_a, tile_a, 0)); // A-A +x
    assert!(rules.check(tile_b, tile_b, 3)); // B-B -y
    assert!(rules.check(tile_a, tile_b, 0)); // A-B +x
    assert!(rules.check(tile_b, tile_a, 1)); // B-A -x

    // Check some disallowed rules
    assert!(!rules.check(tile_a, tile_b, 1)); // A-B -x (not explicitly defined)
    assert!(!rules.check(tile_b, tile_a, 0)); // B-A +x (not explicitly defined)
    assert!(!rules.check(tile_a, tile_b, 2)); // A-B +y
}

#[test]
fn test_load_invalid_dup_name() {
    let path = test_data_path("invalid_dup_name.ron");
    let result = load_from_file(&path);
    assert!(result.is_err());
    match result.err().unwrap() {
        LoadError::InvalidData(msg) => assert!(msg.contains("Duplicate tile name found: A")),
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
                "Error message missing 'non-positive weight': {}",
                msg
            );
            assert!(msg.contains("-1"), "Error message missing '-1': {}", msg);
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
            assert!(msg.contains("Rule references unknown tile name: Unknown"))
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
