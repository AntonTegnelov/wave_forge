use wfc_rules::LoadError;
#[cfg(feature = "serde")]
use wfc_rules::{AdjacencyRules, TileSet};

// Helper function to create the full path to test data
fn test_data_path(filename: &str) -> std::path::PathBuf {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests");
    path.push("rules_data");
    path.push(filename);
    path
}

/// The tile set in the file at `path`.
#[cfg(feature = "serde")]
fn load(path: &std::path::Path) -> Result<(TileSet, AdjacencyRules), LoadError> {
    let content = std::fs::read_to_string(path)?;
    wfc_rules::formats::ron_format::parse_ron_rules(&content)
}

#[test]
#[cfg(feature = "serde")] // RON parsing needs serde
fn test_load_valid_simple() {
    let path = test_data_path("valid_simple.ron");
    let result = load(&path);

    assert!(result.is_ok());
    let (tileset, rules) = result.unwrap();

    assert_eq!(tileset.weights.len(), 2);
    assert!((tileset.weights[0] - 1.0).abs() < f32::EPSILON);
    assert!((tileset.weights[1] - 2.0).abs() < f32::EPSILON);
    let (a, b) = (0, 1);
    assert_eq!(rules.num_tiles(), tileset.num_tiles());
    assert_eq!(rules.num_axes(), 6);

    assert!(rules.check(a, a, 0)); // A-A +x
    assert!(rules.check(b, b, 3)); // B-B -y
    assert!(rules.check(a, b, 0)); // A-B +x
    assert!(rules.check(b, a, 1)); // B-A -x

    assert!(!rules.check(a, b, 1)); // A-B -x (not defined)
    assert!(!rules.check(b, a, 0)); // B-A +x (not defined)
    assert!(!rules.check(a, b, 2)); // A-B +y
}

#[test]
#[cfg(feature = "serde")] // RON parsing needs serde
fn test_load_invalid_dup_name() {
    let path = test_data_path("invalid_dup_name.ron");
    let result = load(&path);
    assert!(result.is_err());
    match result.err().unwrap() {
        LoadError::InvalidData(msg) => assert!(msg.contains("Duplicate tile name: A")),
        _ => panic!("Expected InvalidData error for duplicate name"),
    }
}

#[test]
#[cfg(feature = "serde")] // RON parsing needs serde
fn test_load_invalid_neg_weight() {
    let path = test_data_path("invalid_neg_weight.ron");
    let result = load(&path);
    assert!(result.is_err());
    match result.err().unwrap() {
        LoadError::InvalidData(msg) => {
            // Make assertion less brittle, check for key phrases
            assert!(
                msg.contains("positive finite"),
                "Error message missing 'positive finite': {msg}"
            );
            assert!(msg.contains("-1"), "Error message missing '-1': {msg}");
        }
        _ => panic!("Expected InvalidData error for negative weight"),
    }
}

#[test]
#[cfg(feature = "serde")] // RON parsing needs serde
fn test_load_invalid_bad_axis() {
    let path = test_data_path("invalid_bad_axis.ron");
    let result = load(&path);
    assert!(result.is_err());
    match result.err().unwrap() {
        LoadError::InvalidData(msg) => assert!(msg.contains("Invalid axis name: InvalidAxis")),
        _ => panic!("Expected InvalidData error for bad axis"),
    }
}

#[test]
#[cfg(feature = "serde")] // RON parsing needs serde
fn an_adjacency_listed_from_one_tile_only_is_refused() {
    let path = test_data_path("invalid_one_sided.ron");

    let result = load(&path);

    match result {
        Err(LoadError::InvalidData(message)) => {
            assert!(message.contains(r#"("B", "A", "-x")"#), "{message}");
        }
        other => panic!("expected the one-sided pair to be refused, not {other:?}"),
    }
}

#[test]
#[cfg(feature = "serde")] // RON parsing needs serde
fn test_load_invalid_unknown_tile() {
    let path = test_data_path("invalid_unknown_tile.ron");
    let result = load(&path);
    assert!(result.is_err());
    match result.err().unwrap() {
        LoadError::InvalidData(msg) => {
            assert!(msg.contains("Unknown tile: Unknown"));
        }
        _ => panic!("Expected InvalidData error for unknown tile"),
    }
}

#[test]
#[cfg(feature = "serde")] // RON parsing needs serde
fn test_load_invalid_format() {
    let path = test_data_path("invalid_format.ron");
    let result = load(&path);
    assert!(result.is_err());
    match result.err().unwrap() {
        LoadError::ParseError(msg) => assert!(msg.contains("RON deserialization failed")),
        _ => panic!("Expected ParseError for invalid format"),
    }
}

#[test]
#[cfg(feature = "serde")]
fn test_load_file_not_found() {
    let path = test_data_path("non_existent_file.ron");
    let result = wfc_rules::loader::load_rule_file(&path);
    assert!(result.is_err());
    match result.err().unwrap() {
        LoadError::Io(_) => { /* Expected */ }
        _ => panic!("Expected Io error for non-existent file"),
    }
}

/// The example files at the repository root.
#[cfg(feature = "serde")]
fn example(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../examples")
        .join(name)
}

#[test]
#[cfg(feature = "serde")]
fn a_module_set_file_loads_as_modules() {
    let file = wfc_rules::loader::load_rule_file(&example("city.ron")).expect("the city loads");

    let modules = file.modules().expect("the city is written as modules");
    assert_eq!(modules.variants.len(), file.tileset().weights.len());
    assert_eq!(modules.variants.len(), 81, "the city's rotated variants");
    assert!(modules.connector("walkway").is_some());
}

#[test]
#[cfg(feature = "serde")]
fn a_tile_set_file_loads_as_tiles() {
    let file = wfc_rules::loader::load_rule_file(&example("simple-pattern.ron"))
        .expect("the example loads");

    assert!(file.modules().is_none());
    assert_eq!(file.tileset().weights.len(), 2);
}

#[test]
#[cfg(feature = "serde")]
fn a_file_of_neither_form_is_refused() {
    let result = wfc_rules::loader::parse_rule_file("(colours: [\"red\"])");

    assert!(
        matches!(result, Err(LoadError::ParseError(_))),
        "{result:?}"
    );
}

#[test]
#[cfg(feature = "serde")]
fn every_rotation_of_a_module_shares_its_name() {
    let file = wfc_rules::loader::load_rule_file(&example("city.ron")).expect("the city loads");

    let roads = file.tiles_named("road_straight");

    assert_eq!(roads.len(), 2, "a straight road has two distinct turns");
    let turns: Vec<u8> = roads.iter().map(|&tile| file.rotation(tile)).collect();
    assert_eq!(turns, vec![0, 1]);
    assert!(roads.iter().all(|&tile| file.name(tile) == "road_straight"));
}

#[test]
#[cfg(feature = "serde")]
fn tags_pick_every_rotation_of_every_tagged_module() {
    let file = wfc_rules::loader::load_rule_file(&example("city.ron")).expect("the city loads");

    let street_level = file.tiles_tagged("street_level");

    assert!(street_level.contains(&file.tiles_named("grass")[0]));
    assert!(
        file.tiles_named("road_corner")
            .iter()
            .all(|t| street_level.contains(t))
    );
    assert!(!street_level.contains(&file.tiles_named("air")[0]));
}

#[test]
#[cfg(feature = "serde")]
fn a_tile_set_keeps_its_names_and_has_no_rotations() {
    let file = wfc_rules::loader::load_rule_file(&example("simple-pattern.ron"))
        .expect("the example loads");

    assert_eq!(file.name(0), "Empty");
    assert_eq!(file.tiles_named("Block"), vec![1]);
    assert_eq!(file.rotation(1), 0);
    assert!(file.tiles_tagged("anything").is_empty());
}
