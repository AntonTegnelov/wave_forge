//! A pack written back as text, and as the plain data an engine edits it as: every pack in the
//! repository comes back the same, so a stack in an engine can save a pack without losing any of it.

use std::path::{Path, PathBuf};
use wave_forge::stages::{Pack, PackFile};

/// Every pack file in the repository: `*.world.ron` under `examples` and the Godot project.
fn packs() -> Vec<PathBuf> {
    fn walk(dir: &Path, found: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("a directory") {
            let path = entry.expect("an entry").path();
            if path.is_dir() {
                if path.file_name().is_some_and(|name| name != "addons") {
                    walk(&path, found);
                }
            } else if path.to_string_lossy().ends_with(".world.ron") {
                found.push(path);
            }
        }
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut found = Vec::new();
    walk(&root.join("examples"), &mut found);
    walk(&root.join("wave_forge_godot/godot"), &mut found);
    found.sort();
    found
}

fn read(path: &Path) -> PackFile {
    let text = std::fs::read_to_string(path).expect("the pack");
    ron::from_str(&text).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

/// Every number of `value` as an engine whose numbers are all floating point gives it back, a
/// whole one written as an integer.
fn as_floats(value: serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match value {
        Value::Number(number) => {
            let float = number.as_f64().expect("a number");
            if float.fract() == 0.0 && float.abs() < 9.0e15 {
                Value::from(float as i64)
            } else {
                Value::from(float)
            }
        }
        Value::Array(items) => Value::Array(items.into_iter().map(as_floats).collect()),
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(key, value)| (key, as_floats(value)))
                .collect(),
        ),
        other => other,
    }
}

#[test]
fn every_pack_written_as_text_reads_back_the_same() {
    let packs = packs();

    for path in &packs {
        let file = read(path);

        let text = file.to_text();

        let again: PackFile = ron::from_str(&text).expect("its own text");
        assert_eq!(again, file, "{}", path.display());
        assert!(Pack::parse(&text).is_ok(), "{}", path.display());
    }
    assert!(packs.len() > 20, "only {} packs", packs.len());
}

#[test]
fn every_pack_as_plain_data_reads_back_the_same_with_its_numbers_all_floats() {
    for path in packs() {
        let file = read(&path);

        let data = as_floats(serde_json::to_value(&file).expect("plain data"));

        let again: PackFile = serde_json::from_value(data).expect("the pack from its data");
        assert_eq!(again, file, "{}", path.display());
    }
}
