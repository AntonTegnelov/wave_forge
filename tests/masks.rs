//! N4's check (docs/product/user-stories.md): a mask painted with a brush becomes edits a Rules
//! stage reads, a town is placed on it, its WFC stage fills only the masked columns, and painting
//! again solves only that town again.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::loader::{RuleFile, parse_rule_file};
use wave_forge::stages::brushes::{Brush, stroke};
use wave_forge::stages::{Edits, Pack, Runtime};
use wave_forge::towns::WfcTowns;
use wave_forge::{ChunkCoord, ChunkShape, FocusPoint};
use wfc_core::reference::ReferenceSolver;

const RULES: &str = r#"(
    faces: {
        "air": Side(connector: "air"),
        "ground": Side(connector: "ground", walkable: true),
        "open": Top(connector: "open"),
        "bedrock": Top(connector: "bedrock"),
    },
    modules: [
        (name: "air", sides: ["air", "air", "air", "air"], up: "open", down: "open"),
        (name: "ground", sides: ["ground", "ground", "ground", "ground"], up: "open", down: "bedrock", tags: ["street_level"]),
        (name: "plaza", sides: ["ground", "ground", "ground", "ground"], up: "open", down: "bedrock", tags: ["street_level"], weight: 3.0),
    ],
)"#;

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "height", kind: Field(Constant(4.0))),
        (name: "paint", kind: Field(Constant(0.0))),
        (name: "district", kind: Rules(rules: [
            (category: "town", when: [Greater(Input("paint"), Constant(0.5))]),
        ], otherwise: "wild")),
        (name: "town_mask", kind: Field(Is("district", ["town"]))),
        (name: "sites", kind: Locations(height: "height", region: 4, kinds: [
            (name: "town", quota: 1, size: 2, tries: 200, when: [Greater(Input("town_mask"), Constant(0.5))]),
        ])),
        (name: "buildings", kind: Solve(sites: "sites", rules: "blocks",
            bottom: Some(Tagged("street_level")), top: Some(Named("air")),
            mask: Some((field: "town_mask", above: 0.5, outside: Named("air"), ground: Some(Named("ground")))))),
    ],
)"#;

const CHUNK: ChunkShape = ChunkShape { x: 4, y: 4, z: 3 };

fn runtime() -> Runtime {
    let file = parse_rule_file(RULES).expect("a module set");
    let towns = WfcTowns::new(CHUNK)
        .with_rules("blocks", file, |ruleset| Ok(ReferenceSolver::new(ruleset)))
        .expect("the rules compile");
    Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        3,
        [CHUNK.x, CHUNK.y],
    )
    .with_towns(Box::new(towns))
    .expect("matching chunks")
}

fn area() -> Vec<ChunkCoord> {
    (0..4)
        .flat_map(|y| (0..4).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

fn generate(runtime: &mut Runtime) {
    let focus: Vec<FocusPoint> = area().iter().map(|&c| FocusPoint::new(c, 0)).collect();
    runtime
        .request(&focus, &["buildings", "town_mask"])
        .expect("the stages");
    runtime.run_until_idle().expect("the stages run");
}

/// Paints `brush` along `path` on top of what is painted already.
fn paint(runtime: &mut Runtime, brush: &Brush, path: &[[f32; 3]]) {
    let mut log = runtime.save().edits;
    log.log
        .extend(stroke(runtime, brush, path).expect("a brush of a field"));
    runtime
        .set_edits(&Edits { log: log.log })
        .expect("raises of a field");
}

/// The tile names of every chunk of the town, by chunk, and each column's tiles bottom up.
fn towns(runtime: &Runtime) -> BTreeMap<ChunkCoord, Vec<Vec<String>>> {
    let RuleFile::Modules(modules) = parse_rule_file(RULES).expect("a module set") else {
        panic!("a module set");
    };
    let mut out = BTreeMap::new();
    for chunk in area() {
        let Some(town) = runtime.tiles("buildings", chunk) else {
            continue;
        };
        let columns = (0..CHUNK.x * CHUNK.y)
            .map(|column| {
                (0..CHUNK.z)
                    .map(|z| {
                        let tile = town.tiles[(z * CHUNK.x * CHUNK.y + column) as usize];
                        modules.prototype_of(usize::from(tile)).name.clone()
                    })
                    .collect()
            })
            .collect();
        out.insert(chunk, columns);
    }
    out
}

/// Whether the mask covers the column `column` of `chunk`.
fn masked(runtime: &Runtime, chunk: ChunkCoord, column: u32) -> bool {
    runtime.field("town_mask", chunk).expect("generated").values[column as usize] > 0.5
}

#[test]
fn a_town_fills_only_the_masked_columns_of_its_site() {
    let mut runtime = runtime();
    let brush = Brush::Raise {
        stage: "paint".to_owned(),
        radius: 3.5,
        strength: 1.0,
    };
    paint(&mut runtime, &brush, &[[6.0, 6.0, 0.0], [9.0, 8.0, 0.0]]);

    generate(&mut runtime);

    let towns = towns(&runtime);
    assert!(!towns.is_empty(), "no town on the mask");
    let (mut inside, mut outside, mut plazas) = (0, 0, 0);
    for (&chunk, columns) in &towns {
        for (column, tiles) in columns.iter().enumerate() {
            if masked(&runtime, chunk, column as u32) {
                inside += 1;
                plazas += usize::from(tiles[0] == "plaza");
            } else {
                assert_eq!(
                    tiles[0], "ground",
                    "{chunk:?} column {column} outside the mask"
                );
                assert!(tiles[1..].iter().all(|tile| tile == "air"), "{tiles:?}");
                outside += 1;
            }
        }
    }
    assert!(
        inside > 8 && outside > 8 && plazas > 0,
        "{inside} in, {outside} out, {plazas} plazas"
    );
}

#[test]
fn painting_again_solves_only_that_town_again() {
    let mut runtime = runtime();
    let brush = Brush::Raise {
        stage: "paint".to_owned(),
        radius: 3.5,
        strength: 1.0,
    };
    paint(&mut runtime, &brush, &[[6.0, 6.0, 0.0], [9.0, 8.0, 0.0]]);
    generate(&mut runtime);
    let before = towns(&runtime);
    let site: Vec<ChunkCoord> = before.keys().copied().collect();
    let districts_before: BTreeMap<ChunkCoord, Vec<u8>> = area()
        .into_iter()
        .map(|chunk| {
            let district = runtime.categories("district", chunk).expect("generated");
            (chunk, district.values.clone())
        })
        .collect();

    // Erase a corner of the mask inside the site, away from its centre, so the site stays.
    let eraser = Brush::Raise {
        stage: "paint".to_owned(),
        radius: 1.5,
        strength: -1.0,
    };
    paint(&mut runtime, &eraser, &[[9.5, 8.5, 0.0]]);
    generate(&mut runtime);
    let after = towns(&runtime);

    assert_eq!(
        after.keys().copied().collect::<Vec<_>>(),
        site,
        "the site moved"
    );
    assert_ne!(after, before, "the town was not solved again");
    // Only the chunks the eraser reaches are painted differently.
    for chunk in area() {
        let (x0, y0) = (chunk.x as f32 * 4.0, chunk.y as f32 * 4.0);
        let reached = (x0 - 1.5..x0 + 5.5).contains(&9.5) && (y0 - 1.5..y0 + 5.5).contains(&8.5);
        if !reached {
            let district = runtime.categories("district", chunk).expect("generated");
            assert_eq!(district.values, districts_before[&chunk], "{chunk:?}");
        }
    }
    for (&chunk, columns) in &after {
        for (column, tiles) in columns.iter().enumerate() {
            if !masked(&runtime, chunk, column as u32) {
                assert_eq!(tiles[0], "ground", "{chunk:?} column {column}");
            }
        }
    }
}
