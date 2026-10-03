//! Presets and their parameters (docs/product/user-stories.md, N2): a pack's parameters change
//! what reads them and nothing else, and every value in each preset's ranges gives a sound world,
//! swept over a grid of values and seeds: islands, and hills and forests.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::{Pack, PackError, Runtime, StageError};
use wave_forge::{ChunkCoord, FocusPoint};

const SIZE: [u32; 2] = [8, 8];

/// The preset `examples/presets/<name>.world.ron`.
fn preset(name: &str) -> Arc<Pack> {
    let text = std::fs::read_to_string(format!(
        "{}/examples/presets/{name}.world.ron",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("the preset");
    Arc::new(Pack::parse(&text).expect("a valid pack"))
}

fn islands() -> Arc<Pack> {
    preset("islands")
}

fn area() -> Vec<ChunkCoord> {
    (-4..4)
        .flat_map(|y| (-4..4).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

fn generate(runtime: &mut Runtime, stages: &[&str]) {
    let focus: Vec<FocusPoint> = area().iter().map(|&c| FocusPoint::new(c, 0)).collect();
    runtime.request(&focus, stages).expect("the stages");
    runtime.run_until_idle().expect("the stages run");
}

fn params(values: [(&str, f32); 3]) -> BTreeMap<String, f32> {
    values
        .iter()
        .map(|&(name, value)| (name.to_owned(), value))
        .collect()
}

/// The share of the area's columns above the water, and how many trees stand in it, checking on
/// the way that every height is finite and every tree stands on grass above the water.
fn survey(runtime: &Runtime) -> (f32, usize) {
    let (mut land, mut columns, mut trees) = (0, 0, 0);
    for chunk in area() {
        let height = runtime.field("height", chunk).expect("generated");
        for &value in &height.values {
            assert!(
                value.is_finite() && value.abs() < 40.0,
                "a height of {value}"
            );
            land += usize::from(value > 0.0);
            columns += 1;
        }
        for tree in runtime.points("trees", chunk).expect("generated") {
            assert!(tree.position[2] > 1.2, "a tree at {:?}", tree.position);
            trees += 1;
        }
    }
    (land as f32 / columns as f32, trees)
}

#[test]
fn every_value_in_the_islands_ranges_gives_a_sound_world() {
    let pack = islands();
    let grid = [0.0, 0.5, 1.0];

    for seed in [1, 2] {
        for land in grid {
            for roughness in grid {
                for trees in grid {
                    let mut runtime = Runtime::new(Arc::clone(&pack), seed, SIZE);
                    let values =
                        params([("land", land), ("roughness", roughness), ("trees", trees)]);
                    runtime.set_params(&values).expect("values in range");

                    generate(&mut runtime, &["height", "trees"]);

                    let (share, count) = survey(&runtime);
                    let at =
                        format!("seed {seed}, land {land}, roughness {roughness}, trees {trees}");
                    // Some sea and some land for every amount between the ends.
                    if land == 0.5 {
                        assert!((0.1..0.9).contains(&share), "{share} land at {at}");
                    }
                    if trees == 0.0 {
                        assert_eq!(count, 0, "trees at {at}");
                    }
                    if trees == 1.0 && share > 0.2 {
                        assert!(count > 20, "only {count} trees at {at}");
                    }
                }
            }
        }
    }
}

#[test]
fn more_land_makes_more_land_and_more_trees_more_trees() {
    let pack = islands();
    let survey_at = |land: f32, trees: f32| {
        let mut runtime = Runtime::new(Arc::clone(&pack), 3, SIZE);
        runtime
            .set_params(&params([
                ("land", land),
                ("roughness", 0.4),
                ("trees", trees),
            ]))
            .expect("values in range");
        generate(&mut runtime, &["height", "trees"]);
        survey(&runtime)
    };

    let lands: Vec<f32> = [0.0, 0.25, 0.5, 0.75, 1.0]
        .iter()
        .map(|&land| survey_at(land, 0.5).0)
        .collect();
    let trees: Vec<usize> = [0.0, 0.5, 1.0]
        .iter()
        .map(|&trees| survey_at(0.6, trees).1)
        .collect();

    assert!(lands.windows(2).all(|pair| pair[0] < pair[1]), "{lands:?}");
    assert!(lands[0] < 0.1 && lands[4] > 0.9, "{lands:?}");
    assert!(trees.windows(2).all(|pair| pair[0] < pair[1]), "{trees:?}");
}

#[test]
fn a_changed_parameter_drops_only_what_reads_it() {
    let mut runtime = Runtime::new(islands(), 4, SIZE);
    generate(&mut runtime, &["height", "trees"]);
    let height = runtime
        .field("height", ChunkCoord::new(0, 0, 0))
        .expect("generated")
        .clone();

    let dropped = runtime
        .set_params(&params([
            ("land", 0.45),
            ("roughness", 0.4),
            ("trees", 0.9),
        ]))
        .expect("values in range");

    assert!(
        dropped.iter().all(|(stage, _)| stage == "trees"),
        "{dropped:?}"
    );
    assert!(!dropped.is_empty());
    assert_eq!(
        runtime.field("height", ChunkCoord::new(0, 0, 0)),
        Some(&height)
    );
}

#[test]
fn a_parameter_undeclared_or_out_of_range_is_refused() {
    let mut runtime = Runtime::new(islands(), 4, SIZE);
    let one = |name: &str, value: f32| BTreeMap::from([(name.to_owned(), value)]);

    let results = [
        runtime.set_params(&one("mountains", 0.5)),
        runtime.set_params(&one("land", 1.5)),
    ];
    let undeclared =
        Pack::parse(r#"(version: 1, stages: [(name: "height", kind: Field(Param("land")))])"#);
    let default_outside = Pack::parse(
        r#"(version: 1, params: {"land": (default: 2.0, range: (0.0, 1.0))}, stages: [])"#,
    );

    for result in results {
        assert!(matches!(result, Err(StageError::Param(_))), "{result:?}");
    }
    assert!(
        matches!(&undeclared, Err(PackError::Invalid { stage, .. }) if stage == "height"),
        "{undeclared:?}"
    );
    assert!(
        matches!(default_outside, Err(PackError::Param { .. })),
        "{default_outside:?}"
    );
}

/// The hills' relief over the area (its highest column less its lowest), how many trees stand in
/// it, and the largest grass cover, checking on the way that every height is finite and above the
/// ground's zero, every tree stands in the woods and every cover is between 0 and 1.
fn survey_hills(runtime: &Runtime, pack: &Pack) -> (f32, usize, f32) {
    let names = pack.kind("surface").expect("a surface stage").categories();
    let woods = names
        .iter()
        .position(|&name| name == "forest_floor")
        .expect("woods") as u8;
    let (mut low, mut high, mut trees, mut cover) = (f32::MAX, f32::MIN, 0, 0.0_f32);
    for chunk in area() {
        let height = runtime.field("height", chunk).expect("generated");
        for &value in &height.values {
            assert!(
                value.is_finite() && (0.0..30.0).contains(&value),
                "a height of {value}"
            );
            low = low.min(value);
            high = high.max(value);
        }
        for &value in &runtime.field("cover", chunk).expect("generated").values {
            assert!((0.0..=1.0).contains(&value), "a cover of {value}");
            cover = cover.max(value);
        }
        let surface = runtime.categories("surface", chunk).expect("generated");
        for tree in runtime.points("trees", chunk).expect("generated") {
            let [x, y] = [tree.position[0], tree.position[1]].map(|at| at.floor() as i64);
            let local =
                [x - i64::from(chunk.x) * 8, y - i64::from(chunk.y) * 8].map(|at| at as u32);
            assert_eq!(
                surface.get(local[0], local[1]),
                woods,
                "a tree out of the woods at {:?}",
                tree.position
            );
            trees += 1;
        }
    }
    (high - low, trees, cover)
}

#[test]
fn every_value_in_the_hills_ranges_gives_a_sound_world() {
    let pack = preset("hills");
    let grid = [0.0, 0.5, 1.0];

    for seed in [1, 2] {
        for hills in grid {
            for forest in grid {
                for meadows in grid {
                    let mut runtime = Runtime::new(Arc::clone(&pack), seed, SIZE);
                    let values =
                        params([("hills", hills), ("forest", forest), ("meadows", meadows)]);
                    runtime.set_params(&values).expect("values in range");

                    generate(&mut runtime, &["height", "surface", "cover", "trees"]);

                    let (relief, trees, cover) = survey_hills(&runtime, &pack);
                    let at =
                        format!("seed {seed}, hills {hills}, forest {forest}, meadows {meadows}");
                    assert!(relief > 1.0, "a flat world of relief {relief} at {at}");
                    if forest == 0.0 {
                        assert_eq!(trees, 0, "trees at {at}");
                    }
                    if forest == 1.0 {
                        assert!(trees > 50, "only {trees} trees at {at}");
                    }
                    if meadows == 0.0 {
                        assert_eq!(cover, 0.0, "grass at {at}");
                    }
                }
            }
        }
    }
}

#[test]
fn more_hills_make_taller_hills_and_more_forest_more_trees() {
    let pack = preset("hills");
    let survey_at = |hills: f32, forest: f32| {
        let mut runtime = Runtime::new(Arc::clone(&pack), 3, SIZE);
        runtime
            .set_params(&params([
                ("hills", hills),
                ("forest", forest),
                ("meadows", 0.6),
            ]))
            .expect("values in range");
        generate(&mut runtime, &["height", "surface", "cover", "trees"]);
        survey_hills(&runtime, &pack)
    };

    let reliefs: Vec<f32> = [0.0, 0.5, 1.0]
        .iter()
        .map(|&hills| survey_at(hills, 0.45).0)
        .collect();
    let trees: Vec<usize> = [0.0, 0.5, 1.0]
        .iter()
        .map(|&forest| survey_at(0.5, forest).1)
        .collect();

    assert!(
        reliefs.windows(2).all(|pair| pair[0] < pair[1]),
        "{reliefs:?}"
    );
    assert!(trees.windows(2).all(|pair| pair[0] < pair[1]), "{trees:?}");
}
