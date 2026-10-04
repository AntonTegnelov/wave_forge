//! Presets and their parameters (docs/product/user-stories.md, N2): a pack's parameters change
//! what reads them and nothing else, and every value in each preset's ranges gives a sound world,
//! swept over a grid of values and seeds: islands, hills and forests, the canyon desert, the
//! archipelago and the cave level.

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
                        assert!(trees > 20, "only {trees} trees at {at}");
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

/// The canyon desert's sand share of the area, how many cacti stand in it, its relief, and the
/// share of its columns on level ground, checking on the way that every height is finite and above
/// the ground's zero and every cactus stands on sand.
fn survey_canyon(runtime: &Runtime, pack: &Pack) -> (f32, usize, f32, f32) {
    let names = pack.kind("surface").expect("a surface stage").categories();
    let sand = names.iter().position(|&name| name == "sand").expect("sand") as u8;
    let (mut sands, mut level, mut columns, mut cacti) = (0, 0, 0, 0);
    let (mut low, mut high) = (f32::MAX, f32::MIN);
    for chunk in area() {
        let height = runtime.field("height", chunk).expect("generated");
        for &value in &height.values {
            assert!(
                value.is_finite() && (0.0..20.0).contains(&value),
                "a height of {value}"
            );
            low = low.min(value);
            high = high.max(value);
        }
        for &value in &runtime.field("steep", chunk).expect("generated").values {
            level += usize::from(value < 0.3);
            columns += 1;
        }
        let surface = runtime.categories("surface", chunk).expect("generated");
        sands += (0..8)
            .flat_map(|y| (0..8).map(move |x| (x, y)))
            .filter(|&(x, y)| surface.get(x, y) == sand)
            .count();
        for cactus in runtime.points("cacti", chunk).expect("generated") {
            let [x, y] = [cactus.position[0], cactus.position[1]].map(|at| at.floor() as i64);
            let local =
                [x - i64::from(chunk.x) * 8, y - i64::from(chunk.y) * 8].map(|at| at as u32);
            assert_eq!(
                surface.get(local[0], local[1]),
                sand,
                "a cactus off the sand at {:?}",
                cactus.position
            );
            cacti += 1;
        }
    }
    let columns = columns as f32;
    (
        sands as f32 / columns,
        cacti,
        high - low,
        level as f32 / columns,
    )
}

#[test]
fn every_value_in_the_canyon_ranges_gives_a_sound_world() {
    let pack = preset("canyon");
    let grid = [0.0, 0.5, 1.0];

    for seed in [1, 2] {
        for canyons in grid {
            for strata in grid {
                for cacti in grid {
                    let mut runtime = Runtime::new(Arc::clone(&pack), seed, SIZE);
                    let values =
                        params([("canyons", canyons), ("strata", strata), ("cacti", cacti)]);
                    runtime.set_params(&values).expect("values in range");

                    generate(&mut runtime, &["height", "surface", "cacti"]);

                    let (sand, count, relief, level) = survey_canyon(&runtime, &pack);
                    let at =
                        format!("seed {seed}, canyons {canyons}, strata {strata}, cacti {cacti}");
                    assert!(relief > 8.0, "a flat world of relief {relief} at {at}");
                    assert!((0.05..0.95).contains(&sand), "{sand} sand at {at}");
                    assert!(level > 0.1, "only {level} of the ground level at {at}");
                    if cacti == 0.0 {
                        assert_eq!(count, 0, "cacti at {at}");
                    }
                    if cacti == 1.0 {
                        assert!(count > 20, "only {count} cacti at {at}");
                    }
                }
            }
        }
    }
}

#[test]
fn wider_canyons_more_strata_and_more_cacti_each_make_more_of_theirs() {
    let pack = preset("canyon");
    let survey_at = |canyons: f32, strata: f32, cacti: f32| {
        let mut runtime = Runtime::new(Arc::clone(&pack), 3, SIZE);
        runtime
            .set_params(&params([
                ("canyons", canyons),
                ("strata", strata),
                ("cacti", cacti),
            ]))
            .expect("values in range");
        generate(&mut runtime, &["height", "surface", "cacti"]);
        survey_canyon(&runtime, &pack)
    };
    let amounts = [0.0, 0.5, 1.0];

    let sands: Vec<f32> = amounts.iter().map(|&a| survey_at(a, 0.7, 0.5).0).collect();
    let levels: Vec<f32> = amounts.iter().map(|&a| survey_at(0.4, a, 0.5).3).collect();
    let cacti: Vec<usize> = amounts.iter().map(|&a| survey_at(0.4, 0.7, a).1).collect();

    assert!(sands.windows(2).all(|pair| pair[0] < pair[1]), "{sands:?}");
    assert!(
        levels.windows(2).all(|pair| pair[0] < pair[1]),
        "{levels:?}"
    );
    assert!(cacti.windows(2).all(|pair| pair[0] < pair[1]), "{cacti:?}");
}

/// The archipelago's land share of the area, the share under a reef's shallow water, and how many
/// palms stand in it, checking on the way that every height is finite and every palm stands on land
/// above the waves.
fn survey_archipelago(runtime: &Runtime) -> (f32, f32, usize) {
    let (mut land, mut shallow, mut columns, mut palms) = (0, 0, 0, 0);
    for chunk in area() {
        let height = runtime.field("height", chunk).expect("generated");
        for &value in &height.values {
            assert!(
                value.is_finite() && (-8.5..10.0).contains(&value),
                "a height of {value}"
            );
            land += usize::from(value > 0.0);
            shallow += usize::from((-1.0..0.0).contains(&value));
            columns += 1;
        }
        for palm in runtime.points("palms", chunk).expect("generated") {
            assert!(
                palm.position[2] >= 0.3,
                "a palm in the waves at {:?}",
                palm.position
            );
            palms += 1;
        }
    }
    let columns = columns as f32;
    (land as f32 / columns, shallow as f32 / columns, palms)
}

#[test]
fn every_value_in_the_archipelago_ranges_gives_a_sound_world() {
    let pack = preset("archipelago");
    let grid = [0.0, 0.5, 1.0];

    for seed in [1, 2] {
        for islands in grid {
            for reefs in grid {
                for palms in grid {
                    let mut runtime = Runtime::new(Arc::clone(&pack), seed, SIZE);
                    let values = params([("islands", islands), ("reefs", reefs), ("palms", palms)]);
                    runtime.set_params(&values).expect("values in range");

                    generate(&mut runtime, &["height", "surface", "cover", "palms"]);

                    let (land, _, palm_count) = survey_archipelago(&runtime);
                    let at =
                        format!("seed {seed}, islands {islands}, reefs {reefs}, palms {palms}");
                    assert!(
                        (0.0..0.7).contains(&land) && land > 0.0,
                        "{land} land at {at}"
                    );
                    if palms == 0.0 {
                        assert_eq!(palm_count, 0, "palms at {at}");
                    }
                    if palms == 1.0 && islands >= 0.5 {
                        assert!(palm_count > 15, "only {palm_count} palms at {at}");
                    }
                }
            }
        }
    }
}

#[test]
fn larger_islands_wider_reefs_and_more_palms_each_make_more_of_theirs() {
    let pack = preset("archipelago");
    let survey_at = |islands: f32, reefs: f32, palms: f32| {
        let mut runtime = Runtime::new(Arc::clone(&pack), 3, SIZE);
        runtime
            .set_params(&params([
                ("islands", islands),
                ("reefs", reefs),
                ("palms", palms),
            ]))
            .expect("values in range");
        generate(&mut runtime, &["height", "palms"]);
        survey_archipelago(&runtime)
    };
    let amounts = [0.0, 0.5, 1.0];

    let lands: Vec<f32> = amounts.iter().map(|&a| survey_at(a, 0.5, 0.6).0).collect();
    let shallows: Vec<f32> = amounts.iter().map(|&a| survey_at(0.5, a, 0.6).1).collect();
    let palms: Vec<usize> = amounts.iter().map(|&a| survey_at(0.5, 0.5, a).2).collect();

    assert!(lands.windows(2).all(|pair| pair[0] < pair[1]), "{lands:?}");
    assert!(
        shallows.windows(2).all(|pair| pair[0] < pair[1]),
        "{shallows:?}"
    );
    assert!(palms.windows(2).all(|pair| pair[0] < pair[1]), "{palms:?}");
}

/// The cave level's share of the rock below the hills' surface that its caves hollow out, the
/// share of its columns opened to the sky, the share of its solid voxels that are crystal, and how
/// many trees stand on it, checking on the way that every height and value is finite and every
/// tree stands on the hills' surface, never down a sinkhole.
fn survey_cave(runtime: &Runtime, pack: &Pack) -> (f32, f32, f32, usize) {
    let names = pack.kind("rock").expect("a rock stage").categories();
    let crystal = names
        .iter()
        .position(|&name| name == "crystal")
        .expect("crystal") as u8;
    let (mut below, mut hollow, mut solid, mut crystals) = (0, 0, 0, 0);
    let (mut columns, mut open, mut trees) = (0, 0, 0);
    for chunk in area() {
        let height = runtime.field("height", chunk).expect("generated");
        let top = runtime.field("top", chunk).expect("generated");
        let rock = runtime.volume("rock", chunk).expect("generated");
        let [sx, sy, levels] = rock.size;
        for y in 0..sy {
            for x in 0..sx {
                let surface = height.get(x, y);
                assert!(surface.is_finite(), "a height of {surface}");
                columns += 1;
                open += usize::from(top.get(x, y) < surface - 1.0);
                for level in 0..levels {
                    let value = rock.get(x, y, level);
                    assert!(value.is_finite(), "a value of {value}");
                    let z = (rock.bottom + level as i32) as f32 + 0.5;
                    if value > 0.0 {
                        solid += 1;
                        crystals += usize::from(rock.material(x, y, level) == crystal);
                    } else if z < surface {
                        hollow += 1;
                    }
                    below += usize::from(z < surface);
                }
            }
        }
        for tree in runtime.points("trees", chunk).expect("generated") {
            let [x, y] = [tree.position[0], tree.position[1]].map(|at| at.floor() as i64);
            let local =
                [x - i64::from(chunk.x) * 8, y - i64::from(chunk.y) * 8].map(|at| at as u32);
            assert!(
                (tree.position[2] - height.get(local[0], local[1])).abs() < 1.0,
                "a tree off the hills at {:?}",
                tree.position
            );
            trees += 1;
        }
    }
    (
        hollow as f32 / below as f32,
        open as f32 / columns as f32,
        crystals as f32 / solid as f32,
        trees,
    )
}

#[test]
fn every_value_in_the_cave_ranges_gives_a_sound_world() {
    let pack = preset("cave");
    let grid = [0.0, 0.5, 1.0];

    for seed in [1, 2] {
        for caves in grid {
            for openings in grid {
                for crystals in grid {
                    let mut runtime = Runtime::new(Arc::clone(&pack), seed, SIZE);
                    let values = params([
                        ("caves", caves),
                        ("openings", openings),
                        ("crystals", crystals),
                    ]);
                    runtime.set_params(&values).expect("values in range");

                    generate(&mut runtime, &["height", "rock", "top", "trees"]);

                    let (hollow, open, crystal, trees) = survey_cave(&runtime, &pack);
                    let at = format!(
                        "seed {seed}, caves {caves}, openings {openings}, crystals {crystals}"
                    );
                    assert!(hollow > 0.03, "only {hollow} of the rock hollow at {at}");
                    assert!(crystal > 0.0, "no crystal at {at}");
                    assert!(trees > 20, "only {trees} trees at {at}");
                    if openings == 0.0 {
                        assert_eq!(open, 0.0, "caves open to the sky at {at}");
                    }
                }
            }
        }
    }
}

#[test]
fn more_caves_more_openings_and_more_crystals_each_make_more_of_theirs() {
    let pack = preset("cave");
    let survey_at = |caves: f32, openings: f32, crystals: f32| {
        let mut runtime = Runtime::new(Arc::clone(&pack), 3, SIZE);
        runtime
            .set_params(&params([
                ("caves", caves),
                ("openings", openings),
                ("crystals", crystals),
            ]))
            .expect("values in range");
        generate(&mut runtime, &["height", "rock", "top", "trees"]);
        survey_cave(&runtime, &pack)
    };
    let amounts = [0.0, 0.5, 1.0];

    let hollows: Vec<f32> = amounts.iter().map(|&a| survey_at(a, 0.6, 0.5).0).collect();
    let opens: Vec<f32> = amounts.iter().map(|&a| survey_at(0.5, a, 0.5).1).collect();
    let crystals: Vec<f32> = amounts.iter().map(|&a| survey_at(0.5, 0.6, a).2).collect();

    assert!(
        hollows.windows(2).all(|pair| pair[0] < pair[1]),
        "{hollows:?}"
    );
    assert!(opens.windows(2).all(|pair| pair[0] < pair[1]), "{opens:?}");
    assert!(
        crystals.windows(2).all(|pair| pair[0] < pair[1]),
        "{crystals:?}"
    );
}
