//! The maximal preset (docs/product/user-stories.md, M1, #247): a continent of 2 048 by 2 048
//! cells, 4 by 4 km at cells of 2 m. Its biomes and coast are checked by sampling, which
//! generates no chunk; generating part of it to completion is a measurement, run in release:
//!
//! ```text
//! cargo test --release --test continent -- --ignored --nocapture
//! ```

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::{Facts, GivenRow, Pack, Runtime, Value};
use wave_forge::{ChunkCoord, FocusPoint};

const SIZE: [u32; 2] = [16, 16];
const SIDE: f32 = 2048.0;
const CENTRE: [f32; 2] = [1024.0, 1024.0];
const OCEANS: [&str; 2] = ["ocean", "deep_ocean"];

fn pack() -> Arc<Pack> {
    let text = std::fs::read_to_string(format!(
        "{}/examples/continent/continent.world.ron",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("the continent's pack");
    Arc::new(Pack::parse(&text).expect("a valid pack"))
}

/// The culture whose settlements each biome holds; the biomes left out hold none.
const CULTURES: [(&str, &[&str]); 8] = [
    ("coastfolk", &["beach", "shingle"]),
    (
        "steppe_riders",
        &["cold_steppe", "upland_steppe", "prairie", "savanna"],
    ),
    (
        "woodlanders",
        &[
            "broadleaf_forest",
            "mixed_woodland",
            "meadow",
            "oak_hills",
            "hill_pasture",
        ],
    ),
    (
        "sand_dwellers",
        &["desert", "dry_scrub", "shrubland", "mesa", "chaparral"],
    ),
    (
        "highlanders",
        &[
            "upland_heath",
            "pine_highland",
            "fir_highland",
            "alpine_meadow",
        ],
    ),
    (
        "marsh_folk",
        &["fen", "swamp", "muskeg", "upland_bog", "highland_marsh"],
    ),
    ("jungle_folk", &["rainforest", "cloud_forest", "mangrove"]),
    (
        "frostfolk",
        &[
            "tundra",
            "taiga",
            "spruce_forest",
            "bog_tundra",
            "lichen_highland",
        ],
    ),
];

/// A hash of three numbers, for the history's decisions.
fn hash(a: u64, b: u64, c: u64) -> u64 {
    let mut x = a
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(b.wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
        .wrapping_add(c.wrapping_mul(0x1656_67B1_9E37_79F9));
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^ (x >> 29)
}

/// The continent's history as a game might simulate it over the map before play: a settlement
/// tried in every square of 64 cells, kept where its biome has a culture and the ground is dry,
/// gentle and below the peaks, at least 128 cells from every settlement kept before it; its
/// population, founding year and fate drawn from a hash, the oldest the likeliest to have fallen.
fn history(pack: &Arc<Pack>) -> Vec<GivenRow> {
    let runtime = Runtime::new(Arc::clone(pack), 11, SIZE);
    let names = pack.kind("biome").expect("a biome stage").categories();
    let mut tried: Vec<(u64, [f32; 2], &str)> = Vec::new();
    for gy in 0..32u64 {
        for gx in 0..32u64 {
            let h = hash(gx, gy, 1);
            let at = [
                gx as f32 * 64.0 + 8.0 + (h % 48) as f32,
                gy as f32 * 64.0 + 8.0 + ((h >> 8) % 48) as f32,
            ];
            let biome = names[runtime.sample("biome", at).expect("a biome") as usize];
            let Some((culture, _)) = CULTURES.iter().find(|(_, biomes)| biomes.contains(&biome))
            else {
                continue;
            };
            let height = runtime.sample("terrain", at).expect("a height");
            let rough = runtime.sample("roughness", at).expect("a roughness");
            if (0.5..45.0).contains(&height) && rough < 3.0 {
                tried.push((h, at, culture));
            }
        }
    }
    tried.sort_by_key(|&(h, ..)| h);
    let mut kept: Vec<([f32; 2], &str, u64)> = Vec::new();
    for (h, at, culture) in tried {
        let apart = |other: &[f32; 2]| (other[0] - at[0]).hypot(other[1] - at[1]) >= 128.0;
        if kept.iter().all(|(other, ..)| apart(other)) {
            kept.push((at, culture, h));
        }
    }
    kept.into_iter()
        .enumerate()
        .map(|(id, (at, culture, h))| {
            let population = 150 + (h >> 16) % 4000;
            let founded = (h >> 32) % 500;
            let size = 1 + u64::from(population > 1500) + u64::from(population > 3000);
            let fall = (h >> 40) % 1000;
            let fate = match (founded < 150, fall) {
                (true, 0..=349) => "ruined",
                (true, 350..=499) | (false, 0..=99) => "abandoned",
                (_, 500..=699) => "declining",
                _ => "thriving",
            };
            let number = |value: f32| Value::Number(value);
            GivenRow {
                id: id as u64,
                values: BTreeMap::from([
                    ("x".to_owned(), number(at[0])),
                    ("y".to_owned(), number(at[1])),
                    ("size".to_owned(), number(size as f32)),
                    ("population".to_owned(), number(population as f32)),
                    ("founded".to_owned(), number(founded as f32)),
                    ("culture".to_owned(), Value::Name(culture.to_owned())),
                    ("fate".to_owned(), Value::Name(fate.to_owned())),
                ]),
            }
        })
        .collect()
}

/// A runtime of the continent, given its history.
fn runtime() -> Runtime {
    let pack = pack();
    let mut facts = Facts::new(Arc::clone(&pack), 11).expect("the tables");
    facts
        .give("settlements", history(&pack))
        .expect("a history the table takes");
    let mut runtime = Runtime::new(pack, 11, SIZE);
    runtime
        .set_facts(facts)
        .expect("facts of the pack and seed");
    runtime
}

/// The biome at each of `points`, sampled without generating a chunk.
fn biomes(points: impl Iterator<Item = [f32; 2]>) -> Vec<String> {
    let pack = pack();
    let runtime = Runtime::new(Arc::clone(&pack), 11, SIZE);
    let names = pack.kind("biome").expect("a biome stage").categories();
    points
        .map(|at| {
            let index = runtime.sample("biome", at).expect("a sampled biome") as usize;
            names[index].to_owned()
        })
        .collect()
}

/// `steps` by `steps` points evenly over the continent.
fn grid(steps: u32) -> impl Iterator<Item = [f32; 2]> {
    (0..steps).flat_map(move |j| {
        (0..steps).map(move |i| {
            [
                (i as f32 + 0.5) * SIDE / steps as f32,
                (j as f32 + 0.5) * SIDE / steps as f32,
            ]
        })
    })
}

#[test]
fn the_continent_chooses_among_forty_biomes_by_rules_and_forty_appear() {
    let pack = pack();

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for biome in biomes(grid(128)) {
        *counts.entry(biome).or_default() += 1;
    }

    let declared = pack
        .kind("biome")
        .expect("a biome stage")
        .categories()
        .len();
    assert!(declared >= 40, "{declared} biomes declared");
    assert!(
        counts.len() >= 40,
        "{} biomes appear: {counts:?}",
        counts.len()
    );
}

#[test]
fn the_coast_falls_into_the_ocean_before_the_bound_and_about_half_is_land() {
    let ring = (0..360).map(|degree| {
        let angle = (degree as f32).to_radians();
        [
            CENTRE[0] + 1000.0 * angle.cos(),
            CENTRE[1] + 1000.0 * angle.sin(),
        ]
    });

    let edge = biomes(ring);
    let all = biomes(grid(64));

    let dry = edge
        .iter()
        .filter(|biome| !OCEANS.contains(&biome.as_str()));
    assert_eq!(dry.count(), 0, "land 1 000 cells out: {edge:?}");
    let land = all
        .iter()
        .filter(|biome| !OCEANS.contains(&biome.as_str()) && biome.as_str() != "shallows")
        .count() as f32
        / all.len() as f32;
    assert!(
        (0.35..0.7).contains(&land),
        "{land} of the continent is land"
    );
}

#[test]
#[ignore = "a measurement: run in release with --ignored --nocapture"]
fn a_part_of_the_continent_generates_with_rivers_and_lakes() {
    let mut runtime = runtime();
    let focus: Vec<FocusPoint> = (60..64)
        .flat_map(|y| (60..64).map(move |x| FocusPoint::new(ChunkCoord::new(x, y, 0), 0)))
        .collect();

    let started = std::time::Instant::now();
    runtime
        .request(&focus, &["ground", "biome"])
        .expect("the stages");
    runtime.run_until_idle().expect("the stages run");
    let seconds = started.elapsed().as_secs_f64();

    // Rivers and lakes are region jobs over 32 chunks a side, so the region around the focus holds
    // them wherever they are in it.
    let (mut rivers, mut lake_columns) = (0, 0);
    for y in 32..96 {
        for x in 32..96 {
            let chunk = ChunkCoord::new(x, y, 0);
            rivers += runtime.curves("rivers", chunk).map_or(0, <[_]>::len);
            if let (Some(lakes), Some(terrain)) = (
                runtime.field("lakes", chunk),
                runtime.field("terrain", chunk),
            ) {
                lake_columns += lakes
                    .values
                    .iter()
                    .zip(&terrain.values)
                    .filter(|(water, ground)| *water > *ground)
                    .count();
            }
        }
    }
    eprintln!(
        "continent: 16 chunks of ground in {seconds:.1} s; {rivers} river pieces, {lake_columns} lake columns"
    );
    for (stage, timing) in runtime.timings() {
        eprintln!(
            "continent: {stage}: {} chunks, {:.3} ms each",
            timing.products,
            timing.ms / timing.products.max(1) as f64
        );
    }
    for chunk in &focus {
        assert!(runtime.field("ground", chunk.chunk).is_some(), "{chunk:?}");
    }
    assert!(
        rivers > 0 && lake_columns > 0,
        "{rivers} rivers, {lake_columns} lake columns"
    );
}

#[test]
#[ignore = "a measurement: run in release with --ignored --nocapture"]
fn the_rock_under_cliffs_holds_caves_overhangs_and_ore() {
    let mut runtime = runtime();
    // A chunk and its neighbours on a plateau's rim, cliffs 15 to 50 cells up by sampling.
    let centre = ChunkCoord::new(47, 55, 0);
    let ores = ["coal", "iron", "copper", "gold"];
    let mut targets = vec!["ground", "rock"];
    targets.extend(ores);

    let started = std::time::Instant::now();
    runtime
        .request(&[FocusPoint::new(centre, 1)], &targets)
        .expect("the stages");
    runtime.run_until_idle().expect("the stages run");
    let seconds = started.elapsed().as_secs_f64();

    let (mut caves, mut overhangs) = (0, 0);
    for dy in -1..=1 {
        for dx in -1..=1 {
            let chunk = ChunkCoord::new(centre.x + dx, centre.y + dy, 0);
            let ground = runtime.field("ground", chunk).expect("ground");
            let rock = runtime.volume("rock", chunk).expect("rock");
            let [sx, sy, levels] = rock.size;
            for level in 0..levels {
                let z = (rock.bottom + level as i32) as f32 + 0.5;
                for column in 0..(sx * sy) as usize {
                    let solid = rock.values[level as usize * (sx * sy) as usize + column] > 0.0;
                    let surface = ground.values[column];
                    caves += usize::from(!solid && z < surface - 8.0);
                    overhangs += usize::from(solid && z > surface + 1.0);
                }
            }
        }
    }
    let placed: Vec<usize> = ores
        .iter()
        .map(|ore| {
            (-1..=1)
                .flat_map(|dy| (-1..=1).map(move |dx| (dx, dy)))
                .map(|(dx, dy)| {
                    let chunk = ChunkCoord::new(centre.x + dx, centre.y + dy, 0);
                    runtime.points(ore, chunk).map_or(0, <[_]>::len)
                })
                .sum()
        })
        .collect();
    eprintln!(
        "continent: 3x3 chunks of rock under cliffs in {seconds:.1} s: {caves} cave voxels, {overhangs} overhanging voxels, ores {ores:?} {placed:?}; rock {:.1} ms a chunk",
        runtime
            .timings()
            .iter()
            .find(|(stage, _)| stage == "rock")
            .map_or(0.0, |(_, timing)| timing.ms / timing.products.max(1) as f64)
    );
    assert!(
        caves > 0 && overhangs > 0,
        "{caves} cave voxels, {overhangs} overhanging"
    );
    assert!(
        placed.iter().all(|&count| count > 0),
        "{ores:?}: {placed:?}"
    );
}

#[test]
#[ignore = "a measurement: run in release with --ignored --nocapture"]
fn a_region_places_its_locations_and_joins_them_by_roads() {
    let mut runtime = runtime();
    // The region of 32 chunks a side south-west of the centre, whole.
    let region: Vec<ChunkCoord> = (32..64)
        .flat_map(|y| (32..64).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect();
    let focus: Vec<FocusPoint> = region.iter().map(|&c| FocusPoint::new(c, 0)).collect();

    let started = std::time::Instant::now();
    runtime
        .request(&focus, &["places", "roads", "settlements", "settled"])
        .expect("the stages");
    runtime.run_until_idle().expect("the stages run");
    let seconds = started.elapsed().as_secs_f64();

    let mut places = BTreeMap::new();
    let mut roads = BTreeMap::new();
    for &chunk in &region {
        for site in runtime.sites("places", chunk).expect("placed") {
            places.insert(site.id.clone(), site.kind.clone());
        }
        for road in runtime.curves("roads", chunk).expect("joined") {
            roads.insert(road.id.clone(), road.points.len());
        }
    }
    // Every settlement of the history in the region stands on ground levelled to its height.
    let mut settlements = BTreeMap::new();
    for &chunk in &region {
        for site in runtime.sites("settlements", chunk).expect("settled") {
            settlements.insert(site.id.clone(), site.clone());
        }
    }
    for site in settlements.values() {
        let chunk = ChunkCoord::new(
            (site.min.0 + site.max.0) / 2,
            (site.min.1 + site.max.1) / 2,
            0,
        );
        let settled = runtime.field("settled", chunk).expect("levelled");
        assert!(
            settled
                .values
                .iter()
                .all(|&h| (h - site.height).abs() < 1e-3),
            "{:?} at {}",
            site.id,
            site.height
        );
    }
    let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
    for kind in places.values().flatten() {
        *kinds.entry(kind.to_string()).or_default() += 1;
    }
    eprintln!(
        "continent: a region of 32x32 chunks in {seconds:.1} s: {} places of {} kinds, {} roads, {} settlements; {kinds:?}",
        places.len(),
        kinds.len(),
        roads.len(),
        settlements.len()
    );
    assert!(!settlements.is_empty(), "no settlement in the region");
    assert!(kinds.len() >= 15, "{kinds:?}");
    assert!(
        roads.len() + 1 >= places.len() / 2,
        "{} roads for {} places",
        roads.len(),
        places.len()
    );
}

#[test]
fn the_history_settles_every_culture_on_land_and_the_continent_takes_it() {
    let pack = pack();

    let rows = history(&pack);
    // The runtime refuses a history its table does not take, such as two settlements within a
    // chunk of each other.
    runtime();

    let mut cultures: BTreeMap<String, usize> = BTreeMap::new();
    for row in &rows {
        let Value::Name(culture) = &row.values["culture"] else {
            panic!("a culture is a name");
        };
        *cultures.entry(culture.clone()).or_default() += 1;
    }
    assert!(rows.len() >= 40, "{} settlements", rows.len());
    assert_eq!(cultures.len(), CULTURES.len(), "{cultures:?}");
}
