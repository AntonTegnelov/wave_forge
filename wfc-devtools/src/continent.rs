//! The maximal preset's continent (docs/product/user-stories.md, M1): its pack, the history a game
//! would give it, and its towns' eight cultures, for the checks that generate it.
//!
//! The history is what a game simulates over the continent's map before play and hands the pack as
//! its `settlements` table; here it is a small deterministic simulation, so every check sees the
//! same settlements.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::loader::parse_rule_file;
use wave_forge::stages::{Facts, GivenRow, Pack, Runtime, Value};
use wave_forge::towns::{WfcTowns, gpu_solver};
use wave_forge::{BlockSolver, ChunkShape, WgpuBackend};

/// Columns a chunk of the continent has along each axis: a town's region, a chunk with a halo of
/// one, has to fit a GPU's workgroup memory with a culture's 135 tiles.
pub const SIZE: [u32; 2] = [8, 8];
/// The chunk a town is solved in: the continent's, six layers tall.
pub const TOWN: ChunkShape = ChunkShape { x: 8, y: 8, z: 6 };
/// The seed the checks generate the continent with.
pub const SEED: u64 = 11;

/// The culture whose settlements each biome holds; the biomes left out hold none.
pub const CULTURES: [(&str, &[&str]); 8] = [
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

/// The continent's pack, `examples/continent/continent.world.ron`.
///
/// # Panics
/// If the pack is missing or invalid, which its checks rule out.
#[must_use]
pub fn pack() -> Arc<Pack> {
    let text = std::fs::read_to_string(format!(
        "{}/../examples/continent/continent.world.ron",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("the continent's pack");
    Arc::new(Pack::parse(&text).expect("a valid pack"))
}

/// A hash of three numbers, for the history's decisions.
const fn hash(a: u64, b: u64, c: u64) -> u64 {
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
/// population, founding year and fate drawn from a hash, the oldest the likeliest to have fallen,
/// and its size, two to six chunks a side, from its population.
///
/// # Panics
/// If the pack cannot sample its biome, terrain and roughness, which its checks rule out.
#[must_use]
pub fn history(pack: &Arc<Pack>) -> Vec<GivenRow> {
    let runtime = Runtime::new(Arc::clone(pack), SEED, SIZE);
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
            let size = 2 + 2 * u64::from(population > 1500) + 2 * u64::from(population > 3000);
            let fall = (h >> 40) % 1000;
            let fate = match (founded < 150, fall) {
                (true, 0..=349) => "ruined",
                (true, 350..=499) | (false, 0..=99) => "abandoned",
                (_, 500..=699) => "declining",
                _ => "thriving",
            };
            GivenRow {
                id: id as u64,
                values: BTreeMap::from([
                    ("x".to_owned(), Value::Number(at[0])),
                    ("y".to_owned(), Value::Number(at[1])),
                    ("size".to_owned(), Value::Number(size as f32)),
                    ("population".to_owned(), Value::Number(population as f32)),
                    ("founded".to_owned(), Value::Number(founded as f32)),
                    ("culture".to_owned(), Value::Name((*culture).to_owned())),
                    ("fate".to_owned(), Value::Name(fate.to_owned())),
                ]),
            }
        })
        .collect()
}

/// `rows` as the JSON an engine gives a table from: an array of objects, each with its `id` and a
/// value per column, a number or a name, as Godot's `give_table` takes them.
///
/// # Panics
/// If a number is not finite, which the history never gives.
#[must_use]
pub fn history_json(rows: &[GivenRow]) -> String {
    let rows: Vec<serde_json::Value> = rows
        .iter()
        .map(|row| {
            let mut object = serde_json::Map::new();
            object.insert("id".to_owned(), row.id.into());
            for (column, value) in &row.values {
                let value = match value {
                    Value::Number(number) => serde_json::Number::from_f64(f64::from(*number))
                        .expect("a finite number")
                        .into(),
                    Value::Name(name) => name.clone().into(),
                };
                object.insert(column.clone(), value);
            }
            serde_json::Value::Object(object)
        })
        .collect();
    let mut text = serde_json::to_string_pretty(&rows).expect("plain values");
    text.push('\n');
    text
}

/// A runtime of the continent, given its history, without towns.
///
/// # Panics
/// If the history is one the pack's table refuses, which its checks rule out.
#[must_use]
pub fn runtime() -> Runtime {
    let pack = pack();
    let mut facts = Facts::new(Arc::clone(&pack), SEED).expect("the tables");
    facts
        .give("settlements", history(&pack))
        .expect("a history the table takes");
    let mut runtime = Runtime::new(pack, SEED, SIZE);
    runtime
        .set_facts(facts)
        .expect("facts of the pack and seed");
    runtime
}

/// The towns of the continent's eight cultures, each solved on the GPU with its own module set
/// from `examples/continent/cultures`.
///
/// # Errors
/// If a module set does not compile or no GPU device can be made.
///
/// # Panics
/// If a culture's module set is missing or invalid, which its checks rule out.
pub fn towns() -> Result<WfcTowns<BlockSolver<WgpuBackend>>, wave_forge::Error> {
    let mut towns = WfcTowns::new(TOWN);
    for (culture, _) in CULTURES {
        let text = std::fs::read_to_string(format!(
            "{}/../examples/continent/cultures/{culture}.ron",
            env!("CARGO_MANIFEST_DIR")
        ))
        .expect("the culture's module set");
        let file = parse_rule_file(&text).expect("a valid module set");
        towns = towns.with_rules(culture, file, gpu_solver)?;
    }
    Ok(towns)
}
