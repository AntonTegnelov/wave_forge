//! A finite world with a history (docs/product/user-stories.md, G2): `tests/fixtures/colony.world.ron`,
//! a world map of 32 by 32 tiles generated whole before play, as one region:
//!
//! - its fields, rivers worn and drained over the whole world, biomes and the connected regions of
//!   each, strata under the ground, and a region job of the game's that rejects attempts it does
//!   not like, with the reasons kept;
//! - a history given as tables before play, realised as towns of each culture's module set, ruins
//!   where a place fell, and roads levelled into the ground;
//! - a local area generated from a tile that meets the world map along the tile's edges;
//! - the same world on every run and GPU, its chunks' hashes compared with a record made on another
//!   device (CI runs this on lavapipe; the record was made on an RTX 3070 through dozen).
//!
//! To record the hashes again after a change meant to change the world, run
//!
//! ```text
//! WAVE_FORGE_BLESS=1 cargo test -p wfc-devtools --test colony
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use wave_forge::loader::parse_rule_file;
use wave_forge::stages::regions::{Attempt, Curve, CurveId, RegionInput, RegionJob};
use wave_forge::stages::{Facts, GivenRow, Pack, Runtime, StageError, Value};
use wave_forge::towns::{WfcTowns, gpu_solver};
use wave_forge::{ChunkCoord, ChunkShape, FocusPoint};

const SEED: u64 = 3;
const SIZE: [u32; 2] = [8, 8];
const TOWN: ChunkShape = ChunkShape { x: 8, y: 8, z: 6 };
/// World tiles along each side of the world, one chunk each.
const TILES: i32 = 32;
/// How many settlements the history places.
const SETTLEMENTS: usize = 6;
const HEADER: &str =
    "# wave_forge golden colony: wfc-devtools/tests/fixtures/colony.world.ron, seed 3";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn pack() -> Arc<Pack> {
    let text = std::fs::read_to_string(root().join("wfc-devtools/tests/fixtures/colony.world.ron"))
        .expect("the colony's pack");
    Arc::new(Pack::parse(&text).expect("a valid pack"))
}

/// The world's peaks: three columns its hash stream picks, kept only if none falls in the sea, so
/// some attempts are turned down and the world is chosen again.
struct Peaks;

impl RegionJob for Peaks {
    fn run(&self, input: &RegionInput<'_>) -> Result<Attempt, StageError> {
        let ([x0, y0], [x1, y1]) = input.columns();
        let mut points = Vec::new();
        for peak in 0..3 {
            let hash = input.hash(peak);
            let x = x0 + i64::from(hash % (x1 - x0 + 1) as u32);
            let y = y0 + i64::from((hash >> 16) % (y1 - y0 + 1) as u32);
            let height = input.field("ground", x, y)?;
            if height < 1.0 {
                return Ok(Attempt::Rejected(format!(
                    "peak {peak} fell in the sea, {height:.1} cells high at ({x}, {y})"
                )));
            }
            points.push([x as f32 + 0.5, y as f32 + 0.5]);
        }
        Ok(Attempt::Accepted(vec![Curve {
            id: CurveId::Region {
                region: input.region(),
                index: 0,
            },
            values: vec![1.0; points.len()],
            points,
            heights: Vec::new(),
        }]))
    }
}

/// The game's history, made before play over the world map: places on land, every third left in
/// ruins and the others towns of the culture its biome suits, and a road from each to the next.
fn history(pack: &Arc<Pack>) -> (Vec<GivenRow>, Vec<GivenRow>) {
    let sampler = Runtime::new(Arc::clone(pack), SEED, SIZE);
    let names = pack.kind("biome").expect("a biome stage").categories();
    let mut places: Vec<([f32; 2], &str)> = Vec::new();
    'tiles: for ty in (2..TILES - 2).step_by(5) {
        for tx in (2..TILES - 2).step_by(5) {
            let at = [tx as f32 * 8.0 + 4.0, ty as f32 * 8.0 + 4.0];
            let biome = names[sampler.sample("biome", at).expect("a biome") as usize];
            let culture = match biome {
                "grassland" | "forest" => "woodlanders",
                "tundra" | "desert" => "highlanders",
                _ => continue,
            };
            let built = if places.len() % 3 == 2 {
                "ruins"
            } else {
                culture
            };
            places.push((at, built));
            if places.len() == SETTLEMENTS {
                break 'tiles;
            }
        }
    }
    assert_eq!(
        places.len(),
        SETTLEMENTS,
        "the world has room for the history"
    );
    let settlements = places
        .iter()
        .enumerate()
        .map(|(id, &(at, built))| GivenRow {
            id: id as u64,
            values: BTreeMap::from([
                ("x".to_owned(), Value::Number(at[0])),
                ("y".to_owned(), Value::Number(at[1])),
                ("size".to_owned(), Value::Number(1.0)),
                ("built".to_owned(), Value::Name(built.to_owned())),
            ]),
        })
        .collect();
    let roads = places
        .windows(2)
        .enumerate()
        .map(|(id, pair)| GivenRow {
            id: id as u64,
            values: BTreeMap::from([
                ("x0".to_owned(), Value::Number(pair[0].0[0])),
                ("y0".to_owned(), Value::Number(pair[0].0[1])),
                ("x1".to_owned(), Value::Number(pair[1].0[0])),
                ("y1".to_owned(), Value::Number(pair[1].0[1])),
                ("width".to_owned(), Value::Number(1.0)),
            ]),
        })
        .collect();
    (settlements, roads)
}

fn runtime() -> Runtime {
    let pack = pack();
    let (settlements, roads) = history(&pack);
    let mut facts = Facts::new(Arc::clone(&pack), SEED).expect("the tables");
    facts
        .give("settlements", settlements)
        .expect("a history the table takes");
    facts.give("roads", roads).expect("roads the table takes");
    let mut towns = WfcTowns::new(TOWN);
    for (name, path) in [
        ("woodlanders", "examples/continent/cultures/woodlanders.ron"),
        ("highlanders", "examples/continent/cultures/highlanders.ron"),
        ("ruins", "examples/history/ruins.ron"),
    ] {
        let text = std::fs::read_to_string(root().join(path)).expect("a module set");
        towns = towns
            .with_rules(
                name,
                parse_rule_file(&text).expect("a module set"),
                gpu_solver,
            )
            .expect("a compute device");
    }
    let mut runtime = Runtime::new(pack, SEED, SIZE)
        .with_region_job("peaks", Peaks)
        .with_towns(Box::new(towns))
        .expect("matching chunks");
    runtime
        .set_facts(facts)
        .expect("facts of the pack and seed");
    runtime
}

/// Every chunk of the world, its whole map asked for at once.
fn world(targets: &[&str]) -> Runtime {
    let mut runtime = runtime();
    runtime.request_bound(targets).expect("a bounded world");
    runtime.run_until_idle().expect("the world generates");
    runtime
}

#[test]
fn the_world_map_is_one_region_with_its_rivers_biomes_regions_and_rejections() {
    let runtime = world(&["regions", "rivers", "peaks"]);

    let log = runtime
        .region_log("peaks", (0, 0))
        .expect("the world's one region");
    let rivers: BTreeSet<_> = (0..TILES)
        .flat_map(|y| (0..TILES).map(move |x| ChunkCoord::new(x, y, 0)))
        .flat_map(|chunk| {
            runtime
                .curves("rivers", chunk)
                .unwrap_or(&[])
                .iter()
                .map(|c| c.id.clone())
        })
        .collect();
    let biomes = runtime
        .atlas("biome", [0, 0], [32, 32])
        .expect("the world map");
    let regions = runtime
        .atlas("regions", [0, 0], [32, 32])
        .expect("the world map");
    let kinds: BTreeSet<u32> = biomes.iter().map(|&b| b as u32).collect();
    let areas: BTreeSet<u32> = regions.iter().map(|&r| r as u32).collect();

    assert!(
        !log.is_empty(),
        "the seed's first peaks are accepted, so nothing is logged"
    );
    assert!(
        log.iter().all(|reason| reason.contains("fell in the sea")),
        "{log:?}"
    );
    assert!(rivers.len() >= 2, "{} rivers", rivers.len());
    assert!(kinds.len() >= 4, "{} biomes", kinds.len());
    assert!(
        areas.len() > kinds.len(),
        "{} regions of {} biomes",
        areas.len(),
        kinds.len()
    );
}

#[test]
fn strata_lie_in_order_from_the_ground_down() {
    let mut runtime = runtime();
    let chunk = ChunkCoord::new(TILES / 2, TILES / 2, 0);
    runtime
        .request(&[FocusPoint::new(chunk, 0)], &["rock"])
        .expect("rock");
    runtime.run_until_idle().expect("the rock generates");

    let rock = runtime.volume("rock", chunk).expect("the rock");
    let pack = pack();
    let names = pack.kind("rock").expect("rock").categories();
    let [sx, sy, levels] = rock.size;
    let order = ["soil", "sandstone", "limestone", "granite"];
    let mut seen = BTreeSet::new();
    for column in 0..(sx * sy) as usize {
        let mut deepest = 0;
        for level in (0..levels as usize).rev() {
            let at = level * (sx * sy) as usize + column;
            if rock.values[at] <= 0.0 {
                continue;
            }
            let rank = order
                .iter()
                .position(|n| *n == names[rock.materials[at] as usize])
                .expect("a stratum");
            assert!(
                rank >= deepest,
                "column {column}: {} above a deeper stratum",
                order[rank]
            );
            deepest = rank;
            seen.insert(rank);
        }
    }
    assert_eq!(seen.len(), order.len(), "every stratum shows: {seen:?}");
}

#[test]
fn the_history_stands_as_towns_ruins_and_roads() {
    let runtime = world(&["towns", "roads", "ground"]);

    let mut built = BTreeMap::new();
    for y in 0..TILES {
        for x in 0..TILES {
            if let Some(town) = runtime.tiles("towns", ChunkCoord::new(x, y, 0)) {
                built.insert(town.site.clone(), town.rules.to_string());
            }
        }
    }
    let (settlements, _) = history(&pack());
    let kinds: Vec<&str> = settlements
        .iter()
        .map(|row| match &row.values["built"] {
            Value::Name(name) => name.as_str(),
            Value::Number(_) => unreachable!("built is a names column"),
        })
        .collect();
    let mut realised: Vec<&str> = built.values().map(String::as_str).collect();
    realised.sort_unstable();
    let mut expected = kinds.clone();
    expected.sort_unstable();
    assert_eq!(
        realised, expected,
        "a town or ruins for every settlement, of what it was given"
    );
    let roads: BTreeSet<_> = (0..TILES)
        .flat_map(|y| (0..TILES).map(move |x| ChunkCoord::new(x, y, 0)))
        .flat_map(|chunk| {
            runtime
                .curves("roads", chunk)
                .unwrap_or(&[])
                .iter()
                .map(|c| c.id.clone())
        })
        .collect();
    assert_eq!(
        roads.len(),
        SETTLEMENTS - 1,
        "a road from each place to the next"
    );
}

#[test]
fn a_local_area_meets_the_world_map_along_its_tiles_edges() {
    let mut runtime = runtime();
    let chunk = ChunkCoord::new(TILES / 2, TILES / 2 - 3, 0);
    runtime
        .request(&[FocusPoint::new(chunk, 0)], &["local", "plain"])
        .expect("the ground");
    runtime.run_until_idle().expect("the ground generates");

    let (local, plain) = (
        runtime.field("local", chunk).expect("local"),
        runtime.field("plain", chunk).expect("plain"),
    );
    let mut inside = 0.0_f32;
    for y in 0..SIZE[1] {
        for x in 0..SIZE[0] {
            let apart = (local.get(x, y) - plain.get(x, y)).abs();
            if x == 0 || y == 0 || x == SIZE[0] - 1 || y == SIZE[1] - 1 {
                assert!(
                    apart < 0.05,
                    "edge column ({x}, {y}) {apart} cells from the world map"
                );
            } else {
                inside = inside.max(apart);
            }
        }
    }
    assert!(
        inside > 0.3,
        "the tile's inside has detail of its own: {inside}"
    );
}

#[test]
fn the_world_is_the_same_on_every_device() {
    const STAGES: [&str; 4] = ["regions", "ground", "towns", "peaks"];
    let runtime = world(&STAGES);
    let hashes: BTreeMap<(i32, i32), u64> = (0..TILES)
        .flat_map(|y| (0..TILES).map(move |x| (x, y)))
        .map(|(x, y)| {
            let hash = runtime
                .chunk_hash(ChunkCoord::new(x, y, 0), &["ground", "towns", "peaks"])
                .expect("every stage generated");
            ((x, y), hash)
        })
        .collect();
    let fixture = root().join("wfc-devtools/tests/fixtures/golden_colony.txt");

    if std::env::var_os("WAVE_FORGE_BLESS").is_some() {
        let mut out = format!("{HEADER}\n");
        for ((x, y), hash) in &hashes {
            writeln!(out, "{x} {y}: {hash:016x}").expect("writing to a string");
        }
        std::fs::write(&fixture, out).expect("write the fixture");
        return;
    }
    let recorded: BTreeMap<(i32, i32), u64> = std::fs::read_to_string(&fixture)
        .expect("tests/fixtures/golden_colony.txt; record it with WAVE_FORGE_BLESS=1")
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let (coord, hash) = line.split_once(':').expect("a chunk line has a colon");
            let coord: Vec<i32> = coord
                .split_whitespace()
                .map(|n| n.parse().expect("a number"))
                .collect();
            (
                (coord[0], coord[1]),
                u64::from_str_radix(hash.trim(), 16).expect("a hash"),
            )
        })
        .collect();
    let differ: Vec<_> = hashes
        .iter()
        .filter(|(coord, hash)| recorded.get(coord) != Some(hash))
        .map(|(coord, _)| coord)
        .collect();
    assert_eq!(hashes.len(), recorded.len(), "the same chunks");
    assert!(differ.is_empty(), "chunks whose hash differs: {differ:?}");
}
