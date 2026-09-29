//! G1's check (docs/product/user-stories.md): an unbounded voxel world from
//! `examples/voxel.world.ron`. Its biomes follow its climate, its terrain has overhangs and caves,
//! its aquifers pool water and lava in the caves, its surface takes materials by rule, ore lies in
//! its rock, trees grow on its top and a village stands on levelled ground; two travel orders give
//! the same world, and a changed number changes only the stages that depend on it.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use wave_forge::stages::{Pack, Runtime, Volume};
use wave_forge::{ChunkCoord, FocusPoint};

const SIZE: [u32; 2] = [8, 8];
const SEED: u64 = 12;

/// Every stage of the pack.
const STAGES: [&str; 14] = [
    "continent",
    "temperature",
    "humidity",
    "biomes",
    "height",
    "rock",
    "villages",
    "village",
    "terrain",
    "surface",
    "fluid",
    "coal",
    "iron",
    "trees",
];

fn text() -> String {
    std::fs::read_to_string(format!(
        "{}/examples/voxel.world.ron",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("the voxel pack")
}

/// Each stage's products over `chunks`, asked for one batch at a time, each hashed bit for bit
/// with 64-bit FNV-1a as soon as its batch is generated.
fn world(text: &str, batches: &[Vec<ChunkCoord>]) -> BTreeMap<(&'static str, [i32; 2]), u64> {
    let pack = Arc::new(Pack::parse(text).expect("a valid pack"));
    let mut runtime = Runtime::new(pack, SEED, SIZE);
    let mut hashes = BTreeMap::new();
    for batch in batches {
        let focus: Vec<FocusPoint> = batch.iter().map(|&c| FocusPoint::new(c, 0)).collect();
        runtime.request(&focus, &STAGES).expect("the stages");
        runtime.run_until_idle().expect("the stages run");
        for &chunk in batch {
            for stage in STAGES {
                let product = runtime.product(stage, chunk).expect("generated");
                let hash = format!("{product:?}")
                    .bytes()
                    .fold(0xCBF2_9CE4_8422_2325_u64, |hash, byte| {
                        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01B3)
                    });
                hashes.insert((stage, [chunk.x, chunk.y]), hash);
            }
        }
    }
    hashes
}

/// A runtime with every stage generated over `chunks`.
fn generated(chunks: &[ChunkCoord]) -> Runtime {
    let pack = Arc::new(Pack::parse(&text()).expect("a valid pack"));
    let mut runtime = Runtime::new(pack, SEED, SIZE);
    let focus: Vec<FocusPoint> = chunks.iter().map(|&c| FocusPoint::new(c, 0)).collect();
    runtime.request(&focus, &STAGES).expect("the stages");
    runtime.run_until_idle().expect("the stages run");
    runtime
}

fn area() -> Vec<ChunkCoord> {
    (-5..5)
        .flat_map(|y| (-5..5).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

/// Every voxel of `volume` with its column, level and height in cells.
fn voxels(volume: &Volume) -> impl Iterator<Item = (u32, u32, u32, f32)> + '_ {
    (0..volume.size[2]).flat_map(move |level| {
        (0..volume.size[1]).flat_map(move |y| {
            (0..volume.size[0])
                .map(move |x| (x, y, level, (volume.bottom + level as i32) as f32 + 0.5))
        })
    })
}

#[test]
fn every_biome_takes_the_point_nearest_its_climate_somewhere() {
    let chunks = area();
    let runtime = generated(&chunks);

    let seen: BTreeSet<u8> = chunks
        .iter()
        .flat_map(|&chunk| {
            runtime
                .categories("biomes", chunk)
                .expect("generated")
                .values
                .clone()
        })
        .collect();

    assert_eq!(seen, (0..5).collect(), "biomes seen");
}

#[test]
fn the_terrain_has_overhangs_and_caves_under_its_skin() {
    let chunks = area();
    let runtime = generated(&chunks);

    // An overhang: air under rock in a column, above the height's skin of six cells.
    // A cave: air more than six cells under the height.
    let (mut overhangs, mut caves) = (0, 0);
    for &chunk in &chunks {
        let terrain = runtime.volume("terrain", chunk).expect("generated");
        let height = runtime.field("height", chunk).expect("generated");
        for (x, y, level, z) in voxels(terrain) {
            let skin = height.get(x, y) - 6.0;
            if terrain.get(x, y, level) >= 0.0 {
                continue;
            }
            if z < skin {
                caves += 1;
            } else if (level + 1..terrain.size[2]).any(|above| terrain.get(x, y, above) > 0.0) {
                overhangs += 1;
            }
        }
    }

    assert!(overhangs > 100, "only {overhangs} voxels of air under rock");
    assert!(caves > 1000, "only {caves} voxels of cave");
}

#[test]
fn aquifers_pool_water_and_lava_in_the_caves_and_leave_some_dry() {
    let chunks = area();
    let runtime = generated(&chunks);

    let (mut water, mut lava, mut dry) = (0, 0, 0);
    for &chunk in &chunks {
        let terrain = runtime.volume("terrain", chunk).expect("generated");
        let fluid = runtime.volume("fluid", chunk).expect("generated");
        for (x, y, level, z) in voxels(fluid) {
            if terrain.get(x, y, level) >= 0.0 || z > -8.0 {
                continue;
            }
            match (fluid.get(x, y, level) > 0.0, fluid.material(x, y, level)) {
                (false, _) => dry += 1,
                (true, 0) => lava += 1,
                (true, _) => water += 1,
            }
        }
    }

    assert!(
        water > 100 && lava > 100 && dry > 100,
        "{water} water, {lava} lava and {dry} dry voxels of cave under the highest pool"
    );
}

#[test]
fn the_surface_takes_its_materials_by_rule() {
    let chunks = area();
    let runtime = generated(&chunks);
    let names = Pack::parse(&text())
        .expect("a valid pack")
        .kind("rock")
        .expect("a stage")
        .categories()
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();

    // The material of each column's topmost solid voxel.
    let mut tops: BTreeMap<String, usize> = BTreeMap::new();
    for &chunk in &chunks {
        let rock = runtime.volume("rock", chunk).expect("generated");
        for y in 0..SIZE[1] {
            for x in 0..SIZE[0] {
                if let Some(level) = (0..rock.size[2]).rev().find(|&l| rock.get(x, y, l) > 0.0) {
                    let name = &names[usize::from(rock.material(x, y, level))];
                    *tops.entry(name.clone()).or_default() += 1;
                }
            }
        }
    }

    for material in ["snow", "sand", "grass"] {
        assert!(tops.get(material).copied().unwrap_or(0) > 20, "{tops:?}");
    }
}

#[test]
fn ore_lies_in_the_rock_and_trees_on_the_top() {
    let chunks = area();
    let runtime = generated(&chunks);

    let (mut ore, mut trees) = (0, 0);
    for &chunk in &chunks {
        for stage in ["coal", "iron"] {
            ore += runtime.points(stage, chunk).expect("generated").len();
        }
        let surface = runtime.field("surface", chunk).expect("generated");
        for tree in runtime.points("trees", chunk).expect("generated") {
            let x = tree.position[0].floor() as i64 - i64::from(chunk.x) * 8;
            let y = tree.position[1].floor() as i64 - i64::from(chunk.y) * 8;
            assert_eq!(tree.position[2], surface.get(x as u32, y as u32));
            trees += 1;
        }
    }

    assert!(ore > 500 && trees > 50, "{ore} ore and {trees} trees");
}

#[test]
fn a_village_grows_on_ground_levelled_under_its_pieces() {
    let chunks = area();
    let runtime = generated(&chunks);

    let mut pieces = BTreeSet::new();
    for &chunk in &chunks {
        for stamp in runtime.stamps("village", chunk).expect("generated") {
            pieces.insert(stamp.id);
        }
    }
    let mut levelled = 0;
    for &chunk in &chunks {
        let surface = runtime.field("surface", chunk).expect("generated");
        for stamp in runtime.stamps("village", chunk).expect("generated") {
            let floor = stamp.position[2];
            let x = stamp.position[0].floor() as i64 - i64::from(chunk.x) * 8;
            let y = stamp.position[1].floor() as i64 - i64::from(chunk.y) * 8;
            if (0..8).contains(&x) && (0..8).contains(&y) {
                let top = surface.get(x as u32, y as u32);
                assert!((top - floor).abs() < 1.0, "a piece at {floor} over {top}");
                levelled += 1;
            }
        }
    }

    assert!(pieces.len() >= 5 && levelled > 0, "{} pieces", pieces.len());
}

#[test]
fn two_travel_orders_give_the_same_world() {
    let chunks = area();

    let at_once = world(&text(), std::slice::from_ref(&chunks));
    let backwards: Vec<Vec<ChunkCoord>> = chunks.iter().rev().map(|&c| vec![c]).collect();
    let walked = world(&text(), &backwards);

    assert_eq!(walked, at_once);
}

#[test]
fn a_changed_number_changes_only_the_stages_that_depend_on_it() {
    let chunks = area();
    let before = world(&text(), std::slice::from_ref(&chunks));
    // The stages whose products changed anywhere in the area.
    let changed = |from: &str, to: &str| {
        let text = text();
        assert_eq!(text.matches(from).count(), 1, "{from}");
        let after = world(&text.replace(from, to), std::slice::from_ref(&chunks));
        after
            .iter()
            .filter(|&(key, hash)| before[key] != *hash)
            .map(|(&(stage, _), _)| stage)
            .collect::<BTreeSet<_>>()
    };

    let ore = changed("spacing: 6, between", "spacing: 5, between");
    let overhangs = changed(
        r#"Mul(FastNoise("overhang"), Constant(10.0))"#,
        r#"Mul(FastNoise("overhang"), Constant(11.0))"#,
    );

    // Iron is the only stage that reads its spacing; the overhangs reach the rock and what is
    // built on it, and never the climate, the height or the villages.
    assert_eq!(ore, BTreeSet::from(["iron"]));
    let downstream = BTreeSet::from([
        "rock", "terrain", "surface", "fluid", "coal", "iron", "trees",
    ]);
    assert!(overhangs.contains("rock"), "{overhangs:?}");
    assert!(overhangs.is_subset(&downstream), "{overhangs:?}");
}
