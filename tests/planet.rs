//! G3's check (docs/product/user-stories.md): a planet section from a planet's row in a generated
//! table, `examples/planet.world.ron`. Its terrain is 3D density with overhangs, its outposts stand
//! on an offset grid on levelled ground, its plants and rocks are scattered by biome, an outpost is
//! located without generating the chunks between, and a dig in its terrain is kept in the edits log
//! and survives leaving and returning.

use std::sync::Arc;
use wave_forge::stages::{Edit, Edits, Facts, Pack, RowId, Runtime, Site, Volume};
use wave_forge::{ChunkCoord, FocusPoint};

const SIZE: [u32; 2] = [8, 8];
const SEED: u64 = 31;

fn pack() -> Arc<Pack> {
    let text = std::fs::read_to_string(format!(
        "{}/examples/planet.world.ron",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("the planet pack");
    Arc::new(Pack::parse(&text).expect("a valid pack"))
}

/// The id of the `n`th planet of the generated table.
fn planet(n: usize) -> RowId {
    let facts = Facts::new(pack(), SEED).expect("facts");
    facts.table("planets").expect("planets").rows[n].id.clone()
}

/// A runtime focused on `planet`, with `stages` generated over `chunks`.
fn world(planet: &RowId, stages: &[&str], chunks: &[ChunkCoord]) -> Runtime {
    let pack = pack();
    let mut runtime = Runtime::new(Arc::clone(&pack), SEED, SIZE);
    runtime
        .set_facts(Facts::new(pack, SEED).expect("facts"))
        .expect("facts of this pack");
    runtime.focus("planets", planet.clone()).expect("a planet");
    generate(&mut runtime, stages, chunks);
    runtime
}

fn generate(runtime: &mut Runtime, stages: &[&str], chunks: &[ChunkCoord]) {
    let focus: Vec<FocusPoint> = chunks.iter().map(|&c| FocusPoint::new(c, 0)).collect();
    runtime.request(&focus, stages).expect("the stages");
    runtime.run_until_idle().expect("the stages run");
}

fn section() -> Vec<ChunkCoord> {
    (-4..4)
        .flat_map(|y| (-4..4).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

/// A column's world column and index from its place in a chunk.
fn columns(chunk: ChunkCoord) -> impl Iterator<Item = ([i64; 2], usize)> {
    (0..8).flat_map(move |y| {
        (0..8).map(move |x| {
            (
                [i64::from(chunk.x) * 8 + x, i64::from(chunk.y) * 8 + y],
                (y * 8 + x) as usize,
            )
        })
    })
}

#[test]
fn a_planets_row_decides_its_section_and_the_same_row_gives_the_same_section() {
    let chunks = section();
    let (first, second) = (planet(0), planet(5));

    let a = world(&first, &["terrain"], &chunks);
    let again = world(&first, &["terrain"], &chunks);
    let b = world(&second, &["terrain"], &chunks);

    for &chunk in &chunks {
        assert_eq!(a.volume("terrain", chunk), again.volume("terrain", chunk));
    }
    assert!(
        chunks
            .iter()
            .any(|&chunk| a.volume("terrain", chunk) != b.volume("terrain", chunk)),
        "two planets gave the same section"
    );
}

#[test]
fn the_terrain_has_overhangs() {
    let chunks = section();
    let runtime = world(&planet(0), &["terrain"], &chunks);

    // An overhang: going up a column, solid, then air, then solid again.
    let overhangs = |volume: &Volume| {
        let mut count = 0;
        for y in 0..8 {
            for x in 0..8 {
                let solid: Vec<bool> = (0..volume.size[2])
                    .map(|level| volume.get(x, y, level) > 0.0)
                    .collect();
                let air_under_solid = solid
                    .windows(2)
                    .enumerate()
                    .any(|(level, pair)| !pair[0] && pair[1] && solid[..level].contains(&true));
                count += usize::from(air_under_solid);
            }
        }
        count
    };
    let total: usize = chunks
        .iter()
        .map(|&chunk| overhangs(runtime.volume("terrain", chunk).expect("generated")))
        .sum();

    assert!(total > 20, "only {total} columns have air under rock");
}

#[test]
fn every_outpost_stands_on_level_ground_under_open_sky() {
    let chunks = section();
    let runtime = world(&planet(0), &["surface", "outposts"], &chunks);

    let mut outposts: Vec<Site> = chunks
        .iter()
        .flat_map(|&chunk| {
            runtime
                .sites("outposts", chunk)
                .expect("generated")
                .to_vec()
        })
        .collect();
    outposts.sort_by(|a, b| a.id.cmp(&b.id));
    outposts.dedup_by(|a, b| a.id == b.id);
    assert!(!outposts.is_empty(), "no outpost in the section");
    let mut checked = 0;
    for &chunk in &chunks {
        let surface = runtime.field("surface", chunk).expect("generated");
        let terrain = runtime.volume("terrain", chunk).expect("generated");
        for (column, i) in columns(chunk) {
            // Inside a footprint, a column away from its edges.
            let Some(site) = outposts.iter().find(|site| {
                let inside = |at: i64, low: i32, high: i32| {
                    i64::from(low) * 8 < at && at < i64::from(high) * 8 - 1
                };
                inside(column[0], site.min.0, site.max.0)
                    && inside(column[1], site.min.1, site.max.1)
            }) else {
                continue;
            };
            assert!(
                (surface.values[i] - site.height).abs() < 1.0,
                "{column:?} stands at {}, its outpost at {}",
                surface.values[i],
                site.height
            );
            for level in 0..terrain.size[2] {
                let z = (terrain.bottom + level as i32) as f32 + 0.5;
                if site.height + 0.5 < z {
                    let x = (column[0] - i64::from(chunk.x) * 8) as u32;
                    let y = (column[1] - i64::from(chunk.y) * 8) as u32;
                    assert!(
                        terrain.get(x, y, level) < 0.0,
                        "{column:?} at {z} over its outpost"
                    );
                }
            }
            checked += 1;
        }
    }
    assert!(checked > 20, "only {checked} columns inside outposts");
}

#[test]
fn plants_grow_in_jungles_and_rocks_in_deserts_on_the_terrains_top() {
    let chunks = section();
    let runtime = world(
        &planet(0),
        &["plants", "rocks", "biomes", "surface"],
        &chunks,
    );
    let biomes = pack().kind("biomes").expect("a stage").categories().len();
    assert_eq!(biomes, 2);

    let mut counts = [0, 0];
    for &chunk in &chunks {
        let categories = runtime.categories("biomes", chunk).expect("generated");
        let surface = runtime.field("surface", chunk).expect("generated");
        for (stage, biome) in [("plants", 0), ("rocks", 1)] {
            for point in runtime.points(stage, chunk).expect("generated") {
                let (x, y) = (
                    point.position[0].floor() as i64 - i64::from(chunk.x) * 8,
                    point.position[1].floor() as i64 - i64::from(chunk.y) * 8,
                );
                let i = (y * 8 + x) as usize;
                assert_eq!(
                    categories.values[i], biome,
                    "a {stage} point at {:?}",
                    point.position
                );
                assert_eq!(
                    point.position[2], surface.values[i],
                    "a {stage} point's height"
                );
                counts[biome as usize] += 1;
            }
        }
    }
    assert!(
        counts[0] > 10 && counts[1] > 5,
        "{counts:?} plants and rocks"
    );
}

#[test]
fn the_nearest_outpost_is_located_without_generating_the_chunks_between() {
    let chunks = section();
    let generated = world(&planet(0), &["outposts"], &chunks);
    let fresh = world(&planet(0), &[], &[]);

    let located = fresh
        .locate("outposts", [3.0, 5.0], 4)
        .expect("a Sites stage")
        .expect("an outpost within four regions");

    let placed: Vec<Site> = chunks
        .iter()
        .flat_map(|&chunk| {
            generated
                .sites("outposts", chunk)
                .expect("generated")
                .to_vec()
        })
        .collect();
    assert!(placed.contains(&located), "{located:?}");
    assert!(fresh.volume("terrain", ChunkCoord::new(0, 0, 0)).is_none());
}

#[test]
fn a_dig_is_kept_in_the_edits_log_and_survives_leaving_and_returning() {
    let here = ChunkCoord::new(0, 0, 0);
    let mut runtime = world(&planet(0), &["terrain"], &[here]);
    let before = runtime.volume("terrain", here).expect("generated").clone();
    let surface = before.get(4, 4, 0);
    assert!(surface.is_finite());

    let log = vec![Edit::Dig {
        stage: "terrain".to_owned(),
        at: [4.5, 4.5, 8.0],
        radius: 4.0,
    }];
    runtime
        .set_edits(&Edits { log: log.clone() })
        .expect("a dig of a volume");
    generate(&mut runtime, &["terrain"], &[here]);
    let dug = runtime.volume("terrain", here).expect("generated").clone();
    generate(&mut runtime, &["terrain"], &[ChunkCoord::new(40, 40, 0)]);
    generate(&mut runtime, &["terrain"], &[here]);
    let returned = runtime.volume("terrain", here).expect("generated").clone();

    assert_ne!(dug, before);
    assert_eq!(returned, dug);
    assert_eq!(runtime.save().edits.log, log);
}
