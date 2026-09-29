//! The maximal preset (docs/product/user-stories.md, M1, #247): a continent of 2 048 by 2 048
//! cells, 4 by 4 km at cells of 2 m. Its biomes and coast are checked by sampling, which
//! generates no chunk; generating part of it to completion is a measurement, run in release:
//!
//! ```text
//! cargo test --release --test continent -- --ignored --nocapture
//! ```

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::{Pack, Runtime};
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
    let mut runtime = Runtime::new(pack(), 11, SIZE);
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
