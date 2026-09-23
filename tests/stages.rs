//! A pack's stages come out the same whatever order their chunks are asked for in.
//!
//! The runtime generates every input chunk within a stage's reach before the stage, and the stage
//! reads nothing beyond it, so a chunk's output is a function of the seed and its coordinate
//! (docs/generation-model.md §2). This asks for a 4×4-chunk area of a noise, blur and arithmetic
//! pack all at once and one chunk at a time in raster and reverse raster order, and compares every
//! value bit for bit.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::{Pack, Runtime};
use wave_forge::{ChunkCoord, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "continents", kind: Field(Noise(frequency: 0.02, octaves: 2))),
        (name: "detail", kind: Field(Noise(frequency: 0.3, octaves: 3))),
        (name: "rough", kind: Field(Add(Mul(Input("continents"), Constant(6.0)), Input("detail")))),
        (name: "smooth", kind: Blur(input: "rough", radius: 5)),
        (name: "height", kind: Field(Mul(Input("smooth"), Constant(4.0)))),
        (name: "towns", kind: Sites(height: "height", region: 5, size: (1, 2), chance: 0.8)),
        (name: "level", kind: Flatten(height: "height", sites: "towns", blend: 6)),
        (name: "shore", kind: Blur(input: "level", radius: 2)),
    ],
)"#;

const SEED: u64 = 42;
const SIZE: [u32; 2] = [8, 8];

type Values = BTreeMap<ChunkCoord, Vec<u32>>;

fn area() -> Vec<ChunkCoord> {
    (0..4)
        .flat_map(|y| (0..4).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

/// The target stage's values over the area, as bits so a comparison is exact.
fn generate(order: &[Vec<ChunkCoord>]) -> Values {
    let pack = Arc::new(Pack::parse(PACK).expect("a valid pack"));
    let mut runtime = Runtime::new(pack, SEED, SIZE);
    let mut values = Values::new();
    for request in order {
        let focus: Vec<FocusPoint> = request
            .iter()
            .map(|&chunk| FocusPoint::new(chunk, 0))
            .collect();
        runtime.request(&focus, &["shore"]).expect("a stage");
        runtime.run_until_idle().expect("the stages run");
        for &chunk in request {
            let field = runtime.field("shore", chunk).expect("generated");
            values.insert(chunk, field.values.iter().map(|v| v.to_bits()).collect());
        }
    }
    values
}

#[test]
fn stages_come_out_the_same_in_any_order() {
    let all_at_once = generate(&[area()]);
    let raster = generate(
        &area()
            .into_iter()
            .map(|chunk| vec![chunk])
            .collect::<Vec<_>>(),
    );
    let reverse = generate(
        &area()
            .into_iter()
            .rev()
            .map(|chunk| vec![chunk])
            .collect::<Vec<_>>(),
    );

    assert_eq!(all_at_once.len(), 16);
    assert_eq!(all_at_once, raster, "all at once against raster");
    assert_eq!(all_at_once, reverse, "all at once against reverse raster");
}

#[test]
fn a_pack_reports_how_far_each_stage_is_generated_beyond_the_target() {
    let pack = Pack::parse(PACK).expect("a valid pack");

    let reach = pack.reach("shore", SIZE).expect("a stage");

    assert_eq!(reach["shore"], 0);
    assert_eq!(reach["level"], 2);
    assert_eq!(reach["towns"], 2 + 6);
    // The towns read the height a region (5 chunks of 8) beyond their chunk.
    assert_eq!(reach["height"], 2 + 6 + 40);
    assert_eq!(reach["rough"], 2 + 6 + 40 + 5);
}

/// The sites of a 30×30-chunk area.
fn sites_of_area() -> (Runtime, Vec<wave_forge::stages::Site>) {
    let pack = Arc::new(Pack::parse(PACK).expect("a valid pack"));
    let mut runtime = Runtime::new(pack, SEED, SIZE);
    let chunks: Vec<ChunkCoord> = (0..30)
        .flat_map(|y| (0..30).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect();
    let focus: Vec<FocusPoint> = chunks
        .iter()
        .map(|&chunk| FocusPoint::new(chunk, 0))
        .collect();
    runtime.request(&focus, &["level"]).expect("a stage");
    runtime.run_until_idle().expect("the stages run");
    let mut sites: BTreeMap<(i32, i32), wave_forge::stages::Site> = BTreeMap::new();
    for chunk in chunks {
        for site in runtime.sites("towns", chunk).expect("generated") {
            sites.insert(site.region, *site);
        }
    }
    (runtime, sites.into_values().collect())
}

#[test]
fn sites_stay_inside_their_region_and_two_chunks_apart() {
    let (_, sites) = sites_of_area();

    assert!(sites.len() > 15, "only {} sites", sites.len());
    for site in &sites {
        let region = (site.region.0 * 5, site.region.1 * 5);
        assert!(
            site.min.0 > region.0 && site.max.0 < region.0 + 5,
            "{site:?}"
        );
        assert!(
            site.min.1 > region.1 && site.max.1 < region.1 + 5,
            "{site:?}"
        );
    }
    for (i, a) in sites.iter().enumerate() {
        for b in &sites[i + 1..] {
            let gap_x = (b.min.0 - a.max.0).max(a.min.0 - b.max.0);
            let gap_y = (b.min.1 - a.max.1).max(a.min.1 - b.max.1);
            assert!(gap_x.max(gap_y) >= 2, "{a:?} and {b:?}");
        }
    }
}

#[test]
fn the_ground_inside_a_site_is_level_at_its_height() {
    let (runtime, sites) = sites_of_area();

    for site in sites
        .iter()
        .filter(|site| site.min.0 >= 1 && site.max.0 < 29)
    {
        for x in site.min.0..site.max.0 {
            for y in site.min.1..site.max.1 {
                let field = runtime
                    .field("level", ChunkCoord::new(x, y, 0))
                    .expect("generated");
                assert!(
                    field.values.iter().all(|&v| v == site.height),
                    "chunk ({x}, {y}) of {site:?}"
                );
            }
        }
    }
}
