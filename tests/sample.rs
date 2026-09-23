//! Reading a world without chunks: `Runtime::sample` at a point and `Runtime::atlas` over an area.
//!
//! A sample must be exactly what the chunk holding its column would hold, so every test compares
//! bits against chunks the runtime generated. The ring world (`examples/rings.world.ron`) covers
//! fields, categories and a blended match; a two-level pack covers a coarse world map read by a
//! fine stage; the valley pack shows a stage that needs chunks is refused by name. The last test
//! times an atlas of a world of 256 by 256 world tiles, which docs/research/measurements.md records.

use std::sync::Arc;
use std::time::Instant;
use wave_forge::stages::{Pack, Runtime, StageError};
use wave_forge::{ChunkCoord, FocusPoint};

const SIZE: [u32; 2] = [8, 8];

fn pack(file: &str) -> Arc<Pack> {
    let text = std::fs::read_to_string(format!("{}/examples/{file}", env!("CARGO_MANIFEST_DIR")))
        .expect("an example pack");
    Arc::new(Pack::parse(&text).expect("a valid pack"))
}

const TWO_LEVELS: &str = r#"(
    version: 1,
    stages: [
        (name: "continent", scale: 8, kind: Field(Noise(frequency: 0.01, octaves: 2, name: Some("continent")))),
        (name: "biome", scale: 8, kind: Rules(rules: [(category: "sea", when: [Less(Input("continent"), Constant(0.45))])], otherwise: "land")),
        (name: "shore", scale: 8, kind: Blur(input: "continent", radius: 1)),
        (name: "terrain", kind: Field(Add(Input("shore"), Mul(Noise(frequency: 0.2, octaves: 1), Constant(0.1))))),
    ],
)"#;

/// Every chunked value of `stage` within `radius` chunks of the origin, with the WFC cell at its
/// column's centre.
fn chunked(runtime: &mut Runtime, stage: &str, radius: u32) -> Vec<([f32; 2], u32)> {
    runtime
        .request(
            &[FocusPoint::new(ChunkCoord::new(0, 0, 0), radius)],
            &[stage],
        )
        .expect("a stage");
    runtime.run_until_idle().expect("the stages run");
    let r = radius as i32;
    let mut out = Vec::new();
    for cy in -r..=r {
        for cx in -r..=r {
            let chunk = ChunkCoord::new(cx, cy, 0);
            let values: Vec<f32> = match (
                runtime.field(stage, chunk),
                runtime.categories(stage, chunk),
            ) {
                (Some(field), _) => field.values.clone(),
                (None, Some(categories)) => {
                    categories.values.iter().map(|&c| f32::from(c)).collect()
                }
                (None, None) => continue,
            };
            for (i, value) in values.into_iter().enumerate() {
                let (x, y) = (i as i32 % 8, i as i32 / 8);
                let column = [cx * 8 + x, cy * 8 + y];
                out.push((
                    [column[0] as f32 + 0.5, column[1] as f32 + 0.5],
                    value.to_bits(),
                ));
            }
        }
    }
    out
}

#[test]
fn a_sample_is_exactly_what_the_chunk_holds_for_fields_categories_and_matches() {
    let pack = pack("rings.world.ron");
    let sampler = Runtime::new(Arc::clone(&pack), 7, SIZE);

    for stage in ["height", "biome", "terrain"] {
        let mut runtime = Runtime::new(Arc::clone(&pack), 7, SIZE);
        let values = chunked(&mut runtime, stage, 2);
        assert!(!values.is_empty(), "{stage} generated");

        for (at, bits) in values {
            let got = sampler.sample(stage, at).expect("a sampled stage");
            assert_eq!(
                got.to_bits(),
                bits,
                "{stage} at {at:?}: {got} against {}",
                f32::from_bits(bits)
            );
        }
    }
}

#[test]
fn a_coarse_world_map_is_sampled_as_its_chunks_hold_it() {
    let pack = Arc::new(Pack::parse(TWO_LEVELS).expect("a valid pack"));
    let sampler = Runtime::new(Arc::clone(&pack), 5, SIZE);
    let mut runtime = Runtime::new(Arc::clone(&pack), 5, SIZE);
    runtime
        .request(
            &[FocusPoint::new(ChunkCoord::new(0, 0, 0), 9)],
            &["terrain", "biome"],
        )
        .expect("stages");
    runtime.run_until_idle().expect("the stages run");

    for (cx, cy) in [(0, 0), (-1, 0), (1, -1)] {
        let chunk = ChunkCoord::new(cx, cy, 0);
        let coarse = runtime.categories("biome", chunk).expect("a coarse chunk");
        for (i, &category) in coarse.values.iter().enumerate() {
            let (x, y) = (
                (cx * 8 + i as i32 % 8) as f32,
                (cy * 8 + i as i32 / 8) as f32,
            );
            let at = [(x + 0.5) * 8.0, (y + 0.5) * 8.0];
            assert_eq!(
                sampler.sample("biome", at).expect("sampled"),
                f32::from(category)
            );
        }
    }
    for (at, bits) in chunked(&mut Runtime::new(Arc::clone(&pack), 5, SIZE), "terrain", 1) {
        assert_eq!(
            sampler.sample("terrain", at).expect("sampled").to_bits(),
            bits,
            "{at:?}"
        );
    }
}

#[test]
fn an_atlas_is_a_stages_columns_row_by_row() {
    let pack = pack("rings.world.ron");
    let sampler = Runtime::new(Arc::clone(&pack), 7, SIZE);
    let mut runtime = Runtime::new(Arc::clone(&pack), 7, SIZE);
    let chunked = chunked(&mut runtime, "terrain", 1);

    let atlas = sampler
        .atlas("terrain", [-8, -8], [24, 24])
        .expect("an atlas");

    assert_eq!(atlas.len(), 24 * 24);
    for (at, bits) in chunked {
        let (x, y) = ((at[0] - 0.5) as i64 + 8, (at[1] - 0.5) as i64 + 8);
        assert_eq!(atlas[(y * 24 + x) as usize].to_bits(), bits, "{at:?}");
    }
}

#[test]
fn a_stage_that_needs_chunks_is_refused_by_name() {
    let sampler = Runtime::new(pack("valley.world.ron"), 1, SIZE);

    assert_eq!(sampler.sample("hills", [3.0, 4.0]).map(|_| ()), Ok(()));
    assert_eq!(
        sampler.sample("level", [3.0, 4.0]),
        Err(StageError::NotSampled("level".to_owned()))
    );
    assert_eq!(
        sampler.sample("nothing", [0.0, 0.0]),
        Err(StageError::UnknownStage("nothing".to_owned()))
    );
}

#[test]
fn a_world_map_of_256_by_256_tiles_is_quick_to_read() {
    let pack = Arc::new(Pack::parse(TWO_LEVELS).expect("a valid pack"));
    let sampler = Runtime::new(pack, 5, SIZE);

    let started = Instant::now();
    let atlas = sampler
        .atlas("biome", [-128, -128], [256, 256])
        .expect("an atlas");
    let ms = started.elapsed().as_secs_f64() * 1000.0;

    println!("an atlas of 256 by 256 world tiles: {ms:.0} ms");
    assert_eq!(atlas.len(), 256 * 256);
    assert!(
        atlas.iter().any(|&c| c == 0.0) && atlas.iter().any(|&c| c == 1.0),
        "sea and land"
    );
}
