//! Nearest stages: a biome per column, the one whose point lies nearest the column's climate, as
//! Minecraft's multi-noise biome source picks them.

use std::sync::Arc;
use wave_forge::stages::{Pack, PackError, Runtime};
use wave_forge::{ChunkCoord, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "heat", kind: Field(Remap(X, (0.0, 32.0), (0.0, 1.0)))),
        (name: "wet", kind: Field(Remap(Y, (0.0, 32.0), (0.0, 1.0)))),
        (name: "biomes", kind: Nearest(climate: [Input("heat"), Input("wet")], biomes: [
            (category: "tundra", point: [0.1, 0.2]),
            (category: "desert", point: [0.9, 0.1]),
            (category: "jungle", point: [0.8, 0.9]),
            (category: "taiga", point: [0.2, 0.8]),
            (category: "twin", point: [0.9, 0.1]),
        ])),
        (name: "hot", kind: Field(Is("biomes", ["desert", "jungle"]))),
    ],
)"#;

const POINTS: [[f32; 2]; 5] = [[0.1, 0.2], [0.9, 0.1], [0.8, 0.9], [0.2, 0.8], [0.9, 0.1]];

fn generated(stage: &str, chunks: &[ChunkCoord]) -> Runtime {
    let mut runtime = Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        2,
        [8, 8],
    );
    let focus: Vec<FocusPoint> = chunks.iter().map(|&c| FocusPoint::new(c, 0)).collect();
    runtime
        .request(&focus, &[stage, "heat", "wet"])
        .expect("a stage");
    runtime.run_until_idle().expect("the stages run");
    runtime
}

fn area() -> Vec<ChunkCoord> {
    (0..4)
        .flat_map(|y| (0..4).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

#[test]
fn a_column_takes_the_biome_nearest_its_climate_and_a_tie_the_first_listed() {
    let chunks = area();
    let runtime = generated("biomes", &chunks);

    let mut seen = [false; 5];
    for &chunk in &chunks {
        let biomes = runtime.categories("biomes", chunk).expect("generated");
        let heat = runtime.field("heat", chunk).expect("generated");
        let wet = runtime.field("wet", chunk).expect("generated");
        for i in 0..64 {
            let here = [heat.values[i], wet.values[i]];
            let distance =
                |point: [f32; 2]| (point[0] - here[0]).powi(2) + (point[1] - here[1]).powi(2);
            let nearest = (0..POINTS.len())
                .min_by(|&a, &b| distance(POINTS[a]).total_cmp(&distance(POINTS[b])))
                .expect("biomes");
            assert_eq!(usize::from(biomes.values[i]), nearest, "at {here:?}");
            seen[nearest] = true;
        }
    }
    assert_eq!(
        seen,
        [true, true, true, true, false],
        "the twin never wins its tie"
    );
}

#[test]
fn a_field_reads_the_biomes_and_a_sample_is_what_the_chunk_holds() {
    let chunk = ChunkCoord::new(2, 1, 0);
    let runtime = generated("hot", &[chunk]);

    let biomes = runtime.categories("biomes", chunk).expect("an input");
    let hot = runtime.field("hot", chunk).expect("generated");
    for i in 0..64 {
        let expected = f32::from(u8::from(matches!(biomes.values[i], 1 | 2)));
        assert_eq!(hot.values[i], expected);
    }
    let sampled = runtime
        .sample("biomes", [16.0 + 3.5, 8.0 + 5.5])
        .expect("a Nearest stage samples");
    assert_eq!(sampled, f32::from(biomes.values[5 * 8 + 3]));
}

#[test]
fn a_point_of_another_length_or_a_biome_listed_twice_is_refused() {
    let parse = |biomes: &str| {
        Pack::parse(&format!(
            r#"(version: 1, stages: [
                (name: "heat", kind: Field(X)),
                (name: "biomes", kind: Nearest(climate: [Input("heat")], biomes: [{biomes}])),
            ])"#
        ))
    };

    let long = parse(r#"(category: "a", point: [0.1, 0.2])"#);
    let twice = parse(r#"(category: "a", point: [0.1]), (category: "a", point: [0.5])"#);

    for result in [long, twice] {
        assert!(
            matches!(&result, Err(PackError::Invalid { stage, .. }) if stage == "biomes"),
            "{result:?}"
        );
    }
}
