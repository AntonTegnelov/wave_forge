//! `NoiseConfig` against Godot's own `FastNoiseLite`.
//!
//! `tests/fixtures/fastnoise/godot.json` holds what Godot 4.7.2's `get_noise_2d` returned for a
//! matrix of configurations (every noise type, fractal type, cellular distance function and return
//! type, jitters, weighted strengths, ping pong strengths, offsets, and every domain warp type
//! under every warp fractal type, over seeds up to both ends of `i32`) at points that are negative,
//! fractional, on lattice lines and far from the origin. `generate.gd` beside it writes the file.
//!
//! The tolerance is zero: every sample must equal Godot's exactly. The port does Godot's `f32`
//! operations in Godot's order, and Godot's x86_64 build does not fuse multiply-adds, so any
//! difference is a mistake in the port. The one allowance is the sign of zero, which Godot's JSON
//! drops: a Perlin sample on a lattice point is -0.0 in Godot and in the port alike, and 0.0 in the
//! file.
//!
//! The configuration is Godot's in a pack too: every property is written, and one left out takes
//! Godot's default. A pack names its noises, a Field reads one at its columns' centres, and an
//! engine may replace one, a Godot `FastNoiseLite` resource say.

use serde_json::Value;
use std::sync::Arc;
use wave_forge::noise::{
    CellularDistanceFunction, CellularReturnType, DomainWarpFractalType, DomainWarpType,
    FractalType, NoiseConfig, NoiseType,
};
use wave_forge::stages::{Pack, PackError, Runtime, StageError};
use wave_forge::{ChunkCoord, FocusPoint};

/// Godot's enums in the order of their integers.
const NOISE_TYPES: [NoiseType; 6] = [
    NoiseType::Simplex,
    NoiseType::SimplexSmooth,
    NoiseType::Cellular,
    NoiseType::Perlin,
    NoiseType::ValueCubic,
    NoiseType::Value,
];
const FRACTAL_TYPES: [FractalType; 4] = [
    FractalType::None,
    FractalType::Fbm,
    FractalType::Ridged,
    FractalType::PingPong,
];
const DISTANCE_FUNCTIONS: [CellularDistanceFunction; 4] = [
    CellularDistanceFunction::Euclidean,
    CellularDistanceFunction::EuclideanSquared,
    CellularDistanceFunction::Manhattan,
    CellularDistanceFunction::Hybrid,
];
const RETURN_TYPES: [CellularReturnType; 7] = [
    CellularReturnType::CellValue,
    CellularReturnType::Distance,
    CellularReturnType::Distance2,
    CellularReturnType::Distance2Add,
    CellularReturnType::Distance2Sub,
    CellularReturnType::Distance2Mul,
    CellularReturnType::Distance2Div,
];
const WARP_TYPES: [DomainWarpType; 3] = [
    DomainWarpType::Simplex,
    DomainWarpType::SimplexReduced,
    DomainWarpType::BasicGrid,
];
const WARP_FRACTAL_TYPES: [DomainWarpFractalType; 3] = [
    DomainWarpFractalType::None,
    DomainWarpFractalType::Progressive,
    DomainWarpFractalType::Independent,
];

/// Godot's property names, which the configuration's fields carry.
const PROPERTIES: [&str; 21] = [
    "noise_type",
    "seed",
    "frequency",
    "offset",
    "fractal_type",
    "fractal_octaves",
    "fractal_lacunarity",
    "fractal_gain",
    "fractal_weighted_strength",
    "fractal_ping_pong_strength",
    "cellular_distance_function",
    "cellular_return_type",
    "cellular_jitter",
    "domain_warp_enabled",
    "domain_warp_type",
    "domain_warp_amplitude",
    "domain_warp_frequency",
    "domain_warp_fractal_type",
    "domain_warp_fractal_octaves",
    "domain_warp_fractal_lacunarity",
    "domain_warp_fractal_gain",
];

fn fixture() -> Value {
    let path = format!(
        "{}/tests/fixtures/fastnoise/godot.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("the Godot fixture");
    serde_json::from_str(&text).expect("the Godot fixture is JSON")
}

/// Godot's floats are `f32` widened to `f64`, so narrowing them back is exact.
fn float(value: &Value) -> f32 {
    value.as_f64().expect("a number") as f32
}

fn int(config: &Value, property: &str) -> i64 {
    config[property].as_i64().expect(property)
}

fn index(config: &Value, property: &str) -> usize {
    usize::try_from(int(config, property)).expect(property)
}

fn config_from_godot(config: &Value) -> NoiseConfig {
    let offset = config["offset"].as_array().expect("an offset");
    NoiseConfig {
        noise_type: NOISE_TYPES[index(config, "noise_type")],
        seed: i32::try_from(int(config, "seed")).expect("a seed in i32"),
        frequency: float(&config["frequency"]),
        offset: [float(&offset[0]), float(&offset[1]), float(&offset[2])],
        fractal_type: FRACTAL_TYPES[index(config, "fractal_type")],
        fractal_octaves: i32::try_from(int(config, "fractal_octaves")).expect("octaves"),
        fractal_lacunarity: float(&config["fractal_lacunarity"]),
        fractal_gain: float(&config["fractal_gain"]),
        fractal_weighted_strength: float(&config["fractal_weighted_strength"]),
        fractal_ping_pong_strength: float(&config["fractal_ping_pong_strength"]),
        cellular_distance_function: DISTANCE_FUNCTIONS[index(config, "cellular_distance_function")],
        cellular_return_type: RETURN_TYPES[index(config, "cellular_return_type")],
        cellular_jitter: float(&config["cellular_jitter"]),
        domain_warp_enabled: config["domain_warp_enabled"]
            .as_bool()
            .expect("domain_warp_enabled"),
        domain_warp_type: WARP_TYPES[index(config, "domain_warp_type")],
        domain_warp_amplitude: float(&config["domain_warp_amplitude"]),
        domain_warp_frequency: float(&config["domain_warp_frequency"]),
        domain_warp_fractal_type: WARP_FRACTAL_TYPES[index(config, "domain_warp_fractal_type")],
        domain_warp_fractal_octaves: i32::try_from(int(config, "domain_warp_fractal_octaves"))
            .expect("warp octaves"),
        domain_warp_fractal_lacunarity: float(&config["domain_warp_fractal_lacunarity"]),
        domain_warp_fractal_gain: float(&config["domain_warp_fractal_gain"]),
    }
}

#[test]
fn every_sample_equals_godots() {
    let fixture = fixture();
    let points = fixture["points"].as_array().expect("points");
    let cases = fixture["cases"].as_array().expect("cases");

    let mut mismatches = Vec::new();
    for (case, entry) in cases.iter().enumerate() {
        let config = config_from_godot(&entry["config"]);
        let values = entry["values"].as_array().expect("values");
        assert_eq!(
            values.len(),
            points.len(),
            "case {case} has a value per point"
        );
        for (point, expected) in points.iter().zip(values) {
            let x = float(&point[0]);
            let y = float(&point[1]);
            let expected = float(expected);
            let actual = config.sample(x, y);
            if actual != expected {
                mismatches.push(format!(
                    "case {case} at ({x}, {y}): Godot {expected}, port {actual}, {config:?}"
                ));
            }
        }
    }

    assert!(
        mismatches.is_empty(),
        "{} of {} samples differ from Godot's:\n{}",
        mismatches.len(),
        cases.len() * points.len(),
        mismatches.join("\n")
    );
}

#[test]
fn the_default_is_godots_and_every_property_is_written() {
    let fixture = fixture();
    let godot_default = config_from_godot(&fixture["cases"][0]["config"]);

    let text = ron::to_string(&NoiseConfig::default()).expect("a config serialises");

    assert_eq!(NoiseConfig::default(), godot_default);
    for property in PROPERTIES {
        assert!(
            text.contains(&format!("{property}:")),
            "{property} missing from {text}"
        );
    }
    assert_eq!(
        ron::from_str::<NoiseConfig>(&text),
        Ok(NoiseConfig::default())
    );
}

#[test]
fn a_property_left_out_takes_godots_default() {
    let config: NoiseConfig =
        ron::from_str("(noise_type: Cellular, seed: -7, offset: (1.5, 0.0, 0.0))")
            .expect("a partial config reads");

    assert_eq!(
        config,
        NoiseConfig {
            noise_type: NoiseType::Cellular,
            seed: -7,
            offset: [1.5, 0.0, 0.0],
            ..NoiseConfig::default()
        }
    );
}

#[test]
fn a_misspelt_property_is_refused() {
    let result = ron::from_str::<NoiseConfig>("(fractal_octave: 3)");

    assert!(result.is_err(), "read as {result:?}");
}

const PACK: &str = r#"(
    version: 1,
    noises: {
        "hills": (noise_type: Perlin, seed: 77, frequency: 0.03, fractal_octaves: 3),
    },
    stages: [
        (name: "height", kind: Field(Mul(FastNoise("hills"), Constant(10.0)))),
    ],
)"#;

/// The height stage's value at every column of the chunk at the origin.
fn heights(runtime: &mut Runtime) -> Vec<f32> {
    let chunk = ChunkCoord::new(0, 0, 0);
    runtime
        .request(&[FocusPoint::new(chunk, 0)], &["height"])
        .expect("a stage");
    runtime.run_until_idle().expect("the stages run");
    runtime
        .field("height", chunk)
        .expect("generated")
        .values
        .clone()
}

#[test]
fn a_field_reads_a_named_noise_at_its_columns_centres_in_cells() {
    let pack = Arc::new(Pack::parse(PACK).expect("a valid pack"));
    let noise = NoiseConfig {
        noise_type: NoiseType::Perlin,
        seed: 77,
        frequency: 0.03,
        fractal_octaves: 3,
        ..NoiseConfig::default()
    };

    let values = heights(&mut Runtime::new(pack, 1234, [8, 8]));

    for (i, value) in values.iter().enumerate() {
        let (x, y) = ((i % 8) as f32 + 0.5, (i / 8) as f32 + 0.5);
        assert_eq!(*value, noise.sample(x, y) * 10.0, "column {i}");
    }
}

#[test]
fn an_engine_replaces_a_named_noise_and_the_world_seed_leaves_it_alone() {
    let pack = Arc::new(Pack::parse(PACK).expect("a valid pack"));
    let replaced = NoiseConfig {
        noise_type: NoiseType::Value,
        seed: 5,
        ..NoiseConfig::default()
    };

    let mut runtime = Runtime::new(Arc::clone(&pack), 1, [8, 8])
        .with_noise("hills", replaced)
        .expect("a named noise");
    let values = heights(&mut runtime);
    let other_world = heights(
        &mut Runtime::new(Arc::clone(&pack), 2, [8, 8])
            .with_noise("hills", replaced)
            .expect("a named noise"),
    );
    let unknown = Runtime::new(pack, 1, [8, 8]).with_noise("valleys", replaced);

    assert_eq!(values[0], replaced.sample(0.5, 0.5) * 10.0);
    assert_eq!(values, other_world);
    assert!(matches!(unknown, Err(StageError::UnknownNoise(name)) if name == "valleys"));
}

#[test]
fn a_noise_the_pack_does_not_name_is_refused_by_stage() {
    let result =
        Pack::parse(r#"(version: 1, stages: [(name: "height", kind: Field(FastNoise("hills")))])"#);

    assert!(
        matches!(&result, Err(PackError::Invalid { stage, .. }) if stage == "height"),
        "{result:?}"
    );
}
