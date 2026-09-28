//! The same stages on every platform: a pack whose fields, categories and points depend on the
//! math functions a platform's C library would otherwise compute (a sine, a distance, an angle and
//! the spacing between points), compared bit for bit with a record made on another machine.
//!
//! Stages run on the CPU, and a world has to be the same for every player
//! (docs/architecture/engine-integration.md, "Noise that means the same in both engines"). Every
//! operation the stages use is either correctly rounded by IEEE 754 or computed by the `libm`
//! crate, the same everywhere; a function taken from the platform instead would differ in its last
//! bit between Windows and Linux, and a comparison against a threshold could then flip. CI runs
//! this on Linux; `tools/measure_desktop.ps1` runs it on Windows.
//!
//! To record the fixture again after a change that is meant to change stages, run
//!
//! ```text
//! WAVE_FORGE_BLESS=1 cargo test --test golden_stages
//! ```
//!
//! and say in the commit why the stages changed.

use std::path::PathBuf;
use std::sync::Arc;
use wave_forge::stages::{Pack, Runtime};
use wave_forge::{ChunkCoord, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "ground", kind: Field(Add(
            Mul(Noise(frequency: 0.05, octaves: 3), Constant(10.0)),
            Mul(Sin(Mul(X, Constant(0.3))), Constant(2.0)),
        ))),
        (name: "ring", kind: Rules(rules: [
            (category: "inner", when: [Less(Distance((3.0, 5.0)), Constant(20.0))]),
            (category: "east", when: [Less(Angle((3.0, 5.0)), Constant(0.25))]),
        ], otherwise: "rest")),
        (name: "trees", kind: Scatter(kind: "tree", height: "ground", spacing: 3, apart: 3)),
    ],
)"#;

const HEADER: &str =
    "# wave_forge golden stages: tests/golden_stages.rs's pack, seed 9, 8x8 columns, 5x5 chunks";

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_stages.txt")
}

/// A 64-bit FNV-1a hash of `bytes`, continuing from `hash`.
fn fnv(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01B3)
    })
}

/// Each stage's products over the area, hashed bit for bit, one line per stage.
fn generate() -> String {
    let mut runtime = Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        9,
        [8, 8],
    );
    let area: Vec<ChunkCoord> = (-2..=2)
        .flat_map(|y| (-2..=2).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect();
    let focus: Vec<FocusPoint> = area
        .iter()
        .map(|&chunk| FocusPoint::new(chunk, 0))
        .collect();
    runtime
        .request(&focus, &["ground", "ring", "trees"])
        .expect("stages");
    runtime.run_until_idle().expect("the stages run");

    let start = 0xCBF2_9CE4_8422_2325;
    let (mut ground, mut ring, mut trees) = (start, start, start);
    let mut points = 0;
    for &chunk in &area {
        let field = runtime.field("ground", chunk).expect("generated");
        for value in &field.values {
            ground = fnv(ground, &value.to_bits().to_le_bytes());
        }
        let categories = runtime.categories("ring", chunk).expect("generated");
        ring = fnv(ring, &categories.values);
        for point in runtime.points("trees", chunk).expect("generated") {
            points += 1;
            trees = fnv(trees, &point.id.local.to_le_bytes());
            for axis in point.position {
                trees = fnv(trees, &axis.to_bits().to_le_bytes());
            }
        }
    }
    assert!(
        points > 50,
        "only {points} trees, so the spacing is barely tested"
    );
    format!("{HEADER}\nground {ground:016x}\nring {ring:016x}\ntrees {trees:016x}\n")
}

#[test]
fn the_stages_are_the_record_bit_for_bit() {
    let got = generate();

    if std::env::var_os("WAVE_FORGE_BLESS").is_some() {
        std::fs::write(fixture(), &got).expect("write the fixture");
    }
    let expected = std::fs::read_to_string(fixture())
        .expect("tests/fixtures/golden_stages.txt; record it with WAVE_FORGE_BLESS=1")
        .replace("\r\n", "\n");
    assert_eq!(got, expected);
}
