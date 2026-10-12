//! The same chunk hashes on every GPU (docs/product/user-stories.md, N9): a pack of ground, towns
//! solved on the GPU from the city's module set and trees around them, each chunk's hash compared
//! with one recorded on another device, so two players sharing a seed and a pack can confirm they
//! agree by comparing hashes.
//!
//! The fixture was recorded on an NVIDIA RTX 3070 through Mesa's dozen driver; CI runs this test on
//! Mesa's lavapipe, a second vendor's view of every integer the town kernels compute, as
//! `golden_world.rs` does for a city's tiles. To record it again after a change that is meant to
//! change what the stages give, run
//!
//! ```text
//! WAVE_FORGE_BLESS=1 cargo test -p wfc-devtools --test golden_chunks
//! ```
//!
//! and say in the commit why the hashes changed.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use wave_forge::loader::parse_rule_file;
use wave_forge::stages::{Pack, Runtime};
use wave_forge::towns::{WfcTowns, gpu_solver};
use wave_forge::{ChunkCoord, ChunkShape, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "height", kind: Field(Mul(Noise(frequency: 0.03, octaves: 3), Constant(6.0)))),
        (name: "sites", kind: Sites(height: "height", region: 4, size: (1, 2), chance: 1.0)),
        (name: "towns", kind: Solve(sites: "sites", rules: "city", bottom: Some(Tagged("street_level")), top: Some(Named("air")))),
        (name: "ground", kind: Flatten(height: "height", sites: "sites", blend: 4)),
        (name: "trees", kind: Scatter(kind: "tree", height: "ground", spacing: 4, apart: 3, avoid: Some(("sites", 1)))),
    ],
)"#;

const STAGES: [&str; 5] = ["height", "sites", "towns", "ground", "trees"];
const SEED: u64 = 21;
const TOWN: ChunkShape = ChunkShape { x: 8, y: 8, z: 6 };
/// Eight chunks by eight, two of the Sites stage's regions a side.
const CHUNKS: i32 = 8;
const HEADER: &str =
    "# wave_forge golden chunks: wfc-devtools/tests/golden_chunks.rs's pack, seed 21, 8x8 chunks";

type Hashes = BTreeMap<(i32, i32), u64>;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_chunks.txt")
}

/// Every chunk's hash, and how many chunks hold a town.
fn generate() -> (Hashes, usize) {
    let text = std::fs::read_to_string(format!(
        "{}/../examples/city.ron",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("the city's module set");
    let towns = WfcTowns::new(TOWN)
        .with_rules(
            "city",
            parse_rule_file(&text).expect("a module set"),
            gpu_solver,
        )
        .expect("a compute device");
    let mut runtime = Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        SEED,
        [TOWN.x, TOWN.y],
    )
    .with_towns(Box::new(towns))
    .expect("matching chunks");
    let area: Vec<ChunkCoord> = (0..CHUNKS)
        .flat_map(|y| (0..CHUNKS).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect();
    let focus: Vec<FocusPoint> = area
        .iter()
        .map(|&chunk| FocusPoint::new(chunk, 0))
        .collect();
    runtime.request(&focus, &STAGES).expect("known stages");
    runtime.run_until_idle().expect("the stages run");

    let towns = area
        .iter()
        .filter(|&&chunk| runtime.tiles("towns", chunk).is_some())
        .count();
    let hashes = area
        .iter()
        .map(|&chunk| {
            let hash = runtime
                .chunk_hash(chunk, &STAGES)
                .expect("every stage generated");
            ((chunk.x, chunk.y), hash)
        })
        .collect();
    (hashes, towns)
}

fn write(hashes: &Hashes) -> String {
    let mut out = format!("{HEADER}\n");
    for ((x, y), hash) in hashes {
        writeln!(out, "{x} {y}: {hash:016x}").expect("writing to a string");
    }
    out
}

fn read(text: &str) -> Hashes {
    text.lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let (coord, hash) = line.split_once(':').expect("a chunk line has a colon");
            let coord: Vec<i32> = coord
                .split_whitespace()
                .map(|n| n.parse().expect("a chunk coordinate"))
                .collect();
            let hash = u64::from_str_radix(hash.trim(), 16).expect("a hash");
            ((coord[0], coord[1]), hash)
        })
        .collect()
}

#[test]
fn every_chunk_hashes_the_same_on_every_device() {
    let (hashes, towns) = generate();
    assert!(
        towns > 0,
        "no chunk holds a town, so no GPU work is compared"
    );

    if std::env::var_os("WAVE_FORGE_BLESS").is_some() {
        std::fs::write(fixture(), write(&hashes)).expect("write the fixture");
        eprintln!(
            "golden_chunks: recorded {} chunks, {towns} with a town",
            hashes.len()
        );
        return;
    }
    let recorded = read(
        &std::fs::read_to_string(fixture())
            .expect("tests/fixtures/golden_chunks.txt; record it with WAVE_FORGE_BLESS=1"),
    );

    let differ: Vec<_> = hashes
        .iter()
        .filter(|(coord, hash)| recorded.get(coord) != Some(hash))
        .map(|(coord, _)| coord)
        .collect();
    assert_eq!(hashes.len(), recorded.len(), "the same chunks");
    assert!(differ.is_empty(), "chunks whose hash differs: {differ:?}");
}
