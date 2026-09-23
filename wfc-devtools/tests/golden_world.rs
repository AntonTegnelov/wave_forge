//! The same world on every GPU: a small city compared tile for tile with one recorded on another
//! device.
//!
//! A world is a function of its configuration, independent of backend, vendor and thread count
//! (docs/architecture.md §6.3). The fixture was recorded on an NVIDIA RTX 3070 through Mesa's dozen
//! driver; CI runs this test on Mesa's lavapipe, a CPU implementation of Vulkan, which is a second
//! vendor's view of every integer the kernel computes. A mismatch names the first cell that
//! differs, with both tiles.
//!
//! To record the fixture again after a change that is meant to change worlds, run
//!
//! ```text
//! WAVE_FORGE_BLESS=1 cargo test -p wfc-devtools --test golden_world
//! ```
//!
//! and say in the commit why the world changed.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use wave_forge::{Builder, ChunkCoord, ChunkEvent, ChunkShape, FocusPoint, Ruleset, WorldExtent};
use wfc_devtools::city::{self, city_prior};

const CHUNK: ChunkShape = ChunkShape::cube(8);
/// A seed whose world needs repairs, which rewrite chunks the first attempts already solved: the
/// part of generation most likely to differ between devices, so it has to be in what is compared.
const SEED: u64 = 8;
/// Four chunks by four: both parities, chunks against the world's edges and chunks inside.
const CHUNKS: i32 = 4;
const HEADER: &str = "# wave_forge golden world: examples/city.ron, seed 8, 8x8x8 chunks, 4x4x1";

type World = BTreeMap<ChunkCoord, Vec<u16>>;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_city.txt")
}

/// The city, generated as a game would ask for it, and what it ran on and cost.
fn generate() -> (World, String) {
    let city = city::city();
    let ruleset = Ruleset::from_modules(&city.modules).expect("the city compiles");
    let mut world = Builder::new(ruleset, city_prior(&city, CHUNK.z))
        .seed(SEED)
        .extent(
            WorldExtent::new(CHUNK)
                .with_x(0..CHUNKS)
                .with_y(0..CHUNKS)
                .with_z(0..1),
        )
        .halo(1)
        .build()
        .expect("a compute device");
    world.request(&[FocusPoint::new(ChunkCoord::new(1, 1, 0), 3)]);
    let events = world.run_until_idle().expect("the solver runs");
    let failed: Vec<&ChunkEvent> = events
        .iter()
        .filter(|event| matches!(event, ChunkEvent::Failed { .. }))
        .collect();
    assert!(failed.is_empty(), "chunks given up on: {failed:?}");
    assert!(
        world.stats().repaired > 0,
        "the world must include repairs to test them: {:?}",
        world.stats()
    );
    let tiles = world
        .store()
        .iter()
        .map(|chunk| (chunk.coord, chunk.tiles.to_vec()))
        .collect();
    let ran = format!(
        "{}; {} repairs rewrote {} chunks",
        world.solver().backend().describe(),
        world.stats().repaired,
        world.stats().rewritten_by_repair
    );
    (tiles, ran)
}

fn write(world: &World) -> String {
    let mut out = format!("{HEADER}\n");
    for (coord, tiles) in world {
        write!(out, "{} {} {}:", coord.x, coord.y, coord.z).expect("writing to a string");
        for tile in tiles {
            write!(out, " {tile}").expect("writing to a string");
        }
        out.push('\n');
    }
    out
}

fn read(text: &str) -> World {
    text.lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let (coord, tiles) = line.split_once(':').expect("a chunk line has a colon");
            let coord: Vec<i32> = coord
                .split_whitespace()
                .map(|n| n.parse().expect("a chunk coordinate"))
                .collect();
            let tiles = tiles
                .split_whitespace()
                .map(|n| n.parse().expect("a tile"))
                .collect();
            (ChunkCoord::new(coord[0], coord[1], coord[2]), tiles)
        })
        .collect()
}

#[test]
fn a_city_is_the_same_on_every_device() {
    let (world, adapter) = generate();

    if std::env::var_os("WAVE_FORGE_BLESS").is_some() {
        std::fs::write(fixture(), write(&world)).expect("write the fixture");
        eprintln!("golden_world: recorded {} chunks on {adapter}", world.len());
        return;
    }
    let recorded = read(
        &std::fs::read_to_string(fixture())
            .expect("tests/fixtures/golden_city.txt; record it with WAVE_FORGE_BLESS=1"),
    );

    eprintln!(
        "golden_world: {} chunks generated on {adapter}, {} recorded",
        world.len(),
        recorded.len()
    );
    assert_eq!(
        world.keys().collect::<Vec<_>>(),
        recorded.keys().collect::<Vec<_>>(),
        "the same chunks"
    );
    let names = &city::city().modules.names;
    for (coord, tiles) in &world {
        let expected = &recorded[coord];
        if let Some(cell) = (0..tiles.len()).find(|&cell| tiles[cell] != expected[cell]) {
            let (x, y, z) = (
                cell as u32 % CHUNK.x,
                (cell as u32 / CHUNK.x) % CHUNK.y,
                cell as u32 / (CHUNK.x * CHUNK.y),
            );
            let differing = tiles
                .iter()
                .zip(expected)
                .filter(|(got, want)| got != want)
                .count();
            panic!(
                "on {adapter}, chunk {coord:?} cell ({x}, {y}, {z}) holds {} where the recorded \
                 world has {}; {differing} of the chunk's {} cells differ",
                names[usize::from(tiles[cell])],
                names[usize::from(expected[cell])],
                tiles.len()
            );
        }
    }
}
