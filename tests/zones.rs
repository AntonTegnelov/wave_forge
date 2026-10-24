//! A world of zones (docs/product/user-stories.md, G5): `tests/fixtures/zones.world.ron`, its map
//! of 4 by 3 zones painted here. Every open column reaches every other, within its zone and across
//! the world, the paths of neighbouring zones meet at their shared edge, every zone is cut into
//! segments, each zone's people are drawn from its kind's table by weight and stand on open ground,
//! the world is the same whatever order its chunks come in, and a zone frozen on first emit comes
//! back from its snapshot even under a changed pack.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex};
use wave_forge::stages::{Pack, Painting, Runtime};
use wave_forge::{ChunkCoord, FocusPoint, FrozenStore, StoreError};

const SIZE: [u32; 2] = [8, 8];
/// Columns along a zone's side: 4 chunks of 8.
const ZONE: i64 = 32;
const ZONES: [i64; 2] = [4, 3];
const JUNGLE: [u8; 4] = [40, 140, 40, 255];
const DESERT: [u8; 4] = [200, 180, 90, 255];
const RUINS: [u8; 4] = [120, 120, 130, 255];
const STAGES: [&str; 6] = [
    "zone",
    "segments",
    "open",
    "jungle_folk",
    "desert_folk",
    "ruins_folk",
];

type Column = (i64, i64);

/// The zones' kinds, a row of the map at a time from its first.
fn painting() -> Painting {
    Painting {
        width: 4,
        height: 3,
        pixels: vec![
            JUNGLE, DESERT, RUINS, JUNGLE, //
            RUINS, JUNGLE, DESERT, DESERT, //
            DESERT, RUINS, JUNGLE, RUINS,
        ],
    }
}

fn text() -> String {
    std::fs::read_to_string(format!(
        "{}/tests/fixtures/zones.world.ron",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("the zones' pack")
}

fn pack() -> Arc<Pack> {
    Arc::new(Pack::parse(&text()).expect("a valid pack"))
}

fn runtime() -> Runtime {
    runtime_of(pack())
}

fn runtime_of(pack: Arc<Pack>) -> Runtime {
    let mut runtime = Runtime::new(pack, 9, SIZE);
    runtime
        .give_painting("map", painting())
        .expect("a map its legend reads");
    runtime
}

/// Every chunk of the world, row by row.
fn world() -> Vec<ChunkCoord> {
    let [width, height] = ZONES.map(|zones| (zones * ZONE / i64::from(SIZE[0])) as i32);
    (0..height)
        .flat_map(|y| (0..width).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

/// The world generated whole.
fn generated() -> Runtime {
    let mut runtime = runtime();
    runtime.request_bound(&STAGES).expect("a bounded world");
    runtime.run_until_idle().expect("the world generates");
    runtime
}

fn column_of(chunk: ChunkCoord, x: u32, y: u32) -> Column {
    (
        i64::from(chunk.x) * i64::from(SIZE[0]) + i64::from(x),
        i64::from(chunk.y) * i64::from(SIZE[1]) + i64::from(y),
    )
}

/// A field's value at every column of the world.
fn field(runtime: &Runtime, stage: &str) -> BTreeMap<Column, f32> {
    let mut out = BTreeMap::new();
    for chunk in world() {
        let field = runtime.field(stage, chunk).expect("generated");
        for y in 0..SIZE[1] {
            for x in 0..SIZE[0] {
                out.insert(column_of(chunk, x, y), field.get(x, y));
            }
        }
    }
    out
}

fn zone_of((x, y): Column) -> Column {
    (x.div_euclid(ZONE), y.div_euclid(ZONE))
}

/// The zone kind of every zone, by its name.
fn zone_kinds(runtime: &Runtime) -> BTreeMap<Column, String> {
    let names = pack()
        .kind("zone")
        .expect("a stage")
        .categories()
        .iter()
        .map(|n| (*n).to_owned())
        .collect::<Vec<_>>();
    let mut out = BTreeMap::new();
    for chunk in world() {
        let categories = runtime.categories("zone", chunk).expect("generated");
        let column = column_of(chunk, 0, 0);
        out.insert(
            zone_of(column),
            names[usize::from(categories.get(0, 0))].clone(),
        );
    }
    out
}

/// The areas of `members`, columns joined edge to edge.
fn areas(members: &BTreeSet<Column>) -> Vec<BTreeSet<Column>> {
    let mut left = members.clone();
    let mut out = Vec::new();
    while let Some(start) = left.pop_first() {
        let mut area = BTreeSet::from([start]);
        let mut frontier = vec![start];
        while let Some((x, y)) = frontier.pop() {
            for next in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] {
                if left.remove(&next) {
                    area.insert(next);
                    frontier.push(next);
                }
            }
        }
        out.push(area);
    }
    out
}

fn open(runtime: &Runtime) -> BTreeSet<Column> {
    field(runtime, "open")
        .into_iter()
        .filter(|&(_, value)| value == 1.0)
        .map(|(column, _)| column)
        .collect()
}

fn zones() -> impl Iterator<Item = Column> {
    (0..ZONES[1]).flat_map(|y| (0..ZONES[0]).map(move |x| (x, y)))
}

#[test]
fn every_open_column_reaches_every_other_within_its_zone_and_across_the_world() {
    let runtime = generated();

    let open = open(&runtime);

    assert_eq!(areas(&open).len(), 1, "the world's open ground");
    for zone in zones() {
        let within: BTreeSet<Column> = open
            .iter()
            .copied()
            .filter(|&c| zone_of(c) == zone)
            .collect();
        assert!(
            within.len() > 100,
            "zone {zone:?} has {} open columns",
            within.len()
        );
        assert_eq!(areas(&within).len(), 1, "zone {zone:?}");
    }
}

#[test]
fn the_paths_of_neighbouring_zones_meet_at_their_shared_edge() {
    let runtime = generated();
    let open = open(&runtime);

    for (zx, zy) in zones() {
        if zx + 1 < ZONES[0] {
            let edge = (zx + 1) * ZONE;
            let meets = (zy * ZONE..(zy + 1) * ZONE)
                .any(|y| open.contains(&(edge - 1, y)) && open.contains(&(edge, y)));
            assert!(meets, "zones ({zx}, {zy}) and ({}, {zy})", zx + 1);
        }
        if zy + 1 < ZONES[1] {
            let edge = (zy + 1) * ZONE;
            let meets = (zx * ZONE..(zx + 1) * ZONE)
                .any(|x| open.contains(&(x, edge - 1)) && open.contains(&(x, edge)));
            assert!(meets, "zones ({zx}, {zy}) and ({zx}, {})", zy + 1);
        }
    }
}

#[test]
fn every_zone_is_cut_into_segments_each_one_area() {
    let runtime = generated();
    let segments = field(&runtime, "segments");

    for zone in zones() {
        let mut by_id: BTreeMap<u32, BTreeSet<Column>> = BTreeMap::new();
        for (&column, &id) in &segments {
            if zone_of(column) == zone {
                by_id.entry(id as u32).or_default().insert(column);
            }
        }
        assert!(
            by_id.len() >= 3,
            "zone {zone:?} has {} segments",
            by_id.len()
        );
        for (id, members) in by_id {
            assert_eq!(areas(&members).len(), 1, "zone {zone:?}, segment {id}");
        }
    }
}

#[test]
fn each_zones_people_come_from_its_kinds_table_by_weight_and_stand_on_open_ground() {
    let runtime = generated();
    let open = open(&runtime);
    let kinds = zone_kinds(&runtime);
    let tables = [
        ("jungle_folk", "jungle", "raider", "herder", 0.75),
        ("desert_folk", "desert", "nomad", "jackal", 0.5),
        ("ruins_folk", "ruins", "beetle", "drone", 0.8),
    ];

    for (stage, zone_kind, first, second, share) in tables {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for chunk in world() {
            for point in runtime.points(stage, chunk).expect("generated") {
                let column = (
                    point.position[0].floor() as i64,
                    point.position[1].floor() as i64,
                );
                assert_eq!(kinds[&zone_of(column)], zone_kind, "{stage} at {column:?}");
                assert!(open.contains(&column), "{stage} at {column:?} on rock");
                *counts.entry(point.kind.to_string()).or_default() += 1;
            }
        }
        let total = counts.values().sum::<usize>();
        let seen: BTreeSet<&str> = counts.keys().map(String::as_str).collect();
        assert_eq!(seen, BTreeSet::from([first, second]), "{stage}");
        let drawn = counts[first] as f64 / total as f64;
        assert!(
            total >= 30 && (drawn - share).abs() < 0.15,
            "{stage}: {counts:?}, {first} {drawn:.2} of {total} against {share}"
        );
    }
}

#[test]
fn the_world_is_the_same_whatever_order_its_chunks_come_in() {
    let whole = generated();
    let mut backwards = runtime();
    let mut chunks = world();
    chunks.reverse();

    for chunk in &chunks {
        backwards
            .request(&[FocusPoint::new(*chunk, 0)], &STAGES)
            .expect("the stages");
        backwards.run_until_idle().expect("the world generates");
        assert_eq!(
            backwards
                .chunk_hash(*chunk, &STAGES)
                .expect("every stage there"),
            whole
                .chunk_hash(*chunk, &STAGES)
                .expect("every stage there"),
            "{chunk:?}"
        );
    }
}

/// The bytes kept, by layer and chunk.
type Kept = HashMap<(String, ChunkCoord), Vec<u8>>;

/// A store in memory, shared by every runtime given a clone of it, as a directory on disk is.
#[derive(Clone, Default)]
struct SharedStore(Arc<Mutex<Kept>>);

impl FrozenStore for SharedStore {
    fn keep(&mut self, layer: &str, chunk: ChunkCoord, bytes: Vec<u8>) -> Result<(), StoreError> {
        self.0
            .lock()
            .expect("not poisoned")
            .insert((layer.to_owned(), chunk), bytes);
        Ok(())
    }

    fn fetch(&mut self, layer: &str, chunk: ChunkCoord) -> Result<Option<Vec<u8>>, StoreError> {
        Ok(self
            .0
            .lock()
            .expect("not poisoned")
            .get(&(layer.to_owned(), chunk))
            .cloned())
    }
}

/// The chunks of zone `(x, y)`.
fn zone_chunks((x, y): Column) -> Vec<ChunkCoord> {
    let side = (ZONE / i64::from(SIZE[0])) as i32;
    let (x, y) = (x as i32 * side, y as i32 * side);
    (y..y + side)
        .flat_map(|cy| (x..x + side).map(move |cx| ChunkCoord::new(cx, cy, 0)))
        .collect()
}

/// Each chunk's hash of `stage` over `chunks`, once they are asked for and generated.
fn visit(runtime: &mut Runtime, chunks: &[ChunkCoord], stage: &str) -> Vec<Option<u64>> {
    let focus: Vec<FocusPoint> = chunks.iter().map(|&c| FocusPoint::new(c, 0)).collect();
    runtime.request(&focus, &STAGES).expect("the stages");
    runtime.run_until_idle().expect("the zone generates");
    chunks
        .iter()
        .map(|&chunk| runtime.product_hash(stage, chunk))
        .collect()
}

#[test]
fn a_zone_frozen_on_first_emit_comes_back_from_its_snapshot_under_a_changed_pack() {
    let store = SharedStore::default();
    let (home, away) = (zone_chunks((0, 0)), zone_chunks((3, 2)));
    let mut first = runtime().with_store(Box::new(store.clone()));
    let open = visit(&mut first, &home, "open");
    let folk = visit(&mut first, &home, "jungle_folk");
    let rock = visit(&mut first, &home, "rock");
    // Walking away drops the zone, and its frozen chunks go to the store.
    visit(&mut first, &away, "open");
    let changed = text().replace(
        "Noise(frequency: 0.12, octaves: 2)",
        "Noise(frequency: 0.2, octaves: 2)",
    );
    let mut second = runtime_of(Arc::new(Pack::parse(&changed).expect("a valid pack")))
        .with_store(Box::new(store));

    let back = visit(&mut second, &home, "open");

    assert!(
        open.iter().chain(&folk).all(Option::is_some),
        "the zone was generated"
    );
    assert_eq!(back, open, "the open ground as first emitted");
    assert_eq!(
        visit(&mut second, &home, "jungle_folk"),
        folk,
        "its people as first emitted"
    );
    assert_ne!(
        visit(&mut second, &home, "rock"),
        rock,
        "the changed pack's rock"
    );
}
