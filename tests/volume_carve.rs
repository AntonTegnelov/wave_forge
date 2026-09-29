//! Carve stages: tunnels and rooms carved out of a volume, empty inside them and unchanged
//! elsewhere, the same whatever order chunks are asked for in.
//!
//! The volume is rock below a height field. A tunnel from a table's row runs six cells under the
//! ground, and a dungeon's rooms stand twelve cells under their entrance.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::regions::{Attempt, Curve, CurveId, RegionInput, RegionJob};
use wave_forge::stages::{Facts, GivenRow, Pack, PackError, Runtime, StageError, Value, Volume};
use wave_forge::{ChunkCoord, FocusPoint};

const SIZE: [u32; 2] = [8, 8];

const PACK: &str = r#"(
    version: 1,
    tables: [(name: "tunnels", kind: Given(columns: [
        ("x0", Number), ("y0", Number), ("x1", Number), ("y1", Number), ("width", Number),
    ]))],
    stages: [
        (name: "ground", kind: Field(Add(Mul(Noise(frequency: 0.03, octaves: 2), Constant(12.0)), Constant(20.0)))),
        (name: "rock", kind: Volume(
            density: Sub(Input("ground"), Z),
            bottom: -8,
            top: 40,
            materials: Some((rules: [(category: "soil", when: [Greater(Z, Sub(Input("ground"), Constant(2.0)))])], otherwise: "stone")),
        )),
        (name: "tunnels", kind: TableCurves(table: "tunnels", from: ("x0", "y0"), to: ("x1", "y1"), radius: "width")),
        (name: "entrances", kind: Locations(height: "ground", region: 6, kinds: [
            (name: "crypt", priority: 1, quota: 1, size: 3),
        ])),
        (name: "dungeon", kind: Assemble(sites: "entrances", kinds: ["crypt"], start: "entry", lift: -12.0,
            max: 12, min: 4, rerolls: 8, pieces: [
            (name: "entry", size: (3, 3, 2), doors: [(at: (1, 2, 0), facing: North, kind: "hall")]),
            (name: "room", size: (5, 5, 3), weight: 2, doors: [
                (at: (2, 0, 0), facing: South, kind: "hall"),
                (at: (2, 4, 0), facing: North, kind: "hall"),
                (at: (4, 2, 0), facing: East, kind: "hall"),
                (at: (0, 2, 0), facing: West, kind: "hall"),
            ]),
            (name: "corridor", size: (1, 4, 2), weight: 3, doors: [
                (at: (0, 0, 0), facing: South, kind: "hall"),
                (at: (0, 3, 0), facing: North, kind: "hall"),
            ]),
            (name: "cap", size: (1, 1, 2), end: true, doors: [(at: (0, 0, 0), facing: South, kind: "hall")]),
        ])),
        (name: "outposts", kind: Sites(height: "ground", region: 4, size: (1, 2), chance: 1.0)),
        (name: "levelled", kind: Carve(volume: "rock", level: Some((sites: "outposts", depth: 3, clear: 5)))),
        (name: "caves", kind: Carve(
            volume: "rock",
            tunnels: Some((curves: "tunnels", height: Some("ground"), depth: 6.0, max_radius: 3)),
            rooms: Some("dungeon"),
        )),
    ],
)"#;

const TUNNEL: ((f32, f32), (f32, f32), f32) = ((4.0, 5.0), (60.0, 30.0), 2.0);

fn tunnel(width: f32) -> GivenRow {
    let ((x0, y0), (x1, y1), _) = TUNNEL;
    GivenRow {
        id: 1,
        values: BTreeMap::from([
            ("x0".to_owned(), Value::Number(x0)),
            ("y0".to_owned(), Value::Number(y0)),
            ("x1".to_owned(), Value::Number(x1)),
            ("y1".to_owned(), Value::Number(y1)),
            ("width".to_owned(), Value::Number(width)),
        ]),
    }
}

fn runtime(width: f32) -> Result<Runtime, StageError> {
    let pack = Arc::new(Pack::parse(PACK).expect("a valid pack"));
    let mut facts = Facts::new(Arc::clone(&pack), 9)?;
    facts.give("tunnels", vec![tunnel(width)])?;
    let mut runtime = Runtime::new(pack, 9, SIZE);
    runtime.set_facts(facts)?;
    Ok(runtime)
}

fn area() -> Vec<ChunkCoord> {
    (-6..8)
        .flat_map(|y| (-6..8).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

/// `stages` generated over `chunks`.
fn generate(runtime: &mut Runtime, stages: &[&str], chunks: &[ChunkCoord]) {
    let focus: Vec<FocusPoint> = chunks.iter().map(|&c| FocusPoint::new(c, 0)).collect();
    runtime.request(&focus, stages).expect("the stages");
    runtime.run_until_idle().expect("the stages run");
}

/// Every voxel of `volume` with its column and its height in cells.
fn voxels(volume: &Volume) -> impl Iterator<Item = ([i64; 2], f32, usize)> + '_ {
    let [sx, sy, levels] = volume.size;
    (0..levels).flat_map(move |level| {
        (0..sy).flat_map(move |y| {
            (0..sx).map(move |x| {
                let column = [
                    i64::from(volume.chunk.x) * i64::from(sx) + i64::from(x),
                    i64::from(volume.chunk.y) * i64::from(sy) + i64::from(y),
                ];
                let z = (volume.bottom + level as i32) as f32 + 0.5;
                (column, z, ((level * sy + y) * sx + x) as usize)
            })
        })
    })
}

/// How far a column's centre is from the tunnel's centre line on the ground plane.
fn from_tunnel(column: [i64; 2]) -> f32 {
    let ((x0, y0), (x1, y1), _) = TUNNEL;
    let at = [column[0] as f32 + 0.5, column[1] as f32 + 0.5];
    let along = [x1 - x0, y1 - y0];
    let t = (((at[0] - x0) * along[0] + (at[1] - y0) * along[1])
        / (along[0] * along[0] + along[1] * along[1]))
        .clamp(0.0, 1.0);
    ((at[0] - x0 - along[0] * t).powi(2) + (at[1] - y0 - along[1] * t).powi(2)).sqrt()
}

#[test]
fn a_tunnel_is_empty_along_its_centre_six_cells_under_the_ground() {
    let mut runtime = runtime(2.0).expect("a runtime");
    let chunks = area();
    generate(&mut runtime, &["caves", "ground"], &chunks);

    let mut centre_voxels = 0;
    for &chunk in &chunks {
        let caves = runtime.volume("caves", chunk).expect("generated");
        let ground = runtime.field("ground", chunk).expect("generated");
        for (column, z, i) in voxels(caves) {
            let local = [
                column[0].rem_euclid(8) as u32,
                column[1].rem_euclid(8) as u32,
            ];
            let centre = ground.get(local[0], local[1]) - 6.0;
            if from_tunnel(column) < 0.25 && (z - centre).abs() <= 0.5 {
                assert!(caves.values[i] < 0.0, "voxel of {column:?} at {z}");
                centre_voxels += 1;
            }
        }
    }
    assert!(
        centre_voxels > 20,
        "only {centre_voxels} voxels on the centre line"
    );
}

#[test]
fn a_room_is_empty_inside_its_box_and_the_rock_is_unchanged_far_from_tunnels_and_rooms() {
    let mut runtime = runtime(2.0).expect("a runtime");
    let chunks = area();
    // The dungeon a chunk beyond the area too, whose rooms can reach into it.
    let around: Vec<ChunkCoord> = (-7..9)
        .flat_map(|y| (-7..9).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect();
    generate(&mut runtime, &["caves", "dungeon"], &around);

    let pieces: Vec<([i64; 2], [i64; 2], f32, f32)> = around
        .iter()
        .flat_map(|&chunk| runtime.stamps("dungeon", chunk).expect("generated"))
        .map(|stamp| {
            let height = match &*stamp.piece {
                "room" => 3.0,
                _ => 2.0,
            };
            (stamp.min, stamp.max, stamp.position[2], height)
        })
        .collect();
    assert!(pieces.len() >= 4, "{} pieces", pieces.len());
    let (mut inside, mut untouched) = (0, 0);
    for &chunk in &chunks {
        let caves = runtime.volume("caves", chunk).expect("generated");
        let rock = runtime.volume("rock", chunk).expect("generated");
        for (column, z, i) in voxels(caves) {
            let in_room = |margin: f32| {
                pieces.iter().any(|&(min, max, floor, height)| {
                    let at = [column[0] as f32 + 0.5, column[1] as f32 + 0.5];
                    (min[0] as f32 - margin..max[0] as f32 + margin).contains(&at[0])
                        && (min[1] as f32 - margin..max[1] as f32 + margin).contains(&at[1])
                        && (floor - margin..floor + height + margin).contains(&z)
                })
            };
            if in_room(0.0) {
                assert!(
                    caves.values[i] < 0.0,
                    "voxel of {column:?} at {z} in a room"
                );
                inside += 1;
            } else if !in_room(2.0) && from_tunnel(column) > 3.0 + 2.0 {
                assert_eq!(caves.values[i], rock.values[i], "{column:?} at {z}");
                untouched += 1;
            }
        }
        assert_eq!(
            caves.materials, rock.materials,
            "a carve keeps its volume's materials"
        );
    }
    assert!(inside > 50, "only {inside} voxels inside rooms");
    assert!(untouched > 1000, "only {untouched} voxels far from them");
}

#[test]
fn a_carve_is_the_same_whatever_order_its_chunks_are_asked_for_in() {
    let chunks = area();
    let mut together = runtime(2.0).expect("a runtime");
    let mut apart = runtime(2.0).expect("a runtime");

    generate(&mut together, &["caves"], &chunks);
    let mut one_by_one = BTreeMap::new();
    for &chunk in chunks.iter().rev() {
        generate(&mut apart, &["caves"], &[chunk]);
        one_by_one.insert(
            chunk,
            apart.volume("caves", chunk).expect("generated").clone(),
        );
    }

    for &chunk in &chunks {
        assert_eq!(
            together.volume("caves", chunk),
            one_by_one.get(&chunk),
            "{chunk:?}"
        );
    }
}

#[test]
fn a_tunnel_wider_than_its_stage_allows_is_refused() {
    let mut runtime = runtime(3.5).expect("a runtime");

    runtime
        .request(&[FocusPoint::new(ChunkCoord::new(0, 0, 0), 0)], &["caves"])
        .expect("a stage");
    let result = runtime.run_until_idle();

    assert!(
        matches!(&result, Err(StageError::Curve { stage, .. }) if stage == "caves"),
        "{result:?}"
    );
}

#[test]
fn carving_a_volume_of_another_scale_is_refused() {
    let result = Pack::parse(
        r#"(version: 1, stages: [
            (name: "coarse", scale: 2, kind: Volume(density: Z, bottom: 0, top: 4)),
            (name: "carved", kind: Carve(volume: "coarse")),
        ])"#,
    );

    assert!(
        matches!(&result, Err(PackError::Invalid { stage, .. }) if stage == "carved"),
        "{result:?}"
    );
}

#[test]
fn a_levelled_site_is_solid_under_its_floor_and_open_above_it() {
    let mut runtime = runtime(2.0).expect("a runtime");
    let chunks = area();
    generate(&mut runtime, &["levelled", "outposts"], &chunks);

    let sites: Vec<_> = chunks
        .iter()
        .flat_map(|&chunk| {
            runtime
                .sites("outposts", chunk)
                .expect("generated")
                .to_vec()
        })
        .collect();
    let (mut under, mut over) = (0, 0);
    for &chunk in &chunks {
        let levelled = runtime.volume("levelled", chunk).expect("generated");
        for (column, z, i) in voxels(levelled) {
            let floor = sites.iter().find_map(|site| {
                let inside = (i64::from(site.min.0) * 8..i64::from(site.max.0) * 8)
                    .contains(&column[0])
                    && (i64::from(site.min.1) * 8..i64::from(site.max.1) * 8).contains(&column[1]);
                inside.then_some(site.height)
            });
            let Some(floor) = floor else { continue };
            if (floor - 3.0 < z) && (z < floor) {
                assert!(
                    levelled.values[i] > 0.0,
                    "{column:?} at {z} under a floor at {floor}"
                );
                under += 1;
            } else if (floor < z) && (z < floor + 5.0) {
                assert!(
                    levelled.values[i] < 0.0,
                    "{column:?} at {z} over a floor at {floor}"
                );
                over += 1;
            }
        }
    }
    assert!(
        under > 100 && over > 100,
        "{under} under and {over} over the floors"
    );
}

/// A worm tunnel through 3D space: from (4, 4) at height 2 down to (40, 20) at height -4.
const WORM: [[f32; 3]; 2] = [[4.0, 4.0, 2.0], [40.0, 20.0, -4.0]];

/// Gives one curve with heights, [`WORM`], in the region at the origin.
struct Worm;

impl RegionJob for Worm {
    fn run(&self, input: &RegionInput<'_>) -> Result<Attempt, StageError> {
        if input.region() != (0, 0) {
            return Ok(Attempt::Accepted(Vec::new()));
        }
        Ok(Attempt::Accepted(vec![Curve {
            id: CurveId::Region {
                region: (0, 0),
                index: 0,
            },
            points: WORM.iter().map(|p| [p[0], p[1]]).collect(),
            values: vec![1.5, 1.5],
            heights: WORM.iter().map(|p| p[2]).collect(),
        }]))
    }
}

/// How far `at` is from the worm's centre line in 3D.
fn from_worm(at: [f32; 3]) -> f32 {
    let [a, b] = WORM;
    let along: [f32; 3] = std::array::from_fn(|i| b[i] - a[i]);
    let length: f32 = along.iter().map(|d| d * d).sum();
    let t = ((0..3).map(|i| (at[i] - a[i]) * along[i]).sum::<f32>() / length).clamp(0.0, 1.0);
    (0..3)
        .map(|i| (at[i] - a[i] - along[i] * t).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[test]
fn a_curve_with_heights_carves_a_tube_along_them() {
    let pack = r#"(version: 1, stages: [
        (name: "rock", kind: Volume(density: Sub(Constant(20.0), Z), bottom: -12, top: 12)),
        (name: "worm", kind: Region(job: "worm", region: 8, inputs: [])),
        (name: "caves", kind: Carve(volume: "rock", tunnels: Some((curves: "worm", max_radius: 2)))),
    ])"#;
    let mut runtime = Runtime::new(Arc::new(Pack::parse(pack).expect("a valid pack")), 3, SIZE)
        .with_region_job("worm", Worm);
    let chunks: Vec<ChunkCoord> = (-1..4)
        .flat_map(|y| (-1..7).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect();

    generate(&mut runtime, &["caves"], &chunks);

    let (mut inside, mut outside) = (0, 0);
    for &chunk in &chunks {
        let caves = runtime.volume("caves", chunk).expect("generated");
        for (column, z, i) in voxels(caves) {
            let away = from_worm([column[0] as f32 + 0.5, column[1] as f32 + 0.5, z]);
            if away < 1.0 {
                assert!(caves.values[i] < 0.0, "{column:?} at {z} in the tube");
                inside += 1;
            } else if away > 1.5 + 1.0 {
                assert!(caves.values[i] > 0.0, "{column:?} at {z} beyond the tube");
                outside += 1;
            }
        }
    }
    assert!(
        inside > 50 && outside > 1000,
        "{inside} inside and {outside} outside"
    );
}

#[test]
fn tunnels_along_curves_on_the_ground_plane_without_a_height_field_are_refused() {
    let result = Pack::parse(
        r#"(version: 1,
            tables: [(name: "tunnels", kind: Given(columns: [
                ("x0", Number), ("y0", Number), ("x1", Number), ("y1", Number), ("width", Number),
            ]))],
            stages: [
                (name: "rock", kind: Volume(density: Sub(Constant(20.0), Z), bottom: 0, top: 4)),
                (name: "tunnels", kind: TableCurves(table: "tunnels", from: ("x0", "y0"), to: ("x1", "y1"), radius: "width")),
                (name: "caves", kind: Carve(volume: "rock", tunnels: Some((curves: "tunnels", max_radius: 2)))),
            ])"#,
    );

    assert!(
        matches!(&result, Err(PackError::Invalid { stage, .. }) if stage == "caves"),
        "{result:?}"
    );
}
