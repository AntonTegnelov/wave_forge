//! G8's check (docs/product/user-stories.md): a finite cave level from
//! `examples/cave_level.world.ron`. The level is one region planned whole: a room graph from
//! authored patterns, rooms as stamps, tunnels as curves, carved into a density volume. Its
//! crystal meets the level's quota exactly and its enemies keep to each room's budget, and a dig
//! goes through the edits log and survives a save and a reload.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use wave_forge::stages::regions::Curve;
use wave_forge::stages::{Edit, Edits, Pack, Point, Runtime, SiteId, Stamp, Volume};
use wave_forge::{ChunkCoord, FocusPoint};

const SIZE: [u32; 2] = [16, 16];
const STAGES: [&str; 5] = ["level", "tunnels", "terrain", "crystal", "enemies"];

fn pack() -> Arc<Pack> {
    let text = std::fs::read_to_string(format!(
        "{}/examples/cave_level.world.ron",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("the cave level pack");
    Arc::new(Pack::parse(&text).expect("a valid pack"))
}

/// The level's chunks: the bound's 4 by 4.
fn chunks() -> Vec<ChunkCoord> {
    (0..4)
        .flat_map(|y| (0..4).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

/// A runtime with the whole level generated.
fn level(seed: u64) -> Runtime {
    let mut runtime = Runtime::new(pack(), seed, SIZE);
    runtime.request_bound(&STAGES).expect("a bound");
    runtime.run_until_idle().expect("the stages run");
    runtime
}

/// Every room, tunnel, crystal and enemy of the level, each once.
fn contents(runtime: &Runtime) -> (Vec<Stamp>, Vec<Curve>, Vec<Point>, Vec<Point>) {
    let (mut rooms, mut tunnels, mut crystal, mut enemies) = (
        BTreeMap::new(),
        BTreeMap::new(),
        BTreeMap::new(),
        BTreeMap::new(),
    );
    for chunk in chunks() {
        for stamp in runtime.stamps("level", chunk).expect("generated") {
            rooms.insert(stamp.id, stamp.clone());
        }
        for curve in runtime.curves("tunnels", chunk).expect("generated") {
            tunnels.insert(curve.id.clone(), curve.clone());
        }
        for point in runtime.points("crystal", chunk).expect("generated") {
            crystal.insert(point.id, point.clone());
        }
        for point in runtime.points("enemies", chunk).expect("generated") {
            enemies.insert(point.id, point.clone());
        }
    }
    (
        rooms.into_values().collect(),
        tunnels.into_values().collect(),
        crystal.into_values().collect(),
        enemies.into_values().collect(),
    )
}

fn height(room: &Stamp) -> f32 {
    match &*room.piece {
        "cavern" => 6.0,
        "hall" => 4.0,
        _ => 10.0,
    }
}

fn centre(room: &Stamp) -> [f32; 3] {
    [
        room.position[0],
        room.position[1],
        room.position[2] + height(room) / 2.0,
    ]
}

/// The value of `volume`, one of the level's, at the voxel holding `at`.
fn value_at(runtime: &Runtime, stage: &str, at: [f32; 3]) -> f32 {
    let chunk = ChunkCoord::new(
        (at[0] / 16.0).floor() as i32,
        (at[1] / 16.0).floor() as i32,
        0,
    );
    let volume: &Volume = runtime.volume(stage, chunk).expect("generated");
    volume.get(
        at[0].rem_euclid(16.0) as u32,
        at[1].rem_euclid(16.0) as u32,
        (at[2].floor() as i32 - volume.bottom) as u32,
    )
}

#[test]
fn the_level_is_one_region_whose_rooms_the_tunnels_join_into_one_graph() {
    for seed in 0..4 {
        let runtime = level(seed);
        let (rooms, tunnels, _, _) = contents(&runtime);

        assert!((5..=8).contains(&rooms.len()), "{} rooms", rooms.len());
        assert!(rooms.iter().all(|room| room.site == SiteId::Region(0, 0)));
        assert_eq!(tunnels.len(), rooms.len() - 1);
        // Joined through the tunnels, every room reaches every other.
        let at_room = |x: f32, y: f32, z: f32| {
            rooms
                .iter()
                .position(|room| centre(room) == [x, y, z])
                .expect("a tunnel ends at a room's centre")
        };
        let mut group: Vec<usize> = (0..rooms.len()).collect();
        for tunnel in &tunnels {
            let last = tunnel.points.len() - 1;
            let a = at_room(tunnel.points[0][0], tunnel.points[0][1], tunnel.heights[0]);
            let b = at_room(
                tunnel.points[last][0],
                tunnel.points[last][1],
                tunnel.heights[last],
            );
            let (from, to) = (group[a], group[b]);
            group
                .iter_mut()
                .filter(|g| **g == from)
                .for_each(|g| *g = to);
        }
        assert_eq!(
            group.iter().collect::<BTreeSet<_>>().len(),
            1,
            "seed {seed}"
        );
    }
}

#[test]
fn rooms_and_tunnels_are_carved_into_the_rock() {
    let runtime = level(1);
    let (rooms, tunnels, _, _) = contents(&runtime);

    for room in &rooms {
        assert!(
            value_at(&runtime, "terrain", centre(room)) < 0.0,
            "{room:?}"
        );
    }
    let mut along = 0;
    for tunnel in &tunnels {
        for (point, height) in tunnel.points.iter().zip(&tunnel.heights) {
            assert!(value_at(&runtime, "terrain", [point[0], point[1], *height]) < 0.0);
            along += 1;
        }
    }
    assert!(along > 20, "only {along} points along the tunnels");
    // The rock just under the surface is untouched.
    for chunk in chunks() {
        let terrain = runtime.volume("terrain", chunk).expect("generated");
        let top = terrain.size[2] - 1;
        for y in 0..16 {
            for x in 0..16 {
                assert!(terrain.get(x, y, top) > 0.0, "{chunk:?} {x} {y}");
            }
        }
    }
}

#[test]
fn the_crystal_meets_the_quota_and_every_room_keeps_to_its_budget() {
    for seed in 0..4 {
        let runtime = level(seed);
        let (rooms, _, crystal, enemies) = contents(&runtime);

        assert_eq!(crystal.len(), 30, "seed {seed}");
        for point in &crystal {
            assert!(
                value_at(&runtime, "caves", point.position) > 0.0,
                "{point:?}"
            );
        }
        let cost = |kind: &str| match kind {
            "crawler" => 1,
            "spitter" => 3,
            "brute" => 8,
            other => panic!("a {other}"),
        };
        let mut spent = vec![0; rooms.len()];
        for enemy in &enemies {
            let room = rooms
                .iter()
                .position(|room| {
                    enemy.position[2] == room.position[2]
                        && (room.min[0] as f32..room.max[0] as f32).contains(&enemy.position[0])
                        && (room.min[1] as f32..room.max[1] as f32).contains(&enemy.position[1])
                })
                .expect("an enemy on a room's floor");
            spent[room] += cost(&enemy.kind);
        }
        assert_eq!(spent, vec![12; rooms.len()], "seed {seed}");
    }
}

#[test]
fn the_level_is_the_same_whatever_order_its_chunks_come_in() {
    let at_once = level(2);
    let mut walked = Runtime::new(pack(), 2, SIZE);

    for chunk in chunks().into_iter().rev() {
        walked
            .request(&[FocusPoint::new(chunk, 0)], &STAGES)
            .expect("the stages");
        walked.run_until_idle().expect("the stages run");
        for stage in STAGES {
            assert_eq!(
                walked.product(stage, chunk),
                at_once.product(stage, chunk),
                "{stage} at {chunk:?}"
            );
        }
    }
}

#[test]
fn a_dig_goes_through_the_edits_log_and_survives_a_save_and_a_reload() {
    let mut runtime = level(3);
    let (rooms, _, crystal, _) = contents(&runtime);
    // Into the wall of the first room, from its centre outward.
    let wall = centre(&rooms[0]);
    let at = [rooms[0].min[0] as f32 - 1.0, wall[1], wall[2]];
    assert!(value_at(&runtime, "terrain", at) > 0.0, "the wall is rock");
    let log = vec![Edit::Dig {
        stage: "terrain".to_owned(),
        at,
        radius: 2.0,
    }];

    runtime
        .set_edits(&Edits { log: log.clone() })
        .expect("a dig of a volume");
    runtime.request_bound(&STAGES).expect("a bound");
    runtime.run_until_idle().expect("the stages run");
    let save = runtime.save();
    let mut reloaded = Runtime::new(pack(), 3, SIZE);
    reloaded.load(&save).expect("a save of this pack");
    reloaded.request_bound(&STAGES).expect("a bound");
    reloaded.run_until_idle().expect("the stages run");

    assert_eq!(save.edits.log, log);
    assert!(value_at(&runtime, "terrain", at) < 0.0, "the dig is empty");
    for chunk in chunks() {
        assert_eq!(
            reloaded.volume("terrain", chunk),
            runtime.volume("terrain", chunk),
            "{chunk:?}"
        );
    }
    assert_eq!(contents(&reloaded).2, crystal, "a dig moves no crystal");
}
