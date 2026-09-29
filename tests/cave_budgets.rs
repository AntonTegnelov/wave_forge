//! Budgets in a cave level: a Deposit stage places exactly its total in the rock around the
//! rooms, and a Spawn stage spends each room's budget on the room's floor, both the same whatever
//! order chunks come in.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::{Pack, PackError, Point, Runtime, StageError, Stamp};
use wave_forge::{ChunkCoord, FocusPoint};

const SIZE: [u32; 2] = [8, 8];

const PACK: &str = r#"(
    version: 1,
    noises: {"wander": (noise_type: SimplexSmooth, seed: 7, frequency: 0.08, fractal_octaves: 2)},
    stages: [
        (name: "rock", kind: Volume(density: Sub(Constant(30.0), Z), bottom: -40, top: 8)),
        (name: "level", kind: Cave(region: 8, depth: (-30.0, -10.0), patterns: [Linear, Star, Hub],
            count: (5, 7), apart: 12.0, rerolls: 8, rooms: [
                (name: "cavern", size: (7, 7, 5), weight: 2),
                (name: "hall", size: (5, 9, 4)),
            ])),
        (name: "tunnels", kind: Tunnels(cave: "level", noise: "wander", radius: (1.5, 2.5))),
        (name: "caves", kind: Carve(volume: "rock", tunnels: Some((curves: "tunnels", max_radius: 3)),
            rooms: Some("level"))),
        (name: "gold", kind: Deposit(kind: "gold", cave: "level", volume: "caves", total: 40, depth: 2, apart: 2.0)),
        (name: "enemies", kind: Spawn(cave: "level", budget: 10, kinds: [
            (kind: "grunt", cost: 1, weight: 4),
            (kind: "guard", cost: 3, weight: 2),
            (kind: "brute", cost: 7),
        ])),
    ],
)"#;

const COSTS: [(&str, u32); 3] = [("grunt", 1), ("guard", 3), ("brute", 7)];

/// The region at the origin: 8 by 8 chunks.
fn region() -> Vec<ChunkCoord> {
    (0..8)
        .flat_map(|y| (0..8).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

fn runtime(text: &str, seed: u64) -> Runtime {
    Runtime::new(
        Arc::new(Pack::parse(text).expect("a valid pack")),
        seed,
        SIZE,
    )
}

fn generate(runtime: &mut Runtime, stages: &[&str], chunks: &[ChunkCoord]) {
    let focus: Vec<FocusPoint> = chunks.iter().map(|&c| FocusPoint::new(c, 0)).collect();
    runtime.request(&focus, stages).expect("the stages");
    runtime.run_until_idle().expect("the stages run");
}

/// Every point of `stage` over the region, each once.
fn points(runtime: &Runtime, stage: &str) -> Vec<Point> {
    let mut all = BTreeMap::new();
    for chunk in region() {
        for point in runtime.points(stage, chunk).expect("generated") {
            assert!(
                all.insert(point.id, point.clone()).is_none(),
                "{point:?} twice"
            );
        }
    }
    all.into_values().collect()
}

/// Every room of the level, each once.
fn rooms(runtime: &Runtime) -> Vec<Stamp> {
    let mut all = BTreeMap::new();
    for chunk in region() {
        for stamp in runtime.stamps("level", chunk).expect("generated") {
            all.insert(stamp.id, stamp.clone());
        }
    }
    all.into_values().collect()
}

/// A room's box, from its lowest corner to its highest.
fn bounds(room: &Stamp) -> ([f32; 3], [f32; 3]) {
    let height = if &*room.piece == "cavern" { 5.0 } else { 4.0 };
    let floor = room.position[2];
    (
        [room.min[0] as f32, room.min[1] as f32, floor],
        [room.max[0] as f32, room.max[1] as f32, floor + height],
    )
}

/// How far `at` lies outside a box, 0 inside it.
fn outside(at: [f32; 3], (low, high): ([f32; 3], [f32; 3])) -> f32 {
    (0..3)
        .map(|axis| {
            (low[axis] - at[axis])
                .max(at[axis] - high[axis])
                .max(0.0)
                .powi(2)
        })
        .sum::<f32>()
        .sqrt()
}

#[test]
fn a_level_holds_exactly_its_deposits_in_solid_rock_around_its_rooms() {
    for seed in [1, 2, 3] {
        let mut runtime = runtime(PACK, seed);
        generate(&mut runtime, &["gold", "caves", "level"], &region());

        let gold = points(&runtime, "gold");
        let rooms = rooms(&runtime);

        assert_eq!(gold.len(), 40, "seed {seed}");
        for (index, point) in gold.iter().enumerate() {
            let [x, y, z] = point.position;
            let chunk = ChunkCoord::new((x / 8.0).floor() as i32, (y / 8.0).floor() as i32, 0);
            let caves = runtime.volume("caves", chunk).expect("generated");
            let level = (z.floor() as i32 - caves.bottom) as u32;
            let value = caves.get(x.rem_euclid(8.0) as u32, y.rem_euclid(8.0) as u32, level);
            assert!(value > 0.0, "{point:?} in empty rock");
            let nearest = rooms
                .iter()
                .map(|room| outside(point.position, bounds(room)))
                .fold(f32::INFINITY, f32::min);
            assert!(
                nearest > 0.0 && nearest < 2.0,
                "{point:?} {nearest} from a room"
            );
            for other in &gold[..index] {
                let apart = (0..3)
                    .map(|axis| (other.position[axis] - point.position[axis]).powi(2))
                    .sum::<f32>()
                    .sqrt();
                assert!(apart >= 2.0, "{point:?} near {other:?}");
            }
        }
    }
}

#[test]
fn each_room_spends_its_budget_on_its_floor_and_no_more() {
    let mut runtime = runtime(PACK, 4);
    generate(&mut runtime, &["enemies", "level"], &region());

    let enemies = points(&runtime, "enemies");
    let rooms = rooms(&runtime);

    let cost = |kind: &str| {
        COSTS
            .iter()
            .find(|(name, _)| *name == kind)
            .expect("a kind")
            .1
    };
    let mut spent = vec![0; rooms.len()];
    for enemy in &enemies {
        let room = rooms
            .iter()
            .position(|room| {
                let (low, high) = bounds(room);
                enemy.position[2] == low[2]
                    && (low[0]..high[0]).contains(&enemy.position[0])
                    && (low[1]..high[1]).contains(&enemy.position[1])
            })
            .expect("an enemy stands on a room's floor");
        spent[room] += cost(&enemy.kind);
    }
    // A room stops when it cannot afford even a grunt, so it spends its whole budget.
    assert_eq!(spent, vec![10; rooms.len()]);
    let kinds: std::collections::BTreeSet<&str> = enemies.iter().map(|e| &*e.kind).collect();
    assert!(kinds.len() >= 2, "{kinds:?}");
}

#[test]
fn a_room_short_of_its_cheapest_kind_keeps_what_is_left() {
    let text = PACK.replace("(kind: \"grunt\", cost: 1, weight: 4),", "");
    let mut runtime = runtime(&text, 4);
    generate(&mut runtime, &["enemies", "level"], &region());

    let enemies = points(&runtime, "enemies");
    let rooms = rooms(&runtime);

    let mut spent = vec![0; rooms.len()];
    for enemy in &enemies {
        let room = rooms
            .iter()
            .position(|room| {
                let (low, high) = bounds(room);
                (low[0]..high[0]).contains(&enemy.position[0])
                    && (low[1]..high[1]).contains(&enemy.position[1])
            })
            .expect("an enemy stands in a room");
        spent[room] += if &*enemy.kind == "guard" { 3 } else { 7 };
    }
    for spent in spent {
        assert!(spent <= 10 && 10 - spent < 3, "a room spent {spent}");
    }
}

#[test]
fn deposits_and_spawns_are_the_same_whatever_order_their_chunks_come_in() {
    let mut together = runtime(PACK, 5);
    let mut apart = runtime(PACK, 5);

    generate(&mut together, &["gold", "enemies"], &region());
    let mut one_by_one = BTreeMap::new();
    for chunk in region().into_iter().rev() {
        generate(&mut apart, &["gold", "enemies"], &[chunk]);
        one_by_one.insert(
            chunk,
            (
                apart.points("gold", chunk).map(<[Point]>::to_vec),
                apart.points("enemies", chunk).map(<[Point]>::to_vec),
            ),
        );
    }

    for chunk in region() {
        let here = (
            together.points("gold", chunk).map(<[Point]>::to_vec),
            together.points("enemies", chunk).map(<[Point]>::to_vec),
        );
        assert_eq!(Some(&here), one_by_one.get(&chunk), "{chunk:?}");
    }
}

#[test]
fn a_level_without_room_for_its_total_is_refused() {
    let text = PACK.replace("total: 40, depth: 2", "total: 4000, depth: 1, tries: 4000");
    let mut runtime = runtime(&text, 1);

    runtime
        .request(&[FocusPoint::new(ChunkCoord::new(0, 0, 0), 0)], &["gold"])
        .expect("a stage");
    let result = runtime.run_until_idle();

    assert!(
        matches!(&result, Err(StageError::RegionRejected { stage, .. }) if stage == "gold"),
        "{result:?}"
    );
}

#[test]
fn deposits_and_spawns_out_of_range_are_refused() {
    let parse = |from: &str, to: &str| {
        assert_eq!(PACK.matches(from).count(), 1, "{from}");
        Pack::parse(&PACK.replace(from, to))
    };

    let results = [
        (parse("total: 40,", "total: 0,"), "gold"),
        (parse("total: 40,", "total: 5000,"), "gold"),
        (parse("depth: 2,", "depth: 0,"), "gold"),
        (
            parse(
                r#"kind: "gold", cave: "level""#,
                r#"kind: "gold", cave: "rock""#,
            ),
            "gold",
        ),
        (parse("budget: 10", "budget: 1001"), "enemies"),
        (parse("cost: 7", "cost: 0"), "enemies"),
        (
            parse(r#"Spawn(cave: "level""#, r#"Spawn(cave: "tunnels""#),
            "enemies",
        ),
    ];

    for (result, name) in results {
        assert!(
            matches!(&result, Err(PackError::Invalid { stage, .. }) if stage == name),
            "{result:?}"
        );
    }
}
