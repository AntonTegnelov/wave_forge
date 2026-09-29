//! Cave levels: rooms planned once per region in a pattern, joined by tunnels found through 3D
//! noise, and carved into rock, the same whatever order chunks are asked for in.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::regions::Curve;
use wave_forge::stages::{Pack, PackError, Runtime, StageError, Stamp};
use wave_forge::{ChunkCoord, FocusPoint};

const SIZE: [u32; 2] = [8, 8];

/// A pack whose Cave stage draws from `patterns`.
fn pack(patterns: &str) -> String {
    format!(
        r#"(
        version: 1,
        noises: {{"wander": (noise_type: SimplexSmooth, seed: 7, frequency: 0.08, fractal_octaves: 2)}},
        stages: [
            (name: "rock", kind: Volume(density: Sub(Constant(30.0), Z), bottom: -40, top: 8)),
            (name: "level", kind: Cave(region: 8, depth: (-30.0, -10.0), patterns: [{patterns}],
                count: (5, 7), apart: 12.0, rerolls: 8, rooms: [
                    (name: "cavern", size: (7, 7, 5), weight: 2),
                    (name: "hall", size: (5, 9, 4)),
                ])),
            (name: "tunnels", kind: Tunnels(cave: "level", noise: "wander", radius: (1.5, 2.5))),
            (name: "caves", kind: Carve(volume: "rock", tunnels: Some((curves: "tunnels", max_radius: 3)),
                rooms: Some("level"))),
        ],
    )"#
    )
}

/// The region at the origin: 8 by 8 chunks.
fn region() -> Vec<ChunkCoord> {
    (0..8)
        .flat_map(|y| (0..8).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

fn generate(runtime: &mut Runtime, stages: &[&str], chunks: &[ChunkCoord]) {
    let focus: Vec<FocusPoint> = chunks.iter().map(|&c| FocusPoint::new(c, 0)).collect();
    runtime.request(&focus, stages).expect("the stages");
    runtime.run_until_idle().expect("the stages run");
}

fn runtime(patterns: &str, seed: u64) -> Runtime {
    let pack = Arc::new(Pack::parse(&pack(patterns)).expect("a valid pack"));
    Runtime::new(pack, seed, SIZE)
}

/// The level's rooms and tunnels over the region, each once.
fn level(runtime: &Runtime) -> (Vec<Stamp>, Vec<Curve>) {
    let mut rooms = BTreeMap::new();
    let mut tunnels = BTreeMap::new();
    for chunk in region() {
        for stamp in runtime.stamps("level", chunk).expect("generated") {
            rooms.insert(stamp.id, stamp.clone());
        }
        for curve in runtime.curves("tunnels", chunk).expect("generated") {
            tunnels.insert(curve.id.clone(), curve.clone());
        }
    }
    (
        rooms.into_values().collect(),
        tunnels.into_values().collect(),
    )
}

/// The centre of a room's box.
fn centre(stamp: &Stamp) -> [f32; 3] {
    let height = if &*stamp.piece == "cavern" { 5.0 } else { 4.0 };
    [
        stamp.position[0],
        stamp.position[1],
        stamp.position[2] + height / 2.0,
    ]
}

/// The rooms each tunnel joins, by their place in `rooms`.
fn links(rooms: &[Stamp], tunnels: &[Curve]) -> Vec<(usize, usize)> {
    let room_at = |x: f32, y: f32, z: f32| {
        rooms
            .iter()
            .position(|room| centre(room) == [x, y, z])
            .expect("a tunnel ends at a room's centre")
    };
    tunnels
        .iter()
        .map(|tunnel| {
            let last = tunnel.points.len() - 1;
            (
                room_at(tunnel.points[0][0], tunnel.points[0][1], tunnel.heights[0]),
                room_at(
                    tunnel.points[last][0],
                    tunnel.points[last][1],
                    tunnel.heights[last],
                ),
            )
        })
        .collect()
}

/// How many tunnels meet each room.
fn degrees(rooms: &[Stamp], tunnels: &[Curve]) -> Vec<usize> {
    let mut degrees = vec![0; rooms.len()];
    for (a, b) in links(rooms, tunnels) {
        degrees[a] += 1;
        degrees[b] += 1;
    }
    degrees
}

#[test]
fn rooms_lie_apart_inside_their_region_at_their_depths() {
    let mut runtime = runtime("Linear, Star, Hub", 3);
    generate(&mut runtime, &["level", "tunnels"], &region());

    let (rooms, _) = level(&runtime);

    assert!((5..=7).contains(&rooms.len()), "{} rooms", rooms.len());
    for (index, room) in rooms.iter().enumerate() {
        assert!(room.min[0] >= 0 && room.min[1] >= 0, "{room:?}");
        assert!(room.max[0] <= 64 && room.max[1] <= 64, "{room:?}");
        assert!((-30.0..=-10.0).contains(&room.position[2]), "{room:?}");
        for other in &rooms[..index] {
            let (a, b) = (centre(room), centre(other));
            assert!(
                libm::hypotf(a[0] - b[0], a[1] - b[1]) >= 12.0,
                "{room:?} near {other:?}"
            );
        }
    }
}

#[test]
fn each_pattern_links_the_rooms_as_it_says() {
    for seed in 0..4 {
        let mut chain = runtime("Linear", seed);
        let mut star = runtime("Star", seed);
        let mut hub = runtime("Hub", seed);
        for runtime in [&mut chain, &mut star, &mut hub] {
            generate(runtime, &["level", "tunnels"], &region());
        }

        let (rooms, tunnels) = level(&chain);
        let mut chain_degrees = degrees(&rooms, &tunnels);
        chain_degrees.sort_unstable();
        let (rooms, tunnels) = level(&star);
        let star_degrees = degrees(&rooms, &tunnels);
        let (rooms, tunnels) = level(&hub);
        let hub_degrees = degrees(&rooms, &tunnels);

        // A chain: two ends, every other room between two tunnels.
        let n = chain_degrees.len();
        assert_eq!(chain_degrees[..2], [1, 1]);
        assert!(
            chain_degrees[2..].iter().all(|&d| d == 2),
            "{chain_degrees:?}"
        );
        assert_eq!(chain_degrees.iter().sum::<usize>(), 2 * (n - 1));
        // A star: one room joined to all the others.
        assert_eq!(star_degrees.iter().max(), Some(&(star_degrees.len() - 1)));
        // A hub: one room joined to three branches.
        assert_eq!(hub_degrees.iter().max(), Some(&3), "{hub_degrees:?}");
        assert_eq!(
            hub_degrees.iter().sum::<usize>(),
            2 * (hub_degrees.len() - 1)
        );
    }
}

#[test]
fn tunnels_and_rooms_are_carved_empty_and_the_rock_far_from_them_is_not() {
    let mut runtime = runtime("Linear, Star, Hub", 5);
    generate(&mut runtime, &["caves", "level", "tunnels"], &region());
    let (rooms, tunnels) = level(&runtime);

    let empty = |at: [f32; 3]| {
        let chunk = ChunkCoord::new(
            (at[0] / 8.0).floor() as i32,
            (at[1] / 8.0).floor() as i32,
            0,
        );
        let caves = runtime.volume("caves", chunk).expect("generated");
        let level = (at[2].floor() as i32 - caves.bottom) as u32;
        caves.get(
            at[0].floor().rem_euclid(8.0) as u32,
            at[1].floor().rem_euclid(8.0) as u32,
            level,
        ) < 0.0
    };

    for tunnel in &tunnels {
        for (point, height) in tunnel.points.iter().zip(&tunnel.heights) {
            assert!(
                empty([point[0], point[1], *height]),
                "{point:?} at {height}"
            );
        }
    }
    for room in &rooms {
        assert!(empty(centre(room)), "{room:?}");
    }
    // Rock above the highest ceiling and tunnel stays solid.
    let rock_over = (0..64)
        .flat_map(|y| (0..64).map(move |x| [x as f32 + 0.5, y as f32 + 0.5, 2.5]))
        .all(|at| !empty(at));
    assert!(rock_over, "the rock over the level is carved");
    assert!(
        tunnels.len() == rooms.len() - 1,
        "{} tunnels",
        tunnels.len()
    );
}

#[test]
fn a_level_is_the_same_whatever_order_its_chunks_come_in() {
    let mut together = runtime("Linear, Star, Hub", 9);
    let mut apart = runtime("Linear, Star, Hub", 9);

    generate(&mut together, &["caves", "level", "tunnels"], &region());
    let mut one_by_one = BTreeMap::new();
    for chunk in region().into_iter().rev() {
        generate(&mut apart, &["caves", "level", "tunnels"], &[chunk]);
        one_by_one.insert(
            chunk,
            (
                apart.volume("caves", chunk).cloned(),
                apart.stamps("level", chunk).map(<[Stamp]>::to_vec),
                apart.curves("tunnels", chunk).map(<[Curve]>::to_vec),
            ),
        );
    }

    for chunk in region() {
        let here = (
            together.volume("caves", chunk).cloned(),
            together.stamps("level", chunk).map(<[Stamp]>::to_vec),
            together.curves("tunnels", chunk).map(<[Curve]>::to_vec),
        );
        assert_eq!(Some(&here), one_by_one.get(&chunk), "{chunk:?}");
    }
}

#[test]
fn a_level_whose_rooms_cannot_all_be_placed_is_refused_with_every_attempt() {
    let text = pack("Linear")
        .replace("apart: 12.0", "apart: 40.0")
        .replace("rerolls: 8", "rerolls: 2")
        .replace("Cave(region: 8,", "Cave(region: 2,");
    let mut runtime = Runtime::new(Arc::new(Pack::parse(&text).expect("a valid pack")), 1, SIZE);

    runtime
        .request(&[FocusPoint::new(ChunkCoord::new(0, 0, 0), 0)], &["level"])
        .expect("a stage");
    let result = runtime.run_until_idle();

    assert!(
        matches!(&result, Err(StageError::RegionRejected { stage, log, .. }) if stage == "level" && log.len() == 3),
        "{result:?}"
    );
}

#[test]
fn caves_and_tunnels_out_of_range_are_refused() {
    let parse = |from: &str, to: &str| Pack::parse(&pack("Linear").replace(from, to));

    let results = [
        (parse("count: (5, 7)", "count: (0, 7)"), "level"),
        (parse("count: (5, 7)", "count: (5, 65)"), "level"),
        (parse("region: 8", "region: 0"), "level"),
        (parse("apart: 12.0", "apart: 0.0"), "level"),
        (parse("patterns: [Linear]", "patterns: []"), "level"),
        (parse("size: (7, 7, 5)", "size: (7, 0, 5)"), "level"),
        (
            parse("depth: (-30.0, -10.0)", "depth: (-10.0, -30.0)"),
            "level",
        ),
        (
            parse(r#"noise: "wander""#, r#"noise: "missing""#),
            "tunnels",
        ),
        (parse(r#"cave: "level""#, r#"cave: "rock""#), "tunnels"),
        (parse("radius: (1.5, 2.5)", "radius: (2.5, 1.5)"), "tunnels"),
    ];

    for (result, name) in results {
        assert!(
            matches!(&result, Err(PackError::Invalid { stage, .. }) if stage == name),
            "{result:?}"
        );
    }
}
