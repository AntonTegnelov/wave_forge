//! A Scatter stage's modifier chain: several candidates per block, groups scattered around a
//! candidate, conditions and water depth at each point's column, and a scale, a tilt and a stance
//! along the ground for each point. Ids stay positional and a group's points agree across chunk
//! seams, whatever order chunks are asked for in.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use wave_forge::stages::{Pack, PackError, Point, Runtime};
use wave_forge::{ChunkCoord, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "ground", kind: Field(Mul(Noise(frequency: 0.05, octaves: 3), Constant(20.0)))),
        (name: "side", kind: Rules(rules: [(category: "west", when: [Less(X, Constant(40.0))])], otherwise: "east")),
        (name: "counted", kind: Scatter(kind: "stone", height: "ground", spacing: 8, count: (2, 5))),
        (name: "groves", kind: Scatter(kind: "birch", height: "ground", spacing: 16,
            group: Some((size: (3, 6), radius: 5.0)), apart: 6, scale: (0.8, 1.2), align: 0.5)),
        (name: "westerly", kind: Scatter(kind: "ore", height: "ground", spacing: 6,
            when: [Greater(Is("side", ["west"]), Constant(0.5)), Greater(Input("ground"), Constant(8.0))])),
        (name: "shallows", kind: Scatter(kind: "reed", height: "ground", spacing: 4,
            water: Some((level: 10.0, depth: (0.5, 3.0))))),
        (name: "tilted", kind: Scatter(kind: "rock", height: "ground", spacing: 4, tilt: (10.0, 30.0))),
    ],
)"#;

const SIZE: [u32; 2] = [8, 8];

fn area() -> Vec<ChunkCoord> {
    (0..10)
        .flat_map(|y| (0..10).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

/// Every point of `stage` over the area, by chunk, asked for in the order of `requests`, and the
/// runtime that made them.
fn scatter(
    stage: &str,
    requests: &[Vec<ChunkCoord>],
) -> (Runtime, BTreeMap<ChunkCoord, Vec<Point>>) {
    let mut runtime = Runtime::new(Arc::new(Pack::parse(PACK).expect("a valid pack")), 6, SIZE);
    let mut points = BTreeMap::new();
    for request in requests {
        let focus: Vec<FocusPoint> = request.iter().map(|&c| FocusPoint::new(c, 0)).collect();
        runtime.request(&focus, &[stage, "ground"]).expect("stages");
        runtime.run_until_idle().expect("the stages run");
        for &chunk in request {
            points.insert(
                chunk,
                runtime.points(stage, chunk).expect("generated").to_vec(),
            );
        }
    }
    (runtime, points)
}

fn all(points: &BTreeMap<ChunkCoord, Vec<Point>>) -> Vec<Point> {
    points.values().flatten().cloned().collect()
}

/// The ground's value at a world column, from the runtime's chunks.
fn ground(runtime: &Runtime, x: i64, y: i64) -> f32 {
    let chunk = ChunkCoord::new(x.div_euclid(8) as i32, y.div_euclid(8) as i32, 0);
    runtime
        .field("ground", chunk)
        .expect("generated")
        .get(x.rem_euclid(8) as u32, y.rem_euclid(8) as u32)
}

#[test]
fn a_block_holds_as_many_candidates_as_its_count_allows() {
    let (_, points) = scatter("counted", &[area()]);

    let mut per_block: BTreeMap<(i64, i64), usize> = BTreeMap::new();
    for point in all(&points) {
        let block = (
            (point.position[0] / 8.0).floor() as i64,
            (point.position[1] / 8.0).floor() as i64,
        );
        *per_block.entry(block).or_default() += 1;
    }
    let ids: BTreeSet<_> = all(&points).iter().map(|point| point.id).collect();

    assert_eq!(per_block.len(), 100);
    assert!(
        per_block.values().all(|&n| (2..=5).contains(&n)),
        "{per_block:?}"
    );
    assert!(
        per_block.values().any(|&n| n != per_block[&(0, 0)]),
        "counts vary by block"
    );
    assert_eq!(ids.len(), all(&points).len(), "every id is its own");
}

#[test]
fn a_group_scatters_around_its_first_point_within_its_radius() {
    let (_, points) = scatter("groves", &[area()]);

    let mut groups: BTreeMap<(ChunkCoord, u32, u16), Vec<Point>> = BTreeMap::new();
    for point in all(&points) {
        let key = (point.id.chunk, point.id.cell(), point.id.slot() / 256);
        groups.entry(key).or_default().push(point);
    }

    assert!(groups.len() > 10, "{} groups", groups.len());
    for members in groups.values() {
        // A group whose first point stands outside the area has only its members here.
        let Some(first) = members.iter().find(|point| point.id.slot() % 256 == 0) else {
            continue;
        };
        for member in members {
            let apart = (member.position[0] - first.position[0])
                .hypot(member.position[1] - first.position[1]);
            assert!(apart <= 5.0 + 1e-4, "{apart} from the group's first point");
            assert!((0.8..=1.2).contains(&member.scale), "{}", member.scale);
        }
    }
    let inner = groups
        .values()
        .filter(|members| {
            let first = &members[0].position;
            (8.0..72.0).contains(&first[0]) && (8.0..72.0).contains(&first[1])
        })
        .map(Vec::len);
    assert!(
        inner.clone().all(|n| (3..=6).contains(&n)),
        "{:?}",
        inner.collect::<Vec<_>>()
    );
}

#[test]
fn groups_across_chunk_seams_are_the_same_in_any_order() {
    let one_by_one: Vec<Vec<ChunkCoord>> = area().into_iter().map(|c| vec![c]).collect();
    let backwards: Vec<Vec<ChunkCoord>> = area().into_iter().rev().map(|c| vec![c]).collect();

    let (_, at_once) = scatter("groves", &[area()]);
    let (_, forwards) = scatter("groves", &one_by_one);
    let (_, reversed) = scatter("groves", &backwards);

    assert_eq!(at_once, forwards);
    assert_eq!(at_once, reversed);
    assert!(
        all(&at_once).iter().any(|point| {
            let owner = point.id.chunk;
            let standing = (
                (point.position[0] / 8.0).floor() as i32,
                (point.position[1] / 8.0).floor() as i32,
            );
            standing != (owner.x, owner.y)
        }),
        "some group crosses a seam"
    );
}

#[test]
fn a_point_stands_only_where_its_conditions_hold_at_its_column() {
    let (runtime, points) = scatter("westerly", &[area()]);

    let points = all(&points);
    assert!(points.len() > 20, "{} points", points.len());
    for point in points {
        let (x, y) = (
            point.position[0].floor() as i64,
            point.position[1].floor() as i64,
        );
        assert!(x < 40, "{x} is east");
        assert!(ground(&runtime, x, y) > 8.0);
    }
}

#[test]
fn a_point_in_water_stands_as_deep_as_its_range_allows() {
    let (_, points) = scatter("shallows", &[area()]);

    let points = all(&points);
    assert!(points.len() > 20, "{} points", points.len());
    for point in points {
        let depth = 10.0 - point.position[2];
        assert!((0.5..=3.0).contains(&depth), "{depth}");
    }
}

#[test]
fn a_point_leans_within_its_tilt_or_along_the_ground() {
    let (_, tilted) = scatter("tilted", &[area()]);
    let (runtime, groves) = scatter("groves", &[area()]);

    for point in all(&tilted) {
        let degrees = point.up[2].clamp(-1.0, 1.0).acos().to_degrees();
        assert!((10.0 - 1e-3..=30.0 + 1e-3).contains(&degrees), "{degrees}");
    }
    let (mut along, mut upright) = (0, 0);
    for point in all(&groves) {
        if point.up == [0.0, 0.0, 1.0] {
            upright += 1;
            continue;
        }
        let (x, y) = (
            point.position[0].floor() as i64,
            point.position[1].floor() as i64,
        );
        let along_x = (ground(&runtime, x + 1, y) - ground(&runtime, x - 1, y)) / 2.0;
        let along_y = (ground(&runtime, x, y + 1) - ground(&runtime, x, y - 1)) / 2.0;
        let length = (along_x * along_x + along_y * along_y + 1.0).sqrt();
        let normal = [-along_x / length, -along_y / length, 1.0 / length];
        for axis in 0..3 {
            assert!(
                (point.up[axis] - normal[axis]).abs() < 1e-5,
                "{:?} against {normal:?}",
                point.up
            );
        }
        along += 1;
    }
    assert!(
        along > 5 && upright > 5,
        "{along} along the ground, {upright} upright"
    );
}

#[test]
fn a_chain_out_of_range_is_refused_by_stage() {
    let with = |scatter: &str| {
        Pack::parse(&format!(
            r#"(version: 1, stages: [
                (name: "ground", kind: Field(Constant(1.0))),
                (name: "s", kind: Scatter(kind: "k", height: "ground", spacing: 4, {scatter})),
            ])"#
        ))
    };

    for scatter in [
        "count: (0, 1)",
        "count: (3, 2)",
        "count: (1, 256)",
        "group: Some((size: (2, 1), radius: 3.0))",
        "group: Some((size: (1, 2), radius: -1.0))",
        "tilt: (30.0, 10.0)",
        "tilt: (0.0, 200.0)",
        "align: 2.0",
        "scale: (0.0, 1.0)",
        "water: Some((level: 1.0, depth: (3.0, 1.0)))",
        "when: [Less(Input(\"nowhere\"), Constant(1.0))]",
    ] {
        let result = with(scatter);
        assert!(
            matches!(&result, Err(PackError::Invalid { stage, .. } | PackError::UnknownInput { stage, .. }) if stage == "s"),
            "{scatter}: {result:?}"
        );
    }
}

#[test]
fn a_points_basis_carries_the_vertical_to_its_up_and_keeps_its_scale() {
    let (_, tilted) = scatter("tilted", &[area()]);
    let (_, groves) = scatter("groves", &[area()]);

    for point in all(&tilted).iter().chain(&all(&groves)) {
        let rows = point.y_up_basis();
        let column = |c: usize| [rows[0][c], rows[1][c], rows[2][c]];
        // The vertical of a Y-up engine is its y; `up` is along the lattice's x, y and height.
        let up = [point.up[0], point.up[2], point.up[1]];
        for axis in 0..3 {
            assert!(
                (column(1)[axis] - up[axis] * point.scale).abs() < 1e-4,
                "{rows:?} for {up:?}"
            );
        }
        for (a, b) in [(0, 1), (1, 2), (0, 2)] {
            let dot: f32 = (0..3).map(|k| column(a)[k] * column(b)[k]).sum();
            assert!(dot.abs() < 1e-4, "columns {a} and {b} of {rows:?}");
        }
        for c in 0..3 {
            let length: f32 = column(c).iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((length - point.scale).abs() < 1e-4);
        }
    }
}

fn ring_world() -> Arc<Pack> {
    let text = std::fs::read_to_string(format!(
        "{}/examples/rings.world.ron",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("the ring world");
    Arc::new(Pack::parse(&text).expect("a valid pack"))
}

/// Every point of `stage` over the ring island, and the runtime that placed them.
fn ring_points(stage: &str) -> (Runtime, Vec<Point>) {
    let mut runtime = Runtime::new(ring_world(), 7, SIZE);
    let focus: Vec<FocusPoint> = (-13..13)
        .flat_map(|y| (-13..13).map(move |x| FocusPoint::new(ChunkCoord::new(x, y, 0), 0)))
        .collect();
    runtime
        .request(&focus, &[stage, "biome", "terrain", "roughness"])
        .expect("stages");
    runtime.run_until_idle().expect("the stages run");
    let points = focus
        .iter()
        .flat_map(|focus| {
            runtime
                .points(stage, focus.chunk)
                .expect("generated")
                .to_vec()
        })
        .collect();
    (runtime, points)
}

/// A stage's value at a world column of the ring world: a field's value, or a category's index.
fn ring_value(runtime: &Runtime, stage: &str, x: i64, y: i64) -> f32 {
    let chunk = ChunkCoord::new(x.div_euclid(8) as i32, y.div_euclid(8) as i32, 0);
    let (cx, cy) = (x.rem_euclid(8) as u32, y.rem_euclid(8) as u32);
    match runtime.field(stage, chunk) {
        Some(field) => field.get(cx, cy),
        None => f32::from(
            runtime
                .categories(stage, chunk)
                .expect("generated")
                .get(cx, cy),
        ),
    }
}

/// The name of the ring world's biome at a world column.
fn biome(runtime: &Runtime, names: &[&str], x: i64, y: i64) -> String {
    names[ring_value(runtime, "biome", x, y) as usize].to_owned()
}

#[test]
fn the_ring_worlds_ore_and_groves_stand_where_their_rules_say() {
    let (runtime, woods) = ring_points("woods_ore");
    let (_, peaks) = ring_points("peak_ore");
    let (_, groves) = ring_points("birch_groves");
    let pack = ring_world();
    let names = pack.kind("biome").expect("a stage").categories();

    let column = |point: &Point| {
        (
            point.position[0].floor() as i64,
            point.position[1].floor() as i64,
        )
    };
    assert!(!woods.is_empty() && !peaks.is_empty() && !groves.is_empty());
    for point in &woods {
        let (x, y) = column(point);
        assert_eq!(biome(&runtime, &names, x, y), "woods");
        assert!(ring_value(&runtime, "roughness", x, y) < 0.25);
    }
    for point in &peaks {
        let (x, y) = column(point);
        assert_eq!(biome(&runtime, &names, x, y), "peaks");
        assert!(ring_value(&runtime, "terrain", x, y) > 0.8);
    }
    let mut groups: BTreeMap<(ChunkCoord, u32, u16), usize> = BTreeMap::new();
    for point in &groves {
        let (x, y) = column(point);
        assert_eq!(biome(&runtime, &names, x, y), "grassland");
        *groups
            .entry((point.id.chunk, point.id.cell(), point.id.slot() / 256))
            .or_default() += 1;
    }
    assert!(groups.values().any(|&n| n >= 3), "{groups:?}");
}

const BLOCKING: &str = r#"(
    version: 1,
    stages: [
        (name: "ground", kind: Field(Mul(Noise(frequency: 0.05, octaves: 3), Constant(20.0)))),
        (name: "rocks", kind: Scatter(kind: "rock", height: "ground", spacing: 5, count: (1, 2))),
        (name: "trees", kind: Scatter(kind: "tree", height: "ground", spacing: 4,
            group: Some((size: (2, 4), radius: 3.0)), block: [("rocks", 2.5)])),
        (name: "free", kind: Scatter(kind: "tree", height: "ground", spacing: 4,
            group: Some((size: (2, 4), radius: 3.0)))),
    ],
)"#;

/// Every point of `stage` in the blocking pack over the area, asked for in `requests`' order.
fn blocking(stage: &str, requests: &[Vec<ChunkCoord>]) -> BTreeMap<ChunkCoord, Vec<Point>> {
    let mut runtime = Runtime::new(
        Arc::new(Pack::parse(BLOCKING).expect("a valid pack")),
        6,
        SIZE,
    );
    let mut points = BTreeMap::new();
    for request in requests {
        let focus: Vec<FocusPoint> = request.iter().map(|&c| FocusPoint::new(c, 0)).collect();
        runtime.request(&focus, &[stage]).expect("a stage");
        runtime.run_until_idle().expect("the stages run");
        for &chunk in request {
            points.insert(
                chunk,
                runtime.points(stage, chunk).expect("generated").to_vec(),
            );
        }
    }
    points
}

/// The least distance from any point of `a` to any point of `b`, across the ground.
fn closest(a: &[Point], b: &[Point]) -> f32 {
    a.iter()
        .flat_map(|p| {
            b.iter()
                .map(move |q| (p.position[0] - q.position[0]).hypot(p.position[1] - q.position[1]))
        })
        .fold(f32::INFINITY, f32::min)
}

#[test]
fn a_stage_keeps_clear_of_the_points_of_a_stage_it_defers_to_across_seams() {
    let rocks = all(&blocking("rocks", &[area()]));

    let trees = all(&blocking("trees", &[area()]));
    let free = all(&blocking("free", &[area()]));

    assert!(trees.len() > 100, "{} trees", trees.len());
    assert!(closest(&trees, &rocks) >= 2.5);
    assert!(
        closest(&free, &rocks) < 2.5,
        "without blocking, trees stand by rocks"
    );
}

#[test]
fn blocking_comes_out_the_same_in_any_order() {
    let one_by_one: Vec<Vec<ChunkCoord>> = area().into_iter().map(|c| vec![c]).collect();
    let backwards: Vec<Vec<ChunkCoord>> = area().into_iter().rev().map(|c| vec![c]).collect();

    let at_once = blocking("trees", &[area()]);

    assert_eq!(at_once, blocking("trees", &one_by_one));
    assert_eq!(at_once, blocking("trees", &backwards));
}
