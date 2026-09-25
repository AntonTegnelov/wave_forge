//! A Lakes stage: water standing in a height field's hollows, per region and the same in any
//! order, which Scatter's depth tests and rivers read through the pack's water.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::{Pack, PackError, Runtime};
use wave_forge::{ChunkCoord, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    water: Some((level: 0.0, lakes: Some("lakes"))),
    stages: [
        (name: "ground", kind: Field(Sub(
            Mul(Noise(frequency: 0.06, octaves: 3), Constant(20.0)),
            Constant(2.0),
        ))),
        (name: "lakes", kind: Lakes(height: "ground", region: 4, min_columns: 4)),
        (name: "rivers", kind: Rivers(height: "ground", region: 4, sources: 3,
            width: (1.0, 2.0), step: 1)),
        (name: "reeds", kind: Scatter(kind: "reed", height: "ground", spacing: 2,
            water: Some((depth: (0.2, 2.0))))),
    ],
)"#;

const SIZE: [u32; 2] = [8, 8];
/// Columns along a region's side: 4 chunks of 8.
const REGION: i64 = 32;

fn area() -> Vec<ChunkCoord> {
    (-4..4)
        .flat_map(|y| (-4..4).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

fn pack() -> Arc<Pack> {
    Arc::new(Pack::parse(PACK).expect("a valid pack"))
}

/// A runtime that has generated `stages` over the area, asked for in the order of `requests`.
fn run(requests: &[Vec<ChunkCoord>], stages: &[&str]) -> Runtime {
    let mut runtime = Runtime::new(pack(), 6, SIZE);
    for request in requests {
        let focus: Vec<FocusPoint> = request.iter().map(|&c| FocusPoint::new(c, 0)).collect();
        runtime.request(&focus, stages).expect("the stages");
        runtime.run_until_idle().expect("the stages run");
    }
    runtime
}

/// Every column of `chunks`: its ground and the water's surface over it.
fn columns_of(runtime: &Runtime, chunks: &[ChunkCoord]) -> BTreeMap<(i64, i64), (f32, f32)> {
    let mut out = BTreeMap::new();
    for &chunk in chunks {
        let ground = runtime.field("ground", chunk).expect("ground");
        let lakes = runtime.field("lakes", chunk).expect("lakes");
        for y in 0..SIZE[1] {
            for x in 0..SIZE[0] {
                let column = (
                    i64::from(chunk.x) * i64::from(SIZE[0]) + i64::from(x),
                    i64::from(chunk.y) * i64::from(SIZE[1]) + i64::from(y),
                );
                out.insert(column, (ground.get(x, y), lakes.get(x, y)));
            }
        }
    }
    out
}

/// Every column of the area, which `runtime` has generated.
fn columns(runtime: &Runtime) -> BTreeMap<(i64, i64), (f32, f32)> {
    columns_of(runtime, &area())
}

/// Every column of the area, each read once its request has run, in the order of `requests`.
fn columns_asked(requests: &[Vec<ChunkCoord>]) -> BTreeMap<(i64, i64), (f32, f32)> {
    let mut runtime = Runtime::new(pack(), 6, SIZE);
    let mut out = BTreeMap::new();
    for request in requests {
        let focus: Vec<FocusPoint> = request.iter().map(|&c| FocusPoint::new(c, 0)).collect();
        runtime
            .request(&focus, &["lakes", "ground"])
            .expect("the stages");
        runtime.run_until_idle().expect("the stages run");
        out.extend(columns_of(&runtime, request));
    }
    out
}

#[test]
fn lakes_are_the_same_in_any_order() {
    let one_by_one: Vec<Vec<ChunkCoord>> = area().into_iter().map(|c| vec![c]).collect();
    let backwards: Vec<Vec<ChunkCoord>> = area().into_iter().rev().map(|c| vec![c]).collect();

    let at_once = columns_asked(&[area()]);

    assert_eq!(at_once, columns_asked(&one_by_one));
    assert_eq!(at_once, columns_asked(&backwards));
}

#[test]
fn a_lake_is_level_above_the_sea_and_never_at_its_regions_edge() {
    let columns = columns(&run(&[area()], &["lakes", "ground"]));

    let mut lake_columns = 0;
    for (&(x, y), &(ground, surface)) in &columns {
        assert!(surface >= ground, "({x}, {y}) under its ground");
        if surface == ground {
            continue;
        }
        lake_columns += 1;
        assert!(surface > 0.0, "({x}, {y}) is a lake below the sea");
        let (rx, ry) = (x.rem_euclid(REGION), y.rem_euclid(REGION));
        assert!(
            rx != 0 && ry != 0 && rx != REGION - 1 && ry != REGION - 1,
            "a lake at its region's edge, ({x}, {y})"
        );
        for next in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] {
            let &(next_ground, next_surface) = &columns[&next];
            let under_the_same = next_surface > next_ground && next_surface == surface;
            assert!(
                under_the_same || next_ground >= surface,
                "({x}, {y}) at {surface} beside {next:?}, ground {next_ground}, water {next_surface}"
            );
        }
    }
    assert!(lake_columns > 20, "{lake_columns} columns under lakes");
}

#[test]
fn a_reed_stands_as_deep_as_its_range_below_the_sea_or_a_lake() {
    let runtime = run(&[area()], &["reeds", "lakes", "ground"]);
    let columns = columns(&runtime);

    let (mut in_lakes, mut total) = (0, 0);
    for chunk in area() {
        for point in runtime.points("reeds", chunk).expect("reeds") {
            let column = (
                point.position[0].floor() as i64,
                point.position[1].floor() as i64,
            );
            let (ground, lake) = columns[&column];
            let depth = lake.max(0.0) - ground;
            assert!((0.2..=2.0).contains(&depth), "{column:?}: {depth}");
            in_lakes += usize::from(lake > ground && lake > 0.0);
            total += 1;
        }
    }
    assert!(in_lakes > 0, "none of {total} reeds stands in a lake");
}

#[test]
fn a_river_ends_where_it_reaches_a_lake() {
    let runtime = run(&[area()], &["rivers", "lakes", "ground"]);
    let columns = columns(&runtime);

    let mut ending_in_lakes = 0;
    for chunk in area() {
        for river in runtime.curves("rivers", chunk).expect("rivers") {
            let wet = |point: [f32; 2]| {
                columns
                    .get(&(point[0].floor() as i64, point[1].floor() as i64))
                    .is_some_and(|&(ground, lake)| lake > ground)
            };
            let (last, before) = river.points.split_last().expect("a river has points");
            assert!(
                !before.iter().any(|&point| wet(point)),
                "{:?} runs on through a lake",
                river.id
            );
            ending_in_lakes += usize::from(wet(*last));
        }
    }
    assert!(ending_in_lakes > 0, "no river reaches a lake");
}

#[test]
fn lakes_that_cannot_be_filled_are_refused() {
    let with = |water: &str, lakes: &str| {
        Pack::parse(&format!(
            r#"(version: 1, {water} stages: [
                (name: "ground", kind: Field(Constant(1.0))),
                (name: "lakes", kind: Lakes(height: "ground", {lakes})),
            ])"#
        ))
    };

    for (water, lakes) in [
        ("water: Some((level: 0.0)),", "region: 0"),
        ("water: Some((level: 0.0)),", "region: 2, min_columns: 0"),
        ("", "region: 2"),
    ] {
        let result = with(water, lakes);
        assert!(
            matches!(&result, Err(PackError::Invalid { stage, .. }) if stage == "lakes"),
            "{water} {lakes}: {result:?}"
        );
    }
    let wrong = with(
        "water: Some((level: 0.0, lakes: Some(\"ground\"))),",
        "region: 2",
    );
    assert!(matches!(wrong, Err(PackError::Water(_))), "{wrong:?}");
    assert!(
        with(
            "water: Some((level: 0.0, lakes: Some(\"lakes\"))),",
            "region: 2"
        )
        .is_ok()
    );
}

#[test]
fn lakes_fill_a_coarse_lattice_as_a_fine_one() {
    let pack = Pack::parse(
        r#"(
            version: 1,
            water: Some((level: 0.0, lakes: Some("lakes"))),
            stages: [
                (name: "ground", scale: 4, kind: Field(Sub(
                    Mul(Noise(frequency: 0.015, octaves: 3), Constant(20.0)),
                    Constant(2.0),
                ))),
                (name: "lakes", scale: 4, kind: Lakes(height: "ground", region: 2, min_columns: 2)),
            ],
        )"#,
    )
    .expect("a valid pack");
    let mut runtime = Runtime::new(Arc::new(pack), 6, SIZE);
    // Focus points are in the WFC lattice's chunks: 8 coarse chunks a side cover 32 fine ones.
    let focus: Vec<FocusPoint> = (-16..16)
        .flat_map(|y| (-16..16).map(move |x| FocusPoint::new(ChunkCoord::new(x, y, 0), 0)))
        .collect();

    runtime
        .request(&focus, &["lakes", "ground"])
        .expect("the stages");
    runtime.run_until_idle().expect("the stages run");

    let mut lake_columns = 0;
    for cy in -4..4 {
        for cx in -4..4 {
            let chunk = ChunkCoord::new(cx, cy, 0);
            let ground = runtime.field("ground", chunk).expect("ground");
            let lakes = runtime.field("lakes", chunk).expect("lakes");
            for y in 0..SIZE[1] {
                for x in 0..SIZE[0] {
                    let (g, w) = (ground.get(x, y), lakes.get(x, y));
                    assert!(w >= g);
                    if w > g {
                        lake_columns += 1;
                        // A coarse region is 2 chunks of 8 coarse columns.
                        let (rx, ry) = (
                            (i64::from(cx) * 8 + i64::from(x)).rem_euclid(16),
                            (i64::from(cy) * 8 + i64::from(y)).rem_euclid(16),
                        );
                        assert!(
                            rx != 0 && ry != 0 && rx != 15 && ry != 15,
                            "at its region's edge"
                        );
                    }
                }
            }
        }
    }
    assert!(lake_columns > 0, "no lake on the coarse lattice");
}

#[test]
fn the_ring_world_has_lakes_in_its_hollows_and_rivers_that_reach_them() {
    let text = std::fs::read_to_string(format!(
        "{}/examples/rings.world.ron",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("the ring world");
    let mut runtime = Runtime::new(Arc::new(Pack::parse(&text).expect("a valid pack")), 7, SIZE);

    runtime
        .request_bound(&["lakes", "terrain", "rivers"])
        .expect("a bounded island");
    runtime.run_until_idle().expect("the stages run");

    let mut lake_columns = 0;
    let mut wet = std::collections::BTreeSet::new();
    for y in -14..14 {
        for x in -14..14 {
            let chunk = ChunkCoord::new(x, y, 0);
            let (Some(terrain), Some(lakes)) = (
                runtime.field("terrain", chunk),
                runtime.field("lakes", chunk),
            ) else {
                continue;
            };
            for (at, (&ground, &surface)) in terrain.values.iter().zip(&lakes.values).enumerate() {
                assert!(surface >= ground);
                if surface > ground {
                    lake_columns += 1;
                    assert!(surface > 0.05, "a lake below the sea");
                    let (cx, cy) = (at as i64 % 8, at as i64 / 8);
                    wet.insert((i64::from(x) * 8 + cx, i64::from(y) * 8 + cy));
                }
            }
        }
    }
    let mut ends = BTreeMap::new();
    for y in -14..14 {
        for x in -14..14 {
            for river in runtime
                .curves("rivers", ChunkCoord::new(x, y, 0))
                .unwrap_or_default()
            {
                let last = river.points.last().expect("a river has points");
                let end = (last[0].floor() as i64, last[1].floor() as i64);
                ends.insert(river.id.clone(), wet.contains(&end));
            }
        }
    }
    let ends_in_lakes = ends.values().filter(|&&in_lake| in_lake).count();
    println!(
        "{lake_columns} columns under lakes; {ends_in_lakes} of {} rivers end in one",
        ends.len()
    );
    assert!(lake_columns >= 20, "{lake_columns} columns under lakes");
    assert!(ends_in_lakes > 0, "no river reaches a lake");
}
