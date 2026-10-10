//! A Rivers stage: rivers that run downhill from high ground in every region, on through hollows,
//! to the sea, the region's edge or a river they join, widening as they go, the same in any order.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::regions::{Curve, CurveId};
use wave_forge::stages::{Pack, PackError, Runtime};
use wave_forge::{ChunkCoord, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    water: Some((level: 0.0)),
    stages: [
        (name: "ground", kind: Field(Sub(
            Mul(Noise(frequency: 0.03, octaves: 3), Constant(30.0)),
            Mul(Distance((0.0, 0.0)), Constant(0.2)),
        ))),
        (name: "rivers", kind: Rivers(height: "ground", region: 4, sources: 3,
            width: (1.0, 3.0), step: 2)),
    ],
)"#;

const SIZE: [u32; 2] = [8, 8];
const STEP: f32 = 2.0;

fn area() -> Vec<ChunkCoord> {
    (-8..8)
        .flat_map(|y| (-8..8).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

fn pack() -> Arc<Pack> {
    Arc::new(Pack::parse(PACK).expect("a valid pack"))
}

/// Every river over the area once, asked for in the order of `requests`.
fn rivers(requests: &[Vec<ChunkCoord>]) -> Vec<Curve> {
    let mut runtime = Runtime::new(pack(), 4, SIZE);
    let mut curves: BTreeMap<CurveId, Curve> = BTreeMap::new();
    for request in requests {
        let focus: Vec<FocusPoint> = request.iter().map(|&c| FocusPoint::new(c, 0)).collect();
        runtime.request(&focus, &["rivers"]).expect("a stage");
        runtime.run_until_idle().expect("the stages run");
        for &chunk in request {
            for curve in runtime.curves("rivers", chunk).expect("generated") {
                curves.insert(curve.id.clone(), curve.clone());
            }
        }
    }
    curves.into_values().collect()
}

fn region(curve: &Curve) -> (i32, i32) {
    match curve.id {
        CurveId::Region { region, .. } => region,
        CurveId::Row(_) => panic!("a river is named by its region"),
    }
}

#[test]
fn a_river_runs_on_through_hollows_to_the_sea_its_regions_edge_or_another_river() {
    let sampler = Runtime::new(pack(), 4, SIZE);
    let ground = |[x, y]: [f32; 2]| sampler.sample("ground", [x, y]).expect("a sample");

    let rivers = rivers(&[area()]);

    // Three sources in each of 16 regions, less those that start at the sea or the edge.
    assert!(rivers.len() > 40, "{} rivers", rivers.len());
    let mut ends = BTreeMap::new();
    for river in &rivers {
        let (rx, ry) = region(river);
        let (low, high) = (
            [rx as f32 * 32.0, ry as f32 * 32.0],
            [rx as f32 * 32.0 + 32.0, ry as f32 * 32.0 + 32.0],
        );
        let within =
            |[x, y]: [f32; 2]| (low[0]..high[0]).contains(&x) && (low[1]..high[1]).contains(&y);
        assert!(
            river.points.iter().all(|&point| within(point)),
            "{:?} leaves its region",
            river.id
        );
        let (source, mouth) = (river.points[0], *river.points.last().expect("points"));
        // A river from a source on high ground ends lower; one from a crossing may start in a
        // hollow.
        let from_source = matches!(river.id, CurveId::Region { index, .. } if index < 3);
        assert!(
            !from_source || ground(mouth) < ground(source),
            "{:?} ends higher than it starts",
            river.id
        );
        // No hollow stops a river: it ends in the sea, on its region's outermost columns, or where
        // it joins another river.
        let on_edge = |[x, y]: [f32; 2]| {
            x < low[0] + 1.0 || y < low[1] + 1.0 || x > high[0] - 1.0 || y > high[1] - 1.0
        };
        // A river joining another ends on it, within a step of one of its points.
        let joins = rivers.iter().any(|other| {
            other.id != river.id
                && region(other) == region(river)
                && other
                    .points
                    .iter()
                    .any(|p| (p[0] - mouth[0]).hypot(p[1] - mouth[1]) <= 2.0 * STEP)
        });
        let why = if ground(mouth) <= 0.0 {
            "sea"
        } else if on_edge(mouth) {
            "edge"
        } else {
            assert!(joins, "{:?} stops inland at {mouth:?}", river.id);
            "another river"
        };
        *ends.entry(why).or_insert(0) += 1;
    }
    assert!(ends.contains_key("sea"), "{ends:?}");
}

#[test]
fn a_river_climbs_out_of_a_hollow_no_higher_than_it_came_in() {
    let sampler = Runtime::new(pack(), 4, SIZE);
    let ground = |[x, y]: [f32; 2]| sampler.sample("ground", [x, y]).expect("a sample");

    let rivers = rivers(&[area()]);

    // Water fills a hollow only to where it spills, which is below every point it came down from,
    // for a river from a source on high ground; one from a crossing may start in a hollow.
    let mut climbing = 0;
    for river in rivers
        .iter()
        .filter(|river| matches!(river.id, CurveId::Region { index, .. } if index < 3))
    {
        let heights: Vec<f32> = river.points.iter().map(|&point| ground(point)).collect();
        for (i, &height) in heights.iter().enumerate().skip(1) {
            if height > heights[i - 1] {
                climbing += 1;
                let came_in = heights[..i]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                assert!(
                    height <= came_in,
                    "{:?} climbs to {height}, above all it came down from",
                    river.id
                );
            }
        }
    }
    assert!(climbing > 0, "no river crossed a hollow");
}

#[test]
fn a_river_that_joins_another_ends_where_it_joins() {
    let rivers = rivers(&[area()]);

    // Every point a river runs through but its mouth, each with the river it belongs to.
    let mut owner = BTreeMap::new();
    for river in &rivers {
        for point in &river.points[..river.points.len() - 1] {
            let key = (region(river), point[0].to_bits(), point[1].to_bits());
            let earlier = owner.insert(key, river.id.clone());
            assert!(
                earlier.is_none(),
                "{:?} and {:?} run on together at {point:?}",
                earlier,
                river.id
            );
        }
    }
}

#[test]
fn a_river_that_comes_into_its_region_continues_one_that_left_the_next() {
    let rivers = rivers(&[area()]);

    // A river starting on a side of its region as wide as a mouth came in by a crossing: a river
    // of the region across ends on the column facing its first point.
    let inside = |c: f32| c.rem_euclid(32.0);
    let mut continued = 0;
    for river in rivers.iter().filter(|river| river.values[0] == 3.0) {
        let first = river.points[0];
        let (x, y) = (inside(first[0]), inside(first[1]));
        let across = if x < 1.0 {
            [first[0] - 1.0, first[1]]
        } else if x > 31.0 {
            [first[0] + 1.0, first[1]]
        } else if y < 1.0 {
            [first[0], first[1] - 1.0]
        } else if y > 31.0 {
            [first[0], first[1] + 1.0]
        } else {
            panic!("{:?} starts as wide as a mouth inside its region", river.id);
        };
        if across.iter().any(|c| !(-64.0..64.0).contains(c)) {
            continue;
        }
        let fed = rivers
            .iter()
            .any(|other| region(other) != region(river) && other.points.last() == Some(&across));
        assert!(
            fed,
            "{:?} comes in at {first:?}, but no river ends across at {across:?}",
            river.id
        );
        continued += 1;
    }
    assert!(continued > 0, "no river comes into a region");
}

#[test]
fn a_river_widens_from_its_source_to_its_mouth() {
    for river in rivers(&[area()]) {
        // A river that comes in across its region's side is already as wide as a mouth.
        let first = river.values[0];
        assert!(
            first == 1.0 || first == 3.0,
            "{:?} starts {first} wide",
            river.id
        );
        assert_eq!(river.values.last(), Some(&3.0));
        assert!(river.values.windows(2).all(|pair| pair[1] >= pair[0]));
    }
}

#[test]
fn a_river_that_leaves_its_region_carries_on_in_the_next_from_where_it_left() {
    let sampler = Runtime::new(pack(), 4, SIZE);
    let ground = |[x, y]: [f32; 2]| sampler.sample("ground", [x, y]).expect("a sample");

    let rivers = rivers(&[area()]);

    // A mouth on a side of its region, above the sea, faces a river in the region across that
    // side: one that starts on the column across, or that runs past it, where the river coming in
    // joins it at once.
    let inside = |c: f32| c.rem_euclid(32.0);
    let mut carried = 0;
    for river in &rivers {
        let mouth = *river.points.last().expect("points");
        let (x, y) = (inside(mouth[0]), inside(mouth[1]));
        let mut across = Vec::new();
        if x < 1.0 {
            across.push([mouth[0] - 1.0, mouth[1]]);
        }
        if x > 31.0 {
            across.push([mouth[0] + 1.0, mouth[1]]);
        }
        if y < 1.0 {
            across.push([mouth[0], mouth[1] - 1.0]);
        }
        if y > 31.0 {
            across.push([mouth[0], mouth[1] + 1.0]);
        }
        across.retain(|point| point.iter().all(|c| (-64.0..64.0).contains(c)));
        let near = |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).hypot(a[1] - b[1]) <= STEP;
        // A river that ends where it joins another of its region does not leave it.
        let joins = rivers.iter().any(|other| {
            other.id != river.id
                && region(other) == region(river)
                && other.points.iter().any(|&p| near(p, mouth))
        });
        if across.is_empty() || ground(mouth) <= 0.0 || joins {
            continue;
        }
        let next = rivers.iter().any(|other| {
            region(other) != region(river)
                && across
                    .iter()
                    .any(|&point| other.points.iter().any(|&p| near(p, point)))
        });
        assert!(
            next,
            "{:?} leaves its region at {mouth:?}, but no river is across at {across:?}",
            river.id
        );
        carried += 1;
    }
    assert!(carried > 0, "no river leaves its region");
}

#[test]
fn rivers_are_the_same_in_any_order() {
    let one_by_one: Vec<Vec<ChunkCoord>> = area().into_iter().map(|c| vec![c]).collect();
    let backwards: Vec<Vec<ChunkCoord>> = area().into_iter().rev().map(|c| vec![c]).collect();

    let at_once = rivers(&[area()]);

    assert_eq!(at_once, rivers(&one_by_one));
    assert_eq!(at_once, rivers(&backwards));
}

#[test]
fn rivers_that_cannot_run_are_refused() {
    let with = |water: &str, rivers: &str| {
        Pack::parse(&format!(
            r#"(version: 1, {water} stages: [
                (name: "ground", kind: Field(Constant(1.0))),
                (name: "rivers", kind: Rivers(height: "ground", region: 4, {rivers})),
            ])"#
        ))
    };

    for (water, rivers) in [
        ("water: Some((level: 0.0)),", "sources: 0"),
        ("water: Some((level: 0.0)),", "sources: 65"),
        ("water: Some((level: 0.0)),", "sources: 1, step: 0"),
        (
            "water: Some((level: 0.0)),",
            "sources: 1, width: (-1.0, 2.0)",
        ),
        ("", "sources: 1"),
    ] {
        let result = with(water, rivers);
        assert!(
            matches!(&result, Err(PackError::Invalid { stage, .. }) if stage == "rivers"),
            "{water} {rivers}: {result:?}"
        );
    }
    assert!(with("water: Some((level: 0.0)),", "sources: 1").is_ok());
}

#[test]
fn the_ring_worlds_rivers_are_carved_into_its_ground() {
    let text = std::fs::read_to_string(format!(
        "{}/examples/rings.world.ron",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("the ring world");
    let mut runtime = Runtime::new(Arc::new(Pack::parse(&text).expect("a valid pack")), 7, SIZE);

    runtime
        .request_bound(&["ground", "terrain", "rivers"])
        .expect("a bounded island");
    runtime.run_until_idle().expect("the stages run");

    let (mut carved, mut rivers) = (0, BTreeMap::new());
    for y in -14..14 {
        for x in -14..14 {
            let chunk = ChunkCoord::new(x, y, 0);
            let (Some(ground), Some(terrain)) = (
                runtime.field("ground", chunk),
                runtime.field("terrain", chunk),
            ) else {
                continue;
            };
            carved += ground
                .values
                .iter()
                .zip(&terrain.values)
                .filter(|(g, t)| g < t)
                .count();
            for river in runtime.curves("rivers", chunk).expect("generated") {
                rivers.insert(river.id.clone(), river.points.len());
            }
        }
    }
    assert!(rivers.len() >= 4, "{rivers:?}");
    assert!(carved > 100, "only {carved} columns carved");
}

#[test]
fn a_chunk_of_rivers_generates_the_height_over_its_region_and_the_columns_beside_it_alone() {
    let mut runtime = Runtime::new(pack(), 4, SIZE);

    runtime
        .request(&[FocusPoint::new(ChunkCoord::new(5, 6, 0), 0)], &["rivers"])
        .expect("a stage");
    runtime.run_until_idle().expect("the stages run");

    // Its region of 4 chunks a side runs from chunk 4 to chunk 7 each way, and the column beyond
    // each side, where the crossings are, lies in the ring of chunks around it, 3 and 8.
    for y in -2..12 {
        for x in -2..12 {
            let inside = (3..9).contains(&x) && (3..9).contains(&y);
            let held = runtime.field("ground", ChunkCoord::new(x, y, 0)).is_some();
            assert_eq!(held, inside, "the height of chunk ({x}, {y})");
        }
    }
}
