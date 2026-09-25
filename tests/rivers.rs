//! A Rivers stage: rivers that run downhill from high ground in every region to the sea, a hollow
//! or the region's edge, widening as they go, the same in any order.

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
fn a_river_runs_downhill_to_the_sea_a_hollow_or_its_regions_edge() {
    let sampler = Runtime::new(pack(), 4, SIZE);
    let ground = |[x, y]: [f32; 2]| sampler.sample("ground", [x, y]).expect("a sample");

    let rivers = rivers(&[area()]);

    // Three sources in each of 16 regions, less those that start where no step goes lower.
    assert!(rivers.len() > 40, "{} rivers", rivers.len());
    let mut ends = BTreeMap::new();
    for river in &rivers {
        for pair in river.points.windows(2) {
            assert!(
                ground(pair[1]) < ground(pair[0]),
                "uphill in {:?}",
                river.id
            );
        }
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
        let end = *river.points.last().expect("a river has points");
        let around: Vec<[f32; 2]> = [
            (-1.0, -1.0),
            (0.0, -1.0),
            (1.0, -1.0),
            (-1.0, 0.0),
            (1.0, 0.0),
            (-1.0, 1.0),
            (0.0, 1.0),
            (1.0, 1.0),
        ]
        .iter()
        .map(|(dx, dy)| [end[0] + dx * STEP, end[1] + dy * STEP])
        .collect();
        let why = if ground(end) < 0.0 {
            "sea"
        } else if around.iter().any(|&next| !within(next)) {
            "edge"
        } else {
            assert!(
                around.iter().all(|&next| ground(next) >= ground(end)),
                "{:?} stops on a slope",
                river.id
            );
            "hollow"
        };
        *ends.entry(why).or_insert(0) += 1;
    }
    assert!(ends.contains_key("sea"), "{ends:?}");
}

#[test]
fn a_river_widens_from_its_source_to_its_mouth() {
    for river in rivers(&[area()]) {
        assert_eq!(river.values.first(), Some(&1.0));
        assert_eq!(river.values.last(), Some(&3.0));
        assert!(river.values.windows(2).all(|pair| pair[1] >= pair[0]));
    }
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
