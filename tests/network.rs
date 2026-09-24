//! Network stages: paths between sites over the ground, going round what is steep, joining every
//! site of a region, the same in any order.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::regions::{Curve, CurveId};
use wave_forge::stages::{Facts, GivenRow, Pack, PackError, Runtime, Site, SiteId, Value};
use wave_forge::{ChunkCoord, FocusPoint};

const SIZE: [u32; 2] = [8, 8];

/// Two villages 72 cells apart across a ridge 3 cells wide with a gap at y = 12, and roads between
/// villages in regions of 12 chunks.
fn ridge_pack(height: &str, dry: &str) -> String {
    format!(
        r#"(
    version: 1,
    tables: [(name: "villages", kind: Given(columns: [("x", Number), ("y", Number), ("size", Number)]))],
    stages: [
        (name: "height", kind: Field({height})),
        (name: "villages", kind: TableSites(table: "villages", height: "height", at: ("x", "y"), size: "size", max_size: 1)),
        (name: "roads", kind: Network(sites: "villages", height: "height", region: 12, climb: 4.0{dry})),
    ],
)"#
    )
}

const RIDGE: &str = "Mul(Constant(30.0), Mul(\
    Curve(Abs(Sub(X, Constant(48.0))), [(0.0, 1.0), (1.5, 1.0), (2.5, 0.0)]), \
    Curve(Abs(Sub(Y, Constant(12.0))), [(0.0, 0.0), (2.0, 0.0), (3.0, 1.0)])))";

fn village(id: u64, x: f32, y: f32) -> GivenRow {
    GivenRow {
        id,
        values: BTreeMap::from([
            ("x".to_owned(), Value::Number(x)),
            ("y".to_owned(), Value::Number(y)),
            ("size".to_owned(), Value::Number(1.0)),
        ]),
    }
}

fn region_chunks() -> Vec<ChunkCoord> {
    (0..12)
        .flat_map(|y| (0..12).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

/// Every curve `stage` holds over `chunks`, asked for in the order of `requests`, by id.
fn curves(
    runtime: &mut Runtime,
    stage: &str,
    requests: &[Vec<ChunkCoord>],
) -> BTreeMap<CurveId, Curve> {
    let mut out = BTreeMap::new();
    for request in requests {
        let focus: Vec<FocusPoint> = request.iter().map(|&c| FocusPoint::new(c, 0)).collect();
        runtime.request(&focus, &[stage]).expect("a stage");
        runtime.run_until_idle().expect("the stages run");
        for &chunk in request {
            for curve in runtime.curves(stage, chunk).expect("generated") {
                out.insert(curve.id.clone(), curve.clone());
            }
        }
    }
    out
}

fn ridge_roads(height: &str, dry: &str) -> (Runtime, Vec<Curve>) {
    let pack = Arc::new(Pack::parse(&ridge_pack(height, dry)).expect("a valid pack"));
    let mut runtime = Runtime::new(Arc::clone(&pack), 3, SIZE);
    let mut facts = Facts::new(pack, 3).expect("facts");
    facts
        .give(
            "villages",
            vec![village(1, 12.0, 48.0), village(2, 84.0, 48.0)],
        )
        .expect("villages");
    runtime.set_facts(facts).expect("facts");
    let roads = curves(&mut runtime, "roads", &[region_chunks()]);
    (runtime, roads.into_values().collect())
}

fn height_at(runtime: &Runtime, [x, y]: [f32; 2]) -> f32 {
    runtime.sample("height", [x, y]).expect("a sample")
}

#[test]
fn a_road_goes_round_a_ridge_through_its_gap() {
    let (runtime, roads) = ridge_roads(RIDGE, "");

    assert_eq!(roads.len(), 1, "{roads:?}");
    let road = &roads[0];
    assert!(
        road.points.iter().all(|&p| height_at(&runtime, p) < 1.0),
        "{road:?}"
    );
    assert!(
        road.points
            .iter()
            .any(|p| (p[1] - 12.5).abs() <= 2.0 && (p[0] - 48.0).abs() < 1.0)
    );
    for pair in road.points.windows(2) {
        let step = [
            (pair[1][0] - pair[0][0]).abs(),
            (pair[1][1] - pair[0][1]).abs(),
        ];
        assert!(
            step[0] <= 1.0 && step[1] <= 1.0 && step != [0.0, 0.0],
            "{pair:?}"
        );
    }
}

#[test]
fn a_road_over_flat_ground_runs_straight_and_stops_at_the_villages() {
    let (_, roads) = ridge_roads("Constant(0.0)", "");

    assert_eq!(roads.len(), 1);
    let road = &roads[0];
    // The villages are the chunks holding their rows' positions, columns 8 to 16 and 80 to 88 on
    // rows 48 to 56, so the road runs between their centres on row 52 from one to the other.
    assert!(road.points.iter().all(|p| p[1] == 52.5), "{road:?}");
    assert_eq!(road.points.first(), Some(&[16.5, 52.5]));
    assert_eq!(road.points.last(), Some(&[79.5, 52.5]));
    assert!(road.values.iter().all(|&width| width == 1.5));
}

#[test]
fn no_road_crosses_ground_below_dry() {
    let (_, roads) = ridge_roads("Constant(0.0)", ", dry: Some(1.0)");

    assert!(roads.is_empty(), "{roads:?}");
}

const SITES: &str = r#"(
    version: 1,
    stages: [
        (name: "height", kind: Field(Mul(Noise(frequency: 0.04, octaves: 2), Constant(12.0)))),
        (name: "towns", kind: Sites(height: "height", region: 4, size: (1, 2), chance: 1.0)),
        (name: "roads", kind: Network(sites: "towns", height: "height", region: 12)),
    ],
)"#;

fn sites_runtime() -> Runtime {
    Runtime::new(Arc::new(Pack::parse(SITES).expect("a valid pack")), 9, SIZE)
}

#[test]
fn every_town_of_a_region_is_joined_to_the_others() {
    let mut runtime = sites_runtime();
    let roads = curves(&mut runtime, "roads", &[region_chunks()]);
    let mut towns: BTreeMap<SiteId, Site> = BTreeMap::new();
    for chunk in region_chunks() {
        runtime
            .request(&[FocusPoint::new(chunk, 0)], &["towns"])
            .expect("a stage");
        runtime.run_until_idle().expect("the stages run");
        for site in runtime.sites("towns", chunk).expect("generated") {
            towns.insert(site.id.clone(), site.clone());
        }
    }
    let towns: Vec<Site> = towns.into_values().collect();
    // Which town a road's end lies next to: the one whose footprint is a step away.
    let next_to = |p: [f32; 2]| {
        towns
            .iter()
            .position(|town| {
                let near = |at: f32, low: i32, high: i32| {
                    at >= (low * 8) as f32 - 1.0 && at <= (high * 8) as f32 + 1.0
                };
                near(p[0], town.min.0, town.max.0) && near(p[1], town.min.1, town.max.1)
            })
            .expect("a road ends next to a town")
    };

    let mut group: Vec<usize> = (0..towns.len()).collect();
    fn root(group: &mut [usize], at: usize) -> usize {
        if group[at] == at {
            at
        } else {
            let top = root(group, group[at]);
            group[at] = top;
            top
        }
    }
    for road in roads.values() {
        let (a, b) = (
            next_to(road.points[0]),
            next_to(*road.points.last().expect("a road has points")),
        );
        let (a, b) = (root(&mut group, a), root(&mut group, b));
        group[a] = b;
    }

    assert!(towns.len() >= 6, "{} towns", towns.len());
    assert_eq!(roads.len(), towns.len() - 1);
    let first = root(&mut group, 0);
    assert!((0..towns.len()).all(|town| root(&mut group, town) == first));
}

#[test]
fn roads_are_the_same_in_any_order() {
    let all_at_once = curves(&mut sites_runtime(), "roads", &[region_chunks()]);
    let backwards: Vec<Vec<ChunkCoord>> = region_chunks()
        .into_iter()
        .rev()
        .map(|chunk| vec![chunk])
        .collect();

    let one_by_one = curves(&mut sites_runtime(), "roads", &backwards);

    assert!(!all_at_once.is_empty());
    assert_eq!(all_at_once, one_by_one);
}

#[test]
fn a_network_that_cannot_run_is_refused() {
    for change in [("region: 12", "region: 0"), ("climb: 4.0", "climb: -1.0")] {
        let pack = ridge_pack(RIDGE, "").replace(change.0, change.1);

        let result = Pack::parse(&pack);

        assert!(
            matches!(&result, Err(PackError::Invalid { stage, .. }) if stage == "roads"),
            "{change:?}: {result:?}"
        );
    }
}
