//! Apply stages: curves drawn into a height field, a road levelled across and a river carved in,
//! from a table's rows or from a region job, the same whatever order chunks are asked for in.
//!
//! The ground is a tilted plane, so the height a curve takes is easy to state: a column within a
//! road's radius takes the height at the nearest point of the road's centre line.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::regions::{Attempt, Curve, CurveId, Edge, RegionInput, RegionJob};
use wave_forge::stages::{Facts, GivenRow, Pack, Runtime, StageError, Value};
use wave_forge::{ChunkCoord, FocusPoint};

const SIZE: [u32; 2] = [8, 8];

const PACK: &str = r#"(
    version: 1,
    tables: [(name: "roads", kind: Given(columns: [
        ("x0", Number), ("y0", Number), ("x1", Number), ("y1", Number), ("width", Number),
    ]))],
    stages: [
        (name: "ground", kind: Field(Add(Mul(X, Constant(0.1)), Mul(Y, Constant(0.5))))),
        (name: "roads", kind: TableCurves(table: "roads", from: ("x0", "y0"), to: ("x1", "y1"), radius: "width")),
        (name: "paved", kind: Apply(height: "ground", curves: "roads", max_radius: 4, blend: 3, profile: Level)),
        (name: "ditched", kind: Apply(height: "ground", curves: "roads", max_radius: 4, profile: Carve(3.0))),
    ],
)"#;

fn road(id: u64, from: (f32, f32), to: (f32, f32), width: f32) -> GivenRow {
    GivenRow {
        id,
        values: BTreeMap::from([
            ("x0".to_owned(), Value::Number(from.0)),
            ("y0".to_owned(), Value::Number(from.1)),
            ("x1".to_owned(), Value::Number(to.0)),
            ("y1".to_owned(), Value::Number(to.1)),
            ("width".to_owned(), Value::Number(width)),
        ]),
    }
}

fn runtime(roads: Vec<GivenRow>) -> Result<Runtime, StageError> {
    let pack = Arc::new(Pack::parse(PACK).expect("a valid pack"));
    let mut facts = Facts::new(Arc::clone(&pack), 9)?;
    facts.give("roads", roads)?;
    let mut runtime = Runtime::new(pack, 9, SIZE);
    runtime.set_facts(facts)?;
    Ok(runtime)
}

/// `stage`'s value at every column of the chunks `chunks`, asked for in the order of `requests`.
fn columns(
    runtime: &mut Runtime,
    stage: &str,
    requests: &[Vec<ChunkCoord>],
) -> BTreeMap<(i64, i64), f32> {
    let mut out = BTreeMap::new();
    for request in requests {
        let focus: Vec<FocusPoint> = request.iter().map(|&c| FocusPoint::new(c, 0)).collect();
        runtime.request(&focus, &[stage]).expect("a stage");
        runtime.run_until_idle().expect("the stages run");
        for &chunk in request {
            let field = runtime.field(stage, chunk).expect("generated");
            for (i, &value) in field.values.iter().enumerate() {
                let (x, y) = (i as i64 % 8, i as i64 / 8);
                out.insert(
                    (i64::from(chunk.x) * 8 + x, i64::from(chunk.y) * 8 + y),
                    value,
                );
            }
        }
    }
    out
}

fn area() -> Vec<ChunkCoord> {
    (0..8)
        .flat_map(|y| (0..8).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

fn ground(x: i64, y: i64) -> f32 {
    (x as f32 + 0.5) * 0.1 + (y as f32 + 0.5) * 0.5
}

#[test]
fn a_road_levels_the_ground_across_it_and_blends_back_beyond_its_radius() {
    let mut runtime = runtime(vec![road(1, (0.0, 20.0), (60.0, 20.0), 2.0)]).expect("roads");

    let paved = columns(&mut runtime, "paved", &[area()]);

    for x in 4..56 {
        // Column centres 18.5 to 21.5 lie within 2 of the centre line, at y = 20.
        for y in 18..=21 {
            assert_eq!(paved[&(x, y)], ground(x, 20), "({x}, {y})");
        }
        for y in [15, 16, 17, 22, 23, 24] {
            let (base, level) = (ground(x, y), ground(x, 20));
            let value = paved[&(x, y)];
            assert!(
                value >= base.min(level) && value <= base.max(level) && value != base,
                "({x}, {y}): {value} between {base} and {level}"
            );
        }
        for y in [0, 13, 26, 40] {
            assert_eq!(paved[&(x, y)], ground(x, y), "({x}, {y})");
        }
    }
}

#[test]
fn a_carve_lowers_the_ground_under_its_curve_by_its_depth() {
    let mut runtime = runtime(vec![road(1, (0.0, 20.0), (60.0, 20.0), 2.0)]).expect("roads");

    let ditched = columns(&mut runtime, "ditched", &[area()]);

    for x in 4..56 {
        assert_eq!(ditched[&(x, 20)], ground(x, 20) - 3.0);
        assert_eq!(
            ditched[&(x, 23)],
            ground(x, 23),
            "no blend: untouched beyond the radius"
        );
    }
}

#[test]
fn curves_crossing_chunks_draw_the_same_in_any_order() {
    let roads = || {
        vec![
            road(1, (3.0, 5.0), (61.0, 50.0), 3.0),
            road(2, (10.0, 60.0), (50.0, 2.0), 1.5),
        ]
    };
    let at_once = columns(&mut runtime(roads()).expect("roads"), "paved", &[area()]);
    let one_by_one: Vec<Vec<ChunkCoord>> = area().into_iter().map(|c| vec![c]).collect();
    let backwards: Vec<Vec<ChunkCoord>> = area().into_iter().rev().map(|c| vec![c]).collect();

    let forwards = columns(&mut runtime(roads()).expect("roads"), "paved", &one_by_one);
    let reversed = columns(&mut runtime(roads()).expect("roads"), "paved", &backwards);

    assert_eq!(at_once, forwards);
    assert_eq!(at_once, reversed);
    assert!(
        at_once
            .iter()
            .any(|(&(x, y), &value)| value != ground(x, y))
    );
}

#[test]
fn a_road_wider_than_its_apply_stages_allow_is_refused_by_row() {
    let result = runtime(vec![road(7, (0.0, 20.0), (60.0, 20.0), 4.5)]).map(|_| ());

    assert!(
        matches!(&result, Err(StageError::Table { table, message }) if table == "roads" && message.contains("[7]")),
        "{result:?}"
    );
}

#[test]
fn moving_a_road_drops_only_the_chunks_within_reach_of_where_it_was_and_is() {
    let pack = Arc::new(Pack::parse(PACK).expect("a valid pack"));
    let mut facts = Facts::new(Arc::clone(&pack), 9).expect("facts");
    facts
        .give("roads", vec![road(1, (4.0, 4.0), (4.0, 60.0), 1.0)])
        .expect("roads");
    let mut runtime = Runtime::new(pack, 9, SIZE);
    runtime.set_facts(facts.clone()).expect("facts");
    columns(&mut runtime, "paved", &[area()]);

    facts
        .give("roads", vec![road(1, (12.0, 4.0), (12.0, 60.0), 1.0)])
        .expect("roads");
    let dropped = runtime.set_facts(facts).expect("facts");

    // The road ran through chunk column 0 and now runs through 1; a paved chunk reads 7 cells
    // around it, so column 2 sees the new road too, and nothing further.
    assert!(!dropped.is_empty());
    assert!(dropped.iter().all(|(_, chunk)| chunk.x <= 2), "{dropped:?}");
}

/// A river across each region from west to east, entering and leaving at rows hashed from the
/// edge it crosses, so neighbouring regions meet without reading each other.
struct Rivers;

impl RegionJob for Rivers {
    fn run(&self, input: &RegionInput<'_>) -> Result<Attempt, StageError> {
        let ([x0, y0], [x1, y1]) = input.columns();
        let rows = (y1 - y0 + 1) as u32;
        let row = |hash: u32| y0 as f32 + (hash % rows) as f32 + 0.5;
        Ok(Attempt::Accepted(vec![Curve {
            id: CurveId::Region {
                region: input.region(),
                index: 0,
            },
            points: vec![
                [x0 as f32, row(input.edge_hash(Edge::West, 0))],
                [(x0 + x1) as f32 / 2.0, row(input.hash(1))],
                [x1 as f32 + 1.0, row(input.edge_hash(Edge::East, 0))],
            ],
            values: vec![2.0, 1.0, 2.0],
        }]))
    }
}

#[test]
fn rivers_carve_their_beds_without_seams_across_region_borders() {
    const RIVERS: &str = r#"(
        version: 1,
        stages: [
            (name: "ground", kind: Field(Mul(Noise(frequency: 0.05, octaves: 2), Constant(10.0)))),
            (name: "rivers", kind: Region(job: "rivers", region: 2)),
            (name: "carved", kind: Apply(height: "ground", curves: "rivers", max_radius: 3, blend: 2, profile: Carve(1.5))),
        ],
    )"#;
    let make = || {
        Runtime::new(
            Arc::new(Pack::parse(RIVERS).expect("a valid pack")),
            4,
            SIZE,
        )
        .with_region_job("rivers", Rivers)
    };
    let one_by_one: Vec<Vec<ChunkCoord>> = area().into_iter().map(|c| vec![c]).collect();

    let at_once = columns(&mut make(), "carved", &[area()]);
    let chunk_by_chunk = columns(&mut make(), "carved", &one_by_one);
    let mut direct = make();
    let ground = columns(&mut direct, "ground", &[area()]);

    assert_eq!(at_once, chunk_by_chunk);
    // Every region border in the area is crossed by a river, so every one has carved columns
    // on both sides of it.
    for border in [16, 32, 48] {
        let carved = |x: i64| (0..64).any(|y| at_once[&(x, y)] < ground[&(x, y)]);
        assert!(
            carved(border - 1) && carved(border),
            "the border at x = {border}"
        );
    }
}
