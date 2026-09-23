//! Filter stages that read a neighbourhood: terrain delta, how uneven a field is around a column,
//! and biome area, whether a column lies at the edge of its category or in its middle, as Valheim's
//! location filters use them.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::{Pack, PackError, Runtime};
use wave_forge::{ChunkCoord, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "ground", kind: Field(Mul(Noise(frequency: 0.1, octaves: 2), Constant(10.0)))),
        (name: "delta", kind: Delta(input: "ground", radius: 3)),
        (name: "side", kind: Rules(rules: [(category: "west", when: [Less(X, Constant(20.0))])], otherwise: "east")),
        (name: "area", kind: Area(input: "side", distance: 2)),
        (name: "edges", kind: Field(Is("area", ["edge"]))),
    ],
)"#;

const SIZE: [u32; 2] = [8, 8];

fn pack() -> Arc<Pack> {
    Arc::new(Pack::parse(PACK).expect("a valid pack"))
}

/// Every column's value of `stage` over the chunks from (0, 0) to (5, 5), by world column.
fn columns(stage: &str) -> BTreeMap<(i64, i64), f32> {
    let mut runtime = Runtime::new(pack(), 3, SIZE);
    let chunks: Vec<ChunkCoord> = (0..6)
        .flat_map(|y| (0..6).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect();
    let focus: Vec<FocusPoint> = chunks.iter().map(|&c| FocusPoint::new(c, 0)).collect();
    runtime.request(&focus, &[stage]).expect("a stage");
    runtime.run_until_idle().expect("the stages run");
    let mut out = BTreeMap::new();
    for chunk in chunks {
        let values: Vec<f32> = match runtime.field(stage, chunk) {
            Some(field) => field.values.clone(),
            None => runtime
                .categories(stage, chunk)
                .expect("generated")
                .values
                .iter()
                .map(|&c| f32::from(c))
                .collect(),
        };
        for (i, value) in values.into_iter().enumerate() {
            let (x, y) = (i as i64 % 8, i as i64 / 8);
            out.insert(
                (i64::from(chunk.x) * 8 + x, i64::from(chunk.y) * 8 + y),
                value,
            );
        }
    }
    out
}

#[test]
fn a_delta_is_the_highest_value_less_the_lowest_within_its_radius() {
    let ground = columns("ground");

    let delta = columns("delta");

    for y in 3..45 {
        for x in 3..45 {
            let around: Vec<f32> = (-3..=3)
                .flat_map(|dy| (-3..=3).map(move |dx| (x + dx, y + dy)))
                .map(|at| ground[&at])
                .collect();
            let high = around.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let low = around.iter().copied().fold(f32::INFINITY, f32::min);
            assert_eq!(delta[&(x, y)], high - low, "({x}, {y})");
        }
    }
}

#[test]
fn a_column_is_at_the_edge_where_a_neighbour_that_far_out_has_another_category() {
    let area = columns("area");

    // Columns 0 to 19 are west, 20 on east; two columns either side of that line see across it.
    for y in 2..46 {
        for x in 2..46 {
            let expected = u8::from((18..=21).contains(&x));
            assert_eq!(area[&(x, y)], f32::from(expected), "({x}, {y})");
        }
    }
}

#[test]
fn an_area_names_its_categories_and_a_field_tests_them() {
    let pack = pack();

    let names = pack.kind("area").expect("a stage").categories();
    let edges = columns("edges");

    assert_eq!(names, vec!["median", "edge"]);
    assert_eq!(edges[&(19, 10)], 1.0);
    assert_eq!(edges[&(10, 10)], 0.0);
}

#[test]
fn a_filter_reaches_as_far_as_its_radius_and_samples_as_its_chunks_hold() {
    let pack = pack();
    let sampler = Runtime::new(Arc::clone(&pack), 3, SIZE);
    let (delta, area) = (columns("delta"), columns("area"));

    let reach_delta = pack.reach("delta", SIZE).expect("a stage");
    let reach_area = pack.reach("area", SIZE).expect("a stage");

    assert_eq!(reach_delta["ground"], 3);
    assert_eq!(reach_area["side"], 2);
    for (x, y) in [(5, 7), (19, 30), (40, 2)] {
        let at = [x as f32 + 0.5, y as f32 + 0.5];
        assert_eq!(sampler.sample("delta", at), Ok(delta[&(x, y)]));
        assert_eq!(sampler.sample("area", at), Ok(area[&(x, y)]));
    }
}

#[test]
fn an_area_measured_no_distance_out_is_refused() {
    let result = Pack::parse(
        r#"(version: 1, stages: [
            (name: "side", kind: Rules(rules: [], otherwise: "all")),
            (name: "area", kind: Area(input: "side", distance: 0)),
        ])"#,
    );

    assert!(
        matches!(&result, Err(PackError::Invalid { stage, .. }) if stage == "area"),
        "{result:?}"
    );
}
