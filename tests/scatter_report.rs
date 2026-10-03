//! A Scatter stage's report (docs/product/user-stories.md, N5): every candidate of a chunk with
//! whether it became a point or which modifier rejected it, decided as generating decides, with
//! what the modifiers read there.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::{Pack, Rejection, Runtime, StageError};
use wave_forge::{ChunkCoord, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "height", kind: Field(Mul(Noise(frequency: 0.15, octaves: 2), Constant(10.0)))),
        (name: "mask", kind: Field(Noise(frequency: 0.2, octaves: 1))),
        (name: "trees", kind: Scatter(kind: "tree", height: "height", spacing: 2, count: (1, 2),
            chance: 0.8, between: Some((2.0, 8.0)), max_slope: Some(1.5), apart: 3,
            when: [Greater(Input("mask"), Constant(0.35))])),
    ],
)"#;

fn area() -> Vec<ChunkCoord> {
    (-2..2)
        .flat_map(|y| (-2..2).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

fn runtime() -> Runtime {
    let mut runtime = Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        9,
        [8, 8],
    );
    let focus: Vec<FocusPoint> = area().iter().map(|&c| FocusPoint::new(c, 0)).collect();
    runtime
        .request(&focus, &["trees", "height", "mask"])
        .expect("the stages");
    runtime.run_until_idle().expect("the stages run");
    runtime
}

/// A field's value at the column holding `at`.
fn value(runtime: &Runtime, stage: &str, column: (i64, i64)) -> f32 {
    let chunk = ChunkCoord::new(
        column.0.div_euclid(8) as i32,
        column.1.div_euclid(8) as i32,
        0,
    );
    runtime
        .field(stage, chunk)
        .expect("generated")
        .get(column.0.rem_euclid(8) as u32, column.1.rem_euclid(8) as u32)
}

#[test]
fn a_chunks_kept_candidates_are_exactly_its_points() {
    let runtime = runtime();

    for chunk in area() {
        let report = runtime
            .scatter_report("trees", chunk)
            .expect("a Scatter stage");

        let mut kept: Vec<[f32; 2]> = report
            .iter()
            .filter(|judged| judged.verdict.is_ok())
            .map(|judged| judged.at)
            .collect();
        let mut points: Vec<[f32; 2]> = runtime
            .points("trees", chunk)
            .expect("generated")
            .iter()
            .map(|point| [point.position[0], point.position[1]])
            .collect();
        kept.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        points.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        assert_eq!(kept, points, "{chunk:?}");
    }
}

#[test]
fn every_rejection_names_the_modifier_that_failed() {
    let runtime = runtime();
    let mut counts: BTreeMap<Rejection, usize> = BTreeMap::new();

    for chunk in area() {
        let report = runtime
            .scatter_report("trees", chunk)
            .expect("a Scatter stage");
        for judged in &report {
            let column = (judged.at[0].floor() as i64, judged.at[1].floor() as i64);
            let height = value(&runtime, "height", column);
            let Err(rejection) = judged.verdict else {
                continue;
            };
            *counts.entry(rejection).or_default() += 1;
            match rejection {
                Rejection::Height => {
                    assert!(!(2.0..=8.0).contains(&height), "{judged:?} at {height}");
                }
                Rejection::Slope => {
                    let slope = |a: (i64, i64), b: (i64, i64)| {
                        (value(&runtime, "height", a) - value(&runtime, "height", b)).abs() / 2.0
                    };
                    let (x, y) = column;
                    let steepest = slope((x + 1, y), (x - 1, y)).max(slope((x, y + 1), (x, y - 1)));
                    assert!(steepest > 1.5, "{judged:?} at a slope of {steepest}");
                }
                Rejection::Condition(0) => {
                    assert!(value(&runtime, "mask", column) <= 0.35, "{judged:?}");
                }
                Rejection::Spacing => {
                    // Some other candidate that passed lies closer than `apart`: in the report,
                    // or beyond the chunk's edge.
                    let (x0, y0) = (chunk.x as f32 * 8.0, chunk.y as f32 * 8.0);
                    let near_edge = [
                        judged.at[0] - x0,
                        x0 + 8.0 - judged.at[0],
                        judged.at[1] - y0,
                        y0 + 8.0 - judged.at[1],
                    ]
                    .iter()
                    .any(|&d| d < 3.0);
                    let crowded = report.iter().any(|other| {
                        !matches!(other.verdict, Err(reason) if reason != Rejection::Spacing)
                            && other.at != judged.at
                            && (other.at[0] - judged.at[0]).hypot(other.at[1] - judged.at[1]) < 3.0
                    });
                    assert!(crowded || near_edge, "{judged:?}");
                }
                Rejection::Chance => {}
                other => panic!("{judged:?} rejected by {other:?}, which the stage lacks"),
            }
        }
    }
    for reason in [
        Rejection::Chance,
        Rejection::Height,
        Rejection::Slope,
        Rejection::Condition(0),
        Rejection::Spacing,
    ] {
        assert!(counts.get(&reason).copied().unwrap_or(0) > 0, "{counts:?}");
    }
}

// A viewer shows the readings to say why a candidate went; they have to be the ones it was judged by.
#[test]
fn every_verdict_agrees_with_what_the_modifiers_read() {
    let runtime = runtime();
    let mut seen = BTreeMap::new();

    for chunk in area() {
        for judged in runtime
            .scatter_report("trees", chunk)
            .expect("a Scatter stage")
        {
            let column = (judged.at[0].floor() as i64, judged.at[1].floor() as i64);
            let readings = &judged.readings;
            let slope = readings.slope.expect("the stage has a max_slope");
            let (mask, mask_holds) = readings.conditions[0];
            let in_range = (2.0..=8.0).contains(&readings.height);

            assert_eq!(readings.height, value(&runtime, "height", column));
            assert_eq!(mask, value(&runtime, "mask", column));
            assert_eq!(mask_holds, mask > 0.35);
            assert_eq!(readings.water_depth, None);
            match judged.verdict {
                Err(Rejection::Height) => assert!(!in_range, "{judged:?}"),
                Err(Rejection::Slope) => assert!(in_range && slope > 1.5, "{judged:?}"),
                Err(Rejection::Condition(0)) => {
                    assert!(in_range && slope <= 1.5 && !mask_holds, "{judged:?}");
                }
                Ok(()) | Err(Rejection::Spacing) => {
                    assert!(in_range && slope <= 1.5 && mask_holds, "{judged:?}");
                }
                Err(Rejection::Chance) => {}
                Err(other) => panic!("a stage without it rejected by {other:?}"),
            }
            *seen.entry(format!("{:?}", judged.verdict)).or_insert(0) += 1;
        }
    }

    // Every verdict the stage can reach was checked, not only a few.
    assert!(seen.len() >= 6, "{seen:?}");
}

#[test]
fn a_report_of_a_stage_that_scatters_nothing_is_refused() {
    let runtime = runtime();

    let result = runtime.scatter_report("height", ChunkCoord::new(0, 0, 0));

    assert!(matches!(result, Err(StageError::Edit(_))), "{result:?}");
}
