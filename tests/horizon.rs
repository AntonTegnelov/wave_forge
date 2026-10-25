//! The Horizon stage: how much of the sky the ground around each column hides, from the angle of
//! its horizon in each of several directions out to a bounded reach, the large-scale occlusion
//! terrain shades with (Timonen and Westerholm 2010).

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::{Pack, PackError, Runtime};
use wave_forge::{ChunkCoord, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "flat", kind: Field(Constant(3.0))),
        (name: "open", kind: Horizon(input: "flat", reach: 8)),
        (name: "bowl", kind: Field(Mul(Distance((24.5, 24.5)), Constant(0.5)))),
        (name: "in_bowl", kind: Horizon(input: "bowl", reach: 8)),
        (name: "peak", kind: Field(Max(Sub(Constant(50.0), Mul(Distance((40.5, 24.5)), Constant(10.0))), Constant(0.0)))),
        (name: "by_peak", kind: Horizon(input: "peak", reach: 8, directions: 16)),
        (name: "coarse_bowl", scale: 4, kind: Field(Mul(Distance((26.0, 26.0)), Constant(0.5)))),
        (name: "in_coarse_bowl", scale: 4, kind: Horizon(input: "coarse_bowl", reach: 4)),
    ],
)"#;

const SIZE: [u32; 2] = [8, 8];

/// How much a bowl rising half a cell for every cell out hides from its lowest column: the sine of
/// its walls' angle in every direction.
fn bowl_occlusion() -> f32 {
    libm::sinf(libm::atanf(0.5))
}

fn pack() -> Arc<Pack> {
    Arc::new(Pack::parse(PACK).expect("a valid pack"))
}

/// Every column's value of `stage` over the chunks from (0, 0) to (7, 7), by column of the stage.
fn columns(stage: &str) -> BTreeMap<(i64, i64), f32> {
    let mut runtime = Runtime::new(pack(), 3, SIZE);
    let chunks: Vec<ChunkCoord> = (0..8)
        .flat_map(|y| (0..8).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect();
    let focus: Vec<FocusPoint> = chunks.iter().map(|&c| FocusPoint::new(c, 0)).collect();
    runtime.request(&focus, &[stage]).expect("a stage");
    runtime.run_until_idle().expect("the stages run");
    let mut out = BTreeMap::new();
    for chunk in chunks {
        let Some(field) = runtime.field(stage, chunk) else {
            continue;
        };
        for (i, &value) in field.values.iter().enumerate() {
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
fn flat_ground_hides_none_of_the_sky() {
    let open = columns("open");

    assert_eq!(open.len(), 64 * 64, "every column of the 8 by 8 chunks");
    assert!(open.values().all(|&value| value == 0.0), "{open:?}");
}

#[test]
fn the_lowest_column_of_a_bowl_sees_its_walls_at_their_angle_all_round() {
    let in_bowl = columns("in_bowl");

    let lowest = in_bowl[&(24, 24)];

    assert!(
        (lowest - bowl_occlusion()).abs() < 1e-4,
        "{lowest} against {}",
        bowl_occlusion()
    );
}

#[test]
fn a_column_up_a_bowls_wall_sees_more_sky_than_its_lowest() {
    let in_bowl = columns("in_bowl");

    let (lowest, up_the_wall) = (in_bowl[&(24, 24)], in_bowl[&(30, 24)]);

    assert!(
        up_the_wall > 0.0 && up_the_wall < lowest,
        "{up_the_wall} against {lowest}"
    );
}

#[test]
fn a_peak_beyond_the_reach_hides_nothing_and_one_within_it_does() {
    let by_peak = columns("by_peak");

    // A cone 5 columns in radius stands at (40, 24); the reach is 8 columns.
    let (beyond, within) = (by_peak[&(26, 24)], by_peak[&(34, 24)]);

    assert_eq!(beyond, 0.0);
    assert!(within > 0.0, "{within}");
}

#[test]
fn a_coarse_stage_measures_its_angles_in_cells_not_in_its_columns() {
    // The same bowl at a scale of 4: its lowest column, at (26, 26) in cells, is coarse column 6.
    let in_coarse_bowl = columns("in_coarse_bowl");

    let lowest = in_coarse_bowl[&(6, 6)];

    assert!(
        (lowest - bowl_occlusion()).abs() < 1e-4,
        "{lowest} against {}",
        bowl_occlusion()
    );
}

#[test]
fn a_horizon_reaches_as_far_as_its_reach_and_samples_as_its_chunks_hold() {
    let pack = pack();
    let sampler = Runtime::new(Arc::clone(&pack), 3, SIZE);
    let in_bowl = columns("in_bowl");

    let reach = pack.reach("in_bowl", SIZE).expect("a stage");

    assert_eq!(reach["bowl"], [8; 2]);
    for (x, y) in [(24, 24), (30, 19), (5, 40), (41, 63)] {
        let at = [x as f32 + 0.5, y as f32 + 0.5];
        assert_eq!(sampler.sample("in_bowl", at), Ok(in_bowl[&(x, y)]));
    }
}

#[test]
fn a_horizon_with_no_reach_or_too_few_directions_is_refused() {
    let parse = |reach: u32, directions: u32| {
        Pack::parse(&format!(
            r#"(version: 1, stages: [
                (name: "flat", kind: Field(Constant(0.0))),
                (name: "open", kind: Horizon(input: "flat", reach: {reach}, directions: {directions})),
            ])"#
        ))
    };

    assert!(matches!(parse(0, 8), Err(PackError::Invalid { .. })));
    assert!(matches!(parse(8, 2), Err(PackError::Invalid { .. })));
    assert!(parse(8, 4).is_ok());
}
