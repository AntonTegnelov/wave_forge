//! Erode stages: gullies cut into a height field from its slope around each column alone, the same
//! whichever order chunks are asked for in, the same sampled as generated, reaching only as far as
//! the slope is fitted, and at a coarse scale following the fine field's erosion.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::{Pack, PackError, Runtime};
use wave_forge::{ChunkCoord, FocusPoint};

const SIZE: [u32; 2] = [8, 8];

/// Hills, eroded at full detail with four octaves and with the first two alone, and the same hills
/// at a scale of 4, eroded with the first two from the same named lattice, as a far ground would
/// be. The hills' finest features are a few coarse columns wide, so the coarse field resolves
/// them, and the two with the first octaves alone cut fully over most of the hills: a feature the
/// coarse field cannot resolve, or a slope in the band where gullies fade in, turns the two scales'
/// fitted slopes apart.
const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "height", kind: Field(Mul(Noise(frequency: 0.02, octaves: 2, name: Some("hills")), Constant(40.0)))),
        (name: "eroded", kind: Erode(input: "height", spacing: 16.0, octaves: 4, depth: 2.0, slope: 0.3)),
        (name: "eroded_twice", kind: Erode(input: "height", spacing: 16.0, octaves: 2, depth: 2.0, slope: 0.1, smooth: 4, name: Some("gullies"))),
        (name: "coarse", scale: 4, kind: Field(Mul(Noise(frequency: 0.02, octaves: 2, name: Some("hills")), Constant(40.0)))),
        (name: "coarse_eroded", scale: 4, kind: Erode(input: "coarse", spacing: 16.0, octaves: 2, depth: 2.0, slope: 0.1, smooth: 1, name: Some("gullies"))),
    ],
)"#;

fn pack() -> Arc<Pack> {
    Arc::new(Pack::parse(PACK).expect("a valid pack"))
}

fn area() -> Vec<ChunkCoord> {
    (-3..3)
        .flat_map(|y| (-3..3).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

/// `stage`'s value at every column of the chunks, asked for in the order of `requests`.
fn columns(stage: &str, requests: &[Vec<ChunkCoord>]) -> BTreeMap<(i64, i64), f32> {
    let mut runtime = Runtime::new(pack(), 4, SIZE);
    let mut out = BTreeMap::new();
    for request in requests {
        let focus: Vec<FocusPoint> = request.iter().map(|&c| FocusPoint::new(c, 0)).collect();
        runtime.request(&focus, &[stage]).expect("a stage");
        runtime.run_until_idle().expect("the stages run");
        for &chunk in request {
            let field = runtime.field(stage, chunk).expect("generated");
            for (i, &value) in field.values.iter().enumerate() {
                let x = i64::from(chunk.x) * 8 + (i % 8) as i64;
                let y = i64::from(chunk.y) * 8 + (i / 8) as i64;
                out.insert((x, y), value);
            }
        }
    }
    out
}

#[test]
fn eroded_ground_is_the_same_at_once_and_chunk_by_chunk_in_either_order() {
    let one_by_one: Vec<Vec<ChunkCoord>> = area().into_iter().map(|c| vec![c]).collect();
    let backwards: Vec<Vec<ChunkCoord>> = area().into_iter().rev().map(|c| vec![c]).collect();

    let at_once = columns("eroded", &[area()]);
    let forwards = columns("eroded", &one_by_one);
    let reversed = columns("eroded", &backwards);

    assert_eq!(at_once, forwards);
    assert_eq!(at_once, reversed);
}

#[test]
fn erosion_cuts_into_sloping_ground() {
    let height = columns("height", &[area()]);
    let eroded = columns("eroded", &[area()]);

    let moved: Vec<f32> = height.iter().map(|(at, h)| eroded[at] - h).collect();
    let deepest = moved.iter().copied().fold(0.0_f32, f32::min);
    let highest = moved.iter().copied().fold(0.0_f32, f32::max);

    assert!(
        deepest < -1.0 && highest > 1.0,
        "moved from {deepest} to {highest}"
    );
}

#[test]
fn a_sampled_column_is_the_generated_one() {
    let mut runtime = Runtime::new(pack(), 4, SIZE);
    let chunk = ChunkCoord::new(1, -2, 0);
    runtime
        .request(&[FocusPoint::new(chunk, 0)], &["eroded"])
        .expect("a stage");
    runtime.run_until_idle().expect("the stages run");
    let field = runtime.field("eroded", chunk).expect("generated").clone();

    for (i, &value) in field.values.iter().enumerate() {
        let at = [8.0 + (i % 8) as f32 + 0.5, -16.0 + (i / 8) as f32 + 0.5];
        let sampled = runtime.sample("eroded", at).expect("a sample");
        assert_eq!(sampled, value, "at {at:?}");
    }
}

#[test]
fn the_reach_is_the_slope_fit() {
    let narrow = pack().reach("eroded", SIZE).expect("a stage");
    let wide = Pack::parse(&PACK.replace("slope: 0.3)", "slope: 0.3, smooth: 9)"))
        .expect("a valid pack")
        .reach("eroded", SIZE)
        .expect("a stage");

    assert_eq!(narrow.get("height"), Some(&2));
    assert_eq!(wide.get("height"), Some(&9));
}

#[test]
fn at_a_coarse_scale_the_first_octaves_follow_the_fine_erosion() {
    let mut runtime = Runtime::new(pack(), 4, SIZE);
    let chunk = ChunkCoord::new(0, 0, 0);
    runtime
        .request(&[FocusPoint::new(chunk, 0)], &["coarse_eroded"])
        .expect("a stage");
    runtime.run_until_idle().expect("the stages run");
    let coarse = runtime
        .field("coarse_eroded", chunk)
        .expect("generated")
        .clone();
    let fine = |at: [f32; 2]| {
        let first_two = runtime.sample("eroded_twice", at).expect("eroded");
        (first_two, runtime.sample("height", at).expect("height"))
    };

    let (mut apart, mut erosion) = (0.0, 0.0);
    for (i, &value) in coarse.values.iter().enumerate() {
        let at = [(i % 8) as f32 * 4.0 + 2.0, (i / 8) as f32 * 4.0 + 2.0];
        let (eroded, height) = fine(at);
        apart += (value - eroded).abs();
        erosion += (eroded - height).abs();
    }

    assert!(
        apart < 0.5 * erosion,
        "{apart} apart against {erosion} of erosion"
    );
}

#[test]
fn loading_refuses_gullies_that_cannot_be_cut() {
    let refused = [
        "spacing: 0.0, octaves: 4, depth: 2.0, slope: 0.3",
        "spacing: 16.0, octaves: 0, depth: 2.0, slope: 0.3",
        "spacing: 16.0, octaves: 9, depth: 2.0, slope: 0.3",
        "spacing: 16.0, octaves: 4, depth: -1.0, slope: 0.3",
        "spacing: 16.0, octaves: 4, depth: 2.0, slope: 0.0",
        "spacing: 16.0, octaves: 4, depth: 2.0, slope: 0.3, gain: 1.5",
        "spacing: 16.0, octaves: 4, depth: 2.0, slope: 0.3, smooth: 0",
    ];

    for parameters in refused {
        let text = PACK.replace(
            "spacing: 16.0, octaves: 4, depth: 2.0, slope: 0.3",
            parameters,
        );
        let result = Pack::parse(&text);

        assert!(
            matches!(&result, Err(PackError::Invalid { stage, .. }) if stage == "eroded"),
            "{parameters}: {result:?}"
        );
    }
}

#[test]
#[ignore = "a measurement: run in release with --ignored --nocapture"]
fn what_erosion_costs_a_chunk() {
    let chunks: Vec<ChunkCoord> = (0..16)
        .flat_map(|y| (0..16).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect();
    let focus: Vec<FocusPoint> = chunks.iter().map(|&c| FocusPoint::new(c, 0)).collect();
    for octaves in [4, 6] {
        let text = PACK.replace(
            "spacing: 16.0, octaves: 4, depth: 2.0, slope: 0.3",
            &format!("spacing: 16.0, octaves: {octaves}, depth: 2.0, slope: 0.3"),
        );
        let mut runtime =
            Runtime::new(Arc::new(Pack::parse(&text).expect("a valid pack")), 4, SIZE);

        runtime.request(&focus, &["eroded"]).expect("a stage");
        runtime.run_until_idle().expect("the stages run");

        let timings = runtime.timings();
        let per_chunk = |name: &str| {
            let (_, timing) = timings
                .iter()
                .find(|(stage, _)| stage == name)
                .expect("timed");
            timing.ms / timing.products.max(1) as f64
        };
        eprintln!(
            "erode: {octaves} octaves, smooth 2, chunks of 8 by 8 columns: {:.3} ms a chunk, its input {:.3} ms",
            per_chunk("eroded"),
            per_chunk("height")
        );
    }
}
