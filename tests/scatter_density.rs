//! A Scatter stage's density and clumps (docs/reference/packs.md, "Scatter"): points kept as often
//! as a density field says, so a forest thins toward its edge, the same in any order; and a group
//! of one size.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::{Pack, Point, Rejection, Runtime};
use wave_forge::{ChunkCoord, FocusPoint};

/// Trees whose density falls from 1 at x = 0 to 0 at x = 160 cells, with no spacing, so the
/// density alone decides; and clumps of ferns.
const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "height", kind: Field(Constant(0.0))),
        (name: "forest", kind: Field(Clamp(Sub(Constant(1.0), Mul(X, Constant(0.00625))), 0.0, 1.0))),
        (name: "trees", kind: Scatter(kind: "tree", height: "height", spacing: 2, density: Some("forest"))),
        (name: "ferns", kind: Scatter(kind: "fern", height: "height", spacing: 8,
            group: Some((size: (4, 8), radius: 2.0)), scale: (0.95, 1.05), clump: (0.5, 2.0))),
    ],
)"#;

const SIZE: [u32; 2] = [8, 8];

/// The chunks of a strip 160 cells long along x and 40 wide.
fn strip() -> Vec<ChunkCoord> {
    (0..5)
        .flat_map(|y| (0..20).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

fn points(stage: &str, order: &[Vec<ChunkCoord>]) -> BTreeMap<ChunkCoord, Vec<Point>> {
    let mut runtime = Runtime::new(Arc::new(Pack::parse(PACK).expect("a valid pack")), 3, SIZE);
    let mut points = BTreeMap::new();
    for request in order {
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

#[test]
fn a_forest_thins_as_its_density_falls() {
    let trees: Vec<Point> = points("trees", &[strip()])
        .into_values()
        .flatten()
        .collect();

    // The share of a block's candidates kept, a candidate a 2 by 2 block, in four bands along x.
    let candidates_per_band = (40 / 2) * (40 / 2);
    for band in 0..4 {
        let (low, high) = (band as f32 * 40.0, band as f32 * 40.0 + 40.0);
        let kept = trees
            .iter()
            .filter(|p| (low..high).contains(&p.position[0]))
            .count() as f32
            / candidates_per_band as f32;
        let density = 1.0 - (low + 20.0) * 0.00625;
        println!("cells {low} to {high}: {kept} kept, a density of {density}");
        assert!(
            (kept - density).abs() < 0.07,
            "cells {low} to {high}: {kept} kept for a density of {density}"
        );
    }
}

#[test]
fn density_keeps_the_same_points_in_any_order() {
    let at_once = points("trees", &[strip()]);

    let one_by_one = points(
        "trees",
        &strip()
            .into_iter()
            .rev()
            .map(|c| vec![c])
            .collect::<Vec<_>>(),
    );

    assert_eq!(at_once, one_by_one);
}

#[test]
fn a_candidate_thinned_out_is_reported_as_density() {
    let mut runtime = Runtime::new(Arc::new(Pack::parse(PACK).expect("a valid pack")), 3, SIZE);
    let chunk = ChunkCoord::new(10, 0, 0);
    runtime
        .request(&[FocusPoint::new(chunk, 1)], &["trees"])
        .expect("a stage");
    runtime.run_until_idle().expect("the stages run");

    let report = runtime.scatter_report("trees", chunk).expect("a report");

    assert!(report.iter().any(|j| j.verdict == Err(Rejection::Density)));
    assert!(report.iter().any(|j| j.verdict.is_ok()));
    // Halfway along the strip the density is about a half, and every candidate reads it.
    for judgement in &report {
        let density = judgement
            .readings
            .density
            .expect("a density stage reads its density");
        assert!((0.4..0.55).contains(&density), "{density}");
    }
}

#[test]
fn a_clump_is_of_one_size() {
    let ferns: Vec<Point> = points("ferns", &[strip()])
        .into_values()
        .flatten()
        .collect();

    // A group's members share their candidate's chunk, cell and slot in the id.
    let mut clumps: BTreeMap<(ChunkCoord, u32, u64), Vec<f32>> = BTreeMap::new();
    for fern in &ferns {
        let candidate = (fern.id.local >> 32 & 0xFFFF) / 256;
        clumps
            .entry((fern.id.chunk, fern.id.local as u32, candidate))
            .or_default()
            .push(fern.scale);
    }
    let sizes: Vec<f32> = clumps
        .values()
        .filter(|scales| scales.len() > 1)
        .map(|scales| {
            let (low, high) = scales
                .iter()
                .fold((f32::MAX, f32::MIN), |(l, h), &s| (l.min(s), h.max(s)));
            // Within a clump sizes differ by the scale's range alone.
            assert!(high / low <= 1.05 / 0.95 + 1e-4, "{scales:?}");
            low
        })
        .collect();
    let (smallest, largest) = sizes
        .iter()
        .fold((f32::MAX, f32::MIN), |(l, h), &s| (l.min(s), h.max(s)));
    // Across clumps they differ by the clump's range.
    assert!(
        sizes.len() > 20 && largest / smallest > 2.5,
        "{smallest} to {largest}"
    );
}
