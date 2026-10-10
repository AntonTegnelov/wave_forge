//! A Scatter stage's variation (docs/reference/packs.md, "Scatter"): kinds drawn by weight, one per
//! group, a variant of the kind and a random per point, a stretch of its height, and the values of
//! its appearance fields, all the same whatever order chunks are asked for in.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::{Pack, PackError, Point, Runtime};
use wave_forge::{ChunkCoord, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "height", kind: Field(Mul(Noise(frequency: 0.04, octaves: 3), Constant(20.0)))),
        (name: "dryness", kind: Field(Clamp(Mul(X, Constant(0.01)), 0.0, 1.0))),
        (name: "trees", kind: Scatter(height: "height", spacing: 2,
            kinds: [(kind: "oak", weight: 3.0, variants: 4), (kind: "birch", variants: 2)],
            stretch: (0.8, 1.25), appearance: ["dryness"])),
        (name: "clumps", kind: Scatter(kind: "fern", variants: 3, height: "height", spacing: 6,
            group: Some((size: (3, 6), radius: 2.0)))),
        (name: "mixed", kind: Scatter(height: "height", spacing: 6,
            kinds: [(kind: "oak"), (kind: "birch")], group: Some((size: (3, 6), radius: 2.0)))),
    ],
)"#;

const SIZE: [u32; 2] = [8, 8];

fn area() -> Vec<ChunkCoord> {
    (0..10)
        .flat_map(|y| (0..10).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

/// Every point of `stage` over the area, asking for the chunks in `order`.
fn points(stage: &str, order: &[Vec<ChunkCoord>]) -> BTreeMap<ChunkCoord, Vec<Point>> {
    let pack = Arc::new(Pack::parse(PACK).expect("a valid pack"));
    let mut runtime = Runtime::new(pack, 5, SIZE);
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

fn trees() -> Vec<Point> {
    points("trees", &[area()]).into_values().flatten().collect()
}

#[test]
fn kinds_variants_and_randoms_come_out_the_same_in_any_order() {
    let at_once = points("trees", &[area()]);

    let one_by_one = points(
        "trees",
        &area()
            .into_iter()
            .rev()
            .map(|c| vec![c])
            .collect::<Vec<_>>(),
    );

    assert_eq!(at_once, one_by_one);
}

#[test]
fn kinds_are_drawn_by_their_weights() {
    let trees = trees();

    let oaks = trees.iter().filter(|p| &*p.kind == "oak").count();
    let share = oaks as f32 / trees.len() as f32;
    assert!(trees.len() > 1000, "{} trees", trees.len());
    assert!((share - 0.75).abs() < 0.04, "{share} of the trees are oaks");
    assert!(
        trees
            .iter()
            .all(|p| &*p.kind == "oak" || &*p.kind == "birch")
    );
}

#[test]
fn each_kind_draws_its_variants_evenly() {
    let trees = trees();

    for (kind, variants) in [("oak", 4), ("birch", 2)] {
        let of_kind: Vec<&Point> = trees.iter().filter(|p| &*p.kind == kind).collect();
        for variant in 0..variants {
            let share = of_kind.iter().filter(|p| p.variant == variant).count() as f32
                / of_kind.len() as f32;
            let even = 1.0 / f32::from(variants);
            assert!(
                (share - even).abs() < 0.05,
                "{kind} variant {variant}: {share} of {}",
                of_kind.len()
            );
        }
        assert!(of_kind.iter().all(|p| p.variant < variants));
    }
}

#[test]
fn randoms_spread_over_zero_to_one() {
    let trees = trees();

    let mean = trees.iter().map(|p| p.random).sum::<f32>() / trees.len() as f32;
    assert!(trees.iter().all(|p| (0.0..1.0).contains(&p.random)));
    assert!((mean - 0.5).abs() < 0.03, "a mean random of {mean}");
    let below_a_tenth = trees.iter().filter(|p| p.random < 0.1).count() as f32;
    assert!((below_a_tenth / trees.len() as f32 - 0.1).abs() < 0.03);
}

#[test]
fn a_stretch_lengthens_the_height_alone() {
    let trees = trees();

    assert!(trees.iter().all(|p| (0.8..=1.25).contains(&p.stretch)));
    assert!(trees.iter().any(|p| p.stretch < 0.9) && trees.iter().any(|p| p.stretch > 1.15));
    for point in &trees {
        let basis = point.y_up_basis();
        let column = |c: usize| {
            (0..3)
                .map(|r| basis[r][c] * basis[r][c])
                .sum::<f32>()
                .sqrt()
        };
        assert!((column(1) - point.scale * point.stretch).abs() < 1e-4);
        assert!((column(0) - point.scale).abs() < 1e-4);
        assert!((column(2) - point.scale).abs() < 1e-4);
    }
}

#[test]
fn a_point_carries_its_appearance_fields_at_its_column() {
    let trees = trees();

    for point in &trees {
        // A field's X is its column's centre.
        let centre = point.position[0].floor() + 0.5;
        let dryness = (centre * 0.01).clamp(0.0, 1.0);
        assert!(
            (point.appearance[0] - dryness).abs() < 1e-5,
            "{:?} at {:?}",
            point.appearance,
            point.position
        );
        assert_eq!(point.appearance[1], 0.0);
    }
}

#[test]
fn a_single_kind_draws_among_its_variants() {
    let ferns: Vec<Point> = points("clumps", &[area()])
        .into_values()
        .flatten()
        .collect();

    assert!(ferns.iter().all(|p| &*p.kind == "fern" && p.variant < 3));
    for variant in 0..3 {
        assert!(ferns.iter().any(|p| p.variant == variant));
    }
}

#[test]
fn a_group_is_of_one_kind() {
    let mixed: Vec<Point> = points("mixed", &[area()]).into_values().flatten().collect();

    // A group's members share their candidate's chunk, cell and slot in the id; the slot's low
    // byte is the member's place in the group.
    let mut by_candidate: BTreeMap<(ChunkCoord, u32, u64), Vec<&Point>> = BTreeMap::new();
    for point in &mixed {
        let id = point.id;
        let candidate = (id.local >> 32 & 0xFFFF) / 256;
        by_candidate
            .entry((id.chunk, id.local as u32, candidate))
            .or_default()
            .push(point);
    }
    let groups = by_candidate.values().filter(|members| members.len() > 1);
    let mut kinds = Vec::new();
    for members in groups {
        assert!(members.iter().all(|p| p.kind == members[0].kind));
        kinds.push(members[0].kind.clone());
    }
    assert!(kinds.iter().any(|k| &**k == "oak") && kinds.iter().any(|k| &**k == "birch"));
}

/// Why the pack is refused with `from` in its trees stage replaced by `to`.
fn refused_with(from: &str, to: &str) -> String {
    let text = PACK.replace(from, to);
    assert_ne!(text, PACK, "{from} is in the pack");
    match Pack::parse(&text) {
        Err(PackError::Invalid { message, .. }) => message,
        other => panic!("{to} was not refused: {other:?}"),
    }
}

/// Why the pack is refused with its trees stage given `settings` in place of its kinds.
fn refused(settings: &str) -> String {
    refused_with(
        r#"kinds: [(kind: "oak", weight: 3.0, variants: 4), (kind: "birch", variants: 2)],"#,
        settings,
    )
}

#[test]
fn a_stage_names_one_kind_or_several_and_not_both() {
    assert!(refused(r#"kind: "oak", kinds: [(kind: "birch")],"#).contains("not both"));
    assert!(refused("").contains("not both"));
}

#[test]
fn weights_variants_stretches_and_appearance_are_checked() {
    assert!(refused(r#"kinds: [(kind: "oak", weight: 0.0)],"#).contains("weighs more than 0"));
    assert!(refused(r#"kinds: [(kind: "oak", variants: 0)],"#).contains("0 variants"));
    assert!(refused(r#"kinds: [(kind: "oak", variants: 257)],"#).contains("257 variants"));
    assert!(refused(r#"kinds: [(kind: "oak")], variants: 2,"#).contains("each kind gives"));
    assert!(refused_with("stretch: (0.8, 1.25)", "stretch: (0.0, 1.0)").contains("stretch"));
    assert!(
        refused_with(
            r#"appearance: ["dryness"]"#,
            r#"appearance: ["dryness", "dryness", "dryness"]"#
        )
        .contains("appearance fields")
    );
}

#[test]
fn a_look_packs_the_random_and_the_appearance_in_255ths() {
    let trees = trees();

    for point in &trees {
        let look = point.look();
        let byte = |shift: u32| ((look >> shift) & 0xFF) as f32 / 255.0;
        assert!(look < 1 << 24);
        assert!((byte(0) - point.random).abs() <= 0.5 / 255.0 + 1e-6);
        assert!((byte(8) - point.appearance[0].clamp(0.0, 1.0)).abs() <= 0.5 / 255.0 + 1e-6);
        assert_eq!(byte(16), 0.0);
    }
}
