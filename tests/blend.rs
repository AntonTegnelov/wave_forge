//! A `Match` picks an expression per category and blends categories where they meet.
//!
//! Each category within `blend` cells of a column weighs in with a tent, `(b + 1 - |dx|) *
//! (b + 1 - |dy|)`, so moving one column shifts at most `2 / (b + 1)` of the weight. The tests pin
//! that formula on a straight border, check the loader's refusals and the reach it derives, and
//! check that the ring world's per-biome terrain (`examples/rings.world.ron`, G7) steps by no more
//! than the blend allows between any two neighbouring columns, across biome borders and chunk seams.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::{
    Condition, Expr, Pack, PackError, PackFile, Rule, Runtime, StageDef, StageKind,
};
use wave_forge::{ChunkCoord, FocusPoint};

const SIZE: [u32; 2] = [8, 8];
const SEED: u64 = 11;

fn b(expr: Expr) -> Box<Expr> {
    Box::new(expr)
}

fn constant(value: f32) -> Box<Expr> {
    b(Expr::Constant(value))
}

fn stage(name: &str, kind: StageKind) -> StageDef {
    StageDef {
        name: name.to_owned(),
        scale: 1,
        kind,
    }
}

fn pack(stages: Vec<StageDef>) -> Result<Pack, PackError> {
    Pack::from_file(PackFile {
        version: 1,
        stages,
        tables: Vec::new(),
    })
}

/// Columns west of x = 0 are "low", the rest "high".
fn halves() -> StageDef {
    stage(
        "halves",
        StageKind::Rules {
            rules: vec![Rule {
                category: "low".to_owned(),
                when: vec![Condition::Less(b(Expr::X), constant(0.0))],
            }],
            otherwise: "high".to_owned(),
        },
    )
}

fn step(blend: u32) -> StageDef {
    stage(
        "step",
        StageKind::Field(Expr::Match {
            input: "halves".to_owned(),
            cases: vec![("low".to_owned(), Expr::Constant(0.0))],
            otherwise: constant(10.0),
            blend,
        }),
    )
}

fn value(runtime: &Runtime, stage: &str, (x, y): (i64, i64)) -> f32 {
    let s = i64::from(SIZE[0]);
    let chunk = ChunkCoord::new(x.div_euclid(s) as i32, y.div_euclid(s) as i32, 0);
    runtime
        .field(stage, chunk)
        .expect("generated")
        .get(x.rem_euclid(s) as u32, y.rem_euclid(s) as u32)
}

fn category(runtime: &Runtime, stage: &str, (x, y): (i64, i64)) -> u8 {
    let s = i64::from(SIZE[0]);
    let chunk = ChunkCoord::new(x.div_euclid(s) as i32, y.div_euclid(s) as i32, 0);
    runtime
        .categories(stage, chunk)
        .expect("generated")
        .get(x.rem_euclid(s) as u32, y.rem_euclid(s) as u32)
}

fn run(pack: Pack, targets: &[&str], centre: ChunkCoord, radius: u32) -> Runtime {
    let mut runtime = Runtime::new(Arc::new(pack), SEED, SIZE);
    runtime
        .request(&[FocusPoint::new(centre, radius)], targets)
        .expect("known stages");
    runtime.run_until_idle().expect("the stages run");
    runtime
}

#[test]
fn a_straight_border_blends_by_the_tent_and_steps_by_at_most_its_share() {
    let blend = 4;
    let runtime = run(
        pack(vec![halves(), step(blend)]).expect("a valid pack"),
        &["step"],
        ChunkCoord::new(0, 0, 0),
        1,
    );
    let reach = i64::from(blend);
    let tent = |d: i64| (reach + 1 - d.abs()) as f32;
    let total: f32 = (-reach..=reach).map(tent).sum();

    let row: Vec<f32> = (-8..16).map(|x| value(&runtime, "step", (x, 3))).collect();

    for (x, got) in (-8..16).zip(&row) {
        let high: f32 = (-reach..=reach).filter(|d| x + d >= 0).map(tent).sum();
        let expected = 10.0 * high / total;
        assert!(
            (got - expected).abs() < 1e-5,
            "x {x}: {got} against {expected}"
        );
    }
    let largest = row
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).abs())
        .fold(0.0, f32::max);
    assert!(
        (largest - 10.0 / (blend + 1) as f32).abs() < 1e-5,
        "the largest step is {largest}"
    );
}

#[test]
fn without_a_blend_each_column_takes_its_own_categorys_expression() {
    let runtime = run(
        pack(vec![halves(), step(0)]).expect("a valid pack"),
        &["step"],
        ChunkCoord::new(0, 0, 0),
        1,
    );

    for x in -8..16 {
        let expected = if x < 0 { 0.0 } else { 10.0 };
        assert_eq!(value(&runtime, "step", (x, 2)), expected, "x {x}");
    }
}

#[test]
fn the_categories_a_match_reads_are_generated_as_far_out_as_it_blends() {
    let pack = pack(vec![halves(), step(5)]).expect("a valid pack");

    let reach = pack.reach("step", SIZE).expect("a stage");

    assert_eq!(reach.get("halves"), Some(&5), "{reach:?}");
}

#[test]
fn loading_refuses_a_match_it_cannot_evaluate_and_names_the_stage() {
    let with = |expr: Expr| vec![halves(), stage("reader", StageKind::Field(expr))];
    let refused = [
        (
            "a blend wider than allowed",
            with(Expr::Match {
                input: "halves".to_owned(),
                cases: Vec::new(),
                otherwise: constant(0.0),
                blend: 33,
            }),
        ),
        (
            "two cases for one category",
            with(Expr::Match {
                input: "halves".to_owned(),
                cases: vec![
                    ("low".to_owned(), Expr::Constant(0.0)),
                    ("low".to_owned(), Expr::Constant(1.0)),
                ],
                otherwise: constant(0.0),
                blend: 1,
            }),
        ),
        (
            "a case for a category the rules do not give",
            with(Expr::Match {
                input: "halves".to_owned(),
                cases: vec![("middle".to_owned(), Expr::Constant(0.0))],
                otherwise: constant(0.0),
                blend: 1,
            }),
        ),
        (
            "a match over a field",
            vec![
                stage("height", StageKind::Field(Expr::X)),
                stage(
                    "reader",
                    StageKind::Field(Expr::Match {
                        input: "height".to_owned(),
                        cases: Vec::new(),
                        otherwise: constant(0.0),
                        blend: 1,
                    }),
                ),
            ],
        ),
        (
            "a category stage read as a field beside a match of it",
            with(Expr::Add(
                b(Expr::Match {
                    input: "halves".to_owned(),
                    cases: Vec::new(),
                    otherwise: constant(0.0),
                    blend: 2,
                }),
                b(Expr::Input("halves".to_owned())),
            )),
        ),
    ];

    for (what, stages) in refused {
        let error = pack(stages).expect_err(what);

        assert!(
            matches!(&error, PackError::Invalid { stage, .. } if stage == "reader"),
            "{what}: {error}"
        );
    }
}

/// The ring world's terrain for each biome, as a function of the island's height.
fn shape(biome: &str, height: f32) -> f32 {
    match biome {
        "sea" => height * 2.0,
        "peaks" => 0.45 + (height - 0.45) * 2.5,
        "marsh" => 0.1 + (height - 0.1) * 0.3,
        _ => height,
    }
}

#[test]
fn the_ring_worlds_terrain_steps_no_more_than_its_blend_allows() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/rings.world.ron"
    ))
    .expect("the ring-world test pack");
    let pack = Pack::parse(&text).expect("a valid pack");
    let names: Vec<String> = pack
        .kind("biome")
        .expect("a biome stage")
        .categories()
        .into_iter()
        .map(str::to_owned)
        .collect();
    let blend = 3_i64;
    let centre = ChunkCoord::new(6, 0, 0);
    let runtime = run(pack, &["terrain", "biome", "height"], centre, 6);
    let s = i64::from(SIZE[0]);
    let columns: Vec<(i64, i64)> = (-6 * s..7 * s)
        .flat_map(|y| (0..13 * s).map(move |x| (x, y)))
        .collect();
    let biome_at =
        |column: (i64, i64)| names[usize::from(category(&runtime, "biome", column))].as_str();

    let mut uniform = 0;
    let (mut borders, mut seams) = (0, 0);
    for &(x, y) in &columns {
        let height = value(&runtime, "height", (x, y));
        let terrain = value(&runtime, "terrain", (x, y));
        let own = biome_at((x, y));
        // Inside one biome's whole neighbourhood, the terrain is that biome's shape exactly.
        let alone = (-blend..=blend).all(|dy| {
            (-blend..=blend).all(|dx| {
                let near = (x + dx, y + dy);
                columns.contains(&near) && biome_at(near) == own
            })
        });
        if alone {
            uniform += 1;
            let expected = shape(own, height);
            assert!(
                (terrain - expected).abs() < 1e-5,
                "({x}, {y}) in {own}: {terrain} against {expected}"
            );
        }
        for next in [(x + 1, y), (x, y + 1)] {
            if !columns.contains(&next) {
                continue;
            }
            let (h, h_next) = (height, value(&runtime, "height", next));
            let shapes = ["sea", "peaks", "marsh", "grassland"];
            let change = shapes
                .iter()
                .map(|biome| (shape(biome, h) - shape(biome, h_next)).abs())
                .fold(0.0, f32::max);
            let values: Vec<f32> = shapes.iter().map(|biome| shape(biome, h_next)).collect();
            let spread = values.iter().fold(f32::MIN, |a, &v| a.max(v))
                - values.iter().fold(f32::MAX, |a, &v| a.min(v));
            let bound = change + spread / (blend + 1) as f32 + 1e-5;

            let step = (value(&runtime, "terrain", next) - terrain).abs();

            assert!(
                step <= bound,
                "({x}, {y}) to {next:?}: a step of {step}, at most {bound}"
            );
            if biome_at(next) != own {
                borders += 1;
            }
            if next.0.rem_euclid(s) == 0 && next.0 != x || next.1.rem_euclid(s) == 0 && next.1 != y
            {
                seams += 1;
            }
        }
    }
    let biomes: BTreeMap<&str, usize> = columns.iter().fold(BTreeMap::new(), |mut seen, &c| {
        *seen.entry(biome_at(c)).or_default() += 1;
        seen
    });
    assert!(
        uniform > 0 && borders > 0 && seams > 0,
        "{uniform} uniform, {borders} border and {seams} seam pairs"
    );
    assert!(
        biomes.contains_key("sea") && biomes.len() >= 4,
        "{biomes:?}"
    );
}
