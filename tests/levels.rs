//! Stages at different scales: a coarse stage's columns span several WFC cells, fine stages read
//! it between its columns, and data flows only from coarse to fine.
//!
//! Positions are in WFC cells at every scale, so a coarse `X` read by a fine stage gives the fine
//! stage's own `X` back, a linear field being exact between columns. The last tests generate a coarse
//! biome feeding a fine terrain, as a world map feeds the local area a player walks, and check it is
//! the same in any generation order.

use std::sync::Arc;
use wave_forge::stages::{
    Condition, Expr, Pack, PackError, PackFile, Rule, Runtime, StageDef, StageKind,
};
use wave_forge::{ChunkCoord, FocusPoint};

const SIZE: [u32; 2] = [8, 8];

fn b(expr: Expr) -> Box<Expr> {
    Box::new(expr)
}

fn stage(name: &str, scale: u32, kind: StageKind) -> StageDef {
    StageDef {
        name: name.to_owned(),
        scale,
        kind,
    }
}

fn field(name: &str, scale: u32, expr: Expr) -> StageDef {
    stage(name, scale, StageKind::Field(expr))
}

fn pack(stages: Vec<StageDef>) -> Result<Pack, PackError> {
    Pack::from_file(PackFile {
        version: 1,
        stages,
        tables: Vec::new(),
    })
}

fn run(stages: Vec<StageDef>, targets: &[&str], radius: u32) -> Runtime {
    let mut runtime = Runtime::new(Arc::new(pack(stages).expect("a valid pack")), 9, SIZE);
    runtime
        .request(
            &[FocusPoint::new(ChunkCoord::new(0, 0, 0), radius)],
            targets,
        )
        .expect("known stages");
    runtime.run_until_idle().expect("the stages run");
    runtime
}

/// A stage's value at one of its own columns.
fn value(runtime: &Runtime, stage: &str, (x, y): (i64, i64)) -> f32 {
    let s = i64::from(SIZE[0]);
    let chunk = ChunkCoord::new(x.div_euclid(s) as i32, y.div_euclid(s) as i32, 0);
    runtime
        .field(stage, chunk)
        .expect("generated")
        .get(x.rem_euclid(s) as u32, y.rem_euclid(s) as u32)
}

#[test]
fn a_coarse_stages_columns_are_measured_in_wfc_cells() {
    let runtime = run(vec![field("x", 4, Expr::X)], &["x"], 2);

    for column in -8..8 {
        assert_eq!(
            value(&runtime, "x", (column, 0)),
            (column as f32 + 0.5) * 4.0
        );
    }
}

#[test]
fn a_fine_stage_reads_a_coarse_field_between_its_columns() {
    let runtime = run(
        vec![
            field("coarse", 4, Expr::X),
            field("fine", 1, Expr::Input("coarse".to_owned())),
        ],
        &["fine"],
        2,
    );

    for column in -16..24 {
        let got = value(&runtime, "fine", (column, 5));
        let expected = column as f32 + 0.5;
        assert!(
            (got - expected).abs() < 1e-4,
            "column {column}: {got} against {expected}"
        );
    }
}

#[test]
fn a_fine_stage_reads_the_coarse_category_its_column_lies_in() {
    let zones = stage(
        "zones",
        4,
        StageKind::Rules {
            rules: vec![Rule {
                category: "west".to_owned(),
                when: vec![Condition::Less(b(Expr::X), b(Expr::Constant(0.0)))],
            }],
            otherwise: "east".to_owned(),
        },
    );
    let runtime = run(
        vec![
            zones,
            field(
                "east",
                1,
                Expr::Is("zones".to_owned(), vec!["east".to_owned()]),
            ),
        ],
        &["east"],
        2,
    );

    for column in -16_i64..16 {
        let coarse_centre = (column.div_euclid(4) as f32 + 0.5) * 4.0;
        let expected = if coarse_centre < 0.0 { 0.0 } else { 1.0 };
        assert_eq!(
            value(&runtime, "east", (column, 1)),
            expected,
            "column {column}"
        );
    }
}

#[test]
fn loading_lets_data_flow_only_from_coarse_to_fine() {
    let refused = [
        (
            "a coarse stage reading a fine one",
            vec![
                field("fine", 1, Expr::X),
                field("reader", 4, Expr::Input("fine".to_owned())),
            ],
        ),
        (
            "a scale that is not a whole factor",
            vec![
                field("coarse", 6, Expr::X),
                field("reader", 4, Expr::Input("coarse".to_owned())),
            ],
        ),
        ("a scale of 0", vec![field("reader", 0, Expr::X)]),
        (
            "scatter at a coarse scale",
            vec![
                field("height", 1, Expr::X),
                stage(
                    "reader",
                    2,
                    StageKind::Scatter {
                        kind: "tree".to_owned(),
                        height: "height".to_owned(),
                        spacing: 3,
                        count: (1, 1),
                        group: None,
                        chance: 1.0,
                        between: None,
                        max_slope: None,
                        when: Vec::new(),
                        water: None,
                        avoid: None,
                        apart: 0,
                        scale: (1.0, 1.0),
                        tilt: (0.0, 0.0),
                        align: 0.0,
                    },
                ),
            ],
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

#[test]
fn reach_across_levels_is_counted_in_wfc_cells() {
    let pack = pack(vec![
        field("coarse", 8, Expr::X),
        stage(
            "smooth",
            4,
            StageKind::Blur {
                input: "coarse".to_owned(),
                radius: 2,
            },
        ),
        stage(
            "fine",
            1,
            StageKind::Blur {
                input: "smooth".to_owned(),
                radius: 3,
            },
        ),
    ])
    .expect("a valid pack");

    let reach = pack.reach("fine", SIZE).expect("a stage");

    // 3 fine columns, one smooth column read between, then 2 smooth columns and one coarse
    // column read between.
    assert_eq!(reach.get("smooth"), Some(&(3 + 4)), "{reach:?}");
    assert_eq!(reach.get("coarse"), Some(&(3 + 4 + 2 * 4 + 8)), "{reach:?}");
}

/// A world map and the land a player walks: a coarse continent and biomes on columns of 8 cells, and
/// a fine terrain shaped by biome and blended where biomes meet.
fn two_levels() -> Vec<StageDef> {
    let noise = |frequency: f32, name: &str| Expr::Noise {
        frequency,
        octaves: 2,
        name: Some(name.to_owned()),
    };
    vec![
        field("continent", 8, noise(0.01, "continent")),
        stage(
            "biome",
            8,
            StageKind::Rules {
                rules: vec![
                    Rule {
                        category: "sea".to_owned(),
                        when: vec![Condition::Less(
                            b(Expr::Input("continent".to_owned())),
                            b(Expr::Constant(0.4)),
                        )],
                    },
                    Rule {
                        category: "hills".to_owned(),
                        when: vec![Condition::Greater(
                            b(Expr::Input("continent".to_owned())),
                            b(Expr::Constant(0.6)),
                        )],
                    },
                ],
                otherwise: "plains".to_owned(),
            },
        ),
        field(
            "terrain",
            1,
            Expr::Match {
                input: "biome".to_owned(),
                cases: vec![
                    ("sea".to_owned(), Expr::Constant(-1.0)),
                    (
                        "hills".to_owned(),
                        Expr::Mul(b(noise(0.1, "hill")), b(Expr::Constant(6.0))),
                    ),
                ],
                otherwise: b(Expr::Mul(b(noise(0.05, "plain")), b(Expr::Constant(2.0)))),
                blend: 4,
            },
        ),
    ]
}

#[test]
fn a_coarse_biome_feeding_a_fine_terrain_is_the_same_in_any_order() {
    let area: Vec<ChunkCoord> = (0..4)
        .flat_map(|y| (0..4).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect();
    let generate = |requests: Vec<Vec<ChunkCoord>>| {
        let mut runtime =
            Runtime::new(Arc::new(pack(two_levels()).expect("a valid pack")), 3, SIZE);
        let mut out = Vec::new();
        for request in requests {
            let focus: Vec<FocusPoint> = request.iter().map(|&c| FocusPoint::new(c, 0)).collect();
            runtime.request(&focus, &["terrain"]).expect("a stage");
            runtime.run_until_idle().expect("the stages run");
            for chunk in request {
                let field = runtime.field("terrain", chunk).expect("generated");
                out.push((
                    chunk,
                    field.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                ));
            }
        }
        out.sort_by_key(|(chunk, _)| *chunk);
        out
    };

    let all_at_once = generate(vec![area.clone()]);
    let forward = generate(area.iter().map(|&c| vec![c]).collect());
    let backward = generate(area.iter().rev().map(|&c| vec![c]).collect());

    assert_eq!(all_at_once, forward);
    assert_eq!(all_at_once, backward);
}

#[test]
fn a_coarse_stage_is_generated_once_for_the_many_fine_chunks_it_covers() {
    let runtime = run(two_levels(), &["terrain"], 4);

    let coarse = runtime
        .timings()
        .into_iter()
        .find(|(name, _)| name == "biome")
        .expect("a stage")
        .1;
    let fine = runtime
        .timings()
        .into_iter()
        .find(|(name, _)| name == "terrain")
        .expect("a stage")
        .1;

    assert_eq!(fine.products, 81, "9 by 9 fine chunks");
    assert!(
        coarse.products <= 9,
        "a coarse chunk covers 8 by 8 fine ones: {}",
        coarse.products
    );
}
