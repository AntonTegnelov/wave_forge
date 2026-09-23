//! Rules stages give each column the first category whose conditions all hold.
//!
//! The tests check first-match order and the fallback against direct computation, that loading
//! keeps categories and fields apart, that `Is` reads categories back as a mask, and that the
//! ring-world test pack's biome rules (`examples/rings.world.ron`, G7) give every column the biome
//! the rules computed directly give it, the same in any generation order.

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
        persist: wave_forge::stages::Persist::Pure,
        kind,
    }
}

fn rule(category: &str, when: Vec<Condition>) -> Rule {
    Rule {
        category: category.to_owned(),
        when,
    }
}

fn pack(stages: Vec<StageDef>) -> Result<Pack, PackError> {
    Pack::from_file(PackFile {
        version: 1,
        stages,
        tables: Vec::new(),
        noises: std::collections::BTreeMap::new(),
        bound: None,
    })
}

/// Every column within `radius` chunks of `centre`, as `(x, y)`.
fn columns(centre: ChunkCoord, radius: i64) -> Vec<(i64, i64)> {
    let s = i64::from(SIZE[0]);
    let (cx, cy) = (i64::from(centre.x), i64::from(centre.y));
    ((cy - radius) * s..(cy + radius + 1) * s)
        .flat_map(|y| ((cx - radius) * s..(cx + radius + 1) * s).map(move |x| (x, y)))
        .collect()
}

fn chunk_of((x, y): (i64, i64)) -> (ChunkCoord, u32, u32) {
    let s = i64::from(SIZE[0]);
    (
        ChunkCoord::new(x.div_euclid(s) as i32, y.div_euclid(s) as i32, 0),
        x.rem_euclid(s) as u32,
        y.rem_euclid(s) as u32,
    )
}

fn category(runtime: &Runtime, stage: &str, column: (i64, i64)) -> u8 {
    let (chunk, x, y) = chunk_of(column);
    runtime
        .categories(stage, chunk)
        .expect("generated")
        .get(x, y)
}

fn value(runtime: &Runtime, stage: &str, column: (i64, i64)) -> f32 {
    let (chunk, x, y) = chunk_of(column);
    runtime.field(stage, chunk).expect("generated").get(x, y)
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
fn the_first_rule_that_holds_gives_the_category_and_otherwise_the_fallback() {
    let rules = StageKind::Rules {
        rules: vec![
            rule("west", vec![Condition::Less(b(Expr::X), constant(0.0))]),
            rule(
                "band",
                vec![
                    Condition::Between(b(Expr::Y), 2.0, 6.0),
                    Condition::Greater(b(Expr::X), constant(-4.0)),
                ],
            ),
            rule("west", vec![Condition::Greater(b(Expr::Y), constant(10.0))]),
        ],
        otherwise: "rest".to_owned(),
    };
    assert_eq!(rules.categories(), ["west", "band", "rest"]);
    let runtime = run(
        pack(vec![stage("zones", rules)]).expect("a valid pack"),
        &["zones"],
        ChunkCoord::new(0, 0, 0),
        1,
    );

    for column in columns(ChunkCoord::new(0, 0, 0), 1) {
        let (x, y) = (column.0 as f32 + 0.5, column.1 as f32 + 0.5);
        let expected = if x < 0.0 {
            0
        } else if (2.0..=6.0).contains(&y) && x > -4.0 {
            1
        } else if y > 10.0 {
            0
        } else {
            2
        };

        assert_eq!(category(&runtime, "zones", column), expected, "{column:?}");
    }
}

#[test]
fn is_reads_a_category_back_as_a_mask() {
    let rules = StageKind::Rules {
        rules: vec![
            rule("low", vec![Condition::Less(b(Expr::X), constant(-2.0))]),
            rule("mid", vec![Condition::Less(b(Expr::X), constant(3.0))]),
        ],
        otherwise: "high".to_owned(),
    };
    let mask = StageKind::Field(Expr::Is(
        "zones".to_owned(),
        vec!["low".to_owned(), "high".to_owned()],
    ));
    let runtime = run(
        pack(vec![stage("zones", rules), stage("mask", mask)]).expect("a valid pack"),
        &["mask"],
        ChunkCoord::new(0, 0, 0),
        1,
    );

    for column in columns(ChunkCoord::new(0, 0, 0), 1) {
        let x = column.0 as f32 + 0.5;
        let expected = if (-2.0..3.0).contains(&x) { 0.0 } else { 1.0 };

        assert_eq!(value(&runtime, "mask", column), expected, "{column:?}");
    }
}

#[test]
fn loading_keeps_categories_and_fields_apart() {
    let zones = || {
        stage(
            "zones",
            StageKind::Rules {
                rules: vec![rule("a", vec![Condition::Less(b(Expr::X), constant(0.0))])],
                otherwise: "b".to_owned(),
            },
        )
    };
    let height = || stage("height", StageKind::Field(Expr::X));
    let refused = [
        (
            "a category read as a height",
            vec![
                zones(),
                stage("reader", StageKind::Field(Expr::Input("zones".to_owned()))),
            ],
        ),
        (
            "a height read as a category",
            vec![
                height(),
                stage(
                    "reader",
                    StageKind::Field(Expr::Is("height".to_owned(), vec!["a".to_owned()])),
                ),
            ],
        ),
        (
            "a category the rules do not name",
            vec![
                zones(),
                stage(
                    "reader",
                    StageKind::Field(Expr::Is("zones".to_owned(), vec!["c".to_owned()])),
                ),
            ],
        ),
        (
            "a category read as a height inside a rule",
            vec![
                zones(),
                stage(
                    "reader",
                    StageKind::Rules {
                        rules: vec![rule(
                            "x",
                            vec![Condition::Less(
                                b(Expr::Input("zones".to_owned())),
                                constant(1.0),
                            )],
                        )],
                        otherwise: "y".to_owned(),
                    },
                ),
            ],
        ),
        (
            "an empty range",
            vec![stage(
                "reader",
                StageKind::Rules {
                    rules: vec![rule("x", vec![Condition::Between(b(Expr::X), 2.0, 1.0)])],
                    otherwise: "y".to_owned(),
                },
            )],
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
fn loading_refuses_more_categories_than_a_byte_holds() {
    let rules = (0..256)
        .map(|index| rule(&format!("c{index}"), Vec::new()))
        .collect();

    let error = pack(vec![stage(
        "many",
        StageKind::Rules {
            rules,
            otherwise: "last".to_owned(),
        },
    )])
    .expect_err("257 categories");

    assert!(
        matches!(&error, PackError::Invalid { stage, .. } if stage == "many"),
        "{error}"
    );
}

/// The ring-world test pack, with the noises its biome rules read added as stages of their own, so
/// the rules can be computed directly from what those stages hold.
fn rings() -> Pack {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/rings.world.ron"
    ))
    .expect("the ring-world test pack");
    let mut file: PackFile = ron::from_str(&text).expect("a pack file");
    for name in ["wet", "moor", "dry"] {
        file.stages.push(stage(
            &format!("{name}_noise"),
            StageKind::Field(Expr::Noise {
                frequency: 0.08,
                octaves: 1,
                name: Some(name.to_owned()),
            }),
        ));
    }
    Pack::from_file(file).expect("a valid pack")
}

/// The biome at a column, from the ring world's rules written out directly.
fn ring_biome(runtime: &Runtime, column: (i64, i64)) -> &'static str {
    let (x, y) = (column.0 as f32 + 0.5, column.1 as f32 + 0.5);
    let height = value(runtime, "height", column);
    let wobble = value(runtime, "wobble", column);
    let noise = |name: &str| value(runtime, &format!("{name}_noise"), column);
    let distance = x.hypot(y);
    if height < 0.05 {
        "sea"
    } else if x.hypot(y + 35.0) > 110.0 + wobble {
        "glacier"
    } else if x.hypot(y - 35.0) > 110.0 + wobble {
        "volcanic"
    } else if height > 0.45 {
        "peaks"
    } else if noise("wet") > 0.62
        && (15.0..=70.0).contains(&distance)
        && (0.06..=0.2).contains(&height)
    {
        "marsh"
    } else if noise("moor") > 0.55 && distance > 55.0 + wobble {
        "moor"
    } else if noise("dry") > 0.45 && distance > 25.0 + wobble && distance < 75.0 {
        "steppe"
    } else if distance > 40.0 + wobble {
        "woods"
    } else {
        "grassland"
    }
}

#[test]
fn the_ring_worlds_biomes_follow_its_rules() {
    let pack = rings();
    let names: Vec<String> = pack
        .kind("biome")
        .expect("a biome stage")
        .categories()
        .into_iter()
        .map(str::to_owned)
        .collect();
    let targets = [
        "biome",
        "height",
        "wobble",
        "wet_noise",
        "moor_noise",
        "dry_noise",
    ];
    // A band from the centre to past the island's edge, through every ring.
    let centre = ChunkCoord::new(7, 0, 0);
    let runtime = run(pack, &targets, centre, 7);

    let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
    for column in columns(centre, 7) {
        // The island's bound leaves out chunks wholly beyond 110 cells from its centre.
        if runtime.categories("biome", chunk_of(column).0).is_none() {
            assert!(
                (column.0 as f32).hypot(column.1 as f32) > 100.0,
                "{column:?}"
            );
            continue;
        }
        let expected = ring_biome(&runtime, column);

        let got = &names[usize::from(category(&runtime, "biome", column))];

        assert_eq!(got, expected, "{column:?}");
        *seen.entry(expected).or_default() += 1;
    }
    for biome in ["sea", "grassland", "woods", "steppe", "moor"] {
        assert!(seen.contains_key(biome), "no {biome} in the band: {seen:?}");
    }
}

#[test]
fn the_ring_worlds_biomes_are_the_same_in_any_generation_order() {
    let area: Vec<ChunkCoord> = (-2..2)
        .flat_map(|y| (8..12).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect();
    let generate = |requests: Vec<Vec<ChunkCoord>>| {
        let mut runtime = Runtime::new(Arc::new(rings()), SEED, SIZE);
        let mut out = BTreeMap::new();
        for request in requests {
            let focus: Vec<FocusPoint> = request.iter().map(|&c| FocusPoint::new(c, 0)).collect();
            runtime.request(&focus, &["biome"]).expect("a stage");
            runtime.run_until_idle().expect("the stages run");
            for chunk in request {
                out.insert(
                    chunk,
                    runtime
                        .categories("biome", chunk)
                        .expect("generated")
                        .clone(),
                );
            }
        }
        out
    };

    let all_at_once = generate(vec![area.clone()]);
    let raster = generate(area.iter().map(|&c| vec![c]).collect());
    let reverse = generate(area.iter().rev().map(|&c| vec![c]).collect());

    assert_eq!(all_at_once.len(), 16);
    assert_eq!(all_at_once, raster, "all at once against raster");
    assert_eq!(all_at_once, reverse, "all at once against reverse raster");
}
