//! Field expressions compute what their formulas say, column by column.
//!
//! Each test builds a pack, generates an area, and compares every column's value with the same
//! formula computed directly from the column's coordinates or from other stages' values. The last
//! test writes Valheim's base height (docs/product/user-stories.md, G7) as one Field stage and
//! checks it against the formula computed from its noises.

use std::sync::Arc;
use wave_forge::stages::{
    Condition, Expr, Pack, PackError, PackFile, Runtime, StageDef, StageKind,
};
use wave_forge::{ChunkCoord, FocusPoint};

const SIZE: [u32; 2] = [8, 8];
const SEED: u64 = 7;

fn field(name: &str, expr: Expr) -> StageDef {
    StageDef {
        name: name.to_owned(),
        kind: StageKind::Field(expr),
    }
}

fn pack(stages: Vec<StageDef>) -> Result<Pack, PackError> {
    Pack::from_file(PackFile { version: 1, stages })
}

/// Generates `targets` over the chunks within `radius` of `centre`, and returns the runtime with
/// every column of that area as `(x, y)`.
fn generate(
    stages: Vec<StageDef>,
    targets: &[&str],
    centre: ChunkCoord,
    radius: u32,
) -> (Runtime, Vec<(i64, i64)>) {
    let mut runtime = Runtime::new(Arc::new(pack(stages).expect("a valid pack")), SEED, SIZE);
    runtime
        .request(&[FocusPoint::new(centre, radius)], targets)
        .expect("known stages");
    runtime.run_until_idle().expect("the stages run");
    let r = i64::from(radius);
    let s = i64::from(SIZE[0]);
    let columns = ((i64::from(centre.y) - r) * s..(i64::from(centre.y) + r + 1) * s)
        .flat_map(|y| {
            ((i64::from(centre.x) - r) * s..(i64::from(centre.x) + r + 1) * s).map(move |x| (x, y))
        })
        .collect();
    (runtime, columns)
}

fn value(runtime: &Runtime, stage: &str, (x, y): (i64, i64)) -> f32 {
    let s = i64::from(SIZE[0]);
    let chunk = ChunkCoord::new(x.div_euclid(s) as i32, y.div_euclid(s) as i32, 0);
    runtime
        .field(stage, chunk)
        .expect("generated")
        .get(x.rem_euclid(s) as u32, y.rem_euclid(s) as u32)
}

fn b(expr: Expr) -> Box<Expr> {
    Box::new(expr)
}

fn constant(value: f32) -> Box<Expr> {
    b(Expr::Constant(value))
}

fn noise(frequency: f32, name: &str) -> Expr {
    Expr::Noise {
        frequency,
        octaves: 1,
        name: Some(name.to_owned()),
    }
}

fn assert_close(got: f32, want: f32, what: &str) {
    assert!(
        (got - want).abs() <= 1e-5 * want.abs().max(1.0),
        "{what}: {got} against {want}"
    );
}

#[test]
fn coordinates_are_the_columns_centre_its_distance_and_its_angle_about_a_point() {
    let point = (3.0, -5.0);
    let (runtime, columns) = generate(
        vec![
            field("x", Expr::X),
            field("y", Expr::Y),
            field("distance", Expr::Distance(point)),
            field("angle", Expr::Angle(point)),
        ],
        &["x", "y", "distance", "angle"],
        ChunkCoord::new(0, 0, 0),
        1,
    );

    for column in columns {
        let (x, y) = (column.0 as f32 + 0.5, column.1 as f32 + 0.5);
        let turn = (y - point.1).atan2(x - point.0) / std::f32::consts::TAU;
        assert_eq!(value(&runtime, "x", column), x);
        assert_eq!(value(&runtime, "y", column), y);
        assert_close(
            value(&runtime, "distance", column),
            (x - point.0).hypot(y - point.1),
            "distance",
        );
        assert_close(
            value(&runtime, "angle", column),
            turn.rem_euclid(1.0),
            "angle",
        );
    }
}

#[test]
fn every_operation_computes_its_formula() {
    let x = || b(Expr::Remap(b(Expr::X), (-8.0, 16.0), (-1.5, 2.5)));
    let curve = vec![(-1.0, 4.0), (0.0, 0.0), (0.5, 1.0), (2.0, -3.0)];
    let (runtime, columns) = generate(
        vec![
            field("sub", Expr::Sub(x(), constant(0.25))),
            field("min", Expr::Min(x(), constant(0.3))),
            field("max", Expr::Max(x(), constant(0.3))),
            field("abs", Expr::Abs(x())),
            field("floor", Expr::Floor(x())),
            field("clamp", Expr::Clamp(x(), -0.5, 1.25)),
            field("smooth", Expr::Smoothstep(-0.5, 1.5, x())),
            field("curve", Expr::Curve(x(), curve.clone())),
            field(
                "select",
                Expr::Select {
                    when: Condition::Less(x(), b(Expr::Y)),
                    then: constant(1.0),
                    otherwise: b(Expr::Mul(x(), constant(2.0))),
                },
            ),
            field(
                "greater",
                Expr::Select {
                    when: Condition::Greater(x(), constant(0.0)),
                    then: constant(1.0),
                    otherwise: constant(-1.0),
                },
            ),
        ],
        &[
            "sub", "min", "max", "abs", "floor", "clamp", "smooth", "curve", "select", "greater",
        ],
        ChunkCoord::new(0, 0, 0),
        1,
    );

    for column in columns {
        let v = -1.5 + (column.0 as f32 + 0.5 + 8.0) / 24.0 * 4.0;
        let y = column.1 as f32 + 0.5;
        let t = ((v + 0.5) / 2.0).clamp(0.0, 1.0);
        let on_curve = if v <= -1.0 {
            4.0
        } else if v >= 2.0 {
            -3.0
        } else {
            let after = curve
                .iter()
                .position(|p| p.0 > v)
                .expect("inside the curve");
            let ((x0, y0), (x1, y1)) = (curve[after - 1], curve[after]);
            y0 + (v - x0) / (x1 - x0) * (y1 - y0)
        };
        let expected = [
            ("sub", v - 0.25),
            ("min", v.min(0.3)),
            ("max", v.max(0.3)),
            ("abs", v.abs()),
            ("floor", v.floor()),
            ("clamp", v.clamp(-0.5, 1.25)),
            ("smooth", t * t * (3.0 - 2.0 * t)),
            ("curve", on_curve),
            ("select", if v < y { 1.0 } else { v * 2.0 }),
            ("greater", if v > 0.0 { 1.0 } else { -1.0 }),
        ];
        for (stage, want) in expected {
            assert_close(
                value(&runtime, stage, column),
                want,
                &format!("{stage} at {column:?}"),
            );
        }
    }
}

#[test]
fn a_named_noise_is_the_same_in_every_stage_and_apart_from_other_names() {
    let unnamed = || Expr::Noise {
        frequency: 0.2,
        octaves: 2,
        name: None,
    };
    let named = |name: &str| Expr::Noise {
        frequency: 0.2,
        octaves: 2,
        name: Some(name.to_owned()),
    };
    let (runtime, columns) = generate(
        vec![
            field("first", named("hills")),
            field("second", Expr::Add(b(named("hills")), constant(0.0))),
            field("other", named("rivers")),
            field("plain_a", unnamed()),
            field("plain_b", unnamed()),
        ],
        &["first", "second", "other", "plain_a", "plain_b"],
        ChunkCoord::new(0, 0, 0),
        1,
    );

    let differ = |a: &str, b: &str| {
        columns
            .iter()
            .any(|&c| value(&runtime, a, c) != value(&runtime, b, c))
    };
    for &column in &columns {
        assert_eq!(
            value(&runtime, "first", column),
            value(&runtime, "second", column)
        );
    }
    assert!(differ("first", "other"), "two names are two streams");
    assert!(
        differ("plain_a", "plain_b"),
        "an unnamed noise is its stage's own"
    );
}

#[test]
fn loading_refuses_parameters_an_expression_cannot_use_and_names_the_stage() {
    let refused = [
        Expr::Curve(constant(0.0), vec![(0.0, 1.0)]),
        Expr::Curve(constant(0.0), vec![(0.0, 1.0), (0.0, 2.0)]),
        Expr::Clamp(constant(0.0), 2.0, 1.0),
        Expr::Smoothstep(1.0, 1.0, constant(0.0)),
        Expr::Remap(constant(0.0), (3.0, 3.0), (0.0, 1.0)),
        Expr::Distance((f32::NAN, 0.0)),
        Expr::Select {
            when: Condition::Less(constant(0.0), b(Expr::Clamp(constant(0.0), 1.0, 0.0))),
            then: constant(0.0),
            otherwise: constant(0.0),
        },
    ];

    for expr in refused {
        let error =
            pack(vec![field("bad", expr.clone())]).expect_err(&format!("{expr:?} is refused"));

        assert!(
            matches!(&error, PackError::Invalid { stage, .. } if stage == "bad"),
            "{expr:?}: {error}"
        );
    }
}

#[test]
fn a_select_reads_the_fields_its_branches_and_condition_name() {
    let result = pack(vec![field(
        "choice",
        Expr::Select {
            when: Condition::Greater(b(Expr::Input("missing".to_owned())), constant(0.0)),
            then: constant(1.0),
            otherwise: constant(0.0),
        },
    )]);

    assert!(
        matches!(&result, Err(PackError::UnknownInput { stage, input }) if stage == "choice" && input == "missing"),
        "{result:?}"
    );
}

/// Valheim's base height, scaled so an area of a few hundred cells holds the flattened spawn and
/// the world's edge: products of noises, a ridge mask that flattens the land near the centre, and
/// a fall to -0.2 past the edge. After the community port of `WorldGenerator.GetBaseHeight`, with
/// the game's Perlin noise replaced by the library's value noise and every distance divided by 100.
fn valheim_base_height() -> Expr {
    let n = |frequency: f32, name: &str| b(noise(frequency, name));
    let mul = |a: Box<Expr>, c: Box<Expr>| b(Expr::Mul(a, c));
    let add = |a: Box<Expr>, c: Box<Expr>| b(Expr::Add(a, c));
    let first = mul(n(0.05, "a1"), n(0.075, "a2"));
    let second = add(
        first.clone(),
        mul(
            mul(mul(n(0.1, "b1"), n(0.15, "b2")), first.clone()),
            constant(0.9),
        ),
    );
    let third = add(
        second.clone(),
        mul(mul(mul(n(0.25, "c1"), n(0.5, "c2")), constant(0.5)), second),
    );
    let land = b(Expr::Sub(third, constant(0.07)));
    let ridge = b(Expr::Abs(b(Expr::Sub(n(0.025, "r1"), n(0.025, "r2")))));
    let ridge_step = b(Expr::Clamp(
        b(Expr::Remap(ridge, (0.02, 0.12), (0.0, 1.0))),
        0.0,
        1.0,
    ));
    let flatten = mul(
        b(Expr::Sub(constant(1.0), ridge_step)),
        b(Expr::Smoothstep(7.44, 10.0, b(Expr::Distance((0.0, 0.0))))),
    );
    let kept = mul(land, b(Expr::Sub(constant(1.0), flatten)));
    let edge = b(Expr::Clamp(
        b(Expr::Remap(
            b(Expr::Distance((0.0, 0.0))),
            (100.0, 105.0),
            (0.0, 1.0),
        )),
        0.0,
        1.0,
    ));
    Expr::Add(kept.clone(), mul(b(Expr::Sub(constant(-0.2), kept)), edge))
}

#[test]
fn valheims_base_height_as_one_field_stage_matches_its_formula() {
    let names = ["a1", "a2", "b1", "b2", "c1", "c2", "r1", "r2"];
    let frequencies = [0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 0.025, 0.025];
    let mut stages = vec![field("height", valheim_base_height())];
    for (name, frequency) in names.iter().zip(frequencies) {
        stages.push(field(name, noise(frequency, name)));
    }
    let mut targets = vec!["height"];
    targets.extend(names);
    let (runtime, columns) = generate(stages, &targets, ChunkCoord::new(6, 0, 0), 7);

    let (mut near, mut beyond) = (0, 0);
    for column in columns {
        let n = |name: &str| value(&runtime, name, column);
        let (x, y) = (column.0 as f32 + 0.5, column.1 as f32 + 0.5);
        let distance = x.hypot(y);
        let first = n("a1") * n("a2");
        let second = first + n("b1") * n("b2") * first * 0.9;
        let third = second + n("c1") * n("c2") * 0.5 * second;
        let land = third - 0.07;
        let ridge = (n("r1") - n("r2")).abs();
        let ridge_step = ((ridge - 0.02) / 0.1).clamp(0.0, 1.0);
        let t = ((distance - 7.44) / (10.0 - 7.44)).clamp(0.0, 1.0);
        let flatten = (1.0 - ridge_step) * (t * t * (3.0 - 2.0 * t));
        let kept = land * (1.0 - flatten);
        let edge = ((distance - 100.0) / 5.0).clamp(0.0, 1.0);
        let expected = kept + (-0.2 - kept) * edge;

        let got = value(&runtime, "height", column);

        assert_close(got, expected, &format!("column {column:?}"));
        if distance < 7.44 {
            near += 1;
        }
        if distance > 105.0 {
            beyond += 1;
            assert_close(got, -0.2, &format!("past the edge at {column:?}"));
        }
    }
    assert!(
        near > 0 && beyond > 0,
        "the area holds the spawn ({near}) and the edge ({beyond})"
    );
}
