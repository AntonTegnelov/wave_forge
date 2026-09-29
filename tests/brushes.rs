//! Brushes: a stroke along a path becomes the edits it makes, which the runtime then applies as it
//! applies any edit.

use std::sync::Arc;
use wave_forge::stages::brushes::{Brush, stroke};
use wave_forge::stages::{Edit, Edits, Pack, Runtime, StageError};
use wave_forge::{ChunkCoord, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "flat", kind: Field(Constant(4.0))),
        (name: "hills", kind: Field(Mul(Noise(frequency: 0.3, octaves: 2), Constant(6.0)))),
        (name: "rock", kind: Volume(density: Sub(Input("flat"), Z), bottom: -4, top: 8)),
        (name: "trees", kind: Scatter(kind: "tree", height: "flat", spacing: 2)),
    ],
)"#;

fn runtime(stages: &[&str]) -> Runtime {
    let mut runtime = Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        3,
        [8, 8],
    );
    generate(&mut runtime, stages);
    runtime
}

/// `stages` over the chunks from -1 to 2 on each side.
fn generate(runtime: &mut Runtime, stages: &[&str]) {
    let focus: Vec<FocusPoint> = (-1..3)
        .flat_map(|y| (-1..3).map(move |x| FocusPoint::new(ChunkCoord::new(x, y, 0), 0)))
        .collect();
    runtime.request(&focus, stages).expect("the stages");
    runtime.run_until_idle().expect("the stages run");
}

/// The value of field `stage` at a column, from the chunk that holds it.
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

const PATH: [[f32; 3]; 2] = [[4.5, 8.5, 0.0], [20.5, 8.5, 0.0]];

#[test]
fn a_raise_lifts_the_path_by_its_strength_and_fades_to_nothing_at_its_radius() {
    let mut runtime = runtime(&["flat"]);
    let brush = Brush::Raise {
        stage: "flat".to_owned(),
        radius: 3.0,
        strength: 2.0,
    };

    let edits = stroke(&runtime, &brush, &PATH).expect("a field to raise");
    runtime
        .set_edits(&Edits { log: edits })
        .expect("raises of a field");
    generate(&mut runtime, &["flat"]);

    // Across the path, at the middle of the stroke: highest on it, lower further out, flat beyond.
    let across: Vec<f32> = (8..=13).map(|y| value(&runtime, "flat", (12, y))).collect();
    assert_eq!(across[0], 6.0, "on the path");
    assert!(
        across.windows(2).all(|pair| pair[1] <= pair[0]),
        "{across:?}"
    );
    assert_eq!(across[3], 4.0, "at the radius");
    assert_eq!(
        value(&runtime, "flat", (23, 8)),
        4.0,
        "beyond the stroke's end"
    );
}

#[test]
fn a_negative_strength_lowers() {
    let runtime = runtime(&["flat"]);
    let brush = Brush::Raise {
        stage: "flat".to_owned(),
        radius: 2.0,
        strength: -1.0,
    };

    let edits = stroke(&runtime, &brush, &PATH).expect("a field to lower");

    assert!(!edits.is_empty());
    assert!(
        edits
            .iter()
            .all(|edit| matches!(edit, Edit::Raise { by, .. } if *by < 0.0))
    );
}

#[test]
fn a_smooth_brings_each_column_nearer_its_neighbours() {
    let mut runtime = runtime(&["hills"]);
    // How far each column along the path lies from the average of the nine around it.
    let roughness = |runtime: &Runtime| -> f32 {
        (6..=18)
            .map(|x| {
                let mut sum = 0.0;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        sum += value(runtime, "hills", (x + dx, 8 + dy));
                    }
                }
                (sum / 9.0 - value(runtime, "hills", (x, 8))).abs()
            })
            .sum()
    };
    let before = roughness(&runtime);
    let brush = Brush::Smooth {
        stage: "hills".to_owned(),
        radius: 4.0,
        strength: 1.0,
    };

    let edits = stroke(&runtime, &brush, &PATH).expect("a field to smooth");
    runtime
        .set_edits(&Edits { log: edits })
        .expect("raises of a field");
    generate(&mut runtime, &["hills"]);

    let after = roughness(&runtime);
    assert!(after < before * 0.6, "{before} before, {after} after");
}

#[test]
fn a_dig_stroke_leaves_a_tunnel_along_its_path() {
    let mut runtime = runtime(&["rock"]);
    let tunnel = [[2.0, 3.0, 1.5], [14.0, 3.0, 1.5]];
    let brush = Brush::Dig {
        stage: "rock".to_owned(),
        radius: 1.5,
    };

    let edits = stroke(&runtime, &brush, &tunnel).expect("a volume to dig");
    runtime
        .set_edits(&Edits { log: edits })
        .expect("digs of a volume");
    generate(&mut runtime, &["rock"]);

    for x in 2..14 {
        let chunk = ChunkCoord::new(x / 8, 0, 0);
        let rock = runtime.volume("rock", chunk).expect("generated");
        let level = (1 - rock.bottom) as u32;
        assert!(
            rock.get((x % 8) as u32, 3, level) < 0.0,
            "column {x} is not dug"
        );
    }
}

#[test]
fn a_remove_stroke_takes_every_point_near_its_path_and_none_further() {
    let mut runtime = runtime(&["trees"]);
    let brush = Brush::Remove {
        stages: vec!["trees".to_owned()],
        radius: 2.0,
    };
    let all = |runtime: &Runtime| -> Vec<[f32; 3]> {
        (-1..3)
            .flat_map(|y| (-1..3).map(move |x| ChunkCoord::new(x, y, 0)))
            .flat_map(|chunk| {
                runtime
                    .points("trees", chunk)
                    .expect("generated")
                    .iter()
                    .map(|point| point.position)
                    .collect::<Vec<_>>()
            })
            .collect()
    };
    // Within the brush's radius of the path, its rounded ends included.
    let near = |at: [f32; 3]| {
        let x = at[0].clamp(4.5, 20.5);
        (at[0] - x).hypot(at[1] - 8.5) < 2.0
    };
    let before = all(&runtime);

    let edits = stroke(&runtime, &brush, &PATH).expect("points to remove");
    runtime
        .set_edits(&Edits { log: edits })
        .expect("removals of points");
    generate(&mut runtime, &["trees"]);

    let after = all(&runtime);
    assert!(before.iter().filter(|&&at| near(at)).count() >= 5);
    assert!(
        after.iter().all(|&at| !near(at)),
        "a tree near the stroke stands"
    );
    assert_eq!(
        after.len(),
        before.iter().filter(|&&at| !near(at)).count(),
        "a tree away from the stroke went"
    );
}

#[test]
fn a_brush_of_the_wrong_kind_or_size_is_refused() {
    let runtime = runtime(&[]);
    let raise = |stage: &str, radius: f32| Brush::Raise {
        stage: stage.to_owned(),
        radius,
        strength: 1.0,
    };

    let results = [
        stroke(&runtime, &raise("rock", 2.0), &PATH),
        stroke(&runtime, &raise("flat", 0.0), &PATH),
        stroke(&runtime, &raise("flat", 2.0), &[]),
        stroke(
            &runtime,
            &Brush::Dig {
                stage: "flat".to_owned(),
                radius: 1.0,
            },
            &PATH,
        ),
        stroke(
            &runtime,
            &Brush::Remove {
                stages: vec!["rock".to_owned()],
                radius: 1.0,
            },
            &PATH,
        ),
        stroke(
            &runtime,
            &Brush::Smooth {
                stage: "flat".to_owned(),
                radius: 1.0,
                strength: 2.0,
            },
            &PATH,
        ),
    ];

    for result in results {
        assert!(matches!(result, Err(StageError::Edit(_))), "{result:?}");
    }
}
