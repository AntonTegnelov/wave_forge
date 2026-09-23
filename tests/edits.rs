//! Edits: a felled tree, a moved one and raised ground, applied as chunks are generated, surviving
//! eviction and regeneration, and regenerating only what they reach.

use std::sync::Arc;
use wave_forge::stages::{Edit, Edits, Pack, Point, PointId, Runtime, StageError};
use wave_forge::{ChunkCoord, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "ground", kind: Field(Mul(Noise(frequency: 0.05, octaves: 2), Constant(20.0)))),
        (name: "smooth", kind: Blur(input: "ground", radius: 2)),
        (name: "trees", kind: Scatter(kind: "tree", height: "ground", spacing: 3)),
    ],
)"#;

const SIZE: [u32; 2] = [8, 8];

fn runtime() -> Runtime {
    Runtime::new(Arc::new(Pack::parse(PACK).expect("a valid pack")), 5, SIZE)
}

/// Asks for every stage around `centre` and generates it.
fn around(runtime: &mut Runtime, centre: ChunkCoord) {
    runtime
        .request(
            &[FocusPoint::new(centre, 1)],
            &["ground", "smooth", "trees"],
        )
        .expect("stages");
    runtime.run_until_idle().expect("the stages run");
}

fn trees(runtime: &Runtime, chunk: ChunkCoord) -> Vec<Point> {
    runtime.points("trees", chunk).expect("generated").to_vec()
}

fn value(runtime: &Runtime, stage: &str, x: u32, y: u32) -> f32 {
    runtime
        .field(stage, ChunkCoord::new(0, 0, 0))
        .expect("generated")
        .get(x, y)
}

#[test]
fn a_felled_tree_stays_felled_through_eviction_and_drops_only_its_chunk() {
    let mut runtime = runtime();
    let origin = ChunkCoord::new(0, 0, 0);
    around(&mut runtime, origin);
    let before = trees(&runtime, origin);
    let felled = &before[0];
    let mut edits = Edits::default();
    edits.push(Edit::Remove {
        point: PointId::from(felled.id),
        at: [felled.position[0], felled.position[1]],
    });

    let dropped = runtime.set_edits(&edits).expect("edits");
    around(&mut runtime, origin);
    let after = trees(&runtime, origin);
    around(&mut runtime, ChunkCoord::new(40, 0, 0));
    around(&mut runtime, origin);

    assert_eq!(dropped, vec![("trees".to_owned(), origin)]);
    assert_eq!(after, before[1..].to_vec());
    assert_eq!(
        trees(&runtime, origin),
        after,
        "still felled after eviction"
    );
}

#[test]
fn a_moved_tree_stands_where_it_was_put_in_the_chunk_it_came_from() {
    let mut runtime = runtime();
    let origin = ChunkCoord::new(0, 0, 0);
    around(&mut runtime, origin);
    let tree = trees(&runtime, origin)[0].clone();
    let mut edits = Edits::default();
    edits.push(Edit::Move {
        point: PointId::from(tree.id),
        from: [tree.position[0], tree.position[1]],
        to: [9.5, -2.0, 3.0],
        turn: 0.25,
    });

    runtime.set_edits(&edits).expect("edits");
    around(&mut runtime, origin);

    let moved = trees(&runtime, origin)
        .into_iter()
        .find(|point| point.id == tree.id)
        .expect("the moved tree");
    assert_eq!(moved.position, [9.5, -2.0, 3.0]);
    assert_eq!(moved.turn, 0.25);
}

#[test]
fn raised_ground_reaches_what_reads_it_and_nothing_further() {
    let mut runtime = runtime();
    let origin = ChunkCoord::new(0, 0, 0);
    around(&mut runtime, origin);
    let (ground, smooth) = (
        value(&runtime, "ground", 5, 5),
        value(&runtime, "smooth", 5, 5),
    );
    let mut edits = Edits::default();
    edits.push(Edit::Raise {
        stage: "ground".to_owned(),
        column: (5, 5),
        by: 7.0,
    });

    let dropped = runtime.set_edits(&edits).expect("edits");
    around(&mut runtime, origin);

    // Staleness is kept per chunk: the raise dirties ground chunk (0, 0), and every chunk whose
    // reach covers it regenerates, a blur's neighbours and the trees standing on it, no further.
    let ground_only = dropped
        .iter()
        .filter(|(stage, _)| stage == "ground")
        .all(|(_, chunk)| *chunk == origin);
    let near = dropped
        .iter()
        .all(|(_, chunk)| chunk.x.abs() <= 1 && chunk.y.abs() <= 1);
    assert!(ground_only && near, "{dropped:?}");
    assert!(dropped.contains(&("smooth".to_owned(), origin)));
    assert_eq!(value(&runtime, "ground", 5, 5), ground + 7.0);
    assert!((value(&runtime, "smooth", 5, 5) - (smooth + 7.0 / 25.0)).abs() < 1e-4);
}

#[test]
fn a_sample_holds_the_raise_its_chunk_holds() {
    let mut runtime = runtime();
    let mut edits = Edits::default();
    edits.push(Edit::Raise {
        stage: "ground".to_owned(),
        column: (5, 5),
        by: 7.0,
    });
    runtime.set_edits(&edits).expect("edits");
    around(&mut runtime, ChunkCoord::new(0, 0, 0));

    let sampled = runtime.sample("smooth", [4.5, 6.5]).expect("a sample");

    assert_eq!(sampled, value(&runtime, "smooth", 4, 6));
}

#[test]
fn an_edits_log_saves_and_loads_as_it_was() {
    let mut edits = Edits::default();
    edits.push(Edit::Raise {
        stage: "ground".to_owned(),
        column: (-3, 12),
        by: 1.5,
    });
    edits.push(Edit::Remove {
        point: PointId {
            chunk: (2, -1),
            local: 42,
        },
        at: [17.5, -3.25],
    });

    let text = ron::to_string(&edits).expect("RON");
    let loaded: Edits = ron::from_str(&text).expect("RON");

    assert_eq!(loaded, edits);
}

#[test]
fn an_edit_of_nothing_the_pack_makes_is_refused_and_changes_nothing() {
    let mut runtime = runtime();
    let raise_points = Edits {
        log: vec![Edit::Raise {
            stage: "trees".to_owned(),
            column: (0, 0),
            by: 1.0,
        }],
    };
    let unknown_point = Edits {
        log: vec![Edit::Remove {
            point: PointId {
                chunk: (0, 0),
                local: 1,
            },
            at: [0.5, 0.5],
        }],
    };

    let raised = runtime.set_edits(&raise_points);
    let removed = runtime.set_edits(&unknown_point);

    assert!(matches!(raised, Err(StageError::Edit(_))), "{raised:?}");
    assert!(matches!(removed, Err(StageError::Edit(_))), "{removed:?}");
}
