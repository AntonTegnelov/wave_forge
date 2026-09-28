//! Digs and fills: balls a player digs out of a volume or fills into it, kept in the edits log,
//! applied in its order and applied again whenever a chunk is generated.

use std::sync::Arc;
use wave_forge::stages::{Edit, Edits, Pack, Runtime, StageError, Volume};
use wave_forge::{ChunkCoord, FocusPoint};

const SIZE: [u32; 2] = [8, 8];

// Ground ten cells deep, with its surface at 10 cells up.
const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "height", kind: Field(Constant(10.0))),
        (name: "ground", kind: Volume(density: Sub(Input("height"), Z), bottom: 0, top: 20)),
    ],
)"#;

const HERE: ChunkCoord = ChunkCoord::new(1, 1, 0);

fn runtime() -> Runtime {
    Runtime::new(Arc::new(Pack::parse(PACK).expect("a valid pack")), 5, SIZE)
}

fn dig(at: [f32; 3], radius: f32) -> Edit {
    Edit::Dig {
        stage: "ground".to_owned(),
        at,
        radius,
    }
}

fn fill(at: [f32; 3], radius: f32) -> Edit {
    Edit::Fill {
        stage: "ground".to_owned(),
        at,
        radius,
    }
}

fn generate(runtime: &mut Runtime, chunk: ChunkCoord) -> Volume {
    runtime
        .request(&[FocusPoint::new(chunk, 0)], &["ground"])
        .expect("a stage");
    runtime.run_until_idle().expect("the stages run");
    runtime.volume("ground", chunk).expect("generated").clone()
}

fn edited(log: Vec<Edit>) -> Volume {
    let mut runtime = runtime();
    runtime
        .set_edits(&Edits { log })
        .expect("edits of a volume");
    generate(&mut runtime, HERE)
}

/// How far the centre of each voxel of `volume` is from `at`, in cells, with the voxel's index.
fn distances(volume: &Volume, at: [f32; 3]) -> Vec<(f32, usize)> {
    let [sx, sy, levels] = volume.size;
    let mut out = Vec::new();
    for level in 0..levels {
        for y in 0..sy {
            for x in 0..sx {
                let voxel = [
                    (volume.chunk.x * sx as i32 + x as i32) as f32 + 0.5,
                    (volume.chunk.y * sy as i32 + y as i32) as f32 + 0.5,
                    (volume.bottom + level as i32) as f32 + 0.5,
                ];
                let distance = (0..3)
                    .map(|axis| (voxel[axis] - at[axis]).powi(2))
                    .sum::<f32>()
                    .sqrt();
                out.push((distance, ((level * sy + y) * sx + x) as usize));
            }
        }
    }
    out
}

#[test]
fn a_dig_is_empty_inside_its_ball_and_unchanged_beyond_a_cell_of_it() {
    let at = [12.0, 12.0, 9.0];
    let plain = generate(&mut runtime(), HERE);

    let dug = edited(vec![dig(at, 3.0)]);

    let (mut inside, mut beyond) = (0, 0);
    for (distance, i) in distances(&dug, at) {
        if distance < 3.0 {
            assert!(dug.values[i] < 0.0, "a voxel {distance} from the centre");
            inside += 1;
        } else if distance > 4.0 {
            assert_eq!(dug.values[i], plain.values[i]);
            beyond += 1;
        }
    }
    assert!(
        inside > 50 && beyond > 500,
        "{inside} inside, {beyond} beyond"
    );
}

#[test]
fn a_fill_is_solid_inside_its_ball() {
    let at = [12.0, 12.0, 14.0];

    let filled = edited(vec![fill(at, 2.5)]);

    let mut inside = 0;
    for (distance, i) in distances(&filled, at) {
        if distance < 2.5 {
            assert!(filled.values[i] > 0.0, "a voxel {distance} from the centre");
            inside += 1;
        }
    }
    assert!(inside > 20, "{inside} inside");
}

#[test]
fn digs_and_fills_apply_in_the_order_they_were_made() {
    let at = [12.0, 12.0, 9.0];

    let dug_then_filled = edited(vec![dig(at, 3.0), fill(at, 3.0)]);
    let filled_then_dug = edited(vec![fill(at, 3.0), dig(at, 3.0)]);

    let (_, centre) = distances(&dug_then_filled, at)
        .into_iter()
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .expect("voxels");
    assert!(dug_then_filled.values[centre] > 0.0);
    assert!(filled_then_dug.values[centre] < 0.0);
}

#[test]
fn a_dig_stales_only_its_chunks_and_survives_leaving_and_coming_back() {
    let mut runtime = runtime();
    let before = generate(&mut runtime, HERE);
    let far = ChunkCoord::new(9, 9, 0);

    let stale = runtime
        .set_edits(&Edits {
            log: vec![dig([12.0, 12.0, 9.0], 3.0)],
        })
        .expect("edits of a volume");
    let arrived = generate(&mut runtime, HERE);
    generate(&mut runtime, far);
    let returned = generate(&mut runtime, HERE);

    assert!(stale.contains(&("ground".to_owned(), HERE)), "{stale:?}");
    assert!(
        stale
            .iter()
            .all(|(_, chunk)| (0..=2).contains(&chunk.x) && (0..=2).contains(&chunk.y)),
        "{stale:?}"
    );
    assert_ne!(arrived, before);
    assert_eq!(returned, arrived);
}

#[test]
fn a_save_keeps_the_digs_and_fills() {
    let mut runtime = runtime();
    let log = vec![dig([12.0, 12.0, 9.0], 3.0), fill([3.0, 3.0, 12.0], 1.5)];
    runtime
        .set_edits(&Edits { log: log.clone() })
        .expect("edits of a volume");

    let save = runtime.save();

    assert_eq!(save.edits.log, log);
}

#[test]
fn a_dig_of_a_field_or_a_ball_without_size_is_refused() {
    let mut runtime = runtime();

    let field = runtime.set_edits(&Edits {
        log: vec![Edit::Dig {
            stage: "height".to_owned(),
            at: [1.0, 1.0, 1.0],
            radius: 2.0,
        }],
    });
    let empty = runtime.set_edits(&Edits {
        log: vec![dig([1.0, 1.0, 1.0], 0.0)],
    });

    assert!(matches!(field, Err(StageError::Edit(_))), "{field:?}");
    assert!(matches!(empty, Err(StageError::Edit(_))), "{empty:?}");
}
