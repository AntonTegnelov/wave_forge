//! Saves: a frozen stage keeps the chunks it first generated after the pack changes, an
//! ephemeral stage's edits are never saved, and a save records the generator and the pack. The
//! ring world freezes its shrines and never saves its grass.

use std::sync::Arc;
use wave_forge::stages::{Edit, Edits, Pack, Point, PointId, Runtime, Save};
use wave_forge::{ChunkCoord, FocusPoint};

fn pack(tree_spacing: u32) -> Arc<Pack> {
    Arc::new(
        Pack::parse(&format!(
            r#"(version: 1, stages: [
                (name: "ground", kind: Field(Mul(Noise(frequency: 0.05, octaves: 2), Constant(10.0)))),
                (name: "trees", persist: Frozen, kind: Scatter(kind: "tree", height: "ground", spacing: {tree_spacing})),
                (name: "grass", persist: Ephemeral, kind: Scatter(kind: "grass", height: "ground", spacing: 2)),
                (name: "rocks", kind: Scatter(kind: "rock", height: "ground", spacing: 4)),
            ])"#
        ))
        .expect("a valid pack"),
    )
}

/// The points of `stage` in `chunk`, after asking for every scatter around it.
fn points(runtime: &mut Runtime, stage: &str, chunk: ChunkCoord) -> Vec<Point> {
    runtime
        .request(&[FocusPoint::new(chunk, 0)], &["trees", "grass", "rocks"])
        .expect("stages");
    runtime.run_until_idle().expect("the stages run");
    runtime.points(stage, chunk).expect("generated").to_vec()
}

fn removal(point: &Point) -> Edit {
    Edit::Remove {
        point: PointId::from(point.id),
        at: [point.position[0], point.position[1]],
    }
}

#[test]
fn a_frozen_stage_keeps_the_chunks_it_first_generated_after_the_pack_changes() {
    let origin = ChunkCoord::new(0, 0, 0);
    let mut before = Runtime::new(pack(3), 2, [8, 8]);
    let frozen = points(&mut before, "trees", origin);
    let save = before.save();

    let mut after = Runtime::new(pack(5), 2, [8, 8]);
    after.load(&save).expect("a save");
    let changed = points(&mut Runtime::new(pack(5), 2, [8, 8]), "trees", origin);

    assert_ne!(frozen, changed, "the pack's change moves the trees");
    assert_eq!(points(&mut after, "trees", origin), frozen);
    let elsewhere = ChunkCoord::new(6, 0, 0);
    assert_eq!(
        points(&mut after, "trees", elsewhere),
        points(&mut Runtime::new(pack(5), 2, [8, 8]), "trees", elsewhere),
        "a chunk never generated before follows the new pack"
    );
}

#[test]
fn an_ephemeral_stages_edits_are_never_saved() {
    let origin = ChunkCoord::new(0, 0, 0);
    let mut playing = Runtime::new(pack(3), 2, [8, 8]);
    let grass = points(&mut playing, "grass", origin)[0].clone();
    let rock = points(&mut playing, "rocks", origin)[0].clone();
    let edits = Edits {
        log: vec![removal(&grass), removal(&rock)],
    };
    playing.set_edits(&edits).expect("edits");

    let save = playing.save();
    let mut loaded = Runtime::new(pack(3), 2, [8, 8]);
    loaded.load(&save).expect("a save");

    assert_eq!(save.edits.log, vec![removal(&rock)]);
    assert!(
        points(&mut loaded, "grass", origin).contains(&grass),
        "the grass grows back"
    );
    assert!(
        !points(&mut loaded, "rocks", origin).contains(&rock),
        "the rock stays gone"
    );
}

#[test]
fn a_save_records_the_generator_and_the_pack() {
    let reformatted = Arc::new(
        Pack::parse(
            r#"(version:1,stages:[(name:"ground",kind:Field(Mul(Noise(frequency:0.05,octaves:2),Constant(10.0)))),
            (name:"trees",persist:Frozen,kind:Scatter(kind:"tree",height:"ground",spacing:3)),
            (name:"grass",persist:Ephemeral,kind:Scatter(kind:"grass",height:"ground",spacing:2)),
            (name:"rocks",kind:Scatter(kind:"rock",height:"ground",spacing:4))])"#,
        )
        .expect("a valid pack"),
    );

    let save = Runtime::new(pack(3), 2, [8, 8]).save();

    assert_eq!(save.generator, env!("CARGO_PKG_VERSION"));
    assert_eq!(save.pack, pack(3).digest());
    assert_eq!(
        pack(3).digest(),
        reformatted.digest(),
        "the same pack however it is written"
    );
    assert_ne!(pack(3).digest(), pack(5).digest());
}

#[test]
fn a_save_loads_as_it_was_written() {
    let mut runtime = Runtime::new(pack(3), 2, [8, 8]);
    let rock = points(&mut runtime, "rocks", ChunkCoord::new(0, 0, 0))[0].clone();
    runtime
        .set_edits(&Edits {
            log: vec![removal(&rock)],
        })
        .expect("edits");
    let save = runtime.save();

    let loaded = Save::from_ron(&save.to_ron()).expect("a save");

    assert_eq!(loaded, save);
    assert!(!save.frozen.is_empty());
}

#[test]
fn the_ring_world_saves_its_shrines_and_never_its_grass() {
    let text = std::fs::read_to_string(format!(
        "{}/examples/rings.world.ron",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("the ring world");
    let pack = Arc::new(Pack::parse(&text).expect("a valid pack"));
    let mut runtime = Runtime::new(pack, 7, [8, 8]);
    let origin = ChunkCoord::new(0, 0, 0);
    runtime
        .request(&[FocusPoint::new(origin, 1)], &["shrines", "grass"])
        .expect("stages");
    runtime.run_until_idle().expect("the stages run");
    let grass = runtime.points("grass", origin).expect("generated")[0].clone();
    let mut edits = Edits::default();
    edits.push(removal(&grass));
    runtime.set_edits(&edits).expect("edits");

    let save = runtime.save();

    assert!(save.edits.log.is_empty(), "{:?}", save.edits);
    assert!(!save.frozen.is_empty());
    assert!(save.frozen.iter().all(|chunk| chunk.stage == "shrines"));
}
