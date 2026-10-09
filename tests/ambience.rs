//! The terrain's ambience: running water along rivers and lapping at a lake's shore, on the water
//! checks' valley. A river's emitters meet across chunk borders at their spacing, a wider river
//! plays louder than a narrower one falling as fast, the lake's shore has an emitter at the lake's
//! level, and a pack's ambience naming the wrong stages is refused.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::ambience::SHORE_VOLUME;
use wave_forge::stages::{Facts, GivenRow, Pack, PackError, Runtime, Value};
use wave_forge::{ChunkCoord, Emitter, FocusPoint};

const CELL: [f32; 3] = [2.0, 1.0, 2.0];
const AMBIENCE: &str = r#"ambience: [
        (key: "water_river", kind: River(curves: "rivers", height: "terrain")),
        (key: "water_lake", kind: Lake(lakes: "lakes", height: "terrain")),
    ],
    stages: ["#;

/// The water checks' valley with `ambience` in place of its stages' opening.
fn pack_text(ambience: &str) -> String {
    std::fs::read_to_string(format!(
        "{}/tests/fixtures/water.world.ron",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("the water pack")
    .replacen("stages: [", ambience, 1)
}

fn river(id: u64, y: f32, width: f32) -> GivenRow {
    GivenRow {
        id,
        values: BTreeMap::from([
            ("x0".to_owned(), Value::Number(2.0)),
            ("y0".to_owned(), Value::Number(y)),
            ("x1".to_owned(), Value::Number(62.0)),
            ("y1".to_owned(), Value::Number(y)),
            ("width".to_owned(), Value::Number(width)),
        ]),
    }
}

/// The valley with its ambience and `rivers`, generated over its 8 by 8 chunks.
fn runtime(rivers: Vec<GivenRow>) -> Runtime {
    let pack = Arc::new(Pack::parse(&pack_text(AMBIENCE)).expect("a valid pack"));
    let mut facts = Facts::new(Arc::clone(&pack), 9).expect("facts");
    facts.give("rivers", rivers).expect("the rivers");
    let mut runtime = Runtime::new(pack, 9, [8, 8]);
    runtime.set_facts(facts).expect("facts");
    let focus: Vec<FocusPoint> = (0..8)
        .flat_map(|y| (0..8).map(move |x| FocusPoint::new(ChunkCoord::new(x, y, 0), 0)))
        .collect();
    runtime
        .request(&focus, &["rivers", "terrain", "lakes"])
        .expect("the stages");
    runtime.run_until_idle().expect("the stages run");
    runtime
}

/// Every emitter of `key` in the chunks 1 to 6 each way, by chunk.
fn emitters(runtime: &Runtime, key: &str) -> BTreeMap<ChunkCoord, Vec<Emitter>> {
    (1..7)
        .flat_map(|y| (1..7).map(move |x| ChunkCoord::new(x, y, 0)))
        .map(|chunk| {
            let all = runtime.ambience(chunk, CELL).expect("arrived");
            (chunk, all.into_iter().filter(|e| e.key == key).collect())
        })
        .collect()
}

#[test]
fn a_rivers_emitters_meet_across_chunk_borders_at_their_spacing() {
    let runtime = runtime(vec![river(1, 32.0, 1.5)]);

    let by_chunk = emitters(&runtime, "water_river");

    let mut along: Vec<f32> = Vec::new();
    for (chunk, emitters) in &by_chunk {
        for emitter in emitters {
            let (x, z) = (emitter.at[0] / CELL[0], emitter.at[2] / CELL[2]);
            assert_eq!(
                (x.floor() as i32).div_euclid(8),
                chunk.x,
                "an emitter at {x} in {chunk:?}"
            );
            assert!((z - 32.0).abs() < 1e-4, "an emitter off the river at {z}");
            along.push(x);
        }
    }
    along.sort_by(f32::total_cmp);
    assert!(
        along.len() > 4,
        "only {} emitters along the river",
        along.len()
    );
    // Six cells for each of the river's 1.5 cells of radius.
    for pair in along.windows(2) {
        assert!(
            (pair[1] - pair[0] - 9.0).abs() < 1e-3,
            "emitters at {pair:?}"
        );
    }
}

#[test]
fn a_wider_river_plays_louder_than_a_narrower_one_falling_as_fast() {
    let runtime = runtime(vec![river(1, 32.0, 3.0), river(2, 44.0, 1.5)]);

    let by_chunk = emitters(&runtime, "water_river");

    let volume = |z: f32| {
        let mut volumes: Vec<f32> = by_chunk
            .values()
            .flatten()
            .filter(|e| (e.at[2] / CELL[2] - z).abs() < 1e-4)
            .map(|e| e.volume)
            .collect();
        volumes.sort_by(f32::total_cmp);
        volumes[volumes.len() / 2]
    };
    let (wide, narrow) = (volume(32.0), volume(44.0));
    assert!(
        wide > narrow,
        "the wide river at {wide}, the narrow one at {narrow}"
    );
}

#[test]
fn the_lakes_shore_plays_at_the_lakes_level_and_dry_ground_does_not() {
    let runtime = runtime(vec![river(1, 32.0, 1.5)]);

    let by_chunk = emitters(&runtime, "water_lake");

    let shores: Vec<(&ChunkCoord, &Emitter)> = by_chunk
        .iter()
        .flat_map(|(chunk, emitters)| emitters.iter().map(move |e| (chunk, e)))
        .collect();
    assert!(!shores.is_empty(), "no shore in the hollow");
    for (chunk, shore) in shores {
        let (x, y) = (
            (shore.at[0] / CELL[0]).floor() as i64,
            (shore.at[2] / CELL[2]).floor() as i64,
        );
        let at = ChunkCoord::new(x.div_euclid(8) as i32, y.div_euclid(8) as i32, 0);
        let level = runtime
            .field("lakes", at)
            .expect("generated")
            .get(x.rem_euclid(8) as u32, y.rem_euclid(8) as u32);
        assert_eq!(shore.at[1], level * CELL[1], "the shore in {chunk:?}");
        assert_eq!(shore.volume, SHORE_VOLUME);
    }
    assert!(
        by_chunk[&ChunkCoord::new(1, 6, 0)].is_empty(),
        "a shore on the dry valley side"
    );
}

#[test]
fn ambience_naming_the_wrong_stages_is_refused() {
    let refused = [
        r#"ambience: [(key: "a", kind: River(curves: "terrain", height: "terrain"))], stages: ["#,
        r#"ambience: [(key: "a", kind: River(curves: "rivers", height: "rivers"))], stages: ["#,
        r#"ambience: [(key: "a", kind: Lake(lakes: "terrain", height: "terrain"))], stages: ["#,
        r#"ambience: [
            (key: "a", kind: River(curves: "rivers", height: "terrain")),
            (key: "a", kind: Lake(lakes: "lakes", height: "terrain")),
        ], stages: ["#,
    ];

    for ambience in refused {
        let result = Pack::parse(&pack_text(ambience));

        assert!(
            matches!(&result, Err(PackError::Ambience { key, .. }) if key == "a"),
            "{ambience}: {result:?}"
        );
    }
}

#[test]
fn a_river_with_a_point_repeated_still_sounds_along_its_course() {
    use wave_forge::ambience::river_emitters;
    use wave_forge::stages::regions::{Curve, CurveId};
    // Down x across a chunk of 16 columns, its fourth point the third again.
    let river = Curve {
        id: CurveId::Region {
            region: (0, 0),
            index: 0,
        },
        points: vec![
            [0.5, 8.5],
            [5.5, 8.5],
            [10.5, 8.5],
            [10.5, 8.5],
            [15.5, 8.5],
        ],
        values: vec![1.0; 5],
        heights: Vec::new(),
    };

    let emitters = river_emitters(
        "water_river",
        &[river],
        ChunkCoord::new(0, 0, 0),
        [16, 16],
        |x, _| Some(20.0 - x as f32),
        [1.0; 3],
    )
    .expect("every height read");

    assert!(!emitters.is_empty(), "no emitters");
    for emitter in &emitters {
        assert!(
            emitter.at.iter().all(|c| c.is_finite()) && emitter.volume.is_finite(),
            "{emitter:?}"
        );
    }
}
