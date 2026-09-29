//! Embed stages: points inside a volume's rock, ore say, at hashed places and heights, kept where
//! the rock is solid and the stage's conditions hold at the point's height.

use std::collections::BTreeSet;
use std::sync::Arc;
use wave_forge::stages::{Edit, Edits, Pack, PackError, PointId, Runtime};
use wave_forge::{ChunkCoord, FocusPoint};

const PACK: &str = r#"(
    version: 1,
    noises: {"caves": (noise_type: SimplexSmooth, seed: 4, frequency: 0.1, fractal_octaves: 2)},
    stages: [
        (name: "height", kind: Field(Add(Mul(Noise(frequency: 0.05, octaves: 2), Constant(8.0)), Constant(10.0)))),
        (name: "rock", kind: Volume(
            density: Min(Sub(Input("height"), Z), Add(Mul(FastNoise("caves"), Constant(6.0)), Constant(2.0))),
            bottom: -16,
            top: 24,
        )),
        (name: "iron", kind: Embed(kind: "iron", volume: "rock", spacing: 4, count: (2, 4), between: (-16.0, 20.0),
            when: [Less(Z, Sub(Input("height"), Constant(4.0)))])),
    ],
)"#;

fn runtime(chunks: &[ChunkCoord]) -> Runtime {
    let mut runtime = Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        6,
        [8, 8],
    );
    let focus: Vec<FocusPoint> = chunks.iter().map(|&c| FocusPoint::new(c, 0)).collect();
    runtime
        .request(&focus, &["iron", "height"])
        .expect("a stage");
    runtime.run_until_idle().expect("the stages run");
    runtime
}

fn area() -> Vec<ChunkCoord> {
    (-2..2)
        .flat_map(|y| (-2..2).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

#[test]
fn an_embedded_point_lies_in_solid_rock_where_its_conditions_hold() {
    let chunks = area();
    let runtime = runtime(&chunks);

    let mut total = 0;
    for &chunk in &chunks {
        let rock = runtime.volume("rock", chunk).expect("an input");
        let height = runtime.field("height", chunk).expect("generated");
        for point in runtime.points("iron", chunk).expect("generated") {
            let [x, y, z] = point.position;
            let (cx, cy) = (
                x.floor() as i64 - i64::from(chunk.x) * 8,
                y.floor() as i64 - i64::from(chunk.y) * 8,
            );
            assert!(
                (0..8).contains(&cx) && (0..8).contains(&cy),
                "{point:?} outside its chunk"
            );
            // The volume between the voxels below and above the point, as its surface runs.
            let along = z - rock.bottom as f32 - 0.5;
            let below = along.floor().max(0.0) as u32;
            let above = (below + 1).min(rock.size[2] - 1);
            let t = along - along.floor();
            let (low, high) = (
                rock.get(cx as u32, cy as u32, below),
                rock.get(cx as u32, cy as u32, above),
            );
            assert!(low + (high - low) * t > 0.0, "{point:?} in air");
            assert!((-16.0..20.0).contains(&z), "{point:?} outside its heights");
            assert!(
                z < height.get(cx as u32, cy as u32) - 4.0,
                "{point:?} near the surface"
            );
            total += 1;
        }
    }
    assert!(total > 50, "only {total} points");
}

#[test]
fn embedded_points_have_ids_of_their_own_and_come_back_the_same() {
    let chunks = area();
    let first = runtime(&chunks);
    let again = runtime(&chunks);

    let mut ids = BTreeSet::new();
    for &chunk in &chunks {
        let points = first.points("iron", chunk).expect("generated");
        assert_eq!(Some(points), again.points("iron", chunk));
        for point in points {
            assert!(ids.insert(point.id), "{:?} twice", point.id);
        }
    }
}

#[test]
fn a_mined_point_stays_mined_when_its_chunk_is_generated_again() {
    let chunk = ChunkCoord::new(0, 0, 0);
    let mut runtime = runtime(&[chunk]);
    let mined = runtime.points("iron", chunk).expect("generated")[0].clone();

    runtime
        .set_edits(&Edits {
            log: vec![Edit::Remove {
                point: PointId::from(mined.id),
                at: [mined.position[0], mined.position[1]],
            }],
        })
        .expect("an Embed stage's point");
    runtime
        .request(&[FocusPoint::new(ChunkCoord::new(30, 30, 0), 0)], &["iron"])
        .expect("a stage");
    runtime.run_until_idle().expect("the stages run");
    runtime
        .request(&[FocusPoint::new(chunk, 0)], &["iron"])
        .expect("a stage");
    runtime.run_until_idle().expect("the stages run");

    let points = runtime.points("iron", chunk).expect("generated");
    assert!(points.iter().all(|point| point.id != mined.id));
    assert!(!points.is_empty());
}

#[test]
fn a_spacing_of_nothing_heights_the_wrong_way_or_a_coarse_volume_are_refused() {
    let parse = |embed: &str, scale: u32| {
        Pack::parse(&format!(
            r#"(version: 1, stages: [
                (name: "rock", scale: {scale}, kind: Volume(density: Z, bottom: 0, top: 4)),
                (name: "ore", kind: Embed(kind: "ore", volume: "rock", {embed})),
            ])"#
        ))
    };

    let results = [
        parse("spacing: 0, between: (0.0, 1.0)", 1),
        parse("spacing: 4, between: (2.0, 1.0)", 1),
        parse("spacing: 4, between: (0.0, 1.0)", 2),
    ];

    for result in results {
        assert!(
            matches!(&result, Err(PackError::Invalid { stage, .. }) if stage == "ore"),
            "{result:?}"
        );
    }
}

#[test]
fn an_embed_stage_whose_ids_would_share_a_scatter_stages_stage_number_is_refused() {
    // Two names whose salts give the same 15-bit stage number in their points' ids.
    let text = r#"(version: 1, stages: [
        (name: "height", kind: Field(Constant(4.0))),
        (name: "rock", kind: Volume(density: Sub(Input("height"), Z), bottom: 0, top: 8)),
        (name: "ore1169", kind: Scatter(kind: "rock", height: "height", spacing: 4)),
        (name: "ore1486", kind: Embed(kind: "ore", volume: "rock", spacing: 4, between: (0.0, 4.0))),
    ])"#;

    let result = Pack::parse(text);

    assert!(
        matches!(&result, Err(PackError::Invalid { message, .. }) if message.contains("stage number")),
        "{result:?}"
    );
}
