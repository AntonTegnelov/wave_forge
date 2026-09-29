//! Volume stages: a value per voxel, from an expression that reads the voxel's height as `Z`, the
//! fields of its column, and Godot's noise in 3D.

use std::sync::Arc;
use wave_forge::noise::{NoiseConfig, NoiseType};
use wave_forge::stages::{Pack, PackError, Runtime, Volume};
use wave_forge::{ChunkCoord, FocusPoint};

const SIZE: [u32; 2] = [8, 8];

const PACK: &str = r#"(
    version: 1,
    noises: {
        "caves": (noise_type: Perlin, seed: 3, frequency: 0.08, fractal_octaves: 2),
    },
    stages: [
        (name: "height", kind: Field(Add(Mul(Noise(frequency: 0.05, octaves: 2), Constant(8.0)), Constant(4.0)))),
        (name: "ground", kind: Volume(density: Sub(Input("height"), Z), bottom: -2, top: 14)),
        (name: "caves", kind: Volume(density: FastNoise("caves"), bottom: -4, top: 4)),
        (name: "coarse", scale: 4, kind: Volume(density: Z, bottom: -1, top: 2)),
        (name: "surface", kind: Top(volume: "ground")),
        (name: "roof", scale: 4, kind: Top(volume: "coarse")),
        (name: "layered", kind: Volume(
            density: Sub(Input("height"), Z),
            bottom: -2,
            top: 14,
            materials: Some((rules: [
                (category: "grass", when: [Greater(Z, Sub(Input("height"), Constant(1.0)))]),
                (category: "dirt", when: [Greater(Z, Sub(Input("height"), Constant(3.0)))]),
            ], otherwise: "stone")),
        )),
    ],
)"#;

/// Stage `stage` generated for `chunk`: its volume and the runtime that holds its inputs.
fn generate(stage: &str, chunk: ChunkCoord) -> (Runtime, Volume) {
    let mut runtime = Runtime::new(Arc::new(Pack::parse(PACK).expect("a valid pack")), 7, SIZE);
    runtime
        .request(&[FocusPoint::new(chunk, 0)], &[stage])
        .expect("a stage");
    runtime.run_until_idle().expect("the stages run");
    let volume = runtime.volume(stage, chunk).expect("generated").clone();
    (runtime, volume)
}

fn load(stage: &str) -> Result<Pack, PackError> {
    Pack::parse(&format!(
        r#"(version: 1, stages: [
            (name: "height", kind: Field(Constant(1.0))),
            {stage},
        ])"#
    ))
}

#[test]
fn a_volume_holds_every_level_of_every_column_from_its_bottom() {
    let chunk = ChunkCoord::new(-1, 2, 0);

    let (_, volume) = generate("ground", chunk);

    assert_eq!(volume.chunk, chunk);
    assert_eq!(volume.size, [8, 8, 16]);
    assert_eq!(volume.bottom, -2);
    assert_eq!(volume.values.len(), 8 * 8 * 16);
}

#[test]
fn a_voxel_reads_its_height_and_its_columns_fields() {
    let chunk = ChunkCoord::new(-1, 2, 0);

    let (runtime, volume) = generate("ground", chunk);

    let height = runtime
        .field("height", chunk)
        .expect("an input of the volume");
    for level in 0..volume.size[2] {
        let z = (volume.bottom + level as i32) as f32 + 0.5;
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(
                    volume.get(x, y, level),
                    height.get(x, y) - z,
                    "voxel ({x}, {y}, {level})"
                );
            }
        }
    }
}

#[test]
fn a_voxel_reads_godots_noise_in_3d_with_its_height_as_godots_y() {
    let chunk = ChunkCoord::new(1, -1, 0);
    let noise = NoiseConfig {
        noise_type: NoiseType::Perlin,
        seed: 3,
        frequency: 0.08,
        fractal_octaves: 2,
        ..NoiseConfig::default()
    };

    let (_, volume) = generate("caves", chunk);

    for level in 0..volume.size[2] {
        let height = (volume.bottom + level as i32) as f32 + 0.5;
        for y in 0..8 {
            for x in 0..8 {
                let at = [(8 + x) as f32 + 0.5, (-8 + y as i32) as f32 + 0.5];
                assert_eq!(
                    volume.get(x, y, level),
                    noise.sample_3d(at[0], height, at[1]),
                    "voxel ({x}, {y}, {level})"
                );
            }
        }
    }
}

#[test]
fn a_coarse_voxel_is_as_tall_as_its_column_is_wide() {
    let chunk = ChunkCoord::new(0, 0, 0);

    let (_, volume) = generate("coarse", chunk);

    assert_eq!(volume.size, [8, 8, 3]);
    assert_eq!(volume.get(0, 0, 0), -2.0);
    assert_eq!(volume.get(5, 3, 1), 2.0);
    assert_eq!(volume.get(7, 7, 2), 6.0);
}

#[test]
fn z_is_refused_outside_a_volume() {
    let result = load(r#"(name: "tall", kind: Field(Z))"#);

    assert!(
        matches!(&result, Err(PackError::Invalid { stage, message })
            if stage == "tall" && message.contains("Z")),
        "{result:?}"
    );
}

#[test]
fn a_volume_without_levels_is_refused() {
    let result = load(r#"(name: "flat", kind: Volume(density: Z, bottom: 3, top: 3))"#);

    assert!(
        matches!(&result, Err(PackError::Invalid { stage, .. }) if stage == "flat"),
        "{result:?}"
    );
}

#[test]
fn a_volume_is_no_field_for_another_stage_to_read() {
    let result = load(
        r#"(name: "rock", kind: Volume(density: Z, bottom: 0, top: 4)),
            (name: "reads", kind: Field(Input("rock")))"#,
    );

    assert!(
        matches!(&result, Err(PackError::Invalid { stage, message })
            if stage == "reads" && message.contains("rock")),
        "{result:?}"
    );
}

#[test]
fn a_voxel_takes_the_first_material_whose_rules_hold_at_its_height() {
    let chunk = ChunkCoord::new(2, 0, 0);

    let (runtime, volume) = generate("layered", chunk);

    let height = runtime
        .field("height", chunk)
        .expect("an input of the volume");
    assert_eq!(volume.materials.len(), volume.values.len());
    for level in 0..volume.size[2] {
        let z = (volume.bottom + level as i32) as f32 + 0.5;
        for y in 0..8 {
            for x in 0..8 {
                let depth = height.get(x, y) - z;
                let expected = if depth < 1.0 {
                    0
                } else if depth < 3.0 {
                    1
                } else {
                    2
                };
                assert_eq!(
                    volume.material(x, y, level),
                    expected,
                    "voxel ({x}, {y}, {level})"
                );
            }
        }
    }
}

#[test]
fn a_volumes_materials_are_its_categories_and_one_without_has_none() {
    let pack = Pack::parse(PACK).expect("a valid pack");

    let (_, plain) = generate("ground", ChunkCoord::new(0, 0, 0));

    assert_eq!(
        pack.kind("layered").expect("a stage").categories(),
        ["grass", "dirt", "stone"]
    );
    assert!(plain.materials.is_empty());
}

fn top_of(stage: &str, chunk: ChunkCoord) -> (Runtime, Vec<f32>) {
    let mut runtime = Runtime::new(Arc::new(Pack::parse(PACK).expect("a valid pack")), 7, SIZE);
    runtime
        .request(&[FocusPoint::new(chunk, 0)], &[stage])
        .expect("a stage");
    runtime.run_until_idle().expect("the stages run");
    let values = runtime
        .field(stage, chunk)
        .expect("generated")
        .values
        .clone();
    (runtime, values)
}

#[test]
fn the_top_of_a_volume_solid_below_a_height_is_that_height() {
    let chunk = ChunkCoord::new(2, -1, 0);

    let (runtime, top) = top_of("surface", chunk);

    let height = runtime.field("height", chunk).expect("an input");
    for (i, value) in top.iter().enumerate() {
        assert!(
            (value - height.values[i]).abs() < 1e-4,
            "column {i}: top {value}, height {}",
            height.values[i]
        );
    }
}

#[test]
fn the_top_of_a_column_solid_to_its_highest_voxel_is_the_volumes_top() {
    let (_, top) = top_of("roof", ChunkCoord::new(0, 0, 0));

    assert!(top.iter().all(|&value| value == 8.0), "{top:?}");
}

#[test]
fn the_top_of_a_volume_of_another_scale_is_refused() {
    let result = load(
        r#"(name: "rock", scale: 2, kind: Volume(density: Z, bottom: 0, top: 4)),
            (name: "surface", kind: Top(volume: "rock"))"#,
    );

    assert!(
        matches!(&result, Err(PackError::Invalid { stage, .. }) if stage == "surface"),
        "{result:?}"
    );
}
