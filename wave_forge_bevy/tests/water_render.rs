//! The water checks' valley drawn by Bevy's own renderer: its ground and its river's water, unlit,
//! seen from straight above over a magenta background. Water shows along the river and no magenta
//! pixel shows between it and its banks.
//!
//! ```text
//! cargo test -p wave_forge_bevy --release --test water_render -- --ignored --nocapture
//! ```
//!
//! `#[ignore]`d: it needs a device. The app is headless and renders into an image, which it reads
//! back from the GPU; the materials are unlit and the camera does no tonemapping, so what comes
//! back is their colours.

use bevy::app::PluginsState;
use bevy::asset::RenderAssetUsages;
use bevy::camera::{ClearColorConfig, ImageRenderTarget, RenderTarget, ScalingMode};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::view::Msaa;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wave_forge::stages::{Facts, GivenRow, Pack, Runtime, Value};
use wave_forge::{ChunkCoord, FocusPoint, ground, water_surface};
use wave_forge_bevy::stages::{ground_mesh, water_mesh};

const SIZE: u32 = 128;
const CELL: [f32; 3] = [2.0, 1.0, 2.0];
const GROUND: [u8; 3] = [128, 128, 128];
const WATER: [u8; 3] = [26, 90, 230];
/// The square of the ground plane the camera sees: from 50 to 80 along x and z.
const LOW: f32 = 50.0;
const SPAN: f32 = 30.0;

#[derive(Resource, Default)]
struct Pixels(Option<Vec<u8>>);

/// The valley's runtime with its river, generated around the river's middle.
fn runtime() -> Runtime {
    let text = std::fs::read_to_string(format!(
        "{}/../tests/fixtures/water.world.ron",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("the water pack");
    let pack = Arc::new(Pack::parse(&text).expect("a valid pack"));
    let mut facts = Facts::new(Arc::clone(&pack), 9).expect("facts");
    let river = GivenRow {
        id: 1,
        values: BTreeMap::from([
            ("x0".to_owned(), Value::Number(2.0)),
            ("y0".to_owned(), Value::Number(32.0)),
            ("x1".to_owned(), Value::Number(62.0)),
            ("y1".to_owned(), Value::Number(32.0)),
            ("width".to_owned(), Value::Number(1.5)),
        ]),
    };
    facts.give("rivers", vec![river]).expect("the river");
    let mut runtime = Runtime::new(pack, 9, [8, 8]);
    runtime.set_facts(facts).expect("facts");
    runtime
        .request(
            &[FocusPoint::new(ChunkCoord::new(3, 3, 0), 3)],
            &["ground", "water"],
        )
        .expect("stages");
    runtime.run_until_idle().expect("the stages run");
    runtime
}

/// The valley's ground and water around the river, unlit, from straight above: `SIZE` by `SIZE`
/// pixels, RGBA.
fn render() -> Vec<u8> {
    let runtime = runtime();
    let mut app = App::new();
    app.add_plugins(DefaultPlugins).init_resource::<Pixels>();
    // What `App::run` does before the first frame: let the renderer's device future resolve.
    let started = Instant::now();
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "the renderer did not become ready"
        );
    }
    app.finish();
    app.cleanup();
    let world = app.world_mut();
    let mut target = Image::new_fill(
        Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    target.texture_descriptor.usage |=
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC | TextureUsages::TEXTURE_BINDING;
    let target = world.resource_mut::<Assets<Image>>().add(target);
    let unlit = |colour: [u8; 3]| StandardMaterial {
        base_color: Color::srgb_u8(colour[0], colour[1], colour[2]),
        unlit: true,
        ..default()
    };
    let ground_material = world
        .resource_mut::<Assets<StandardMaterial>>()
        .add(unlit(GROUND));
    let water_material = world
        .resource_mut::<Assets<StandardMaterial>>()
        .add(unlit(WATER));
    for y in 2..=5 {
        for x in 2..=5 {
            let chunk = ChunkCoord::new(x, y, 0);
            let corner = Vec3::new(x as f32 * 16.0, 0.0, y as f32 * 16.0);
            let floor = ground(chunk, |at| runtime.field("ground", at), CELL).expect("a ground");
            let water = water_surface(
                chunk,
                |at| runtime.field("water", at),
                |at| runtime.field("ground", at),
                CELL,
            )
            .expect("its water");
            let floor = world
                .resource_mut::<Assets<Mesh>>()
                .add(ground_mesh(&floor));
            world.spawn((
                Mesh3d(floor),
                MeshMaterial3d(ground_material.clone()),
                Transform::from_translation(corner),
            ));
            if !water.indices.is_empty() {
                let water = world.resource_mut::<Assets<Mesh>>().add(water_mesh(&water));
                world.spawn((
                    Mesh3d(water),
                    MeshMaterial3d(water_material.clone()),
                    Transform::from_translation(corner),
                ));
            }
        }
    }
    let centre = LOW + SPAN / 2.0;
    world.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::srgb(1.0, 0.0, 1.0)),
            ..default()
        },
        RenderTarget::Image(ImageRenderTarget::from(target.clone())),
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::Fixed {
                width: SPAN,
                height: SPAN,
            },
            ..OrthographicProjection::default_3d()
        }),
        Tonemapping::None,
        Msaa::Off,
        Transform::from_xyz(centre, 50.0, centre)
            .looking_at(Vec3::new(centre, 0.0, centre), Vec3::NEG_Z),
    ));
    world.spawn(Readback::texture(target)).observe(
        |readback: On<ReadbackComplete>, mut pixels: ResMut<Pixels>| {
            pixels.0 = Some(readback.data.clone());
        },
    );

    let started = Instant::now();
    let mut frames = 0;
    // Pipelines compile over the first frames; keep reading until the picture has settled.
    while frames < 60 || app.world().resource::<Pixels>().0.is_none() {
        app.update();
        frames += 1;
        assert!(started.elapsed() < Duration::from_secs(60), "no picture");
    }
    app.world()
        .resource::<Pixels>()
        .0
        .clone()
        .expect("read back")
}

#[test]
#[ignore = "needs a device; run with --ignored in release mode"]
fn water_shows_along_the_river_with_no_gap_against_its_banks() {
    let pixels = render();

    let pixel = |x: u32, y: u32| {
        let at = ((y * SIZE + x) * 4) as usize;
        [pixels[at], pixels[at + 1], pixels[at + 2]]
    };
    let near = |a: [u8; 3], b: [u8; 3]| a.iter().zip(b).all(|(a, b)| a.abs_diff(b) <= 3);
    let (mut gaps, mut water) = (0, 0);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let colour = pixel(x, y);
            gaps += usize::from(colour[0] > 200 && colour[1] < 50 && colour[2] > 200);
            water += usize::from(near(colour, WATER));
        }
    }
    // The river runs along z = 64 world units, 14 of the 30 the camera sees from its top edge.
    let row = ((64.0 - LOW) / SPAN * SIZE as f32) as u32;
    let across = pixel(SIZE / 2, row);
    println!("{water} water pixels, {gaps} gap pixels, {across:?} on the river's line");

    assert_eq!(gaps, 0, "the background shows through");
    assert!(near(across, WATER), "{across:?} on the river's line");
    assert!(water > 200, "only {water} water pixels");
}
