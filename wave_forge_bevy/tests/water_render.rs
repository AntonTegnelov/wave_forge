//! The water checks' valley drawn by Bevy's own renderer: its ground and its river's water, unlit,
//! seen from straight above over a magenta background. Water shows along the river and no magenta
//! pixel shows between it and its banks; through the reference water material it turns from its
//! shallow colour toward its deep one toward the river's middle and fades into the bank.
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
use wave_forge_bevy::materials::{
    WaterLook, WaterMaterial, WaveForgeMaterialsPlugin, water_material,
};
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
/// pixels, RGBA. The water is drawn in a colour of its own, or with `reference` through the
/// reference water material under that look.
fn render(reference: Option<WaterLook>) -> Vec<u8> {
    let runtime = runtime();
    let mut app = App::new();
    app.add_plugins((DefaultPlugins, WaveForgeMaterialsPlugin))
        .init_resource::<Pixels>();
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
    let plain_water = world
        .resource_mut::<Assets<StandardMaterial>>()
        .add(unlit(WATER));
    let reference_water = reference.map(|look| {
        let mut material = water_material();
        material.base.unlit = true;
        material.extension.look = look;
        world.resource_mut::<Assets<WaterMaterial>>().add(material)
    });
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
                None,
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
                let mut entity = world.spawn((Mesh3d(water), Transform::from_translation(corner)));
                match &reference_water {
                    Some(material) => entity.insert(MeshMaterial3d(material.clone())),
                    None => entity.insert(MeshMaterial3d(plain_water.clone())),
                };
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
    // Pipelines compile over the first frames; keep reading until two readbacks in a row agree,
    // the picture having settled.
    let mut last: Option<Vec<u8>> = None;
    loop {
        app.update();
        frames += 1;
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "no settled picture"
        );
        let now = app.world().resource::<Pixels>().0.clone();
        if frames >= 60 && now.is_some() && now == last {
            break;
        }
        last = now;
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
    let pixels = render(None);

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

#[test]
#[ignore = "needs a device; run with --ignored in release mode"]
fn the_reference_water_deepens_toward_the_rivers_middle_and_fades_into_its_banks() {
    // Red where shallow, blue where deep, and opaque, so a pixel's share of blue is how deep the
    // water reads and any grey in it is the bank showing through at the shore.
    let look = WaterLook {
        shallow_colour: Vec4::new(1.0, 0.0, 0.0, 1.0),
        deep_colour: Vec4::new(0.0, 0.0, 1.0, 1.0),
        opacity: Vec2::ONE,
        clarity: 0.3,
        shore: 0.25,
    };

    let pixels = render(Some(look));

    // Down the image's middle column, across the river at z = 64 world units.
    let column: Vec<[u8; 3]> = (0..SIZE)
        .map(|y| {
            let at = ((y * SIZE + SIZE / 2) * 4) as usize;
            [pixels[at], pixels[at + 1], pixels[at + 2]]
        })
        .collect();
    let blue = |[r, _, b]: [u8; 3]| f32::from(b) / (f32::from(r) + f32::from(b)).max(1.0);
    let watery: Vec<f32> = column
        .iter()
        .filter(|&&[r, g, b]| r.max(b) > g.saturating_add(40))
        .map(|&colour| blue(colour))
        .collect();
    let deepest = watery.iter().copied().fold(0.0, f32::max);
    let shallowest = watery.iter().copied().fold(1.0, f32::min);
    // Between the bank's grey and the water's colour: the water fading in at the shore.
    let fading = column
        .iter()
        .filter(|&&[r, g, b]| {
            g > 40 && r.max(b) > g.saturating_add(10) && r.max(b) < g.saturating_add(60)
        })
        .count();
    println!(
        "{} pixels of water across the river, blue from {shallowest:.2} to {deepest:.2}; {fading} fading into the bank",
        watery.len()
    );

    let middle = watery
        .iter()
        .position(|&share| share == deepest)
        .expect("the deepest is among them");
    assert!(watery.len() >= 4, "{column:?}");
    assert!(deepest - shallowest > 0.1, "{column:?}");
    assert!(
        (watery.len() / 4..watery.len() * 3 / 4).contains(&middle),
        "the deepest at {middle} of {}",
        watery.len()
    );
    assert!(fading > 0, "{column:?}");
}
