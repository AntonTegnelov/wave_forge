//! The ground's levels of detail drawn by Bevy's own renderer: over a wide view, neighbouring chunks
//! at different levels meet without a gap.
//!
//! ```text
//! cargo test -p wave_forge_bevy --release --test ground_levels_render -- --ignored --nocapture
//! ```
//!
//! `#[ignore]`d: it needs a device. The app is headless and renders into an image, which it reads
//! back from the GPU. Ground fills the whole picture over a magenta background, so a magenta pixel
//! is a gap between chunks. The picture is drawn at several thresholds in pixels, each putting
//! neighbours at different levels somewhere in view.

use bevy::app::PluginsState;
use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::ViewVisibility;
use bevy::camera::{ImageRenderTarget, RenderTarget};
use bevy::core_pipeline::tonemapping::{DebandDither, Tonemapping};
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::view::Msaa;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wave_forge::stages::{Pack, Runtime};
use wave_forge::{ChunkCoord, FocusPoint, GroundMesh, ground};
use wave_forge_bevy::levels::LevelDetail;
use wave_forge_bevy::stages::ground_levels;

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "height", kind: Field(Mul(Noise(frequency: 0.05, octaves: 3), Constant(12.0)))),
    ],
)"#;

const CELLS: u32 = 8;
const CELL: [f32; 3] = [2.0, 1.0, 2.0];
/// Chunks of ground along each side of the view, around the origin's.
const REACH: i32 = 7;
/// Pixels across and down: a row of 256 is 1 KiB, which a texture's readback needs no padding for.
const WIDTH: u32 = 256;
const HEIGHT: u32 = 128;
const FOV: f32 = std::f32::consts::FRAC_PI_4;

#[derive(Resource, Default)]
struct Pixels(Option<Vec<u8>>);

/// A ground level's entity, and the step of its level.
#[derive(Component)]
struct Level(u32);

fn grounds() -> Vec<GroundMesh> {
    let mut runtime = Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        3,
        [CELLS, CELLS],
    );
    runtime
        .request(
            &[FocusPoint::new(
                ChunkCoord::new(0, 0, 0),
                (REACH + 1) as u32,
            )],
            &["height"],
        )
        .expect("stages");
    runtime.run_until_idle().expect("the stages run");
    (-REACH..=REACH)
        .flat_map(|y| (-REACH..=REACH).map(move |x| ChunkCoord::new(x, y, 0)))
        .map(|chunk| ground(chunk, |at| runtime.field("height", at), CELL).expect("a ground"))
        .collect()
}

/// Draws a frame with every ground's levels for `pixels` of error, and returns the picture and how
/// many visible entities have each step.
fn draw(
    app: &mut App,
    grounds: &[GroundMesh],
    material: &Handle<StandardMaterial>,
    pixels: f32,
) -> (Vec<u8>, BTreeMap<u32, usize>) {
    let world = app.world_mut();
    let old: Vec<Entity> = world
        .query_filtered::<Entity, With<Level>>()
        .iter(world)
        .collect();
    for entity in old {
        world.despawn(entity);
    }
    let detail = LevelDetail {
        pixels,
        height: HEIGHT as f32,
        fov: FOV,
    };
    for ground in grounds {
        let corner = Vec3::new(
            ground.chunk.x as f32 * CELLS as f32 * CELL[0],
            0.0,
            ground.chunk.y as f32 * CELLS as f32 * CELL[2],
        );
        for level in ground_levels(ground, detail) {
            let mesh = world.resource_mut::<Assets<Mesh>>().add(level.mesh);
            world.spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material.clone()),
                Transform::from_translation(corner),
                level.range,
                Level(level.step),
            ));
        }
    }
    world.resource_mut::<Pixels>().0 = None;
    let started = Instant::now();
    let mut frames = 0;
    // Pipelines compile over the first frames; keep reading until the picture has settled.
    while frames < 30 || app.world().resource::<Pixels>().0.is_none() {
        app.update();
        frames += 1;
        assert!(started.elapsed() < Duration::from_secs(60), "no picture");
    }
    let world = app.world_mut();
    let mut visible = BTreeMap::new();
    for (level, view) in world.query::<(&Level, &ViewVisibility)>().iter(world) {
        if view.get() {
            *visible.entry(level.0).or_insert(0) += 1;
        }
    }
    let picture = world.resource::<Pixels>().0.clone().expect("read back");
    (picture, visible)
}

fn gaps(picture: &[u8]) -> usize {
    picture
        .chunks(4)
        .filter(|pixel| pixel[0] > 200 && pixel[1] < 50 && pixel[2] > 200)
        .count()
}

#[test]
#[ignore = "needs a device; run with --ignored in release mode"]
fn neighbours_at_different_levels_meet_without_a_gap() {
    let grounds = grounds();
    let mut app = App::new();
    app.add_plugins(DefaultPlugins)
        .init_resource::<Pixels>()
        .insert_resource(ClearColor(Color::srgb(1.0, 0.0, 1.0)));
    let started = Instant::now();
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
        assert!(started.elapsed() < Duration::from_secs(60), "no renderer");
    }
    app.finish();
    app.cleanup();
    let world = app.world_mut();
    let mut target = Image::new_fill(
        Extent3d {
            width: WIDTH,
            height: HEIGHT,
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
    let material = world
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            base_color: Color::srgb(0.5, 0.5, 0.5),
            unlit: true,
            ..default()
        });
    world.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: FOV,
            ..default()
        }),
        RenderTarget::Image(ImageRenderTarget::from(target.clone())),
        Tonemapping::None,
        DebandDither::Disabled,
        Msaa::Off,
        Transform::from_xyz(8.0, 50.0, -15.0).looking_at(Vec3::new(8.0, 0.0, 15.0), Vec3::Y),
    ));
    world.spawn(Readback::texture(target)).observe(
        |readback: On<ReadbackComplete>, mut pixels: ResMut<Pixels>| {
            pixels.0 = Some(readback.data.clone());
        },
    );

    let drawn: Vec<(f32, usize, BTreeMap<u32, usize>)> = [1.0, 4.0, 16.0]
        .into_iter()
        .map(|pixels| {
            let (picture, visible) = draw(&mut app, &grounds, &material, pixels);
            (pixels, gaps(&picture), visible)
        })
        .collect();

    for (pixels, gaps, visible) in &drawn {
        println!("{pixels} px: {gaps} gap pixels, visible chunks by step {visible:?}");
    }
    assert!(drawn.iter().all(|(_, gaps, _)| *gaps == 0));
    assert!(
        drawn.iter().any(|(_, _, visible)| visible.len() > 1),
        "no picture mixed levels"
    );
}
