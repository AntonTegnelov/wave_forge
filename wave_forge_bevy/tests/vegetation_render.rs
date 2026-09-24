//! Plants drawn by Bevy's own renderer with the reference vegetation material: they sway while the
//! wind blows and stand still once it drops, and so do their shadows, which the prepass draws.
//!
//! ```text
//! cargo test -p wave_forge_bevy --release --test vegetation_render -- --ignored --nocapture
//! ```
//!
//! `#[ignore]`d: it needs a device. The app is headless and renders into an image, which it reads
//! back from the GPU every frame. The trees are unlit green, the ground lit grey, so a pixel that is
//! grey in two frames and changed between them is a shadow that moved. The ground fills the view, so
//! a pixel left in the clear colour is one where the depth prepass put a tree the main pass did not
//! draw.

use bevy::app::PluginsState;
use bevy::asset::RenderAssetUsages;
use bevy::camera::{ImageRenderTarget, RenderTarget};
use bevy::core_pipeline::prepass::{DepthPrepass, MotionVectorPrepass};
use bevy::core_pipeline::tonemapping::{DebandDither, Tonemapping};
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::view::Msaa;
use std::time::{Duration, Instant};
use wave_forge_bevy::materials::{
    VegetationMaterial, WaveForgeMaterialsPlugin, Wind, vegetation_material,
};

/// Pixels along each side: a row of 128 is 512 bytes, which a texture's readback needs no padding
/// for.
const SIZE: u32 = 128;

#[derive(Resource, Default)]
struct Pixels(Option<Vec<u8>>);

/// Runs frames for `seconds` of the clock, and returns the last picture read back.
fn run_for(app: &mut App, seconds: f32) -> Vec<u8> {
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs_f32(seconds)
        || app.world().resource::<Pixels>().0.is_none()
    {
        app.update();
        assert!(started.elapsed() < Duration::from_secs(60), "no picture");
    }
    app.world()
        .resource::<Pixels>()
        .0
        .clone()
        .expect("read back")
}

fn is_tree(pixel: &[u8]) -> bool {
    pixel[1] > pixel[0].saturating_add(40)
}

fn is_clear(pixel: &[u8]) -> bool {
    pixel[0] > 200 && pixel[1] < 50 && pixel[2] > 200
}

fn is_ground(pixel: &[u8]) -> bool {
    pixel[0].abs_diff(pixel[1]) < 12 && pixel[1].abs_diff(pixel[2]) < 12
}

fn changed(a: &[u8], b: &[u8]) -> usize {
    a.chunks(4).zip(b.chunks(4)).filter(|(a, b)| a != b).count()
}

/// Pixels that are ground in both pictures and changed between them: shadows that moved.
fn shadows_moved(a: &[u8], b: &[u8]) -> usize {
    a.chunks(4)
        .zip(b.chunks(4))
        .filter(|(a, b)| is_ground(a) && is_ground(b) && a != b)
        .count()
}

#[test]
#[ignore = "needs a device; run with --ignored in release mode"]
fn plants_and_their_shadows_sway_in_the_wind() {
    let mut app = App::new();
    app.add_plugins((DefaultPlugins, WaveForgeMaterialsPlugin))
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
    let tree = world
        .resource_mut::<Assets<Mesh>>()
        .add(Mesh::from(Cylinder::new(0.15, 4.0)).translated_by(Vec3::Y * 2.0));
    let plane = world
        .resource_mut::<Assets<Mesh>>()
        .add(Plane3d::default().mesh().size(200.0, 200.0));
    let bark = world
        .resource_mut::<Assets<VegetationMaterial>>()
        .add(vegetation_material(
            StandardMaterial {
                base_color: Color::srgb(0.1, 0.6, 0.1),
                unlit: true,
                ..default()
            },
            4.0,
            0.03,
        ));
    let grey = world
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            base_color: Color::srgb(0.6, 0.6, 0.6),
            perceptual_roughness: 1.0,
            ..default()
        });
    world.spawn((Mesh3d(plane), MeshMaterial3d(grey)));
    for x in 0..3 {
        for z in 0..3 {
            world.spawn((
                Mesh3d(tree.clone()),
                MeshMaterial3d(bark.clone()),
                Transform::from_xyz(x as f32 * 2.0 - 2.0, 0.0, z as f32 * 2.0 - 2.0)
                    .with_scale(Vec3::splat(0.75 + 0.25 * ((x + z) % 3) as f32)),
            ));
        }
    }
    world.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(-3.0, 6.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    world.spawn((
        Camera3d::default(),
        RenderTarget::Image(ImageRenderTarget::from(target.clone())),
        Tonemapping::None,
        // Dithering would change pixels from frame to frame by itself.
        DebandDither::Disabled,
        Msaa::Off,
        DepthPrepass,
        MotionVectorPrepass,
        Transform::from_xyz(0.0, 9.0, -9.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
    ));
    world.spawn(Readback::texture(target)).observe(
        |readback: On<ReadbackComplete>, mut pixels: ResMut<Pixels>| {
            pixels.0 = Some(readback.data.clone());
        },
    );
    app.world_mut()
        .insert_resource(Wind(Vec4::new(1.0, 0.0, 0.6, 3.0)));

    // Pipelines compile over the first seconds, drawing nothing until they are ready.
    let started = Instant::now();
    while !run_for(&mut app, 0.1).chunks(4).any(is_tree) {
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "no tree was drawn"
        );
    }
    let first = run_for(&mut app, 0.1);
    let later = run_for(&mut app, 0.5);
    app.world_mut()
        .insert_resource(Wind(Vec4::new(1.0, 0.0, 0.0, 3.0)));
    let calm = run_for(&mut app, 0.3);
    let still = run_for(&mut app, 0.5);

    let trees = first.chunks(4).filter(|pixel| is_tree(pixel)).count();
    let (swaying, shadows) = (changed(&first, &later), shadows_moved(&first, &later));
    let standing = changed(&calm, &still);
    let unmatched = [&first, &later]
        .iter()
        .map(|picture| picture.chunks(4).filter(|pixel| is_clear(pixel)).count())
        .sum::<usize>();
    println!(
        "{trees} tree pixels; {swaying} pixels changed in the wind, {shadows} of them shadows on the ground; {standing} without; {unmatched} where the passes disagree"
    );
    assert!(trees > 100);
    assert_eq!(unmatched, 0);
    assert!(swaying > 0);
    assert!(shadows > 0);
    assert_eq!(standing, 0);
}
