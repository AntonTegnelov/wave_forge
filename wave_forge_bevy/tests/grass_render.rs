//! Grass drawn by Bevy's own renderer: blades grow only where the cover is, and sway while the wind
//! blows and stand still once it drops.
//!
//! ```text
//! cargo test -p wave_forge_bevy --release --test grass_render -- --ignored --nocapture
//! ```
//!
//! `#[ignore]`d: it needs a device. The app is headless and renders into an image, which it reads
//! back from the GPU every frame. Only the grass is drawn, unlit, on black, so every lit pixel is a
//! blade.

use bevy::app::PluginsState;
use bevy::asset::RenderAssetUsages;
use bevy::camera::{ImageRenderTarget, RenderTarget};
use bevy::core_pipeline::tonemapping::{DebandDither, Tonemapping};
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::view::Msaa;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wave_forge::stages::{Pack, Runtime};
use wave_forge::{ChunkCoord, FocusPoint, ground};
use wave_forge_bevy::materials::{
    GrassMaterial, WaveForgeMaterialsPlugin, Wind, grass_bounds_of, grass_material_of, grass_mesh,
};

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "height", kind: Field(Constant(0.0))),
        (name: "side", kind: Rules(rules: [(category: "west", when: [Less(X, Constant(4.0))])], otherwise: "east")),
        (name: "cover", kind: Field(Is("side", ["west"]))),
    ],
)"#;

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

/// How many pixels are lit in the columns from `from` up to but not including `to`.
fn lit(pixels: &[u8], from: u32, to: u32) -> usize {
    (0..SIZE)
        .flat_map(|y| (from..to).map(move |x| ((y * SIZE + x) * 4) as usize))
        .filter(|&at| pixels[at..at + 3].iter().any(|&channel| channel > 8))
        .count()
}

fn changed(a: &[u8], b: &[u8]) -> usize {
    a.chunks(4).zip(b.chunks(4)).filter(|(a, b)| a != b).count()
}

#[test]
#[ignore = "needs a device; run with --ignored in release mode"]
fn blades_grow_only_where_the_cover_is_and_sway_in_the_wind() {
    let mut runtime = Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        1,
        [8, 8],
    );
    let chunk = ChunkCoord::new(0, 0, 0);
    runtime
        .request(&[FocusPoint::new(chunk, 1)], &["height", "cover"])
        .expect("stages");
    runtime.run_until_idle().expect("the stages run");
    let mesh = ground(chunk, |at| runtime.field("height", at), [1.0; 3]).expect("a ground");
    let cover = runtime.field("cover", chunk).expect("cover").clone();

    let mut app = App::new();
    app.add_plugins((DefaultPlugins, WaveForgeMaterialsPlugin))
        .init_resource::<Pixels>()
        .insert_resource(ClearColor(Color::BLACK));
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
    let material = {
        let mut images = world.resource_mut::<Assets<Image>>();
        grass_material_of(
            &mesh,
            &cover,
            Vec3::ONE,
            16,
            StandardMaterial {
                unlit: true,
                ..default()
            },
            &mut images,
        )
    };
    let material = world.resource_mut::<Assets<GrassMaterial>>().add(material);
    let blades = world
        .resource_mut::<Assets<Mesh>>()
        .add(grass_mesh([8, 8], 16));
    world.spawn((
        Mesh3d(blades),
        MeshMaterial3d(material),
        Transform::IDENTITY,
        grass_bounds_of(&mesh, Vec3::ONE),
    ));
    world.spawn((
        Camera3d::default(),
        RenderTarget::Image(ImageRenderTarget::from(target.clone())),
        Tonemapping::None,
        // Dithering would change pixels from frame to frame by itself.
        DebandDither::Disabled,
        Msaa::Off,
        Transform::from_xyz(4.5, 4.0, -5.0).looking_at(Vec3::new(4.5, 0.0, 4.5), Vec3::Y),
    ));
    world.spawn(Readback::texture(target)).observe(
        |readback: On<ReadbackComplete>, mut pixels: ResMut<Pixels>| {
            pixels.0 = Some(readback.data.clone());
        },
    );
    app.world_mut()
        .insert_resource(Wind(Vec4::new(1.0, 0.0, 0.3, 3.0)));

    // Pipelines compile over the first seconds, drawing nothing until they are ready.
    let started = Instant::now();
    while lit(&run_for(&mut app, 0.1), 0, SIZE) == 0 {
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "nothing was drawn"
        );
    }
    let first = run_for(&mut app, 0.1);
    let later = run_for(&mut app, 0.5);
    app.world_mut()
        .insert_resource(Wind(Vec4::new(1.0, 0.0, 0.0, 3.0)));
    let calm = run_for(&mut app, 0.3);
    let still = run_for(&mut app, 0.5);

    let (left, right) = (lit(&first, 0, SIZE / 3), lit(&first, SIZE * 2 / 3, SIZE));
    let (swaying, standing) = (changed(&first, &later), changed(&calm, &still));
    println!(
        "{left} lit on the left third, {right} on the right; {swaying} pixels changed in the wind, {standing} without"
    );
    assert!(left.max(right) > 100 && left.min(right) == 0);
    assert!(swaying > 0);
    assert_eq!(standing, 0);
}
