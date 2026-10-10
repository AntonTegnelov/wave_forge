//! The hills preset under a sun and a sky: what a game spawns for a first look at its terrain,
//! since Bevy has no sun or sky of its own and the plugin owns no lighting (docs/reference/bevy.md,
//! "First look"). A sun with shadows, an atmosphere for the sky, and a camera that generation
//! follows; the ground comes with its reference material as each chunk is built.
//!
//! ```text
//! cargo run -p wave_forge_bevy --release --example first_look
//! ```
//!
//! Here the app is headless and renders into an image, since the workspace's Bevy has no window;
//! in a game, the same entities go into a windowed app. It runs until the view's ground is drawn
//! and prints how many chunks it drew.

use bevy::app::PluginsState;
use bevy::asset::RenderAssetUsages;
use bevy::camera::{ImageRenderTarget, RenderTarget};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::atmosphere::ScatteringMedium;
use bevy::light::{Atmosphere, light_consts};
use bevy::pbr::AtmosphereSettings;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use std::sync::Arc;
use std::time::{Duration, Instant};
use wave_forge::stages::{Pack, Runtime};
use wave_forge_bevy::GenerationFocus;
use wave_forge_bevy::materials::{WaveForgeMaterialsPlugin, ground_material, palette_image};
use wave_forge_bevy::stages::{
    GroundReady, StagesSettings, WaveForgeStages, WaveForgeStagesPlugin, WaveForgeStagesSystems,
    ground_mesh,
};

const SETTINGS: StagesSettings = StagesSettings {
    chunk: [8, 8],
    cell_size: Vec3::new(2.0, 1.0, 2.0),
};
/// Chunks generated around the camera.
const RADIUS: u32 = 3;

/// The preset's palette, its rock, woods and meadow.
#[derive(Resource)]
struct Palette(Handle<Image>);

fn main() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../examples/presets/hills.world.ron"
    ))
    .expect("the hills preset");
    let pack = Arc::new(Pack::parse(&text).expect("a valid pack"));
    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins,
        WaveForgeMaterialsPlugin,
        WaveForgeStagesPlugin::new(&["height", "surface"], SETTINGS, move || {
            Ok(Runtime::new(pack, 7, SETTINGS.chunk))
        })
        .with_ground("height")
        .with_ground_materials("surface")
        // The ground reads the materials of the chunks beyond its far edges.
        .with_radius("surface", RADIUS + 1),
    ))
    .add_systems(Startup, spawn_sun_sky_and_camera)
    .add_systems(Update, draw_ground.after(WaveForgeStagesSystems));

    let started = Instant::now();
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
        assert!(started.elapsed() < Duration::from_secs(60), "no renderer");
    }
    app.finish();
    app.cleanup();
    let wanted = ((2 * RADIUS + 1) * (2 * RADIUS + 1)) as usize;
    while app.world_mut().query::<&Mesh3d>().iter(app.world()).count() < wanted {
        app.update();
        assert!(
            started.elapsed() < Duration::from_secs(120),
            "the view's ground did not arrive"
        );
    }
    println!("first_look: {wanted} chunks of ground drawn under a sun and an atmosphere");
}

/// The sun, the sky and the camera generation follows.
fn spawn_sun_sky_and_camera(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut media: ResMut<Assets<ScatteringMedium>>,
) {
    // A sun 35 degrees up, casting shadows.
    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::RAW_SUNLIGHT,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(
            EulerRot::YXZ,
            30f32.to_radians(),
            -35f32.to_radians(),
            0.0,
        )),
    ));
    // The sky: an Earth-like atmosphere, which the sun lights.
    commands.spawn(Atmosphere::earth(
        media.add(ScatteringMedium::earth(256, 256)),
    ));
    let mut target = Image::new_fill(
        Extent3d {
            width: 1280,
            height: 720,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba16Float,
        RenderAssetUsages::default(),
    );
    target.texture_descriptor.usage |=
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC | TextureUsages::TEXTURE_BINDING;
    commands.spawn((
        Camera3d::default(),
        RenderTarget::Image(ImageRenderTarget::from(images.add(target))),
        AtmosphereSettings::default(),
        // A tonemapper without lookup tables, which this workspace's Bevy builds without; a game
        // with Bevy's default features keeps its default.
        Tonemapping::AcesFitted,
        Transform::from_xyz(0.0, 30.0, 0.0).looking_at(Vec3::new(60.0, 10.0, 60.0), Vec3::Y),
        GenerationFocus::new(RADIUS),
    ));
    let palette = palette_image(&[
        Color::srgb(0.42, 0.40, 0.37),
        Color::srgb(0.20, 0.27, 0.13),
        Color::srgb(0.27, 0.40, 0.18),
    ]);
    commands.insert_resource(Palette(images.add(palette)));
}

/// Gives each chunk its ground as it is built, with the reference ground material.
fn draw_ground(
    mut commands: Commands,
    mut ready: MessageReader<GroundReady>,
    stages: Res<WaveForgeStages>,
    palette: Res<Palette>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<wave_forge_bevy::materials::GroundMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    for GroundReady(chunk) in ready.read() {
        let (Some(ground), Some(material)) = (
            stages.ground(*chunk),
            ground_material(
                &stages,
                *chunk,
                StandardMaterial::default(),
                palette.0.clone(),
                &mut images,
            ),
        ) else {
            continue;
        };
        commands.spawn((
            Mesh3d(meshes.add(ground_mesh(ground))),
            MeshMaterial3d(materials.add(material)),
            Transform::from_translation(stages.chunk_corner(*chunk)),
        ));
    }
}
