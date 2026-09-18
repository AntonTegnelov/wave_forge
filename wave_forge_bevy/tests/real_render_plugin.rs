//! The plugin in a Bevy app whose own render plugin created the device.
//!
//! ```text
//! cargo test -p wave_forge_bevy --release --test real_render_plugin -- --ignored --nocapture
//! ```
//!
//! `#[ignore]`d: it needs a compute device. `tests/shared_device.rs` checks what comes out of the
//! generator; what is in question here is the handover. Bevy requests its device asynchronously and
//! inserts it when its own `finish` runs, so this pins the two things a game depends on: that the
//! plugin sees the device when it is added after `DefaultPlugins`, and that generation runs on it.
//!
//! The app is headless. This crate's dev-dependency on Bevy has no windowing feature, so
//! `DefaultPlugins` brings no winit and needs no display.

use bevy::app::PluginsState;
use bevy::prelude::*;
use bevy::render::renderer::RenderDevice;
use std::time::{Duration, Instant};
use wave_forge::{BlockSolver, ChunkCoord, ChunkShape, Ruleset, WgpuBackend, WorldExtent};
use wave_forge_bevy::{
    GenerationFocus, WaveForgePlugin, WaveForgeSettings, WaveForgeSystems, WaveForgeWorld,
};
use wfc_devtools::city::{self, city_prior};

const CHUNK: ChunkShape = ChunkShape::cube(8);

type CityWorld = WaveForgeWorld<BlockSolver<WgpuBackend>>;

#[derive(Resource, Default)]
struct Built(Vec<ChunkCoord>);

/// What a game does with a chunk: read its tiles and build something from them. Here it only
/// records that every tile was decided, which is what a mesh builder would rely on.
fn build_chunks(
    world: Res<CityWorld>,
    mut built: ResMut<Built>,
    mut updated: MessageReader<wave_forge_bevy::ChunkUpdated>,
) {
    for event in updated.read() {
        let chunk = world
            .chunk(event.0)
            .expect("an updated chunk is in the store");
        assert_eq!(chunk.tiles.len(), CHUNK.cells() as usize);
        built.0.push(event.0);
    }
}

#[test]
#[ignore = "needs a compute device; run with --ignored in release mode"]
fn bevys_own_device_generates_a_city() {
    let city = city::city();
    let ruleset = Ruleset::from_modules(&city.modules).expect("the city compiles");
    let settings = WaveForgeSettings {
        seed: 11,
        extent: WorldExtent::new(CHUNK)
            .with_x(0..2)
            .with_y(0..2)
            .with_z(0..1),
        halo: 1,
        cell_size: Vec3::splat(2.0),
        ..WaveForgeSettings::default()
    };

    let mut app = App::new();
    app.add_plugins(DefaultPlugins)
        // After Bevy's own plugins, so the device exists by the time this one finishes.
        .add_plugins(
            WaveForgePlugin::new(ruleset, city_prior(&city, CHUNK.z), settings).warm(&[1, 2]),
        )
        .init_resource::<Built>()
        .add_systems(Update, build_chunks.after(WaveForgeSystems))
        .add_systems(Startup, |mut commands: Commands| {
            commands.spawn((
                GlobalTransform::from_translation(Vec3::new(8.0, 0.0, 8.0)),
                GenerationFocus::new(1),
            ));
        });

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

    let adapter = app
        .world()
        .get_resource::<RenderDevice>()
        .expect("Bevy created the device")
        .limits()
        .max_compute_workgroup_storage_size;
    let deadline = Instant::now() + Duration::from_secs(120);
    let mut frames = 0u32;
    loop {
        app.update();
        frames += 1;
        if app.world().resource::<CityWorld>().is_idle() {
            app.update();
            break;
        }
        assert!(
            Instant::now() < deadline,
            "still generating after 120 s: {:?}",
            app.world().resource::<CityWorld>().stats()
        );
    }

    let built = app.world().resource::<Built>();
    let world = app.world().resource::<CityWorld>();
    eprintln!(
        "real_render_plugin: {} chunks over {frames} frames on Bevy's device \
         ({adapter} B workgroup storage), {:?}",
        world.store().len(),
        world.stats()
    );

    assert_eq!(world.store().len(), 4, "the whole 2x2 world");
    assert_eq!(world.stats().failed, 0);
    assert!(
        !built.0.is_empty(),
        "a game's system saw the chunks it would build"
    );
    assert!(built.0.contains(&ChunkCoord::new(0, 0, 0)), "{:?}", built.0);
}
