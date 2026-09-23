//! The plugin on a real device, shared the way Bevy shares it.
//!
//! ```text
//! cargo test -p wave_forge_bevy --release --test shared_device -- --ignored --nocapture
//! ```
//!
//! `#[ignore]`d: it needs a compute device. The device here is created the way `bevy_render` creates
//! its own (an instance from the environment, and the adapter's limits), then handed to the app as
//! `RenderDevice` and `RenderQueue`. That is the whole of what the plugin reads, so this exercises
//! the sharing path without a window, a surface or a render graph.

use bevy_app::{App, Startup, Update};
use bevy_ecs::message::MessageReader;
use bevy_ecs::prelude::{Commands, IntoScheduleConfigs, ResMut, Resource};
use bevy_math::Vec3;
use bevy_render::renderer::{RenderDevice, RenderQueue};
use bevy_transform::components::GlobalTransform;
use std::time::{Duration, Instant};
use wave_forge::{
    BlockSolver, ChunkCoord, ChunkShape, Ruleset, WgpuBackend, WorldExtent, WorldGenerator,
};
use wave_forge_bevy::{
    ChunkFailed, ChunkUpdated, GenerationFocus, WaveForgePlugin, WaveForgeSettings,
    WaveForgeSystems, WaveForgeWorld,
};
use wfc_devtools::city::{self, city_prior};
use wfc_devtools::{BoundaryCondition, TileGrid, adjacency_violations};

const CHUNK: ChunkShape = ChunkShape::cube(8);

type World = WaveForgeWorld<BlockSolver<WgpuBackend>>;

/// A device and queue as `bevy_render::renderer::initialize_renderer` makes them: an instance built
/// from the environment, the high-performance adapter, and that adapter's own limits.
fn bevy_style_device() -> (wgpu::Device, wgpu::Queue) {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("a compute adapter");
    eprintln!(
        "shared_device: {:?}, {} B workgroup storage",
        adapter.get_info().name,
        adapter.limits().max_compute_workgroup_storage_size
    );
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("bevy-like device"),
        required_features: adapter.features(),
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("a device")
}

#[derive(Resource, Default)]
struct Seen {
    updated: Vec<ChunkCoord>,
    failed: Vec<ChunkCoord>,
}

fn collect(
    mut seen: ResMut<Seen>,
    mut updated: MessageReader<ChunkUpdated>,
    mut failed: MessageReader<ChunkFailed>,
) {
    seen.updated.extend(updated.read().map(|event| event.0));
    seen.failed.extend(failed.read().map(|event| event.chunk));
}

/// A city generated in a Bevy app around a focus at the middle of a 4x4-chunk world.
#[test]
#[ignore = "needs a compute device; run with --ignored in release mode"]
fn a_city_generates_on_the_device_bevy_renders_with() {
    let city = city::city();
    let ruleset = Ruleset::from_modules(&city.modules).expect("the city compiles");
    let settings = WaveForgeSettings {
        seed: 11,
        extent: WorldExtent::new(CHUNK)
            .with_x(0..4)
            .with_y(0..4)
            .with_z(0..1),
        halo: 1,
        cell_size: Vec3::splat(2.0),
        evict_margin: None,
        ..WaveForgeSettings::default()
    };
    let (device, queue) = bevy_style_device();

    let mut app = App::new();
    app.insert_resource(RenderDevice::from(device))
        .insert_resource(RenderQueue::new(queue))
        // Sixteen chunks means batches of eight per parity, and a repair is one chunk.
        .add_plugins(
            WaveForgePlugin::new(ruleset, city_prior(&city, CHUNK.z), settings.clone()).warm(2),
        )
        .init_resource::<Seen>()
        .add_systems(Update, collect.after(WaveForgeSystems))
        .add_systems(Startup, |mut commands: Commands| {
            // The middle of the world, so the focus asks for every chunk of it.
            commands.spawn((
                GlobalTransform::from_translation(Vec3::new(32.0, 0.0, 32.0)),
                GenerationFocus::new(2),
            ));
        });
    app.finish();
    app.cleanup();

    // Frames, not blocking calls: a batch spans however many frames the device needs, and the app
    // keeps running through them. The budget is wall time, because a frame here costs microseconds
    // where a real one costs milliseconds.
    let started = Instant::now();
    let deadline = started + Duration::from_secs(120);
    let mut frames = 0u32;
    loop {
        app.update();
        frames += 1;
        if app.world().resource::<World>().is_idle() {
            // The frame after the last batch is the one a game reads its messages in.
            app.update();
            break;
        }
        assert!(
            Instant::now() < deadline,
            "still generating after 120 s and {frames} frames: {:?}",
            app.world().resource::<World>().stats()
        );
    }
    let wall_s = started.elapsed().as_secs_f64();

    let seen = app.world().resource::<Seen>();
    let world = app.world().resource::<World>();
    let stats = *world.stats();
    eprintln!(
        "shared_device: 16 chunks in {frames} frames, {wall_s:.2} s wall, {stats:?}; \
         {} updated, {} given up on",
        seen.updated.len(),
        seen.failed.len()
    );

    // What was generated is valid: the store's tiles, checked against the city's own rules.
    let (width, height, depth) = (4 * CHUNK.x as usize, 4 * CHUNK.y as usize, CHUNK.z as usize);
    let store = world.store();
    let tiles: Vec<usize> = (0..depth)
        .flat_map(|z| {
            (0..height).flat_map(move |y| {
                (0..width).map(move |x| {
                    store
                        .tile([x as i32, y as i32, z as i32])
                        .map_or(city.air, usize::from)
                })
            })
        })
        .collect();
    let grid = TileGrid::new(width, height, depth, tiles).expect("the world's dimensions");
    let decided =
        |x: usize, y: usize, z: usize| store.tile([x as i32, y as i32, z as i32]).is_some();
    let violations = adjacency_violations(&grid, &city.modules.rules, BoundaryCondition::Finite)
        .into_iter()
        .filter(|violation| {
            decided(violation.cell.0, violation.cell.1, violation.cell.2)
                && decided(
                    violation.neighbor.0,
                    violation.neighbor.1,
                    violation.neighbor.2,
                )
        })
        .count();

    assert_eq!(violations, 0, "decided cells never violate the rules");
    assert!(
        stats.solved >= 14,
        "at least 14 of 16 chunks placed: {stats:?}"
    );
    assert_eq!(
        store.len() + seen.failed.len(),
        16,
        "every chunk is either generated or reported as failed"
    );
    assert!(
        seen.updated.len() >= store.len(),
        "a game hears about every chunk it could build"
    );

    // The same world, generated by the library on a device of its own, has the same tiles: sharing
    // Bevy's device changes nothing about what comes out.
    let mut direct: WorldGenerator<BlockSolver<WgpuBackend>> = wave_forge::Builder::new(
        Ruleset::from_modules(&city.modules).expect("compiles"),
        city_prior(&city, CHUNK.z),
    )
    .seed(11)
    .extent(settings.extent.clone())
    .halo(1)
    .build()
    .expect("a device of its own");
    direct.request(&[wave_forge::FocusPoint::new(ChunkCoord::new(2, 2, 0), 2)]);
    direct.run_until_idle().expect("the solver runs");
    for chunk in direct.store().iter() {
        assert_eq!(
            world.chunk(chunk.coord).map(|theirs| theirs.tiles.to_vec()),
            Some(chunk.tiles.to_vec()),
            "{:?} differs between Bevy's device and our own",
            chunk.coord
        );
    }
}
