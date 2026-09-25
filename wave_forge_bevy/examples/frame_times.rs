//! Frame times while a city streams in around a moving focus, for choosing where the solver runs:
//! on Bevy's device, shared with rendering, or on one of its own, and how large a batch may be
//! ([#39](https://github.com/AntonTegnelov/wave_forge/issues/39),
//! docs/guides/desktop-measurements.md).
//!
//! ```text
//! cargo run -p wave_forge_bevy --release --example frame_times -- \
//!     [--device shared|own] [--max-batch N] [--speed M_PER_S] [--seconds S] [--out FILE]
//! ```
//!
//! The app is headless: Bevy renders the city into a 1920 by 1080 image, a lit cube for every solid
//! or indoor cell with shadows, from a camera that follows the focus. It fills the view, then
//! measures two phases of `--seconds` each: `idle`, standing still with nothing to generate, which is
//! the cost of rendering alone, and `streaming`, moving along the city at `--speed` while chunks are
//! generated ahead and evicted behind. Each phase prints one line of `key=value` fields (frames,
//! median, 99th percentile and slowest frame in milliseconds, chunks generated), appended to
//! `--out` as well when it is given. `WGPU_BACKEND` picks the graphics API, as for any Bevy app.

use bevy::app::PluginsState;
use bevy::asset::RenderAssetUsages;
use bevy::camera::{ImageRenderTarget, RenderTarget};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::renderer::RenderAdapterInfo;
use std::collections::HashMap;
use std::io::Write as _;
use std::time::{Duration, Instant};
use wave_forge::loader::{RuleFile, parse_rule_file};
use wave_forge::{BlockSolver, ChunkCoord, ChunkShape, SolverConfig, WgpuBackend, WorldExtent};
use wave_forge_bevy::{
    ChunkEvicted, ChunkUpdated, GenerationFocus, WaveForgePlugin, WaveForgeSettings,
    WaveForgeSystems, WaveForgeWorld,
};
use wfc_devtools::city::{self, city_prior};

const CHUNK: ChunkShape = ChunkShape::cube(8);
const CELL: f32 = 2.0;
/// Chunks around the focus that are generated, as P1's check uses.
const RADIUS: u32 = 4;
/// Chunks along the city, enough for the fastest run to stay inside it.
const LENGTH: i32 = 160;
const WIDTH: i32 = 12;

type CityWorld = WaveForgeWorld<BlockSolver<WgpuBackend>>;

/// What a run measures, from the command line.
struct Options {
    own_device: bool,
    max_batch: Option<u32>,
    speed: f32,
    seconds: f32,
    out: Option<String>,
}

impl Options {
    /// Reads the command line.
    ///
    /// # Panics
    /// On an option it does not know or a value it cannot read, naming it.
    fn parse() -> Self {
        let mut options = Self {
            own_device: false,
            max_batch: None,
            speed: 4.2,
            seconds: 20.0,
            out: None,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            let mut value = || args.next().unwrap_or_else(|| panic!("{arg} needs a value"));
            match arg.as_str() {
                "--device" => {
                    options.own_device = match value().as_str() {
                        "own" => true,
                        "shared" => false,
                        other => panic!("--device is shared or own, not {other}"),
                    };
                }
                "--max-batch" => options.max_batch = Some(number(&arg, &value())),
                "--speed" => options.speed = number(&arg, &value()),
                "--seconds" => options.seconds = number(&arg, &value()),
                "--out" => options.out = Some(value()),
                other => panic!("unknown option {other}"),
            }
        }
        options
    }
}

fn number<T: std::str::FromStr>(name: &str, text: &str) -> T {
    text.parse()
        .unwrap_or_else(|_| panic!("{name} takes a number, not {text}"))
}

/// The median, 99th percentile and slowest of `frames`, in milliseconds.
fn percentiles(frames: &[f64]) -> [f64; 3] {
    let mut sorted = frames.to_vec();
    sorted.sort_by(f64::total_cmp);
    let at = |fraction: f64| sorted[((sorted.len() - 1) as f64 * fraction).round() as usize];
    [at(0.5), at(0.99), sorted[sorted.len() - 1]]
}

/// The cubes drawn for each chunk, and how many chunks have been generated.
#[derive(Resource, Default)]
struct Drawn {
    cubes: HashMap<ChunkCoord, Vec<Entity>>,
    generated: usize,
}

#[derive(Resource)]
struct Tiles(RuleFile);

#[derive(Resource)]
struct Cube(Handle<Mesh>, Handle<StandardMaterial>);

#[derive(Component)]
struct Focus;

/// Draws a cube for each solid or indoor cell of a generated chunk, replacing what it had, and
/// removes an evicted chunk's.
#[allow(clippy::too_many_arguments)]
fn draw_chunks(
    mut commands: Commands,
    mut updated: MessageReader<ChunkUpdated>,
    mut evicted: MessageReader<ChunkEvicted>,
    world: Res<CityWorld>,
    tiles: Res<Tiles>,
    cube: Res<Cube>,
    mut drawn: ResMut<Drawn>,
) {
    for &ChunkEvicted(chunk) in evicted.read() {
        for entity in drawn.cubes.remove(&chunk).unwrap_or_default() {
            commands.entity(entity).despawn();
        }
    }
    for &ChunkUpdated(chunk) in updated.read() {
        drawn.generated += 1;
        for entity in drawn.cubes.remove(&chunk).unwrap_or_default() {
            commands.entity(entity).despawn();
        }
        let tiles_of = world.chunk(chunk).expect("an updated chunk is generated");
        let entities = tiles_of
            .tiles
            .iter()
            .enumerate()
            .filter(|&(_, &tile)| {
                tiles.0.solid(usize::from(tile)) || tiles.0.indoor(usize::from(tile))
            })
            .map(|(cell, _)| {
                let cell = u32::try_from(cell).expect("few cells");
                let at = world.settings().cell_translation(chunk, cell);
                commands
                    .spawn((
                        Mesh3d(cube.0.clone()),
                        MeshMaterial3d(cube.1.clone()),
                        Transform::from_translation(at),
                    ))
                    .id()
            })
            .collect();
        drawn.cubes.insert(chunk, entities);
    }
}

/// Runs frames until `done` holds, and returns each frame's duration in milliseconds.
fn frames_until(app: &mut App, mut done: impl FnMut(&mut App, Duration) -> bool) -> Vec<f64> {
    let started = Instant::now();
    let mut frames = Vec::new();
    loop {
        let frame = Instant::now();
        app.update();
        frames.push(frame.elapsed().as_secs_f64() * 1000.0);
        if done(app, started.elapsed()) {
            return frames;
        }
    }
}

fn report(options: &Options, adapter: &str, phase: &str, frames: &[f64], generated: usize) {
    let [median, p99, slowest] = percentiles(frames);
    let line = format!(
        "frame_times adapter=\"{adapter}\" device={} max_batch={} speed={} phase={phase} frames={} p50_ms={median:.2} p99_ms={p99:.2} max_ms={slowest:.2} chunks={generated}",
        if options.own_device { "own" } else { "shared" },
        options
            .max_batch
            .map_or_else(|| "default".to_owned(), |batch| batch.to_string()),
        options.speed,
        frames.len(),
    );
    println!("{line}");
    if let Some(out) = &options.out {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(out)
            .unwrap_or_else(|error| panic!("{out} cannot be written: {error}"));
        writeln!(file, "{line}").unwrap_or_else(|error| panic!("{out} cannot be written: {error}"));
    }
}

fn main() {
    let options = Options::parse();
    let city = city::city();
    let rules = parse_rule_file(city::CITY_RON).expect("the city's rule file loads");
    let settings = WaveForgeSettings {
        seed: 11,
        extent: WorldExtent::new(CHUNK)
            .with_x(0..LENGTH)
            .with_y(0..WIDTH)
            .with_z(0..1),
        halo: 1,
        cell_size: Vec3::splat(CELL),
        ..WaveForgeSettings::default()
    };
    let mut solver = SolverConfig::default();
    if let Some(batch) = options.max_batch {
        solver.max_batch = batch;
    }
    let prior = city_prior(&city, CHUNK.z);
    let plugin = if options.own_device {
        let ruleset = wave_forge::Ruleset::new(rules.rules(), &rules.tileset().weights)
            .expect("the city's weights make a rule set");
        WaveForgePlugin::on_own_device(ruleset, prior, settings.clone())
    } else {
        WaveForgePlugin::from_rules(rules.clone(), prior, settings.clone())
            .expect("the city's weights make a rule set")
    };

    let mut app = App::new();
    app.add_plugins(DefaultPlugins)
        .add_plugins(plugin.solver_config(solver).warm(RADIUS))
        .insert_resource(Tiles(rules))
        .init_resource::<Drawn>()
        .add_systems(Update, draw_chunks.after(WaveForgeSystems));
    let started = Instant::now();
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
        assert!(started.elapsed() < Duration::from_secs(120), "no renderer");
    }
    app.finish();
    app.cleanup();
    let adapter = app.world().resource::<RenderAdapterInfo>().name.clone();

    let world = app.world_mut();
    let mut target = Image::new_fill(
        Extent3d {
            width: 1920,
            height: 1080,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    target.texture_descriptor.usage |= TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC;
    let target = world.resource_mut::<Assets<Image>>().add(target);
    let mesh = world
        .resource_mut::<Assets<Mesh>>()
        .add(Cuboid::from_length(CELL));
    let material = world
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial::from_color(Color::srgb(0.7, 0.68, 0.62)));
    world.insert_resource(Cube(mesh, material));
    let chunk_length = CELL * CHUNK.x as f32;
    let start = Vec3::new(
        chunk_length * (RADIUS as f32 + 1.5),
        0.0,
        chunk_length * WIDTH as f32 / 2.0,
    );
    world.spawn((
        Transform::from_translation(start),
        GlobalTransform::from_translation(start),
        GenerationFocus::new(RADIUS),
        Focus,
    ));
    world.spawn((
        Camera3d::default(),
        RenderTarget::Image(ImageRenderTarget::from(target)),
        Transform::from_translation(start + Vec3::new(-40.0, 60.0, 0.0)).looking_at(start, Vec3::Y),
    ));
    world.spawn((
        DirectionalLight {
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(1.0, 2.0, 0.5).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Fill the view first: kernels, the first batches and the first draws are not what is asked.
    frames_until(&mut app, |app, elapsed| {
        assert!(elapsed < Duration::from_secs(300), "the view did not fill");
        app.world().resource::<CityWorld>().is_idle()
            && app.world().resource::<Drawn>().generated > 0
    });
    let seconds = Duration::from_secs_f32(options.seconds);

    let before = app.world().resource::<Drawn>().generated;
    let idle = frames_until(&mut app, |_, elapsed| elapsed >= seconds);
    let generated = app.world().resource::<Drawn>().generated - before;
    report(&options, &adapter, "idle", &idle, generated);

    let before = app.world().resource::<Drawn>().generated;
    let mut last = Instant::now();
    let speed = options.speed;
    let streaming = frames_until(&mut app, |app, elapsed| {
        let step = last.elapsed().as_secs_f32() * speed;
        last = Instant::now();
        let world = app.world_mut();
        let mut query =
            world.query_filtered::<(&mut Transform, &mut GlobalTransform), With<Focus>>();
        let (mut transform, mut global) = query.single_mut(world).expect("one focus");
        transform.translation.x += step;
        *global = GlobalTransform::from_translation(transform.translation);
        let at = transform.translation;
        let mut cameras = world.query_filtered::<&mut Transform, With<Camera3d>>();
        for mut camera in cameras.iter_mut(world) {
            *camera = Transform::from_translation(at + Vec3::new(-40.0, 60.0, 0.0))
                .looking_at(at, Vec3::Y);
        }
        elapsed >= seconds
    });
    let generated = app.world().resource::<Drawn>().generated - before;
    report(&options, &adapter, "streaming", &streaming, generated);
}
