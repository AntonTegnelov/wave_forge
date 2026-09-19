//! What the plugin wires into a Bevy app, on the CPU reference solver so no device is needed.
//!
//! The solver itself is tested in the library; what is in question here is the ECS side: that a
//! focus entity makes the world generate around it, that the messages a game builds meshes from
//! arrive, that moving away drops what was left behind, and that a chunk's place in Bevy's Y-up
//! world space is the place the lattice says.

use bevy_app::{App, Startup, Update};
use bevy_ecs::message::MessageReader;
use bevy_ecs::prelude::{Commands, IntoScheduleConfigs, Res, ResMut, Resource};
use bevy_math::Vec3;
use bevy_transform::components::GlobalTransform;
use std::sync::Arc;
use wave_forge::{ChunkCoord, ChunkShape, Prior, Ruleset, WorldExtent};
use wave_forge_bevy::{
    ChunkEvicted, ChunkFailed, ChunkUpdated, GenerationFocus, WaveForgeSettings,
    WaveForgeSolverPlugin, WaveForgeSystems, WaveForgeWorld,
};
use wfc_core::reference::ReferenceSolver;

const TILES: u32 = 3;
const CHUNK: ChunkShape = ChunkShape::cube(4);
/// One cell is two units across and one unit tall, so the two axis orders cannot be confused.
const CELL: Vec3 = Vec3::new(2.0, 1.0, 2.0);

type World = WaveForgeWorld<ReferenceSolver>;

/// Three tiles where anything may sit beside anything except 0 beside 2: every chunk solves, and
/// propagation still has something to do.
fn ruleset() -> Ruleset {
    let allowed = (0..6).flat_map(|axis| {
        (0..TILES as usize).flat_map(move |a| (0..TILES as usize).map(move |b| (axis, a, b)))
    });
    let rules = wave_forge::AdjacencyRules::from_allowed_tuples(
        TILES as usize,
        6,
        allowed.filter(|&(_, a, b)| a.abs_diff(b) != 2),
    );
    Ruleset::new(&rules, &[3.0, 1.0, 2.0]).expect("a rule set of three tiles")
}

fn settings(evict_margin: Option<u32>) -> WaveForgeSettings {
    WaveForgeSettings {
        seed: 7,
        extent: WorldExtent::new(CHUNK)
            .with_x(0..4)
            .with_y(0..4)
            .with_z(0..1),
        halo: 1,
        cell_size: CELL,
        evict_margin,
        ..WaveForgeSettings::default()
    }
}

/// What a game would have built meshes from.
#[derive(Resource, Default)]
struct Seen {
    updated: Vec<ChunkCoord>,
    failed: Vec<ChunkCoord>,
    evicted: Vec<ChunkCoord>,
}

fn collect(
    mut seen: ResMut<Seen>,
    mut updated: MessageReader<ChunkUpdated>,
    mut failed: MessageReader<ChunkFailed>,
    mut evicted: MessageReader<ChunkEvicted>,
) {
    seen.updated.extend(updated.read().map(|event| event.0));
    seen.failed.extend(failed.read().map(|event| event.chunk));
    seen.evicted.extend(evicted.read().map(|event| event.0));
}

/// An app with the plugin, a focus at the origin, and a collector for its messages.
fn app(radius: u32, evict_margin: Option<u32>) -> App {
    let mut app = App::new();
    let solver = ReferenceSolver::new(Arc::new(ruleset()));
    app.add_plugins(WaveForgeSolverPlugin::new(
        ruleset(),
        Prior::open(TILES),
        settings(evict_margin),
        solver,
    ))
    .init_resource::<Seen>()
    .add_systems(Update, collect.after(WaveForgeSystems))
    .add_systems(Startup, move |mut commands: Commands| {
        commands.spawn((GlobalTransform::default(), GenerationFocus::new(radius)));
    });
    app.finish();
    app.cleanup();
    app
}

/// Runs frames until the world has nothing left to do, or gives up so a test fails rather than
/// hangs. One frame starts at most one batch, which is what keeps a frame short.
///
/// One more frame follows the last batch, because that is the frame a game reads its messages in.
fn run_until_idle(app: &mut App) -> usize {
    for frame in 1..=200 {
        app.update();
        if app.world().resource::<World>().is_idle() {
            app.update();
            return frame;
        }
    }
    panic!("the world was still generating after 200 frames");
}

#[test]
fn a_focus_entity_makes_the_world_generate_around_it() {
    let mut app = app(1, None);

    let frames = run_until_idle(&mut app);

    let seen = app.world().resource::<Seen>();
    let world = app.world().resource::<World>();
    assert!(seen.failed.is_empty(), "{:?}", seen.failed);
    // A focus of radius 1 at the origin asks for the four chunks of the world's corner, and the
    // second-parity ones among those bring the two neighbours they read.
    assert_eq!(world.store().len(), 6, "{} frames", frames);
    let mut reported = seen.updated.clone();
    reported.sort_unstable();
    reported.dedup();
    assert_eq!(reported.len(), 6, "every generated chunk was reported once");
    assert!(world.chunk(ChunkCoord::new(0, 0, 0)).is_some());
    assert_eq!(world.stats().solved, 6);
    assert!(frames > 1, "one batch per frame, so this takes several");
}

#[test]
fn a_chunk_that_leaves_every_focus_is_dropped_and_reported() {
    // A margin of one, because the chunks a focus asks for reach one chunk beyond its radius.
    let mut app = app(1, Some(1));
    run_until_idle(&mut app);
    let before = app.world().resource::<World>().store().len();

    // The focus walks three chunks along Bevy's x, which is the lattice's x.
    let chunk_units = 2.0 * 4.0;
    let mut focus = app.world_mut().query::<&mut GlobalTransform>();
    let mut at = focus.single_mut(app.world_mut()).expect("one focus entity");
    *at = GlobalTransform::from_translation(Vec3::new(3.0 * chunk_units, 0.0, 0.0));
    run_until_idle(&mut app);

    let seen = app.world().resource::<Seen>();
    let world = app.world().resource::<World>();
    assert_eq!(before, 6);
    assert!(
        !seen.evicted.is_empty(),
        "the chunks the focus left were dropped"
    );
    assert!(
        seen.evicted.contains(&ChunkCoord::new(0, 0, 0)),
        "including the one it started in: {:?}",
        seen.evicted
    );
    assert!(
        seen.evicted
            .iter()
            .all(|chunk| world.chunk(*chunk).is_none()),
        "what was reported as evicted is gone: {:?}",
        seen.evicted
    );
    assert!(
        world.chunk(ChunkCoord::new(3, 0, 0)).is_some(),
        "and the chunks it walked into are there"
    );
}

#[test]
fn a_chunk_sits_where_bevys_world_space_says_it_does() {
    let settings = settings(None);

    // Bevy is Y-up and the lattice is Z-up, so Bevy's xz plane is the lattice's xy.
    assert_eq!(settings.chunk_size(), Vec3::new(8.0, 4.0, 8.0));
    assert_eq!(
        settings.chunk_at(Vec3::new(9.0, 0.5, 17.0)),
        ChunkCoord::new(1, 2, 0)
    );
    assert_eq!(
        settings.translation_of(ChunkCoord::new(1, 2, 0)),
        Vec3::new(8.0, 0.0, 16.0)
    );
    // A point just below the origin belongs to the chunk below it, not to the one above.
    assert_eq!(
        settings.chunk_at(Vec3::new(-0.5, -0.5, -0.5)),
        ChunkCoord::new(-1, -1, -1)
    );
}

#[test]
fn the_world_a_bevy_app_generates_is_the_world_the_library_generates() {
    let mut app = app(1, None);
    run_until_idle(&mut app);

    let mut direct = wave_forge::Builder::new(ruleset(), Prior::open(TILES))
        .seed(7)
        .extent(settings(None).extent)
        .halo(1)
        .build_with(ReferenceSolver::new(Arc::new(ruleset())));
    direct.request(&[wave_forge::FocusPoint::new(ChunkCoord::new(0, 0, 0), 1)]);
    direct.run_until_idle().expect("the reference solves");

    let world = app.world().resource::<World>();
    for chunk in direct.store().iter() {
        assert_eq!(
            world.chunk(chunk.coord).map(|theirs| theirs.tiles.to_vec()),
            Some(chunk.tiles.to_vec()),
            "{:?}",
            chunk.coord
        );
    }
    assert_eq!(world.store().len(), direct.store().len());
}

/// Nothing in the plugin depends on a focus existing, so an app without one simply generates
/// nothing rather than failing.
#[test]
fn an_app_without_a_focus_generates_nothing() {
    let mut app = App::new();
    app.add_plugins(WaveForgeSolverPlugin::new(
        ruleset(),
        Prior::open(TILES),
        settings(None),
        ReferenceSolver::new(Arc::new(ruleset())),
    ));
    app.finish();
    app.cleanup();

    app.update();
    app.update();

    let world = app.world().resource::<World>();
    assert_eq!(world.store().len(), 0);
    assert_eq!(world.stats().batches, 0);
}

/// A game may want to drive generation itself. The resource hands the generator over for that.
#[test]
fn a_game_can_drive_the_generator_itself() {
    fn ask(mut world: ResMut<World>) {
        world
            .generator_mut()
            .request(&[wave_forge::FocusPoint::new(ChunkCoord::new(2, 2, 0), 0)]);
    }
    fn wait(mut world: ResMut<World>) -> bevy_ecs::error::Result {
        world.generator_mut().wait()?;
        Ok(())
    }
    let mut app = App::new();
    app.add_plugins(WaveForgeSolverPlugin::new(
        ruleset(),
        Prior::open(TILES),
        settings(None),
        ReferenceSolver::new(Arc::new(ruleset())),
    ))
    .add_systems(bevy_app::Startup, ask)
    .add_systems(Update, wait);
    app.finish();
    app.cleanup();

    for _ in 0..8 {
        app.update();
    }

    let world = app.world().resource::<World>();
    assert!(world.chunk(ChunkCoord::new(2, 2, 0)).is_some());
}

/// The systems read the world through a resource, so a game can look at what was generated with
/// nothing more than `Res`.
#[test]
fn a_system_can_read_the_generated_tiles() {
    fn count_tiles(world: Res<World>, mut seen: ResMut<Seen>) {
        if let Some(chunk) = world.chunk(ChunkCoord::new(0, 0, 0)) {
            assert_eq!(chunk.tiles.len(), CHUNK.cells() as usize);
            seen.updated.push(chunk.coord);
        }
    }
    let mut app = app(0, None);
    app.add_systems(Update, count_tiles);

    run_until_idle(&mut app);

    assert!(
        app.world()
            .resource::<Seen>()
            .updated
            .contains(&ChunkCoord::new(0, 0, 0))
    );
}
