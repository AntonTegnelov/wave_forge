//! A pack of stages in a Bevy app: a focus entity makes the stages generate around it, what
//! arrives is what the library's runtime generates, and moving away drops what was left behind.

use bevy_app::{App, Startup, Update};
use bevy_ecs::message::MessageReader;
use bevy_ecs::prelude::{Commands, IntoScheduleConfigs, Query, ResMut, Resource, With};
use bevy_math::Vec3;
use bevy_transform::components::GlobalTransform;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wave_forge::stages::{Pack, Runtime};
use wave_forge::{ChunkCoord, FocusPoint};
use wave_forge_bevy::GenerationFocus;
use wave_forge_bevy::stages::{
    StageDropped, StageReady, StagesSettings, WaveForgeStages, WaveForgeStagesPlugin,
    WaveForgeStagesSystems,
};

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "height", kind: Field(Mul(Noise(frequency: 0.04, octaves: 3), Constant(20.0)))),
        (name: "trees", kind: Scatter(kind: "tree", height: "height", spacing: 3, apart: 3)),
    ],
)"#;

const SETTINGS: StagesSettings = StagesSettings {
    chunk: [8, 8],
    cell_size: Vec3::new(2.0, 1.0, 2.0),
};

fn runtime() -> Runtime {
    Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        4,
        SETTINGS.chunk,
    )
}

#[derive(Resource, Default)]
struct Seen {
    ready: Vec<(String, ChunkCoord)>,
    dropped: Vec<(String, ChunkCoord)>,
}

fn collect(
    mut seen: ResMut<Seen>,
    mut ready: MessageReader<StageReady>,
    mut dropped: MessageReader<StageDropped>,
) {
    seen.ready
        .extend(ready.read().map(|m| (m.stage.clone(), m.chunk)));
    seen.dropped
        .extend(dropped.read().map(|m| (m.stage.clone(), m.chunk)));
}

fn app() -> App {
    let mut app = App::new();
    app.add_plugins(WaveForgeStagesPlugin::new(
        &["height", "trees"],
        SETTINGS,
        || Ok(runtime()),
    ))
    .init_resource::<Seen>()
    .add_systems(Update, collect.after(WaveForgeStagesSystems))
    .add_systems(Startup, |mut commands: Commands| {
        commands.spawn((GlobalTransform::default(), GenerationFocus::new(1)));
    });
    app
}

/// Runs frames until `done`, or fails after ten seconds.
fn run_until(app: &mut App, done: impl Fn(&App) -> bool) {
    let started = Instant::now();
    while !done(app) {
        app.update();
        assert!(
            app.world()
                .resource::<WaveForgeStages>()
                .failure()
                .is_none(),
            "{:?}",
            app.world().resource::<WaveForgeStages>().failure()
        );
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "nothing arrived in 10 s"
        );
    }
    app.update();
}

fn around_origin() -> Vec<ChunkCoord> {
    (-1..=1)
        .flat_map(|y| (-1..=1).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

#[test]
fn what_arrives_is_what_the_runtime_generates() {
    let mut app = app();
    let mut direct = runtime();
    direct
        .request(
            &[FocusPoint::new(ChunkCoord::new(0, 0, 0), 1)],
            &["height", "trees"],
        )
        .expect("stages");
    direct.run_until_idle().expect("the stages run");

    run_until(&mut app, |app| {
        let stages = app.world().resource::<WaveForgeStages>();
        around_origin()
            .iter()
            .all(|&c| stages.points("trees", c).is_some())
    });

    let stages = app.world().resource::<WaveForgeStages>();
    for chunk in around_origin() {
        assert_eq!(stages.field("height", chunk), direct.field("height", chunk));
        assert_eq!(stages.points("trees", chunk), direct.points("trees", chunk));
    }
    let seen = app.world().resource::<Seen>();
    assert!(
        seen.ready
            .contains(&("trees".to_owned(), ChunkCoord::new(0, 0, 0)))
    );
}

#[test]
fn a_point_stands_where_the_lattice_puts_it_in_bevys_world() {
    let mut app = app();
    run_until(&mut app, |app| {
        app.world()
            .resource::<WaveForgeStages>()
            .points("trees", ChunkCoord::new(0, 0, 0))
            .is_some_and(|points| !points.is_empty())
    });
    let stages = app.world().resource::<WaveForgeStages>();
    let point = &stages
        .points("trees", ChunkCoord::new(0, 0, 0))
        .expect("arrived")[0];

    let at = stages.translation_of(point);

    assert_eq!(at.x, point.position[0] * 2.0);
    assert_eq!(at.y, point.position[2]);
    assert_eq!(at.z, point.position[1] * 2.0);
}

#[test]
fn moving_away_drops_what_was_left_behind() {
    let mut app = app();
    run_until(&mut app, |app| {
        app.world()
            .resource::<WaveForgeStages>()
            .points("trees", ChunkCoord::new(0, 0, 0))
            .is_some()
    });

    app.add_systems(
        Update,
        |mut focus: Query<&mut GlobalTransform, With<GenerationFocus>>| {
            for mut at in &mut focus {
                *at = GlobalTransform::from_translation(Vec3::new(800.0, 0.0, 800.0));
            }
        },
    );
    run_until(&mut app, |app| {
        app.world()
            .resource::<Seen>()
            .dropped
            .contains(&("trees".to_owned(), ChunkCoord::new(0, 0, 0)))
    });

    assert!(
        app.world()
            .resource::<WaveForgeStages>()
            .points("trees", ChunkCoord::new(0, 0, 0))
            .is_none()
    );
}
