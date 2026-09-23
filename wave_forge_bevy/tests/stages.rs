//! A pack of stages in a Bevy app: a focus entity makes the stages generate around it, what
//! arrives is what the library's runtime generates, and moving away drops what was left behind.
//! The same holds for each chunk's ground, built from the height field.

use bevy_app::{App, Startup, Update};
use bevy_ecs::message::MessageReader;
use bevy_ecs::prelude::{Commands, IntoScheduleConfigs, Query, ResMut, Resource, With};
use bevy_math::Vec3;
use bevy_transform::components::GlobalTransform;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wave_forge::stages::{Facts, GivenRow, Pack, RowId, Runtime, Value};
use wave_forge::{ChunkCoord, FocusPoint, ground};
use wave_forge_bevy::GenerationFocus;
use wave_forge_bevy::stages::{
    GroundDropped, GroundReady, StageDropped, StageReady, StagesSettings, WaveForgeStages,
    WaveForgeStagesPlugin, WaveForgeStagesSystems, ground_mesh,
};

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "height", kind: Field(Mul(Noise(frequency: 0.04, octaves: 3), Constant(20.0)))),
        (name: "trees", kind: Scatter(kind: "tree", height: "height", spacing: 3, apart: 3)),
        (name: "cover", kind: Rules(rules: [(category: "high", when: [Greater(Input("height"), Constant(10.0))])], otherwise: "low")),
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
    grounds: Vec<ChunkCoord>,
    grounds_dropped: Vec<ChunkCoord>,
}

fn collect(
    mut seen: ResMut<Seen>,
    mut ready: MessageReader<StageReady>,
    mut dropped: MessageReader<StageDropped>,
    mut grounds: MessageReader<GroundReady>,
    mut grounds_dropped: MessageReader<GroundDropped>,
) {
    seen.ready
        .extend(ready.read().map(|m| (m.stage.clone(), m.chunk)));
    seen.dropped
        .extend(dropped.read().map(|m| (m.stage.clone(), m.chunk)));
    seen.grounds.extend(grounds.read().map(|m| m.0));
    seen.grounds_dropped
        .extend(grounds_dropped.read().map(|m| m.0));
}

fn plugin() -> WaveForgeStagesPlugin {
    WaveForgeStagesPlugin::new(&["height", "trees", "cover"], SETTINGS, || Ok(runtime()))
}

fn app() -> App {
    app_with(plugin())
}

fn app_with(plugin: WaveForgeStagesPlugin) -> App {
    let mut app = App::new();
    app.add_plugins(plugin)
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

#[test]
fn the_ground_that_arrives_is_the_librarys_ground_of_the_same_fields() {
    let mut app = app_with(plugin().with_ground("height"));
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
        around_origin().iter().all(|&c| stages.ground(c).is_some())
    });

    let stages = app.world().resource::<WaveForgeStages>();
    let cell = SETTINGS.cell_size.to_array();
    for chunk in around_origin() {
        let expected = ground(chunk, |at| direct.field("height", at), cell);
        assert_eq!(stages.ground(chunk), expected.as_ref(), "chunk {chunk:?}");
    }
    let seen = app.world().resource::<Seen>();
    for chunk in around_origin() {
        assert_eq!(
            seen.grounds.iter().filter(|&&c| c == chunk).count(),
            1,
            "{chunk:?} is announced once"
        );
    }
}

#[test]
fn a_ground_mesh_has_a_vertex_per_column_and_its_neighbours_edge() {
    let mut app = app_with(plugin().with_ground("height"));
    let origin = ChunkCoord::new(0, 0, 0);
    run_until(&mut app, |app| {
        app.world()
            .resource::<WaveForgeStages>()
            .ground(origin)
            .is_some()
    });
    let ground = app
        .world()
        .resource::<WaveForgeStages>()
        .ground(origin)
        .expect("arrived");

    let mesh = ground_mesh(ground);

    assert_eq!(mesh.count_vertices(), 9 * 9);
    assert_eq!(mesh.indices().expect("indexed").len(), 8 * 8 * 6);
}

#[test]
fn moving_away_drops_the_ground() {
    let mut app = app_with(plugin().with_ground("height"));
    let origin = ChunkCoord::new(0, 0, 0);
    run_until(&mut app, |app| {
        app.world()
            .resource::<WaveForgeStages>()
            .ground(origin)
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
            .grounds_dropped
            .contains(&origin)
    });

    assert!(
        app.world()
            .resource::<WaveForgeStages>()
            .ground(origin)
            .is_none()
    );
}

#[test]
fn categories_arrive_as_the_runtime_generates_them() {
    let mut app = app();
    let mut direct = runtime();
    direct
        .request(&[FocusPoint::new(ChunkCoord::new(0, 0, 0), 1)], &["cover"])
        .expect("stages");
    direct.run_until_idle().expect("the stages run");

    run_until(&mut app, |app| {
        let stages = app.world().resource::<WaveForgeStages>();
        around_origin()
            .iter()
            .all(|&c| stages.categories("cover", c).is_some())
    });

    let stages = app.world().resource::<WaveForgeStages>();
    for chunk in around_origin() {
        assert_eq!(
            stages.categories("cover", chunk),
            direct.categories("cover", chunk),
            "{chunk:?}"
        );
    }
}

#[test]
fn each_stage_reports_what_it_has_cost() {
    let mut app = app();

    run_until(&mut app, |app| {
        let stages = app.world().resource::<WaveForgeStages>();
        !stages.timings().is_empty()
            && stages
                .timings()
                .iter()
                .all(|(_, timing)| timing.products >= 9)
    });

    let stages = app.world().resource::<WaveForgeStages>();
    let names: Vec<&str> = stages
        .timings()
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert_eq!(names, ["height", "trees", "cover"]);
}

/// A straight line across each region, at a row hashed from its west edge.
struct Lines;

impl wave_forge::stages::regions::RegionJob for Lines {
    fn run(
        &self,
        input: &wave_forge::stages::regions::RegionInput<'_>,
    ) -> Result<wave_forge::stages::regions::Attempt, wave_forge::stages::StageError> {
        use wave_forge::stages::regions::{Attempt, Curve, CurveId, Edge};
        let ([x0, y0], [x1, _]) = input.columns();
        let row = y0 as f32 + (input.edge_hash(Edge::West, 0) % 8) as f32;
        Ok(Attempt::Accepted(vec![Curve {
            id: CurveId::Region {
                region: input.region(),
                index: 0,
            },
            points: vec![[x0 as f32, row], [x1 as f32 + 1.0, row]],
            values: vec![0.0, 0.0],
        }]))
    }
}

const REGIONS: &str = r#"(
    version: 1,
    stages: [
        (name: "lines", kind: Region(job: "lines", region: 2)),
    ],
)"#;

fn lines_runtime() -> Runtime {
    Runtime::new(
        Arc::new(Pack::parse(REGIONS).expect("a valid pack")),
        4,
        SETTINGS.chunk,
    )
    .with_region_job("lines", Lines)
}

#[test]
fn a_region_jobs_curves_arrive_as_the_runtime_makes_them() {
    let mut app = app_with(WaveForgeStagesPlugin::new(&["lines"], SETTINGS, || {
        Ok(lines_runtime())
    }));
    let mut direct = lines_runtime();
    direct
        .request(&[FocusPoint::new(ChunkCoord::new(0, 0, 0), 1)], &["lines"])
        .expect("stages");
    direct.run_until_idle().expect("the stages run");

    run_until(&mut app, |app| {
        let stages = app.world().resource::<WaveForgeStages>();
        around_origin()
            .iter()
            .all(|&c| stages.curves("lines", c).is_some())
    });

    let stages = app.world().resource::<WaveForgeStages>();
    for chunk in around_origin() {
        assert_eq!(
            stages.curves("lines", chunk),
            direct.curves("lines", chunk),
            "{chunk:?}"
        );
    }
}

const FACTS: &str = r#"(
    version: 1,
    tables: [(name: "villages", kind: Given(columns: [("population", Number)]))],
    stages: [(name: "wealth", kind: Field(Add(Noise(frequency: 0.05, octaves: 2), Row("villages", "population"))))],
)"#;

/// Facts for `pack`, the pack the runtime was made with, with one village of `population`.
fn villages(pack: &Arc<Pack>, population: f32) -> Facts {
    let mut facts = Facts::new(Arc::clone(pack), 4).expect("facts");
    facts
        .give(
            "villages",
            vec![GivenRow {
                id: 1,
                values: BTreeMap::from([("population".to_owned(), Value::Number(population))]),
            }],
        )
        .expect("villages");
    facts
}

#[test]
fn new_facts_drop_the_stages_that_read_them_and_regenerate_them() {
    let pack = Arc::new(Pack::parse(FACTS).expect("a valid pack"));
    let (for_thread, facts) = (Arc::clone(&pack), villages(&pack, 10.0));
    let mut app = app_with(WaveForgeStagesPlugin::new(
        &["wealth"],
        SETTINGS,
        move || {
            let mut runtime = Runtime::new(for_thread, 4, SETTINGS.chunk);
            runtime
                .set_facts(facts)
                .map_err(|error| error.to_string())?;
            runtime
                .focus("villages", RowId(vec![1]))
                .map_err(|error| error.to_string())?;
            Ok(runtime)
        },
    ));
    let origin = ChunkCoord::new(0, 0, 0);
    run_until(&mut app, |app| {
        app.world()
            .resource::<WaveForgeStages>()
            .field("wealth", origin)
            .is_some()
    });
    let before = app
        .world()
        .resource::<WaveForgeStages>()
        .field("wealth", origin)
        .expect("arrived")
        .values
        .clone();

    app.world()
        .resource::<WaveForgeStages>()
        .set_facts(villages(&pack, 60.0));
    run_until(&mut app, |app| {
        app.world()
            .resource::<WaveForgeStages>()
            .field("wealth", origin)
            .is_some_and(|field| field.values != before)
    });

    let after = &app
        .world()
        .resource::<WaveForgeStages>()
        .field("wealth", origin)
        .expect("arrived")
        .values;
    assert!(
        app.world()
            .resource::<Seen>()
            .dropped
            .contains(&("wealth".to_owned(), origin))
    );
    for (low, high) in before.iter().zip(after) {
        assert!((high - low - 50.0).abs() < 1e-3, "{low} then {high}");
    }
}
