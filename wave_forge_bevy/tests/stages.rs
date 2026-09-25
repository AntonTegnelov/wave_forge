//! A pack of stages in a Bevy app: a focus entity makes the stages generate around it, what
//! arrives is what the library's runtime generates, and moving away drops what was left behind.
//! The same holds for each chunk's ground, built from the height field, and a save made in one app
//! brings a player's edits back in another. An assembled piece stands where its stage grew it, and
//! bound kinds get an entity per point or piece, which goes with its chunk.

use bevy_app::{App, Startup, Update};
use bevy_ecs::message::{MessageReader, Messages};
use bevy_ecs::prelude::{Commands, IntoScheduleConfigs, Query, ResMut, Resource, With};
use bevy_math::Vec3;
use bevy_transform::components::{GlobalTransform, Transform};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wave_forge::stages::{
    Edit, Edits, Facts, GivenRow, Pack, PointId, RowId, Runtime, Save, Stamp, Value,
};
use wave_forge::{ChunkCoord, FocusPoint, ground, ground_materials};
use wave_forge_bevy::GenerationFocus;
use wave_forge_bevy::stages::{
    FarGroundDropped, GroundDropped, GroundReady, InstanceSpawned, Placed, StageDropped,
    StagePlacements, StageReady, StagesSaved, StagesSettings, WaveForgeStages,
    WaveForgeStagesPlugin, WaveForgeStagesSystems, far_ground_mesh, ground_mesh,
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
    saves: Vec<Save>,
}

fn collect(
    mut seen: ResMut<Seen>,
    mut ready: MessageReader<StageReady>,
    mut dropped: MessageReader<StageDropped>,
    mut grounds: MessageReader<GroundReady>,
    mut grounds_dropped: MessageReader<GroundDropped>,
    mut saves: MessageReader<StagesSaved>,
) {
    seen.ready
        .extend(ready.read().map(|m| (m.stage.clone(), m.chunk)));
    seen.dropped
        .extend(dropped.read().map(|m| (m.stage.clone(), m.chunk)));
    seen.grounds.extend(grounds.read().map(|m| m.0));
    seen.grounds_dropped
        .extend(grounds_dropped.read().map(|m| m.0));
    seen.saves.extend(saves.read().map(|m| m.0.clone()));
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
    let standing = stages.transform_of(point);
    assert_eq!(standing.translation, at);
    assert!(
        (standing.rotation * Vec3::Y - Vec3::Y).length() < 1e-5,
        "an upright tree"
    );
    assert_eq!(standing.scale, Vec3::ONE);
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
fn the_grounds_materials_are_the_librarys_categories_of_its_vertices() {
    let mut app = app_with(
        plugin()
            .with_ground("height")
            .with_ground_materials("cover")
            // The ground reads the materials of the chunks beyond its far edges too.
            .with_radius("cover", 2),
    );
    let mut direct = runtime();
    direct
        .request(
            &[FocusPoint::new(ChunkCoord::new(0, 0, 0), 2)],
            &["height", "cover"],
        )
        .expect("stages");
    direct.run_until_idle().expect("the stages run");

    run_until(&mut app, |app| {
        let stages = app.world().resource::<WaveForgeStages>();
        around_origin()
            .iter()
            .all(|&c| stages.ground_materials(c).is_some())
    });

    let stages = app.world().resource::<WaveForgeStages>();
    for chunk in around_origin() {
        let expected = ground_materials(chunk, |at| direct.categories("cover", at));
        assert_eq!(
            stages.ground_materials(chunk),
            expected.as_deref(),
            "chunk {chunk:?}"
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

    assert_eq!(ground.size, [9, 9]);
    assert_eq!(mesh.count_vertices(), ground.positions.len());
    assert_eq!(
        mesh.indices().expect("indexed").len(),
        ground.levels[0].indices.len()
    );
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

#[test]
fn a_target_with_a_radius_of_its_own_reaches_past_the_focus() {
    let mut app = app_with(plugin().with_radius("height", 3));

    run_until(&mut app, |app| {
        let stages = app.world().resource::<WaveForgeStages>();
        stages.field("height", ChunkCoord::new(3, 0, 0)).is_some()
            && stages.points("trees", ChunkCoord::new(1, 0, 0)).is_some()
    });

    let stages = app.world().resource::<WaveForgeStages>();
    assert!(stages.points("trees", ChunkCoord::new(2, 0, 0)).is_none());
    assert!(stages.field("height", ChunkCoord::new(4, 0, 0)).is_none());
}

/// The trees in the chunk at the origin, once they have arrived.
fn trees_at_origin(app: &mut App) -> Vec<wave_forge::stages::Point> {
    let origin = ChunkCoord::new(0, 0, 0);
    run_until(app, |app| {
        app.world()
            .resource::<WaveForgeStages>()
            .points("trees", origin)
            .is_some()
    });
    app.world()
        .resource::<WaveForgeStages>()
        .points("trees", origin)
        .expect("arrived")
        .to_vec()
}

#[test]
fn a_save_made_in_one_app_brings_a_felled_tree_back_felled_in_another() {
    let mut first = app();
    let felled = trees_at_origin(&mut first)[0].clone();
    let mut edits = Edits::default();
    edits.push(Edit::Remove {
        point: PointId::from(felled.id),
        at: [felled.position[0], felled.position[1]],
    });
    first
        .world()
        .resource::<WaveForgeStages>()
        .set_edits(edits.clone());
    first.world().resource::<WaveForgeStages>().request_save();
    run_until(&mut first, |app| {
        !app.world().resource::<Seen>().saves.is_empty()
    });
    let save = first.world().resource::<Seen>().saves[0].clone();

    let mut second = app();
    second
        .world()
        .resource::<WaveForgeStages>()
        .load(save.clone());
    let trees = trees_at_origin(&mut second);

    assert_eq!(save.edits, edits);
    assert!(trees.iter().all(|tree| tree.id != felled.id), "{trees:?}");
}

const VILLAGE: &str = r#"(
    version: 1,
    stages: [
        (name: "ground", kind: Field(Mul(Noise(frequency: 0.03, octaves: 2), Constant(12.0)))),
        (name: "villages", kind: Sites(height: "ground", region: 6, size: (2, 3), chance: 1.0)),
        (name: "village", kind: Assemble(sites: "villages", start: "square", max: 14, min: 5, pieces: [
            (name: "square", size: (4, 4, 1), doors: [
                (at: (3, 1, 0), facing: East, kind: "street"),
                (at: (0, 2, 0), facing: West, kind: "street"),
            ]),
            (name: "street", size: (1, 4, 1), weight: 3, doors: [
                (at: (0, 0, 0), facing: South, kind: "street"),
                (at: (0, 3, 0), facing: North, kind: "street"),
                (at: (0, 1, 0), facing: East, kind: "house"),
                (at: (0, 2, 0), facing: West, kind: "house"),
            ]),
            (name: "house", size: (3, 3, 1), weight: 2, doors: [(at: (1, 0, 0), facing: South, kind: "house")]),
        ])),
    ],
)"#;

#[test]
fn an_assembled_house_placed_by_its_transform_opens_its_door_onto_a_street() {
    let pack = Arc::new(Pack::parse(VILLAGE).expect("a valid pack"));
    let for_thread = Arc::clone(&pack);
    let mut app = app_with(
        WaveForgeStagesPlugin::new(&["village"], SETTINGS, move || {
            Ok(Runtime::new(for_thread, 21, SETTINGS.chunk))
        })
        .with_radius("village", 4),
    );
    let area: Vec<ChunkCoord> = (-4..=4)
        .flat_map(|y| (-4..=4).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect();
    run_until(&mut app, |app| {
        let stages = app.world().resource::<WaveForgeStages>();
        area.iter()
            .all(|&chunk| stages.stamps("village", chunk).is_some())
    });
    let stages = app.world().resource::<WaveForgeStages>();
    let mut stamps: Vec<Stamp> = Vec::new();
    for &chunk in &area {
        for stamp in stages.stamps("village", chunk).expect("arrived") {
            if !stamps.contains(stamp) {
                stamps.push(stamp.clone());
            }
        }
    }
    let column = |at: Vec3| {
        (
            (at.x / SETTINGS.cell_size.x).floor() as i64,
            (at.z / SETTINGS.cell_size.z).floor() as i64,
        )
    };
    let covers = |stamp: &Stamp, (x, y): (i64, i64)| {
        (stamp.min[0]..stamp.max[0]).contains(&x) && (stamp.min[1]..stamp.max[1]).contains(&y)
    };

    let mut checked = 0;
    for house in stamps.iter().filter(|stamp| &*stamp.piece == "house") {
        let transform = stages.stamp_transform(house);
        let door = column(transform.transform_point(Vec3::new(0.0, 0.0, -SETTINGS.cell_size.z)));
        let outside =
            column(transform.transform_point(Vec3::new(0.0, 0.0, -2.0 * SETTINGS.cell_size.z)));
        assert!(covers(house, door), "{house:?}: door at {door:?}");
        if (-32..40).contains(&outside.0) && (-32..40).contains(&outside.1) {
            assert!(
                stamps
                    .iter()
                    .any(|other| &*other.piece == "street" && covers(other, outside)),
                "{house:?}: its door at {door:?} opens onto no street"
            );
            checked += 1;
        }
    }
    assert!(checked >= 3, "{checked} houses checked");
}

#[derive(bevy_ecs::prelude::Component)]
struct Tree;

#[derive(bevy_ecs::prelude::Component)]
struct House;

/// The placed entities of a marker, with their transforms and what they stand for.
fn placed<T: bevy_ecs::prelude::Component>(app: &mut App) -> Vec<(Transform, Placed)> {
    let mut query = app
        .world_mut()
        .query_filtered::<(&Transform, &Placed), With<T>>();
    query
        .iter(app.world())
        .map(|(transform, placed)| (*transform, placed.clone()))
        .collect()
}

#[test]
fn every_tree_of_a_bound_kind_gets_an_entity_that_goes_with_its_chunk() {
    let mut app = app();
    app.insert_resource(StagePlacements::default().bind("tree", |entity| {
        entity.insert(Tree);
    }));
    run_until(&mut app, |app| {
        let stages = app.world().resource::<WaveForgeStages>();
        around_origin()
            .iter()
            .all(|&chunk| stages.points("trees", chunk).is_some())
    });
    let spawned = app.world().resource::<Messages<InstanceSpawned>>().len();

    let trees = placed::<Tree>(&mut app);

    let stages = app.world().resource::<WaveForgeStages>();
    let mut expected = 0;
    for chunk in around_origin() {
        for point in stages.points("trees", chunk).expect("arrived") {
            expected += 1;
            let entity = trees
                .iter()
                .find(|(_, placed)| placed.id == point.id)
                .expect("an entity for every tree");
            assert_eq!(entity.0, stages.transform_of(point));
            assert_eq!(entity.1.chunk, chunk);
        }
    }
    assert_eq!(trees.len(), expected);
    assert!(spawned > 0);

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
    let trees = placed::<Tree>(&mut app);
    let stages = app.world().resource::<WaveForgeStages>();
    assert!(
        trees
            .iter()
            .all(|(_, placed)| stages.points("trees", placed.chunk).is_some())
    );
}

#[test]
fn a_piece_overlapping_several_chunks_gets_one_entity() {
    let pack = Arc::new(Pack::parse(VILLAGE).expect("a valid pack"));
    let mut app = app_with(
        WaveForgeStagesPlugin::new(&["village"], SETTINGS, move || {
            Ok(Runtime::new(pack, 21, SETTINGS.chunk))
        })
        .with_radius("village", 4),
    );
    app.insert_resource(StagePlacements::default().bind("house", |entity| {
        entity.insert(House);
    }));
    let area: Vec<ChunkCoord> = (-4..=4)
        .flat_map(|y| (-4..=4).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect();
    run_until(&mut app, |app| {
        let stages = app.world().resource::<WaveForgeStages>();
        area.iter()
            .all(|&chunk| stages.stamps("village", chunk).is_some())
    });

    let houses = placed::<House>(&mut app);

    let stages = app.world().resource::<WaveForgeStages>();
    let mut owned = std::collections::BTreeSet::new();
    for &chunk in &area {
        for stamp in stages.stamps("village", chunk).expect("arrived") {
            if &*stamp.piece == "house" && stamp.id.chunk == chunk {
                owned.insert(stamp.id);
            }
        }
    }
    let ids: std::collections::BTreeSet<_> = houses.iter().map(|(_, placed)| placed.id).collect();
    assert!(owned.len() >= 3, "{} houses", owned.len());
    assert_eq!(houses.len(), ids.len(), "no house placed twice");
    assert_eq!(ids, owned);
}

/// A pack with ground at full detail near the focus and at a coarse scale of 8 beyond it.
const FAR_PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "height", kind: Field(Mul(Noise(frequency: 0.01, octaves: 3), Constant(10.0)))),
        (name: "far", scale: 8, kind: Field(Mul(Noise(frequency: 0.01, octaves: 3), Constant(10.0)))),
    ],
)"#;

fn far_app() -> App {
    app_with(
        WaveForgeStagesPlugin::new(&["height", "far"], SETTINGS, || {
            Ok(Runtime::new(
                Arc::new(Pack::parse(FAR_PACK).expect("a valid pack")),
                4,
                SETTINGS.chunk,
            ))
        })
        .with_ground("height")
        .with_far_ground("far", 8)
        .with_radius("far", 24),
    )
}

#[test]
fn the_far_ground_is_the_librarys_and_leaves_out_the_chunks_with_ground() {
    let mut app = far_app();

    run_until(&mut app, |app| {
        let stages = app.world().resource::<WaveForgeStages>();
        stages.far_ground(ChunkCoord::new(0, 0, 0)).is_some()
            && stages.ground(ChunkCoord::new(0, 0, 0)).is_some()
    });

    let stages = app.world().resource::<WaveForgeStages>();
    let origin = ChunkCoord::new(0, 0, 0);
    let expected = wave_forge::far_ground(
        origin,
        8,
        |at| stages.field("far", at),
        SETTINGS.cell_size.to_array(),
        |fine| stages.ground(fine),
    )
    .expect("the fields around it arrived");
    assert_eq!(stages.far_ground(origin), Some(&expected));
    assert_eq!(
        stages.far_ground_corner(origin),
        stages.chunk_corner(origin)
    );
    let far = far_ground_mesh(stages.far_ground(origin).expect("built"));
    assert_eq!(
        far.count_vertices(),
        expected.positions.len(),
        "the mesh holds the far ground's vertices"
    );
}

#[test]
fn moving_away_drops_the_far_ground() {
    let mut app = far_app();
    let origin = ChunkCoord::new(0, 0, 0);
    run_until(&mut app, |app| {
        app.world()
            .resource::<WaveForgeStages>()
            .far_ground(origin)
            .is_some()
    });

    let mut focus = app
        .world_mut()
        .query_filtered::<&mut GlobalTransform, With<GenerationFocus>>();
    for mut at in focus.iter_mut(app.world_mut()) {
        *at = GlobalTransform::from_translation(Vec3::new(8000.0, 0.0, 8000.0));
    }
    run_until(&mut app, |app| {
        app.world()
            .resource::<WaveForgeStages>()
            .far_ground(origin)
            .is_none()
    });

    let dropped: Vec<ChunkCoord> = app
        .world_mut()
        .resource_mut::<Messages<FarGroundDropped>>()
        .drain()
        .map(|message| message.0)
        .collect();
    assert!(dropped.contains(&origin), "{dropped:?}");
}
