//! A pack of stages in a Bevy app: a focus entity makes the stages generate around it, what
//! arrives is what the library's runtime generates, and moving away drops what was left behind.
//! The same holds for each chunk's ground, built from the height field, and a save made in one app
//! brings a player's edits back in another. An assembled piece stands where its stage grew it, and
//! bound kinds get an entity per point, piece or cave room, which goes with its chunk.

use bevy_app::{App, Startup, Update};
use bevy_color::{Color, ColorToComponents};
use bevy_ecs::message::{MessageReader, Messages};
use bevy_ecs::prelude::{Commands, IntoScheduleConfigs, Query, ResMut, Resource, With};
use bevy_math::Vec3;
use bevy_mesh::{Mesh, VertexAttributeValues};
use bevy_transform::components::{GlobalTransform, Transform};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wave_forge::stages::{
    Edit, Edits, Facts, GivenRow, Pack, PointId, RowId, Runtime, Save, Stamp, Value,
};
use wave_forge::{ChunkCoord, FocusPoint, ground, ground_materials, volume_mesh};
use wave_forge_bevy::GenerationFocus;
use wave_forge_bevy::materials::coloured_surface_mesh;
use wave_forge_bevy::stages::{
    FarGroundDropped, FluidDropped, FluidReady, GroundDropped, GroundReady, InstanceSpawned,
    Placed, StageDropped, StagePlacements, StageReady, StagesSaved, StagesSettings, VolumeDropped,
    WaveForgeStages, WaveForgeStagesPlugin, WaveForgeStagesSystems, far_ground_mesh, ground_mesh,
    surface_mesh,
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
    volumes_dropped: Vec<ChunkCoord>,
    saves: Vec<Save>,
}

fn collect(
    mut seen: ResMut<Seen>,
    mut ready: MessageReader<StageReady>,
    mut dropped: MessageReader<StageDropped>,
    mut grounds: MessageReader<GroundReady>,
    mut grounds_dropped: MessageReader<GroundDropped>,
    mut volumes_dropped: MessageReader<VolumeDropped>,
    mut saves: MessageReader<StagesSaved>,
) {
    seen.ready
        .extend(ready.read().map(|m| (m.stage.clone(), m.chunk)));
    seen.dropped
        .extend(dropped.read().map(|m| (m.stage.clone(), m.chunk)));
    seen.grounds.extend(grounds.read().map(|m| m.0));
    seen.grounds_dropped
        .extend(grounds_dropped.read().map(|m| m.0));
    seen.volumes_dropped
        .extend(volumes_dropped.read().map(|m| m.0));
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
            heights: Vec::new(),
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
const CAVE: &str = r#"(
    version: 1,
    stages: [
        (name: "level", kind: Cave(region: 8, depth: (-30.0, -10.0), patterns: [Star],
            count: (4, 6), apart: 12.0, rerolls: 8, rooms: [(name: "cavern", size: (7, 7, 5))])),
        (name: "enemies", kind: Spawn(cave: "level", budget: 4, kinds: [(kind: "grunt", cost: 1)])),
    ],
)"#;

#[derive(bevy_ecs::prelude::Component)]
struct Cavern;

#[derive(bevy_ecs::prelude::Component)]
struct Grunt;

#[test]
fn a_caves_rooms_and_its_spawned_points_get_an_entity_each() {
    let pack = Arc::new(Pack::parse(CAVE).expect("a valid pack"));
    let mut app = app_with(
        WaveForgeStagesPlugin::new(&["level", "enemies"], SETTINGS, move || {
            Ok(Runtime::new(pack, 5, SETTINGS.chunk))
        })
        .with_radius("level", 4)
        .with_radius("enemies", 4),
    );
    app.insert_resource(
        StagePlacements::default()
            .bind("cavern", |entity| {
                entity.insert(Cavern);
            })
            .bind("grunt", |entity| {
                entity.insert(Grunt);
            }),
    );
    let area: Vec<ChunkCoord> = (-4..=4)
        .flat_map(|y| (-4..=4).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect();
    run_until(&mut app, |app| {
        let stages = app.world().resource::<WaveForgeStages>();
        area.iter().all(|&chunk| {
            stages.stamps("level", chunk).is_some() && stages.points("enemies", chunk).is_some()
        })
    });

    let caverns = placed::<Cavern>(&mut app);
    let grunts = placed::<Grunt>(&mut app);

    let stages = app.world().resource::<WaveForgeStages>();
    let (mut rooms, mut spawns) = (
        std::collections::BTreeSet::new(),
        std::collections::BTreeSet::new(),
    );
    for &chunk in &area {
        for stamp in stages.stamps("level", chunk).expect("arrived") {
            if stamp.id.chunk == chunk {
                rooms.insert(stamp.id);
            }
        }
        for point in stages.points("enemies", chunk).expect("arrived") {
            spawns.insert(point.id);
        }
    }
    let ids = |placed: &[(Transform, Placed)]| {
        placed
            .iter()
            .map(|(_, placed)| placed.id)
            .collect::<std::collections::BTreeSet<_>>()
    };
    assert!(
        rooms.len() >= 4 && spawns.len() >= 16,
        "{rooms:?} {spawns:?}"
    );
    assert_eq!((caverns.len(), ids(&caverns)), (rooms.len(), rooms));
    assert_eq!((grunts.len(), ids(&grunts)), (spawns.len(), spawns));
}

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

#[test]
fn the_ground_height_is_the_librarys_where_the_ground_is() {
    let mut app = app_with(plugin().with_ground("height"));
    let origin = ChunkCoord::new(0, 0, 0);
    run_until(&mut app, |app| {
        app.world()
            .resource::<WaveForgeStages>()
            .ground(origin)
            .is_some()
    });

    let stages = app.world().resource::<WaveForgeStages>();
    for (x, z) in [(1.0, 1.0), (5.3, 9.7), (13.1, 2.2), (15.0, 15.0)] {
        let got = stages.ground_height(Vec3::new(x, 100.0, z));

        let expected = wave_forge::ground_height(
            [x, z],
            SETTINGS.chunk,
            |at| stages.field("height", at),
            SETTINGS.cell_size.to_array(),
        );
        assert!(got.is_some(), "({x}, {z}) is on the ground");
        assert_eq!(got, expected, "({x}, {z})");
    }
}

#[test]
fn the_noise_configuration_is_reflected_and_registered() {
    use bevy_reflect::GetPath;
    use wave_forge::noise::NoiseConfig;
    let app = app();
    let mut noise = NoiseConfig::default();

    *noise
        .path_mut::<f32>("frequency")
        .expect("a reflected field") = 0.25;

    assert_eq!(noise.frequency, 0.25);
    let registry = app
        .world()
        .resource::<bevy_ecs::reflect::AppTypeRegistry>()
        .read();
    assert!(
        registry
            .get(std::any::TypeId::of::<NoiseConfig>())
            .is_some(),
        "the plugin registers the noise configuration"
    );
}

/// A cave volume, solid along its lowest level and empty along its highest.
const VOLUME_PACK: &str = r#"(
    version: 1,
    noises: {
        "caves": (noise_type: SimplexSmooth, seed: 11, frequency: 0.12, fractal_octaves: 2),
    },
    stages: [
        (name: "caves", kind: Volume(
            density: Max(Min(FastNoise("caves"), Sub(Constant(5.0), Z)), Sub(Constant(-3.0), Z)),
            bottom: -4,
            top: 6,
            materials: Some((rules: [(category: "high", when: [Greater(Z, Constant(0.0))])], otherwise: "low")),
        )),
    ],
)"#;

#[test]
fn each_chunks_surface_is_the_librarys_and_goes_with_its_volume() {
    let mut app = app_with(
        WaveForgeStagesPlugin::new(&["caves"], SETTINGS, || {
            Ok(Runtime::new(
                Arc::new(Pack::parse(VOLUME_PACK).expect("a valid pack")),
                4,
                SETTINGS.chunk,
            ))
        })
        .with_volume("caves"),
    );
    let origin = ChunkCoord::new(0, 0, 0);
    run_until(&mut app, |app| {
        app.world()
            .resource::<WaveForgeStages>()
            .surface(origin)
            .is_some()
    });

    let stages = app.world().resource::<WaveForgeStages>();
    let expected = volume_mesh(
        origin,
        |at| stages.volume("caves", at),
        SETTINGS.cell_size.to_array(),
    )
    .expect("the volumes around it arrived");
    assert!(!expected.indices.is_empty());
    assert_eq!(stages.surface(origin), Some(&expected));
    assert_eq!(
        surface_mesh(&expected).count_vertices(),
        expected.positions.len()
    );
    let palette = [Color::srgb(0.2, 0.7, 0.2), Color::srgb(0.5, 0.5, 0.5)];
    let coloured = coloured_surface_mesh(&expected, &palette);
    let Some(VertexAttributeValues::Float32x4(colours)) = coloured.attribute(Mesh::ATTRIBUTE_COLOR)
    else {
        panic!("a coloured surface has vertex colours");
    };
    for (colour, &material) in colours.iter().zip(&expected.materials) {
        assert_eq!(
            *colour,
            palette[usize::from(material)].to_linear().to_f32_array()
        );
    }

    let mut focus = app
        .world_mut()
        .query_filtered::<&mut GlobalTransform, With<GenerationFocus>>();
    for mut at in focus.iter_mut(app.world_mut()) {
        *at = GlobalTransform::from_translation(Vec3::new(8000.0, 0.0, 8000.0));
    }
    run_until(&mut app, |app| {
        app.world()
            .resource::<WaveForgeStages>()
            .surface(origin)
            .is_none()
    });
    let dropped: Vec<ChunkCoord> = app
        .world_mut()
        .resource_mut::<Messages<VolumeDropped>>()
        .drain()
        .map(|message| message.0)
        .collect();
    assert!(dropped.contains(&origin), "{dropped:?}");
}

#[test]
fn each_chunks_fluid_surface_is_the_librarys_and_goes_with_its_fluid() {
    let pack = VOLUME_PACK.replace(
        "    ],\n)",
        r#"        (name: "pools", kind: Aquifer(volume: "caves", cell: (6, 4), level: (-3.0, 2.0),
            materials: Some((rules: [(category: "lava", when: [Less(Z, Constant(-0.5))])], otherwise: "water")))),
    ],
)"#,
    );
    let mut app = app_with(
        WaveForgeStagesPlugin::new(&["caves", "pools"], SETTINGS, move || {
            Ok(Runtime::new(
                Arc::new(Pack::parse(&pack).expect("a valid pack")),
                4,
                SETTINGS.chunk,
            ))
        })
        .with_volume("caves")
        .with_fluid("pools"),
    );
    let origin = ChunkCoord::new(0, 0, 0);
    run_until(&mut app, |app| {
        app.world()
            .resource::<WaveForgeStages>()
            .fluid(origin)
            .is_some()
    });

    let stages = app.world().resource::<WaveForgeStages>();
    let expected = volume_mesh(
        origin,
        |at| stages.volume("pools", at),
        SETTINGS.cell_size.to_array(),
    )
    .expect("the fluid around it arrived");
    assert!(!expected.indices.is_empty());
    assert_eq!(stages.fluid(origin), Some(&expected));
    assert_ne!(stages.fluid(origin), stages.surface(origin));
    let ready: Vec<ChunkCoord> = app
        .world_mut()
        .resource_mut::<Messages<FluidReady>>()
        .drain()
        .map(|message| message.0)
        .collect();
    assert!(ready.contains(&origin), "{ready:?}");

    let mut focus = app
        .world_mut()
        .query_filtered::<&mut GlobalTransform, With<GenerationFocus>>();
    for mut at in focus.iter_mut(app.world_mut()) {
        *at = GlobalTransform::from_translation(Vec3::new(8000.0, 0.0, 8000.0));
    }
    run_until(&mut app, |app| {
        app.world()
            .resource::<WaveForgeStages>()
            .fluid(origin)
            .is_none()
    });
    let dropped: Vec<ChunkCoord> = app
        .world_mut()
        .resource_mut::<Messages<FluidDropped>>()
        .drain()
        .map(|message| message.0)
        .collect();
    assert!(dropped.contains(&origin), "{dropped:?}");
}

#[test]
fn a_dig_in_a_neighbour_builds_again_the_surface_that_reads_it() {
    let mut app = app_with(
        WaveForgeStagesPlugin::new(&["caves"], SETTINGS, || {
            Ok(Runtime::new(
                Arc::new(Pack::parse(VOLUME_PACK).expect("a valid pack")),
                4,
                SETTINGS.chunk,
            ))
        })
        .with_volume("caves"),
    );
    let (origin, beside) = (ChunkCoord::new(0, 0, 0), ChunkCoord::new(1, 0, 0));
    run_until(&mut app, |app| {
        let stages = app.world().resource::<WaveForgeStages>();
        stages.surface(origin).is_some() && stages.volume("caves", beside).is_some()
    });
    let before = app
        .world()
        .resource::<WaveForgeStages>()
        .volume("caves", beside)
        .cloned();
    app.world_mut()
        .resource_mut::<Seen>()
        .volumes_dropped
        .clear();

    // A ball in the chunk beside the origin that reaches only its first column, which the
    // origin's surface reads but the origin's volume does not hold.
    app.world().resource::<WaveForgeStages>().set_edits(Edits {
        log: vec![Edit::Dig {
            stage: "caves".to_owned(),
            at: [10.2, 4.5, 1.5],
            radius: 1.0,
        }],
    });
    run_until(&mut app, |app| {
        let stages = app.world().resource::<WaveForgeStages>();
        app.world()
            .resource::<Seen>()
            .volumes_dropped
            .contains(&origin)
            && stages.volume("caves", beside).is_some()
            && stages.volume("caves", beside).cloned() != before
            && stages.surface(origin).is_some()
    });

    let stages = app.world().resource::<WaveForgeStages>();
    let expected = volume_mesh(
        origin,
        |at| stages.volume("caves", at),
        SETTINGS.cell_size.to_array(),
    );
    assert_eq!(stages.surface(origin), expected.as_ref());
}

#[test]
fn a_raise_in_a_neighbour_builds_again_the_ground_that_reads_it() {
    let mut app = app_with(plugin().with_ground("height"));
    let (origin, beside) = (ChunkCoord::new(0, 0, 0), ChunkCoord::new(1, 0, 0));
    run_until(&mut app, |app| {
        let stages = app.world().resource::<WaveForgeStages>();
        stages.ground(origin).is_some() && stages.field("height", beside).is_some()
    });
    let before = app
        .world()
        .resource::<WaveForgeStages>()
        .field("height", beside)
        .expect("arrived")
        .get(0, 3);

    // The first column of the chunk beside the origin, which the origin's ground reaches to but
    // does not hold.
    app.world().resource::<WaveForgeStages>().set_edits(Edits {
        log: vec![Edit::Raise {
            stage: "height".to_owned(),
            column: (8, 3),
            by: 5.0,
        }],
    });
    run_until(&mut app, |app| {
        let stages = app.world().resource::<WaveForgeStages>();
        stages
            .field("height", beside)
            .is_some_and(|field| field.get(0, 3) == before + 5.0)
            && stages.ground(origin).is_some_and(|mesh| {
                mesh.positions[3 * mesh.size[0] as usize + 8][1]
                    == (before + 5.0) * SETTINGS.cell_size.y
            })
    });

    let stages = app.world().resource::<WaveForgeStages>();
    let expected = ground(
        origin,
        |at| stages.field("height", at),
        SETTINGS.cell_size.to_array(),
    );
    assert_eq!(stages.ground(origin), expected.as_ref());
}

/// A store in memory, by layer and chunk.
#[derive(Default)]
struct Memory(BTreeMap<(String, (i32, i32)), Vec<u8>>);

impl wave_forge::FrozenStore for Memory {
    fn keep(
        &mut self,
        layer: &str,
        chunk: ChunkCoord,
        bytes: Vec<u8>,
    ) -> Result<(), wave_forge::StoreError> {
        self.0.insert((layer.to_owned(), (chunk.x, chunk.y)), bytes);
        Ok(())
    }

    fn fetch(
        &mut self,
        layer: &str,
        chunk: ChunkCoord,
    ) -> Result<Option<Vec<u8>>, wave_forge::StoreError> {
        Ok(self.0.get(&(layer.to_owned(), (chunk.x, chunk.y))).cloned())
    }
}

#[test]
fn a_played_world_arrives_as_generated_with_no_stage_run() {
    let text = PACK.replace(
        "version: 1,",
        "version: 1,\n    bound: Some(Rect(min: (-24.0, -24.0), max: (23.0, 23.0))),",
    );
    let pack = Arc::new(Pack::parse(&text).expect("a valid pack"));
    let targets = ["height", "trees", "cover"];
    let mut store = Memory::default();
    Runtime::new(Arc::clone(&pack), 4, SETTINGS.chunk)
        .run_world(
            &targets,
            &mut store,
            |_| std::ops::ControlFlow::Continue(()),
        )
        .expect("a bounded pack");
    let mut direct = Runtime::new(Arc::clone(&pack), 4, SETTINGS.chunk);
    direct
        .request(
            &[FocusPoint::new(ChunkCoord::new(0, 0, 0), 1)],
            &["height", "trees"],
        )
        .expect("stages");
    direct.run_until_idle().expect("the stages run");
    let mut app = app_with(
        WaveForgeStagesPlugin::play(&targets, SETTINGS, pack, Box::new(store))
            .with_ground("height"),
    );

    run_until(&mut app, |app| {
        let stages = app.world().resource::<WaveForgeStages>();
        around_origin()
            .iter()
            .all(|&c| stages.ground(c).is_some() && stages.points("trees", c).is_some())
    });

    let stages = app.world().resource::<WaveForgeStages>();
    let cell = SETTINGS.cell_size.to_array();
    for chunk in around_origin() {
        let expected = ground(chunk, |at| direct.field("height", at), cell);
        assert_eq!(stages.ground(chunk), expected.as_ref(), "chunk {chunk:?}");
        assert_eq!(stages.points("trees", chunk), direct.points("trees", chunk));
    }
    assert!(
        stages.timings().is_empty(),
        "a stage ran: {:?}",
        stages.timings()
    );
}
