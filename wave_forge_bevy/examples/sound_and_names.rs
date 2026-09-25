//! Sound and place names in a Bevy game, through `bevy_kira_audio` and Fluent.
//!
//! The plugin gives a chunk's rooms and sounds (`WaveForgeWorld::region_tags`) and a location's
//! name as a translation key with arguments (`Site::name`), and depends on no audio or localisation
//! crate. This is the mapping a game writes on top:
//!
//! - Each emitter becomes an entity with a `SpatialAudioEmitter`, playing a looped sound for its
//!   key, and goes when its chunk is evicted or generated again.
//! - A sound on the other side of a wall from the listener is muffled: it is heard over a quarter
//!   of its radius. `bevy_kira_audio` has neither effects nor buses to reverb or route a room's
//!   sounds through (a channel's volume is written onto each sound, which spatial audio overwrites
//!   every frame), so the room reaches the mix through each emitter's `SpatialRadius`.
//! - A location's name is formatted by a Fluent bundle from its key and arguments.
//!
//! ```text
//! cargo run -p wave_forge_bevy --release --example sound_and_names
//! ```
//!
//! It needs a compute device for the city, and plays through the default audio output when there
//! is one; without one, `bevy_kira_audio` warns and the example still runs. It prints what it
//! routed and named, then quits.

use bevy::app::{AppExit, ScheduleRunnerPlugin};
use bevy::prelude::*;
use bevy_kira_audio::SpatialRadius;
use bevy_kira_audio::prelude::*;
use fluent::concurrent::FluentBundle;
use fluent::{FluentArgs, FluentResource, FluentValue};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use wave_forge::cell_boxes::CellBox;
use wave_forge::loader::parse_rule_file;
use wave_forge::stages::{Pack, PlaceName, Runtime};
use wave_forge::{BlockSolver, ChunkCoord, ChunkShape, WgpuBackend, WorldExtent};
use wave_forge_bevy::stages::{StageReady, StagesSettings, WaveForgeStages, WaveForgeStagesPlugin};
use wave_forge_bevy::{
    ChunkEvicted, ChunkUpdated, GenerationFocus, WaveForgePlugin, WaveForgeSettings,
    WaveForgeSystems, WaveForgeTiles, WaveForgeWorld,
};
use wfc_devtools::city::{self, city_prior};

const CHUNK: ChunkShape = ChunkShape::cube(8);

/// How far an emitter is heard, in Bevy's units, when nothing stands between it and the listener.
const HEARD_OVER: f32 = 30.0;

/// A pack with a location table, whose sites are named by translation keys.
const PLACES: &str = r#"(
    version: 1,
    stages: [
        (name: "ground", kind: Field(Constant(4.0))),
        (name: "places", kind: Locations(height: "ground", region: 4, kinds: [
            (name: "stone_circle", quota: 2, tries: 20),
        ])),
    ],
)"#;

/// The game's translations of the pack's place names.
const ENGLISH: &str =
    "wf-place-stone-circle = Stone Circle { $index } of { $region_x }, { $region_y }\n";

type CityWorld = WaveForgeWorld<BlockSolver<WgpuBackend>>;

/// Whether `point` lies inside any of `interiors`.
fn indoors(point: Vec3, interiors: &[CellBox]) -> bool {
    interiors.iter().any(|interior| {
        (0..3).all(|axis| point[axis] >= interior.min[axis] && point[axis] <= interior.max[axis])
    })
}

/// How far a sound is heard: over its whole `radius` when the listener is on the same side of the
/// walls as it, indoors or out, and over a quarter of it through a wall.
fn heard_over(listener_indoors: bool, emitter_indoors: bool, radius: f32) -> f32 {
    if listener_indoors == emitter_indoors {
        radius
    } else {
        radius / 4.0
    }
}

/// A place's name in `bundle`'s language, or `None` if the bundle has no message for its key.
fn place_name(bundle: &FluentBundle<FluentResource>, name: &PlaceName) -> Option<String> {
    let pattern = bundle.get_message(&name.key)?.value()?;
    let mut args = FluentArgs::new();
    for &(arg, value) in &name.args {
        args.set(arg, FluentValue::from(value));
    }
    let mut errors = Vec::new();
    let text = bundle.format_pattern(pattern, Some(&args), &mut errors);
    errors.is_empty().then(|| text.into_owned())
}

/// A bundle of `source`, a Fluent resource in English, without the isolation marks that only a
/// bidirectional text renderer needs.
fn bundle(source: &str) -> FluentBundle<FluentResource> {
    let resource = FluentResource::try_new(source.to_owned()).expect("the translations parse");
    let mut bundle = FluentBundle::new_concurrent(vec!["en-GB".parse().expect("a language")]);
    bundle.set_use_isolating(false);
    bundle
        .add_resource(resource)
        .expect("no message is defined twice");
    bundle
}

#[derive(Resource)]
struct Names(FluentBundle<FluentResource>);

/// A looped sound for each emitter key the module set uses.
#[derive(Resource)]
struct Sounds(HashMap<String, Handle<AudioSource>>);

/// The rooms of each chunk that has sound, to tell whether the listener is indoors.
#[derive(Resource, Default)]
struct Rooms(HashMap<ChunkCoord, Vec<CellBox>>);

/// An emitter entity: its chunk, and whether it stands indoors.
#[derive(Component)]
struct ChunkSound {
    chunk: ChunkCoord,
    indoors: bool,
}

/// What the example routed and named, printed as it quits.
#[derive(Resource, Default)]
struct Report {
    indoors: usize,
    outdoors: usize,
    names: Vec<String>,
}

/// A second of a quiet tone at `hz`, so the example needs no sound files.
fn tone(hz: f32) -> AudioSource {
    let rate = 44_100_u32;
    let frames: Vec<Frame> = (0..rate)
        .map(|index| {
            let turns = index as f32 / rate as f32 * hz;
            Frame::from_mono((turns * std::f32::consts::TAU).sin() * 0.2)
        })
        .collect();
    AudioSource {
        sound: StaticSoundData {
            sample_rate: rate,
            frames: frames.into(),
            settings: StaticSoundSettings::default(),
            slice: None,
        },
    }
}

fn load_sounds(mut commands: Commands, mut sources: ResMut<Assets<AudioSource>>) {
    let fountain = sources.add(tone(440.0));
    commands.insert_resource(Sounds(HashMap::from([("fountain".to_owned(), fountain)])));
}

/// Gives a generated chunk's emitters their sounds, replacing any it had.
#[allow(clippy::too_many_arguments)]
fn play_chunk_sounds(
    mut commands: Commands,
    mut updated: MessageReader<ChunkUpdated>,
    world: Res<CityWorld>,
    tiles: Res<WaveForgeTiles>,
    sounds: Res<Sounds>,
    audio: Res<Audio>,
    mut rooms: ResMut<Rooms>,
    mut report: ResMut<Report>,
    playing: Query<(Entity, &ChunkSound, &SpatialAudioEmitter)>,
    mut instances: ResMut<Assets<AudioInstance>>,
) {
    for &ChunkUpdated(chunk) in updated.read() {
        stop(&mut commands, &playing, &mut instances, chunk);
        let tags = world
            .region_tags(chunk, &tiles)
            .expect("an updated chunk is generated");
        for emitter in &tags.emitters {
            let Some(sound) = sounds.0.get(&emitter.key) else {
                continue;
            };
            let at = Vec3::from_array(emitter.at);
            let inside = indoors(at, &tags.interiors);
            if inside {
                report.indoors += 1;
            } else {
                report.outdoors += 1;
            }
            let entity = commands
                .spawn((
                    Transform::from_translation(at),
                    SpatialRadius { radius: HEARD_OVER },
                    ChunkSound {
                        chunk,
                        indoors: inside,
                    },
                ))
                .id();
            let instance = audio.play(sound.clone()).looped().handle();
            commands.entity(entity).insert(SpatialAudioEmitter {
                instances: vec![instance],
            });
        }
        rooms.0.insert(chunk, tags.interiors);
    }
}

/// Stops and removes an evicted chunk's sounds.
fn stop_evicted_sounds(
    mut commands: Commands,
    mut evicted: MessageReader<ChunkEvicted>,
    mut rooms: ResMut<Rooms>,
    playing: Query<(Entity, &ChunkSound, &SpatialAudioEmitter)>,
    mut instances: ResMut<Assets<AudioInstance>>,
) {
    for &ChunkEvicted(chunk) in evicted.read() {
        stop(&mut commands, &playing, &mut instances, chunk);
        rooms.0.remove(&chunk);
    }
}

fn stop(
    commands: &mut Commands,
    playing: &Query<(Entity, &ChunkSound, &SpatialAudioEmitter)>,
    instances: &mut Assets<AudioInstance>,
    chunk: ChunkCoord,
) {
    for (entity, sound, emitter) in playing {
        if sound.chunk != chunk {
            continue;
        }
        for instance in &emitter.instances {
            if let Some(mut instance) = instances.get_mut(instance) {
                instance.stop(AudioTween::default());
            }
        }
        commands.entity(entity).despawn();
    }
}

/// Muffles the sounds on the other side of a wall from the listener.
fn muffle_through_walls(
    world: Res<CityWorld>,
    rooms: Res<Rooms>,
    listener: Query<&GlobalTransform, With<SpatialAudioReceiver>>,
    mut emitters: Query<(&ChunkSound, &mut SpatialRadius)>,
) {
    let Ok(listener) = listener.single() else {
        return;
    };
    let at = listener.translation();
    let chunk = world.settings().chunk_at(at);
    let inside = rooms
        .0
        .get(&chunk)
        .is_some_and(|interiors| indoors(at, interiors));
    for (sound, mut radius) in &mut emitters {
        radius.radius = heard_over(inside, sound.indoors, HEARD_OVER);
    }
}

/// Names the locations of each chunk of the places stage as it arrives.
fn name_places(
    mut ready: MessageReader<StageReady>,
    stages: Res<WaveForgeStages>,
    names: Res<Names>,
    mut report: ResMut<Report>,
) {
    for message in ready.read() {
        let Some(sites) = stages.sites(&message.stage, message.chunk) else {
            continue;
        };
        for name in sites.iter().filter_map(|site| site.name()) {
            let text = place_name(&names.0, &name).unwrap_or(name.key);
            report.names.push(text);
        }
    }
}

/// Quits once the city around the listener is generated and some places are named.
fn finish(world: Res<CityWorld>, report: Res<Report>, mut exit: MessageWriter<AppExit>) {
    if !world.is_idle() || report.names.is_empty() {
        return;
    }
    println!(
        "{} sounds indoors and {} outdoors, heard over {HEARD_OVER} units, a quarter of it through a wall",
        report.indoors, report.outdoors
    );
    for name in &report.names {
        println!("a place: {name}");
    }
    exit.write(AppExit::Success);
}

fn main() {
    let city = city::city();
    let rules = parse_rule_file(city::CITY_RON).expect("the city's rule file loads");
    let settings = WaveForgeSettings {
        seed: 11,
        extent: WorldExtent::new(CHUNK)
            .with_x(0..4)
            .with_y(0..4)
            .with_z(0..1),
        halo: 1,
        cell_size: Vec3::splat(2.0),
        ..WaveForgeSettings::default()
    };
    let places = StagesSettings {
        chunk: [8, 8],
        cell_size: Vec3::new(2.0, 1.0, 2.0),
    };

    App::new()
        .add_plugins(DefaultPlugins)
        // This crate's Bevy has no window, whose event loop would otherwise run the frames.
        .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
            1.0 / 60.0,
        )))
        .add_plugins((AudioPlugin, SpatialAudioPlugin))
        .add_plugins(
            WaveForgePlugin::from_rules(rules, city_prior(&city, CHUNK.z), settings)
                .expect("the city's weights make a rule set"),
        )
        .add_plugins(WaveForgeStagesPlugin::new(&["places"], places, || {
            let pack = Pack::parse(PLACES).map_err(|error| error.to_string())?;
            Ok(Runtime::new(Arc::new(pack), 5, [8, 8]))
        }))
        .insert_resource(Names(bundle(ENGLISH)))
        .init_resource::<Rooms>()
        .init_resource::<Report>()
        .add_systems(
            Startup,
            (load_sounds, |mut commands: Commands| {
                commands.spawn((
                    Transform::from_translation(Vec3::new(32.0, 1.0, 32.0)),
                    GenerationFocus::new(2),
                    SpatialAudioReceiver,
                ));
            }),
        )
        .add_systems(
            Update,
            (
                (play_chunk_sounds, stop_evicted_sounds, muffle_through_walls)
                    .chain()
                    .after(WaveForgeSystems),
                name_places,
                finish,
            ),
        )
        .run();
}

#[cfg(test)]
mod tests {
    use super::*;
    use wave_forge::FocusPoint;

    const ROOM: CellBox = CellBox {
        min: [0.0, 0.0, 0.0],
        max: [4.0, 2.0, 4.0],
    };

    #[test]
    fn a_point_in_a_room_is_indoors_and_one_beside_it_is_not() {
        let rooms = [ROOM];

        let inside = indoors(Vec3::new(2.0, 1.0, 2.0), &rooms);
        let outside = indoors(Vec3::new(6.0, 1.0, 2.0), &rooms);

        assert!(inside);
        assert!(!outside);
    }

    #[test]
    fn a_sound_is_heard_over_its_radius_on_its_own_side_of_the_walls() {
        let radius = 30.0;

        let both_in = heard_over(true, true, radius);
        let both_out = heard_over(false, false, radius);

        assert_eq!(both_in, radius);
        assert_eq!(both_out, radius);
    }

    #[test]
    fn a_sound_through_a_wall_is_heard_over_a_quarter_of_its_radius() {
        let radius = 30.0;

        let from_outside = heard_over(false, true, radius);
        let from_inside = heard_over(true, false, radius);

        assert_eq!(from_outside, radius / 4.0);
        assert_eq!(from_inside, radius / 4.0);
    }

    #[test]
    fn a_place_name_is_its_translation_with_its_arguments() {
        let bundle = bundle(ENGLISH);
        let name = PlaceName {
            key: "wf-place-stone-circle".to_owned(),
            args: vec![("region_x", 3), ("region_y", -1), ("index", 0)],
        };

        let text = place_name(&bundle, &name);

        assert_eq!(text.as_deref(), Some("Stone Circle 0 of 3, -1"));
    }

    #[test]
    fn a_place_name_without_a_translation_is_none() {
        let bundle = bundle(ENGLISH);
        let name = PlaceName {
            key: "wf-place-ruined-tower".to_owned(),
            args: vec![("region_x", 0), ("region_y", 0), ("index", 0)],
        };

        let text = place_name(&bundle, &name);

        assert_eq!(text, None);
    }

    #[test]
    fn every_place_the_pack_names_has_a_translation() {
        let pack = Arc::new(Pack::parse(PLACES).expect("a valid pack"));
        let mut runtime = Runtime::new(pack, 5, [8, 8]);
        let bundle = bundle(ENGLISH);
        let area: Vec<ChunkCoord> = (-3..=3)
            .flat_map(|y| (-3..=3).map(move |x| ChunkCoord::new(x, y, 0)))
            .collect();
        let focus: Vec<FocusPoint> = area
            .iter()
            .map(|&chunk| FocusPoint::new(chunk, 0))
            .collect();

        runtime
            .request(&focus, &["places"])
            .expect("the pack has places");
        runtime.run_until_idle().expect("the stages run");
        let named: Vec<PlaceName> = area
            .iter()
            .flat_map(|&chunk| runtime.sites("places", chunk).expect("generated"))
            .filter_map(|site| site.name())
            .collect();

        assert!(!named.is_empty(), "the area holds no named place");
        for name in named {
            assert!(
                place_name(&bundle, &name).is_some(),
                "{} has no translation",
                name.key
            );
        }
    }
}
