//! A Bevy app that has generated one chunk of rooms and yards on the CPU reference solver, for the
//! tests of what the plugin makes of a generated chunk.

use bevy_app::{App, Startup};
use bevy_ecs::prelude::Commands;
use bevy_math::Vec3;
use bevy_transform::components::GlobalTransform;
use std::sync::Arc;
use wave_forge::loader::parse_rule_file;
use wave_forge::{ChunkCoord, ChunkShape, Prior, Ruleset, WorldExtent};
use wave_forge_bevy::{
    GenerationFocus, WaveForgeSettings, WaveForgeSolverPlugin, WaveForgeTiles, WaveForgeWorld,
};
use wfc_core::reference::ReferenceSolver;

/// Rooms and yards that may sit anywhere beside each other.
const RULES: &str = r#"(
    faces: {
        "any": Side(connector: "any"),
        "open": Top(connector: "open"),
    },
    modules: [
        (name: "yard", sides: ["any", "any", "any", "any"], up: "open", down: "open",
         surface: "grass"),
        (name: "room", sides: ["any", "any", "any", "any"], up: "open", down: "open",
         surface: "wood", indoor: true, sounds: [(at: (0.5, 0.5, 0.5), key: "clock")]),
    ],
)"#;

pub const CHUNK: ChunkShape = ChunkShape::cube(4);
/// One cell is two units across and one unit tall, so the two axis orders cannot be confused.
pub const CELL: Vec3 = Vec3::new(2.0, 1.0, 2.0);

pub type World = WaveForgeWorld<ReferenceSolver>;

/// An app that has generated the chunk at the origin, and what its tiles are.
pub fn generated() -> (App, WaveForgeTiles) {
    let file = parse_rule_file(RULES).expect("a module set");
    let ruleset = Ruleset::new(file.rules(), &file.tileset().weights).expect("a rule set");
    let settings = WaveForgeSettings {
        seed: 3,
        extent: WorldExtent::new(CHUNK)
            .with_x(0..1)
            .with_y(0..1)
            .with_z(0..1),
        cell_size: CELL,
        ..WaveForgeSettings::default()
    };
    let tiles = u32::try_from(file.num_tiles()).expect("few tiles");
    let mut app = App::new();
    app.add_plugins(WaveForgeSolverPlugin::new(
        ruleset.clone(),
        Prior::open(tiles),
        settings,
        ReferenceSolver::new(Arc::new(ruleset)),
    ))
    .add_systems(Startup, |mut commands: Commands| {
        commands.spawn((GlobalTransform::default(), GenerationFocus::new(0)));
    });
    app.finish();
    app.cleanup();
    for _ in 0..200 {
        app.update();
        if app
            .world()
            .resource::<World>()
            .chunk(ChunkCoord::new(0, 0, 0))
            .is_some()
        {
            return (app, WaveForgeTiles(file));
        }
    }
    panic!("the chunk at the origin was not generated");
}
