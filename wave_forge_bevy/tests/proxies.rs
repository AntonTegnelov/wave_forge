//! Far proxies in a Bevy app: a generated chunk's stand-in as one mesh per level, each drawn over a
//! range of distances that starts where the game's own drawing ends. On the CPU reference solver,
//! so no device is needed.

mod common;

use bevy_color::Color;
use bevy_mesh::{Mesh, VertexAttributeValues};
use common::{World, generated};
use wave_forge::ChunkCoord;
use wave_forge_bevy::levels::LevelDetail;

const DETAIL: LevelDetail = LevelDetail {
    pixels: 1.0,
    height: 720.0,
    fov: std::f32::consts::FRAC_PI_4,
};

fn colour(module: &str) -> Option<Color> {
    match module {
        "room" => Some(Color::srgb(0.8, 0.2, 0.1)),
        "yard" => Some(Color::srgb(0.2, 0.6, 0.2)),
        _ => None,
    }
}

fn colours(mesh: &Mesh) -> usize {
    match mesh.attribute(Mesh::ATTRIBUTE_COLOR) {
        Some(VertexAttributeValues::Float32x4(values)) => values.len(),
        _ => 0,
    }
}

#[test]
fn the_levels_are_drawn_from_the_start_on_one_after_another() {
    let (app, tiles) = generated();
    let world = app.world().resource::<World>();

    let levels = world
        .proxy_levels(ChunkCoord::new(0, 0, 0), &tiles, colour, DETAIL, 50.0)
        .expect("generated");

    assert!(levels.len() > 1, "{} levels", levels.len());
    assert_eq!(levels[0].range.start_margin.start, 50.0);
    for pair in levels.windows(2) {
        assert!(pair[0].cells < pair[1].cells);
        assert_eq!(
            pair[0].range.end_margin.start,
            pair[1].range.start_margin.start
        );
    }
    assert_eq!(
        levels.last().expect("levels").range.end_margin.end,
        f32::INFINITY
    );
    for level in &levels {
        assert!(
            colours(&level.mesh) > 0,
            "{} cells: no colours",
            level.cells
        );
    }
}

#[test]
fn a_chunk_of_uncoloured_modules_has_nothing_to_stand_in_for() {
    let (app, tiles) = generated();
    let world = app.world().resource::<World>();

    let levels = world
        .proxy_levels(ChunkCoord::new(0, 0, 0), &tiles, |_| None, DETAIL, 50.0)
        .expect("generated");

    assert!(levels.is_empty());
}
