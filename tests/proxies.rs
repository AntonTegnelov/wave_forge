//! Far proxies: a chunk's modules as coloured boxes, with coarser levels of bigger boxes.

use wave_forge::loader::{RuleFile, parse_rule_file};
use wave_forge::{Chunk, ChunkCoord, ChunkShape, ProxyMesh, YUpSpace, proxy_mesh};

const RULES: &str = r#"(
    faces: {
        "air": Side(connector: "air"),
        "open": Top(connector: "open"),
    },
    modules: [
        (name: "air", sides: ["air", "air", "air", "air"], up: "open", down: "open"),
        (name: "red", sides: ["air", "air", "air", "air"], up: "open", down: "open"),
        (name: "blue", sides: ["air", "air", "air", "air"], up: "open", down: "open"),
    ],
)"#;

const SHAPE: ChunkShape = ChunkShape { x: 4, y: 4, z: 2 };
const CELL: [f32; 3] = [2.0, 3.0, 1.0];
const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

fn colour(module: &str) -> Option<[f32; 4]> {
    match module {
        "red" => Some(RED),
        "blue" => Some(BLUE),
        _ => None,
    }
}

/// A chunk whose cell (x, y, z) holds the module `pick` names.
fn chunk(rules: &RuleFile, pick: impl Fn(u32, u32, u32) -> &'static str) -> Chunk {
    let tiles: Vec<u16> = (0..SHAPE.cells())
        .map(|cell| {
            let (x, y, z) = (
                cell % SHAPE.x,
                (cell / SHAPE.x) % SHAPE.y,
                cell / (SHAPE.x * SHAPE.y),
            );
            u16::try_from(rules.tiles_named(pick(x, y, z))[0]).expect("few tiles")
        })
        .collect();
    Chunk {
        coord: ChunkCoord::new(0, 0, 0),
        tiles: tiles.into_boxed_slice(),
        version: 0,
    }
}

fn proxy(pick: impl Fn(u32, u32, u32) -> &'static str) -> ProxyMesh {
    let rules = parse_rule_file(RULES).expect("a module set");
    proxy_mesh(
        &chunk(&rules, pick),
        &rules,
        &YUpSpace::new(SHAPE, CELL),
        colour,
    )
}

fn faces(mesh: &ProxyMesh, level: usize) -> usize {
    mesh.levels[level].indices.len() / 6
}

#[test]
fn a_lone_cell_is_a_closed_box_the_size_of_its_cell() {
    let mesh = proxy(|x, y, z| if (x, y, z) == (1, 2, 1) { "red" } else { "air" });

    assert_eq!(faces(&mesh, 0), 6);
    let indices = &mesh.levels[0].indices;
    let (low, high) = indices
        .iter()
        .fold(([f32::MAX; 3], [f32::MIN; 3]), |(low, high), &i| {
            let p = mesh.positions[i as usize];
            (
                [0, 1, 2].map(|a| low[a].min(p[a])),
                [0, 1, 2].map(|a| high[a].max(p[a])),
            )
        });
    // Lattice (1, 2, 1): the engine's x is its x, the engine's y its z, the engine's z its y.
    assert_eq!(low, [2.0, 3.0, 2.0]);
    assert_eq!(high, [4.0, 6.0, 3.0]);
}

#[test]
fn a_face_between_two_filled_cells_is_left_out() {
    let mesh = proxy(|x, y, z| {
        if y == 0 && z == 0 && x < 2 {
            "red"
        } else {
            "air"
        }
    });

    assert_eq!(faces(&mesh, 0), 10);
}

#[test]
fn every_face_of_every_level_faces_out_of_its_box() {
    let mesh = proxy(|x, y, z| match (x * 3 + y * 5 + z * 7) % 4 {
        0 | 1 => "red",
        2 => "blue",
        _ => "air",
    });

    for level in &mesh.levels {
        for triangle in level.indices.chunks(3) {
            let [a, b, c] = [0, 1, 2].map(|k| mesh.positions[triangle[k] as usize]);
            let (u, v) = (
                [0, 1, 2].map(|k| b[k] - a[k]),
                [0, 1, 2].map(|k| c[k] - a[k]),
            );
            let facing = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let normal = mesh.normals[triangle[0] as usize];
            let along: f32 = (0..3).map(|k| facing[k] * normal[k]).sum();
            assert!(along > 0.0, "{} cells: {triangle:?} faces in", level.cells);
        }
    }
}

#[test]
fn a_coarser_box_is_filled_where_half_its_cells_are_in_their_average_colour() {
    // In the first group of two by two by two, four of eight cells: two red, two blue. In the
    // group beside it along x, three of eight: under half.
    let mesh = proxy(|x, y, z| match (x, y, z) {
        (0, 0, 0) | (1, 0, 0) => "red",
        (0, 1, 0) | (1, 1, 0) => "blue",
        (2, 0, 0) | (3, 0, 0) | (2, 1, 0) => "red",
        _ => "air",
    });

    let level = &mesh.levels[1];
    assert_eq!(level.cells, 2);
    assert_eq!(level.indices.len() / 6, 6, "one box");
    let colour = mesh.colours[level.indices[0] as usize];
    assert_eq!(colour, [0.5, 0.0, 0.5, 1.0]);
    let high_x = level
        .indices
        .iter()
        .map(|&i| mesh.positions[i as usize][0])
        .fold(f32::MIN, f32::max);
    assert_eq!(high_x, 2.0 * CELL[0]);
}

#[test]
fn levels_halve_until_one_box_stands_for_the_chunk_and_stray_further_each() {
    let mesh = proxy(|_, _, _| "red");

    let cells: Vec<u32> = mesh.levels.iter().map(|level| level.cells).collect();
    assert_eq!(cells, [1, 2, 4]);
    assert_eq!(mesh.levels[0].error, 0.0);
    assert!(
        mesh.levels
            .windows(2)
            .all(|pair| pair[0].error < pair[1].error)
    );
    // The outer surface of 4 by 4 by 2 cells, then of 2 by 2 by 1 boxes, then of one box.
    let faces: Vec<usize> = mesh
        .levels
        .iter()
        .map(|level| level.indices.len() / 6)
        .collect();
    assert_eq!(faces, [2 * 16 + 2 * 8 + 2 * 8, 2 * 4 + 2 * 2 + 2 * 2, 6]);
}

#[test]
fn a_module_without_a_colour_is_left_out() {
    let mesh = proxy(|_, _, _| "air");

    assert!(mesh.levels.iter().all(|level| level.indices.is_empty()));
}
