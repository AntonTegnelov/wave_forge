//! Occluders: boxes that cover a solved chunk's solid cells, from the module set's `solid`.

use wave_forge::loader::{RuleFile, parse_rule_file};
use wave_forge::{Chunk, ChunkCoord, ChunkShape, YUpSpace, occluders};

const RULES: &str = r#"(
    faces: {
        "air": Side(connector: "air"),
        "open": Top(connector: "open"),
    },
    modules: [
        (name: "air", sides: ["air", "air", "air", "air"], up: "open", down: "open"),
        (name: "block", sides: ["air", "air", "air", "air"], up: "open", down: "open", solid: true),
        (name: "hut", sides: ["air", "air", "air", "air"], up: "open", down: "open", indoor: true),
    ],
)"#;

const SHAPE: ChunkShape = ChunkShape { x: 4, y: 3, z: 2 };
const CELL: [f32; 3] = [2.0, 3.0, 1.0];

/// A chunk whose cell (x, y, z) holds the module `pick` names.
fn chunk(
    rules: &RuleFile,
    coord: ChunkCoord,
    pick: impl Fn(u32, u32, u32) -> &'static str,
) -> Chunk {
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
        coord,
        tiles: tiles.into_boxed_slice(),
        version: 0,
    }
}

#[test]
fn occluders_cover_every_solid_cell_once_and_nothing_else() {
    let rules = parse_rule_file(RULES).expect("a module set");
    let space = YUpSpace::new(SHAPE, CELL);
    let coord = ChunkCoord::new(2, -1, 0);
    let pick = |x: u32, y: u32, z: u32| match (x + 2 * y + 3 * z) % 4 {
        0 | 1 => "block",
        2 => "hut",
        _ => "air",
    };
    let chunk = chunk(&rules, coord, pick);

    let boxes = occluders(&chunk, &rules, &space);

    for cell in 0..SHAPE.cells() {
        let (x, y, z) = (
            cell % SHAPE.x,
            (cell / SHAPE.x) % SHAPE.y,
            cell / (SHAPE.x * SHAPE.y),
        );
        let centre = space.cell_center(coord, cell);
        let holding = boxes
            .iter()
            .filter(|b| (0..3).all(|axis| centre[axis] > b.min[axis] && centre[axis] < b.max[axis]))
            .count();
        assert_eq!(
            holding,
            usize::from(pick(x, y, z) == "block"),
            "cell ({x}, {y}, {z})"
        );
    }
}

#[test]
fn a_chunk_without_solid_cells_has_no_occluders() {
    let rules = parse_rule_file(RULES).expect("a module set");
    let chunk = chunk(&rules, ChunkCoord::new(0, 0, 0), |_, _, _| "hut");

    assert!(occluders(&chunk, &rules, &YUpSpace::new(SHAPE, CELL)).is_empty());
}
