//! Region tags: the rooms and sounds a solved chunk holds, and what walkers stand on, from the
//! module set's `indoor`, `sounds` and `surface`.

use wave_forge::loader::{RuleFile, parse_rule_file};
use wave_forge::{Chunk, ChunkCoord, ChunkShape, YUpSpace, region_tags, surface_at};

const RULES: &str = r#"(
    faces: {
        "air": Side(connector: "air"),
        "open": Top(connector: "open"),
    },
    modules: [
        (name: "air", sides: ["air", "air", "air", "air"], up: "open", down: "open"),
        (name: "room", sides: ["air", "air", "air", "air"], up: "open", down: "open",
         surface: "wood", indoor: true, sounds: [(at: (0.25, 0.5, 0.1), key: "clock")]),
    ],
)"#;

const SHAPE: ChunkShape = ChunkShape { x: 4, y: 3, z: 2 };
const CELL: [f32; 3] = [2.0, 3.0, 1.0];

fn rules() -> RuleFile {
    parse_rule_file(RULES).expect("a module set")
}

/// A chunk whose cell (x, y, z) holds a room where `room` says so, air elsewhere.
fn chunk(rules: &RuleFile, coord: ChunkCoord, room: impl Fn(u32, u32, u32) -> bool) -> Chunk {
    let (air, room_tile) = (rules.tiles_named("air")[0], rules.tiles_named("room")[0]);
    let tiles: Vec<u16> = (0..SHAPE.cells())
        .map(|cell| {
            let (x, y, z) = (
                cell % SHAPE.x,
                (cell / SHAPE.x) % SHAPE.y,
                cell / (SHAPE.x * SHAPE.y),
            );
            let tile = if room(x, y, z) { room_tile } else { air };
            u16::try_from(tile).expect("few tiles")
        })
        .collect();
    Chunk {
        coord,
        tiles: tiles.into_boxed_slice(),
        version: 0,
    }
}

fn space() -> YUpSpace {
    YUpSpace::new(SHAPE, CELL)
}

fn inside(point: [f32; 3], min: [f32; 3], max: [f32; 3]) -> bool {
    (0..3).all(|axis| point[axis] > min[axis] && point[axis] < max[axis])
}

/// How many interiors of the chunk holding rooms where `pattern` says hold each cell, against how
/// many should: one for a room, none for air.
fn coverage(pattern: impl Fn(u32, u32, u32) -> bool) -> Vec<(u32, usize, usize)> {
    let rules = rules();
    let coord = ChunkCoord::new(-1, 2, 0);
    let chunk = chunk(&rules, coord, &pattern);
    let tags = region_tags(&chunk, &rules, &space());
    (0..SHAPE.cells())
        .map(|cell| {
            let (x, y, z) = (
                cell % SHAPE.x,
                (cell / SHAPE.x) % SHAPE.y,
                cell / (SHAPE.x * SHAPE.y),
            );
            let centre = space().cell_center(coord, cell);
            let boxes = tags
                .interiors
                .iter()
                .filter(|interior| inside(centre, interior.min, interior.max))
                .count();
            (cell, boxes, usize::from(pattern(x, y, z)))
        })
        .collect()
}

#[test]
fn interiors_cover_every_indoor_cell_once_and_nothing_else() {
    let scattered = |x: u32, y: u32, z: u32| !(x * 7 + y * 3 + z * 5).is_multiple_of(3);
    // A full row with a narrower one beside it and above it: a box must not grow over air.
    let l_shaped = |x: u32, y: u32, z: u32| (y == 0 && z == 0) || x == 0;

    for (name, pattern) in [
        ("scattered", &scattered as &dyn Fn(u32, u32, u32) -> bool),
        ("L-shaped", &l_shaped),
    ] {
        for (cell, boxes, expected) in coverage(pattern) {
            assert_eq!(boxes, expected, "{name}, cell {cell}");
        }
    }
}

#[test]
fn a_chunk_full_of_rooms_is_one_interior_the_chunks_size() {
    let rules = rules();
    let coord = ChunkCoord::new(1, 0, 0);
    let chunk = chunk(&rules, coord, |_, _, _| true);

    let tags = region_tags(&chunk, &rules, &space());

    let origin = space().chunk_origin(coord);
    let size = space().chunk_size();
    assert_eq!(tags.interiors.len(), 1);
    assert_eq!(tags.interiors[0].min, origin);
    assert_eq!(
        tags.interiors[0].max,
        [
            origin[0] + size[0],
            origin[1] + size[1],
            origin[2] + size[2]
        ]
    );
}

#[test]
fn a_sound_plays_at_its_point_of_its_cell_in_the_engines_axes() {
    let rules = rules();
    let coord = ChunkCoord::new(0, 0, 0);
    let chunk = chunk(&rules, coord, |x, y, z| (x, y, z) == (1, 2, 1));

    let tags = region_tags(&chunk, &rules, &space());

    // Lattice (1 + 0.25, 2 + 0.5, 1 + 0.1) cells: the engine's x, then z up as y, then y as z.
    assert_eq!(tags.emitters.len(), 1);
    assert_eq!(tags.emitters[0].key, "clock");
    let at = tags.emitters[0].at;
    let expected = [1.25 * CELL[0], 1.1 * CELL[1], 2.5 * CELL[2]];
    for axis in 0..3 {
        assert!(
            (at[axis] - expected[axis]).abs() < 1e-5,
            "{at:?} against {expected:?}"
        );
    }
}

#[test]
fn a_point_is_on_the_surface_of_the_module_in_its_cell() {
    let rules = rules();
    let coord = ChunkCoord::new(0, 0, 0);
    let chunk = chunk(&rules, coord, |x, _, _| x == 0);
    let lookup = |at: ChunkCoord| (at == coord).then_some(&chunk);
    let space = space();

    let room = space.cell_center(coord, 0);
    let air = space.cell_center(coord, 1);
    let elsewhere = space.cell_center(ChunkCoord::new(5, 0, 0), 0);

    assert_eq!(surface_at(room, lookup, &rules, &space), Some("wood"));
    assert_eq!(surface_at(air, lookup, &rules, &space), None);
    assert_eq!(surface_at(elsewhere, lookup, &rules, &space), None);
}
