//! A world of stages's navigation source (docs/product/user-stories.md, N1 and M1): each chunk's
//! walkable triangles and its neighbours' out to a border, so chunks baked apart meet.

use std::sync::Arc;
use wave_forge::stages::{Pack, Runtime};
use wave_forge::surface_nav_source;
use wave_forge::{ChunkCoord, FocusPoint, GroundMesh, NavSource, NavSourceError, ground};

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "height", kind: Field(Mul(Noise(frequency: 0.08, octaves: 2), Constant(6.0)))),
    ],
)"#;

const COLUMNS: u32 = 8;
const CELL: [f32; 3] = [2.0, 1.0, 2.0];
const CHUNK: f32 = COLUMNS as f32 * 2.0;

/// The ground of the chunks from -1 to 1 each way.
fn grounds() -> Vec<GroundMesh> {
    let mut runtime = Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        5,
        [COLUMNS, COLUMNS],
    );
    let focus: Vec<FocusPoint> = (-2..=2)
        .flat_map(|y| (-2..=2).map(move |x| FocusPoint::new(ChunkCoord::new(x, y, 0), 0)))
        .collect();
    runtime.request(&focus, &["height"]).expect("the stages");
    runtime.run_until_idle().expect("the stages run");
    (-1..=1)
        .flat_map(|y| (-1..=1).map(move |x| ChunkCoord::new(x, y, 0)))
        .map(|chunk| ground(chunk, |at| runtime.field("height", at), CELL).expect("fields around"))
        .collect()
}

fn corner(chunk: ChunkCoord) -> [f32; 3] {
    [chunk.x as f32 * CHUNK, 0.0, chunk.y as f32 * CHUNK]
}

fn source(
    grounds: &[GroundMesh],
    in_world: impl Fn(ChunkCoord) -> bool,
    border: f32,
) -> Result<NavSource, NavSourceError> {
    surface_nav_source(
        ChunkCoord::new(0, 0, 0),
        [CHUNK, CHUNK],
        in_world,
        |chunk| {
            grounds
                .iter()
                .find(|ground| ground.chunk == chunk)
                .map(|ground| ground.surface_triangles(corner(chunk)))
        },
        border,
        0.25,
    )
}

/// Whether `(x, z)` lies inside the triangle's shadow on the ground plane.
fn covers(triangle: &[f32], x: f32, z: f32) -> bool {
    let [ax, az, bx, bz, cx, cz] = [
        triangle[0],
        triangle[2],
        triangle[3],
        triangle[5],
        triangle[6],
        triangle[8],
    ];
    let side = |px: f32, pz: f32, qx: f32, qz: f32| (qx - px) * (z - pz) - (qz - pz) * (x - px);
    let (d1, d2, d3) = (
        side(ax, az, bx, bz),
        side(bx, bz, cx, cz),
        side(cx, cz, ax, az),
    );
    let negative = d1 < -1e-4 || d2 < -1e-4 || d3 < -1e-4;
    let positive = d1 > 1e-4 || d2 > 1e-4 || d3 > 1e-4;
    !(negative && positive)
}

#[test]
fn a_source_covers_its_chunk_and_border_with_the_grounds_triangles_and_nothing_further() {
    let grounds = grounds();
    let border = 3.0;

    let source = source(&grounds, |_| true, border).expect("every neighbour built");

    assert_eq!(source.bounds_origin[0], -border);
    assert_eq!(source.bounds_origin[2], -border);
    assert_eq!(source.bounds_size[0], CHUNK + 2.0 * border);
    assert_eq!(source.bounds_size[2], CHUNK + 2.0 * border);
    let triangles: Vec<&[f32]> = source.triangles.chunks(9).collect();
    // Every point of the bounds' shadow is on some triangle.
    for step_z in 0..=22 {
        for step_x in 0..=22 {
            let (x, z) = (-border + step_x as f32, -border + step_z as f32);
            assert!(
                triangles.iter().any(|triangle| covers(triangle, x, z)),
                "({x}, {z}) is not covered"
            );
        }
    }
    // The bottom is on a whole cell, and no triangle lies wholly beyond the bounds, every corner
    // inside them upwards.
    assert_eq!(source.bounds_origin[1] % 0.25, 0.0);
    let (low, high) = (
        source.bounds_origin[1],
        source.bounds_origin[1] + source.bounds_size[1],
    );
    for triangle in &triangles {
        let xs = [triangle[0], triangle[3], triangle[6]];
        let zs = [triangle[2], triangle[5], triangle[8]];
        let near = |values: [f32; 3]| {
            values.iter().any(|&v| v >= -border - CELL[0])
                && values.iter().any(|&v| v <= CHUNK + border + CELL[0])
        };
        assert!(near(xs) && near(zs), "{triangle:?}");
        for y in [triangle[1], triangle[4], triangle[7]] {
            assert!(y > low && y < high, "{y} outside {low}..{high}");
        }
    }
}

#[test]
fn a_source_waits_for_every_neighbour_in_the_world_and_skips_those_outside() {
    let grounds = grounds();
    let without_east: Vec<GroundMesh> = grounds
        .iter()
        .filter(|ground| ground.chunk.x < 1)
        .cloned()
        .collect();

    let waiting = source(&without_east, |_| true, 3.0);
    let at_edge = source(&without_east, |chunk| chunk.x < 1, 3.0).expect("the world's neighbours");

    assert!(
        matches!(waiting, Err(NavSourceError::Missing(chunk)) if chunk.x == 1),
        "{waiting:?}"
    );
    // A chunk's own ground reaches half a cell into its neighbour, to the neighbour's first column
    // centre; nothing east of that comes from the chunk beyond the world's edge.
    assert!(
        at_edge
            .triangles
            .chunks(3)
            .all(|at| at[0] <= CHUNK + CELL[0] / 2.0 + 1e-4),
        "a triangle east of the world's edge"
    );
}

#[test]
fn a_border_wider_than_a_chunk_is_refused() {
    let grounds = grounds();

    let result = source(&grounds, |_| true, CHUNK + 1.0);

    assert!(
        matches!(result, Err(NavSourceError::BorderTooWide { .. })),
        "{result:?}"
    );
}
