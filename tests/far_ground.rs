//! Far ground from a coarse height field: its heights are what a fine stage reading the coarse field
//! gets, neighbouring coarse chunks meet exactly, it covers every point of the ground plane that no
//! near ground covers and none that one does, and a wall joins it to every near ground it meets.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::{Pack, Runtime};
use wave_forge::{ChunkCoord, FarGround, FocusPoint, GroundMesh, far_ground, ground};

const PACK: &str = r#"(
    version: 1,
    stages: [
        (name: "far", scale: 8, kind: Field(Mul(Noise(frequency: 0.004, octaves: 3), Constant(40.0)))),
        (name: "near", kind: Field(Mul(Noise(frequency: 0.004, octaves: 3), Constant(40.0)))),
        (name: "read", kind: Field(Input("far"))),
    ],
)"#;

const CELL: [f32; 3] = [2.0, 1.0, 2.0];
const COLUMNS: i32 = 8;
const SCALE: u32 = 8;

fn runtime() -> Runtime {
    let mut runtime = Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        3,
        [8, 8],
    );
    runtime
        .request_each(
            &[FocusPoint::new(ChunkCoord::new(4, 4, 0), 3)],
            &[("far", Some(24)), ("near", None), ("read", Some(5))],
        )
        .expect("stages");
    runtime.run_until_idle().expect("the stages run");
    runtime
}

fn far(runtime: &Runtime, chunk: ChunkCoord, near: &BTreeMap<ChunkCoord, GroundMesh>) -> FarGround {
    far_ground(
        chunk,
        SCALE,
        |at| runtime.field("far", at),
        CELL,
        |at| near.get(&at),
    )
    .expect("the coarse fields around it arrived")
}

/// Where a far ground's vertex is in the world's ground plane and height.
fn world(far: &FarGround, vertex: usize) -> [f32; 3] {
    let [x, y, z] = far.positions[vertex];
    let span = COLUMNS as f32 * CELL[0];
    let corner = [
        far.chunk.x as f32 * SCALE as f32 * span,
        far.chunk.y as f32 * SCALE as f32 * span,
    ];
    [x + corner[0], y, z + corner[1]]
}

#[test]
fn a_far_grounds_corners_are_what_a_fine_stage_reading_the_coarse_field_gets() {
    let runtime = runtime();
    let chunk = ChunkCoord::new(0, 0, 0);

    let far = far(&runtime, chunk, &BTreeMap::new());

    for j in 0..=SCALE as i32 {
        for i in 0..=SCALE as i32 {
            let lattice = ChunkCoord::new(i.min(SCALE as i32 - 1), j.min(SCALE as i32 - 1), 0);
            // A corner past the chunk's last lattice chunk is that chunk's far edge: the first
            // column of the next one, which is read here from the one before.
            let (column_x, column_y) = (
                (i - lattice.x) as u32 * COLUMNS as u32,
                (j - lattice.y) as u32 * COLUMNS as u32,
            );
            let read = runtime.field("read", lattice).expect("generated");
            let expected = if column_x < COLUMNS as u32 && column_y < COLUMNS as u32 {
                read.get(column_x, column_y)
            } else {
                let next = ChunkCoord::new(
                    lattice.x + (column_x / COLUMNS as u32) as i32,
                    lattice.y + (column_y / COLUMNS as u32) as i32,
                    0,
                );
                runtime
                    .field("read", next)
                    .expect("generated")
                    .get(column_x % COLUMNS as u32, column_y % COLUMNS as u32)
            };
            let got = far.positions[(j as u32 * (SCALE + 1) + i as u32) as usize][1];
            assert!(
                (got - expected * CELL[1]).abs() < 1e-4,
                "corner ({i}, {j}): {got} against {expected}"
            );
        }
    }
}

#[test]
fn neighbouring_far_grounds_meet_exactly() {
    let runtime = runtime();
    let (a, b) = (ChunkCoord::new(0, 0, 0), ChunkCoord::new(1, 0, 0));

    let (a, b) = (
        far(&runtime, a, &BTreeMap::new()),
        far(&runtime, b, &BTreeMap::new()),
    );

    let side = SCALE + 1;
    for j in 0..side {
        let on_a = world(&a, (j * side + SCALE) as usize);
        let on_b = world(&b, (j * side) as usize);
        assert!(
            on_a.iter().zip(on_b).all(|(p, q)| (p - q).abs() < 1e-4),
            "row {j}: {on_a:?} against {on_b:?}"
        );
    }
}

/// The near grounds of the lattice chunks from (2, 2) to (4, 4), in the middle of the coarse chunk
/// at the origin.
fn near_grounds(runtime: &Runtime) -> BTreeMap<ChunkCoord, GroundMesh> {
    (2..=4)
        .flat_map(|y| (2..=4).map(move |x| ChunkCoord::new(x, y, 0)))
        .map(|chunk| {
            let mesh = ground(chunk, |at| runtime.field("near", at), CELL).expect("fields around");
            (chunk, mesh)
        })
        .collect()
}

/// Whether the point `p` of the ground plane lies inside the triangle `a`, `b`, `c`.
fn inside(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> bool {
    let cross = |o: [f32; 2], u: [f32; 2], v: [f32; 2]| {
        (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])
    };
    let (d1, d2, d3) = (cross(a, b, p), cross(b, c, p), cross(c, a, p));
    let negative = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let positive = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(negative && positive)
}

/// How many of these triangles, flat on the ground plane, hold `p`; a wall's, which stand on a
/// line, hold none.
fn covering(p: [f32; 2], triangles: &[[[f32; 2]; 3]]) -> usize {
    triangles
        .iter()
        .filter(|[a, b, c]| {
            let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            area.abs() > 1e-6 && inside(p, *a, *b, *c)
        })
        .count()
}

/// Whether `p` lies within a hundredth of a unit of a triangle's edge, where which triangle holds
/// it is a matter of rounding.
fn on_an_edge(p: [f32; 2], triangles: &[[[f32; 2]; 3]]) -> bool {
    let near_segment = |a: [f32; 2], b: [f32; 2]| {
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let length = dx * dx + dy * dy;
        if length == 0.0 {
            return false;
        }
        let t = (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length).clamp(0.0, 1.0);
        let (x, y) = (a[0] + t * dx - p[0], a[1] + t * dy - p[1]);
        (x * x + y * y).sqrt() < 0.01
    };
    triangles
        .iter()
        .any(|[a, b, c]| near_segment(*a, *b) || near_segment(*b, *c) || near_segment(*c, *a))
}

#[test]
fn far_and_near_ground_cover_every_point_once() {
    let runtime = runtime();
    let near = near_grounds(&runtime);
    let far = far(&runtime, ChunkCoord::new(0, 0, 0), &near);
    let span = COLUMNS as f32 * CELL[0];
    let flat = |p: [f32; 3]| [p[0], p[2]];
    let mut triangles: Vec<[[f32; 2]; 3]> = far
        .indices
        .chunks(3)
        .map(|t| [0, 1, 2].map(|k| flat(world(&far, t[k] as usize))))
        .collect();
    for (chunk, mesh) in &near {
        let corner = [chunk.x as f32 * span, chunk.y as f32 * span];
        let at = |vertex: u32| {
            let [x, _, z] = mesh.positions[vertex as usize];
            [x + corner[0], z + corner[1]]
        };
        triangles.extend(
            mesh.levels[0]
                .indices
                .chunks(3)
                .map(|t| [at(t[0]), at(t[1]), at(t[2])]),
        );
    }

    // Points off every vertex and edge, across the coarse chunk's inside.
    let mut checked = 0;
    for j in 0..60 {
        for i in 0..60 {
            let p = [
                0.5 * CELL[0] + (i as f32 + 0.37) * span * SCALE as f32 / 60.0,
                0.5 * CELL[2] + (j as f32 + 0.41) * span * SCALE as f32 / 60.0,
            ];
            if p[0] >= 0.5 * CELL[0] + span * SCALE as f32
                || p[1] >= 0.5 * CELL[2] + span * SCALE as f32
            {
                continue;
            }
            if on_an_edge(p, &triangles) {
                continue;
            }
            assert_eq!(covering(p, &triangles), 1, "the point {p:?}");
            checked += 1;
        }
    }
    assert!(checked > 3000, "only {checked} points checked");
}

#[test]
fn a_wall_joins_every_near_edge_the_far_ground_meets_from_above_it_to_below_its_skirt() {
    let runtime = runtime();
    let near = near_grounds(&runtime);

    let far = far(&runtime, ChunkCoord::new(0, 0, 0), &near);

    let span = COLUMNS as f32 * CELL[0];
    let vertices: Vec<[f32; 3]> = (0..far.positions.len()).map(|v| world(&far, v)).collect();
    // The lowest and highest far ground vertex standing at a place of the ground plane.
    let span_at = |x: f32, z: f32| {
        vertices
            .iter()
            .filter(|v| (v[0] - x).abs() < 1e-4 && (v[2] - z).abs() < 1e-4)
            .fold((f32::MAX, f32::MIN), |(low, high), v| {
                (low.min(v[1]), high.max(v[1]))
            })
    };
    // The block's outer edges: along x = 2 chunks, below the block's first column of chunks.
    let mut edges = 0;
    for (chunk, mesh) in &near {
        let [w, h] = mesh.size;
        let corner = [chunk.x as f32 * span, chunk.y as f32 * span];
        let skirt = mesh.positions[0][1] - mesh.positions[(w * h) as usize][1];
        let outer: Vec<(bool, Vec<u32>)> = vec![
            (chunk.x == 2, (0..h).map(|k| k * w).collect()),
            (chunk.x == 4, (0..h).map(|k| k * w + w - 1).collect()),
            (chunk.y == 2, (0..w).collect()),
            (chunk.y == 4, (0..w).map(|k| (h - 1) * w + k).collect()),
        ];
        for (_, edge) in outer.into_iter().filter(|(faces_far, _)| *faces_far) {
            edges += 1;
            for vertex in edge {
                let [x, edge, z] = mesh.positions[vertex as usize];
                let (low, high) = span_at(x + corner[0], z + corner[1]);
                assert!(
                    high >= edge - 1e-4 && low <= edge - skirt + 1e-4,
                    "the wall at {chunk:?}'s vertex {vertex} spans {low} to {high}, its edge {edge}"
                );
            }
        }
    }
    assert_eq!(edges, 12, "the block's outer edges");
    // Two triangles each way per segment of each of a wall's two strips, above and below its foot on
    // the far edge, eight segments to an edge; a square beside the
    // block fans from its centre through its outline, eight vertices along its walled edge and one
    // for each other corner; the rest are two triangles.
    let walls = 12 * 8 * 2 * 4;
    let beside = 12 * (8 + 3);
    let plain = 2 * (SCALE * SCALE - 9 - 12) as usize;
    assert_eq!(far.indices.len(), 3 * (walls + beside + plain));
}
