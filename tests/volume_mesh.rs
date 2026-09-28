//! The surface of a Volume stage: where it crosses zero, as a mesh that chunks share their sides
//! of, facing from solid to empty.

use std::collections::BTreeMap;
use std::sync::Arc;
use wave_forge::stages::{Pack, Runtime};
use wave_forge::{ChunkCoord, FocusPoint, VolumeMesh, volume_mesh};

const CELL: [f32; 3] = [2.0, 1.5, 2.0];

// "caves" and "hills" are Godot's noise in 3D, held solid along the lowest level and empty along
// the highest so that their surfaces are closed. The caves' features are about a voxel across, the
// hills' several.
const PACK: &str = r#"(
    version: 1,
    noises: {
        "caves": (noise_type: SimplexSmooth, seed: 11, frequency: 0.12, fractal_octaves: 2),
        "hills": (noise_type: SimplexSmooth, seed: 5, frequency: 0.04, fractal_type: None),
    },
    stages: [
        (name: "flat", kind: Volume(
            density: Sub(Constant(4.3), Z),
            bottom: -2,
            top: 10,
            materials: Some((rules: [(category: "top", when: [Greater(Z, Constant(3.0))])], otherwise: "deep")),
        )),
        (name: "caves", kind: Volume(
            density: Max(Min(FastNoise("caves"), Sub(Constant(5.0), Z)), Sub(Constant(-3.0), Z)),
            bottom: -4,
            top: 6,
            materials: Some((rules: [(category: "banded", when: [Less(Sin(Mul(Add(X, Z), Constant(0.7))), Constant(0.0))])], otherwise: "plain")),
        )),
        (name: "hills", kind: Volume(
            density: Max(Min(FastNoise("hills"), Sub(Constant(5.0), Z)), Sub(Constant(-3.0), Z)),
            bottom: -4,
            top: 6,
        )),
    ],
)"#;

/// A runtime of chunks `size` columns wide with `stage` generated over `chunks`.
fn runtime(size: u32, stage: &str, chunks: &[ChunkCoord]) -> Runtime {
    let mut runtime = Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        3,
        [size, size],
    );
    let focus: Vec<FocusPoint> = chunks
        .iter()
        .map(|&chunk| FocusPoint::new(chunk, 0))
        .collect();
    runtime.request(&focus, &[stage]).expect("a stage");
    runtime.run_until_idle().expect("the stages run");
    runtime
}

/// The chunks from `min` to `max` on the ground plane, both included.
fn square(min: i32, max: i32) -> Vec<ChunkCoord> {
    (min..=max)
        .flat_map(|y| (min..=max).map(move |x| ChunkCoord::new(x, y, 0)))
        .collect()
}

fn mesh(runtime: &Runtime, stage: &str, chunk: ChunkCoord) -> VolumeMesh {
    volume_mesh(chunk, |at| runtime.volume(stage, at), CELL).expect("the volumes around it")
}

/// A mesh's triangles in world units with each corner's material, each rounded and started at its
/// smallest corner with its winding kept, as a count per triangle.
fn triangles(mesh: &VolumeMesh, size: u32) -> BTreeMap<[[i64; 4]; 3], u32> {
    let corner = [
        (mesh.chunk.x * size as i32) as f32 * CELL[0],
        (mesh.chunk.y * size as i32) as f32 * CELL[2],
    ];
    let world = |index: u32| {
        let [x, y, z] = mesh.positions[index as usize];
        let [x, y, z] = [x + corner[0], y, z + corner[1]].map(|axis| (axis * 1e3).round() as i64);
        [x, y, z, i64::from(mesh.materials[index as usize])]
    };
    let mut counts = BTreeMap::new();
    for triangle in mesh.indices.chunks(3) {
        let mut corners = [world(triangle[0]), world(triangle[1]), world(triangle[2])];
        let first = (0..3).min_by_key(|&i| corners[i]).expect("three corners");
        corners.rotate_left(first);
        *counts.entry(corners).or_insert(0) += 1;
    }
    counts
}

fn face_normal(mesh: &VolumeMesh, triangle: &[u32]) -> [f32; 3] {
    let [a, b, c] = [0, 1, 2].map(|i| mesh.positions[triangle[i] as usize]);
    let (u, v) = (
        [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
        [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
    );
    [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]
}

#[test]
fn a_volume_solid_below_a_height_is_a_flat_surface_there_facing_up() {
    let chunk = ChunkCoord::new(0, -1, 0);
    let runtime = runtime(8, "flat", &square(-2, 1));

    let mesh = mesh(&runtime, "flat", chunk);

    assert_eq!(
        mesh.indices.len(),
        8 * 8 * 2 * 3,
        "two triangles per column"
    );
    for (position, normal) in mesh.positions.iter().zip(&mesh.normals) {
        assert!(
            (position[1] - 4.3 * CELL[1]).abs() < 1e-4,
            "a vertex at {position:?}"
        );
        assert_eq!(*normal, [0.0, 1.0, 0.0]);
    }
    assert_eq!(
        mesh.materials,
        vec![0; mesh.positions.len()],
        "every vertex takes the solid voxel below it, which is on top"
    );
    for triangle in mesh.indices.chunks(3) {
        assert!(
            face_normal(&mesh, triangle)[1] > 0.0,
            "{triangle:?} faces down"
        );
    }
}

#[test]
fn four_chunks_mesh_the_same_surface_and_materials_as_one_chunk_twice_as_wide() {
    let small = runtime(8, "caves", &square(-1, 2));
    let large = runtime(16, "caves", &square(-1, 1));

    let mut quarters: BTreeMap<[[i64; 4]; 3], u32> = BTreeMap::new();
    for chunk in square(0, 1) {
        for (triangle, count) in triangles(&mesh(&small, "caves", chunk), 8) {
            *quarters.entry(triangle).or_insert(0) += count;
        }
    }
    let whole = triangles(&mesh(&large, "caves", ChunkCoord::new(0, 0, 0)), 16);

    assert!(
        whole.len() > 100,
        "only {} triangles to compare",
        whole.len()
    );
    assert_eq!(quarters, whole);
}

#[test]
fn where_features_are_wider_than_a_voxel_every_triangle_faces_the_way_its_normals_point() {
    let chunk = ChunkCoord::new(0, 0, 0);
    let runtime = runtime(8, "hills", &square(-1, 1));

    let mesh = mesh(&runtime, "hills", chunk);

    assert!(!mesh.indices.is_empty());
    let mut against = 0;
    for triangle in mesh.indices.chunks(3) {
        let face = face_normal(&mesh, triangle);
        let normals: [f32; 3] = std::array::from_fn(|axis| {
            triangle
                .iter()
                .map(|&vertex| mesh.normals[vertex as usize][axis])
                .sum()
        });
        if face.iter().zip(normals).map(|(f, n)| f * n).sum::<f32>() <= 0.0 {
            against += 1;
        }
    }
    assert_eq!(against, 0, "of {} triangles", mesh.indices.len() / 3);
}

#[test]
fn a_chunk_has_no_surface_until_its_neighbours_volumes_have_arrived() {
    let chunk = ChunkCoord::new(0, 0, 0);
    let runtime = runtime(8, "caves", &[chunk]);

    let mesh = volume_mesh(chunk, |at| runtime.volume("caves", at), CELL);

    assert!(runtime.volume("caves", chunk).is_some());
    assert!(mesh.is_none());
}
