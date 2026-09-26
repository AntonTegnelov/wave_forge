//! The ground's height at any point is where the ground mesh's surface at full detail stands.

use std::sync::Arc;
use wave_forge::stages::{Pack, Runtime};
use wave_forge::{ChunkCoord, FocusPoint, ground, ground_height};

const PACK: &str = r#"(
    version: 1,
    stages: [(name: "height", kind: Field(Mul(Noise(frequency: 0.08, octaves: 3), Constant(12.0))))],
)"#;

const CELL: [f32; 3] = [2.0, 0.5, 2.0];

/// The height of the triangle of `mesh`'s first level that holds `p`, with `p` relative to its
/// chunk's corner.
fn on_mesh(mesh: &wave_forge::GroundMesh, p: [f32; 2]) -> Option<f32> {
    mesh.levels[0].indices.chunks(3).find_map(|t| {
        let [a, b, c] = [0, 1, 2].map(|k| mesh.positions[t[k] as usize]);
        let area = (b[0] - a[0]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[0] - a[0]);
        if area.abs() < 1e-6 {
            return None;
        }
        let w_b = ((p[0] - a[0]) * (c[2] - a[2]) - (p[1] - a[2]) * (c[0] - a[0])) / area;
        let w_c = ((b[0] - a[0]) * (p[1] - a[2]) - (b[2] - a[2]) * (p[0] - a[0])) / area;
        let w_a = 1.0 - w_b - w_c;
        (w_a >= -1e-5 && w_b >= -1e-5 && w_c >= -1e-5).then(|| w_a * a[1] + w_b * b[1] + w_c * c[1])
    })
}

#[test]
fn the_ground_height_is_where_the_ground_mesh_stands() {
    let mut runtime = Runtime::new(
        Arc::new(Pack::parse(PACK).expect("a valid pack")),
        6,
        [8, 8],
    );
    runtime
        .request(&[FocusPoint::new(ChunkCoord::new(0, 0, 0), 2)], &["height"])
        .expect("stages");
    runtime.run_until_idle().expect("the stages run");
    let chunk = ChunkCoord::new(-1, 0, 0);
    let mesh = ground(chunk, |at| runtime.field("height", at), CELL).expect("fields around");
    let span = 8.0 * CELL[0];

    let mut checked = 0;
    for j in 0..40 {
        for i in 0..40 {
            // Across the mesh, from its first column centre to past its last one.
            let local = [
                0.5 * CELL[0] + (i as f32 + 0.3) * span / 40.0,
                0.5 * CELL[2] + (j as f32 + 0.7) * span / 40.0,
            ];
            let world = [
                local[0] + chunk.x as f32 * span,
                local[1] + chunk.y as f32 * span,
            ];

            let got = ground_height(world, [8, 8], |at| runtime.field("height", at), CELL)
                .expect("fields around");

            let expected = on_mesh(&mesh, local).expect("the mesh covers its span");
            assert!(
                (got - expected).abs() < 1e-4,
                "{world:?}: {got} against {expected}"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 1600);
}
