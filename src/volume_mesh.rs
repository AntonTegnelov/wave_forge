//! A chunk's surface from a Volume stage: where its values cross zero, as one triangle mesh that is
//! drawn and collided with alike, since a height field cannot hold an overhang.
//!
//! The surface is a naive surface net. Samples sit at the voxels' centres; every cube of eight
//! neighbouring samples that the surface passes through gets one vertex, at the average of the
//! points where its edges cross zero, and every edge between two samples that the surface crosses
//! gets a quad joining the vertices of the four cubes around it. A chunk emits the quads of the
//! edges that start at its own samples, and reads the samples one column beyond each side to place
//! the vertices of the cubes those quads reach, so two neighbouring chunks compute the same vertices
//! along their common side and meet without a seam. It therefore reads its eight neighbours' volumes
//! as well as its own, the same chunks as the ground ([`crate::ground_readers`]).
//!
//! Each quad faces from its edge's solid sample to its empty one, and is split along whichever
//! diagonal keeps both its triangles facing that way. Normals follow the values' gradient across a
//! vertex's cube, so they agree with the triangles wherever the surface's features are wider than a
//! voxel; around a feature about one voxel across they can point every which way.
//!
//! Where a voxel's value is above zero it is solid. The volume's lowest and highest levels bound the
//! surface: an edge whose quad would reach below the lowest level or above the highest is left open,
//! so a volume that is solid along its lowest level and empty along its highest is closed.
//!
//! Everything is in a Y-up engine's axes, as [`crate::YUpSpace`] maps them: the lattice's x is the
//! engine's x, its y the engine's z, and a volume's levels go up the engine's y.

use crate::stages::Volume;
use std::collections::HashMap;
use wfc_core::ChunkCoord;

/// One chunk's surface, ready for an engine to draw and to collide with.
#[derive(Clone, Debug, PartialEq)]
pub struct VolumeMesh {
    pub chunk: ChunkCoord,
    /// Relative to the chunk's corner on the ground plane (the engine's x and z), with the height
    /// absolute.
    pub positions: Vec<[f32; 3]>,
    /// Unit normals, one per vertex, pointing from solid to empty.
    pub normals: Vec<[f32; 3]>,
    /// Triangles into `positions`, counter-clockwise seen from the empty side.
    pub indices: Vec<u32>,
}

/// The surface of `chunk` from a Volume stage whose chunks `volume` looks up, with voxels
/// `voxel_size` along the engine's x, y and z: a cell's size times the stage's scale.
///
/// Returns `None` until the volume of `chunk` and of all eight chunks around it have arrived.
///
/// # Panics
/// If the volumes do not all have the size and bottom of `chunk`'s volume: a stage's chunks are
/// one size.
#[must_use]
pub fn volume_mesh<'a>(
    chunk: ChunkCoord,
    volume: impl Fn(ChunkCoord) -> Option<&'a Volume>,
    voxel_size: [f32; 3],
) -> Option<VolumeMesh> {
    let own = volume(chunk)?;
    let [sx, sy, levels] = own.size;
    let mut around = [[None; 3]; 3];
    for (dy, row) in around.iter_mut().enumerate() {
        for (dx, slot) in row.iter_mut().enumerate() {
            let at = ChunkCoord::new(chunk.x + dx as i32 - 1, chunk.y + dy as i32 - 1, chunk.z);
            let neighbour = volume(at)?;
            assert!(
                neighbour.size == own.size && neighbour.bottom == own.bottom,
                "the volumes of one stage share a size"
            );
            *slot = Some(neighbour);
        }
    }
    // A sample by its place in the engine's axes: x and z are columns relative to this chunk,
    // from -1 to one past the far side, and y is a level from the bottom.
    let sample = |at: [i64; 3]| -> f32 {
        let (cx, cz) = (
            at[0].div_euclid(i64::from(sx)),
            at[2].div_euclid(i64::from(sy)),
        );
        let neighbour = around[(cz + 1) as usize][(cx + 1) as usize]
            .expect("every neighbour was looked up above");
        neighbour.get(
            at[0].rem_euclid(i64::from(sx)) as u32,
            at[2].rem_euclid(i64::from(sy)) as u32,
            at[1] as u32,
        )
    };
    let solid = |at: [i64; 3]| sample(at) > 0.0;
    let bottom = own.bottom as f32;
    let top_cube = i64::from(levels) - 2;
    let mut mesh = VolumeMesh {
        chunk,
        positions: Vec::new(),
        normals: Vec::new(),
        indices: Vec::new(),
    };
    let mut vertices: HashMap<[i64; 3], u32> = HashMap::new();
    let mut vertex = |cube: [i64; 3], mesh: &mut VolumeMesh| -> u32 {
        *vertices.entry(cube).or_insert_with(|| {
            let (position, normal) = cube_vertex(cube, &sample, voxel_size, bottom);
            mesh.positions.push(position);
            mesh.normals.push(normal);
            (mesh.positions.len() - 1) as u32
        })
    };
    for z in 0..i64::from(sy) {
        for up in 0..i64::from(levels) {
            for x in 0..i64::from(sx) {
                let at = [x, up, z];
                for axis in 0..3 {
                    let mut next = at;
                    next[axis] += 1;
                    if next[1] >= i64::from(levels) || solid(at) == solid(next) {
                        continue;
                    }
                    // The other two axes in right-handed order after `axis`, and the four cubes
                    // around the edge, counter-clockwise seen from along `axis`.
                    let (b, c) = ((axis + 1) % 3, (axis + 2) % 3);
                    let cubes = [(1, 1), (0, 1), (0, 0), (1, 0)].map(|(db, dc)| {
                        let mut cube = at;
                        cube[b] -= db;
                        cube[c] -= dc;
                        cube
                    });
                    if cubes.iter().any(|cube| cube[1] < 0 || cube[1] > top_cube) {
                        continue;
                    }
                    let corners = cubes.map(|cube| vertex(cube, &mut mesh));
                    // Seen from along `axis` the quad winds counter-clockwise, so it faces the way
                    // `axis` points: towards `next`, which must then be the empty side.
                    let (quad, facing) = if solid(at) {
                        (corners, 1.0)
                    } else {
                        ([corners[3], corners[2], corners[1], corners[0]], -1.0)
                    };
                    mesh.indices
                        .extend(split(quad, &mesh.positions, |normal| facing * normal[axis]));
                }
            }
        }
    }
    Some(mesh)
}

/// A quad's two triangles, split along whichever diagonal leaves both facing the quad's way best:
/// its corners need not lie in a plane, and along the wrong diagonal one triangle can fold over
/// and face the solid side. `facing` measures how far a unit normal faces the quad's way.
fn split(quad: [u32; 4], positions: &[[f32; 3]], facing: impl Fn([f32; 3]) -> f32) -> [u32; 6] {
    let [v0, v1, v2, v3] = quad;
    let worst = |triangles: [[u32; 3]; 2]| {
        triangles
            .map(|triangle| facing(unit_normal(triangle.map(|v| positions[v as usize]))))
            .into_iter()
            .fold(f32::INFINITY, f32::min)
    };
    let along_02 = [[v0, v1, v2], [v0, v2, v3]];
    let along_13 = [[v1, v2, v3], [v1, v3, v0]];
    let [a, b] = if worst(along_13) > worst(along_02) {
        along_13
    } else {
        along_02
    };
    [a[0], a[1], a[2], b[0], b[1], b[2]]
}

/// The unit normal of a triangle whose corners wind counter-clockwise seen from its front, or zero
/// for one without area.
fn unit_normal([a, b, c]: [[f32; 3]; 3]) -> [f32; 3] {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let length = n.iter().map(|x| x * x).sum::<f32>().sqrt();
    if length > 0.0 {
        n.map(|x| x / length)
    } else {
        [0.0; 3]
    }
}

/// The vertex of the cube whose lowest corner is sample `cube`: at the average of the points where
/// its edges cross zero, with a normal down the values' gradient.
fn cube_vertex(
    cube: [i64; 3],
    sample: &impl Fn([i64; 3]) -> f32,
    voxel_size: [f32; 3],
    bottom: f32,
) -> ([f32; 3], [f32; 3]) {
    let corner = |bits: usize| -> [i64; 3] {
        [
            cube[0] + (bits & 1) as i64,
            cube[1] + ((bits >> 1) & 1) as i64,
            cube[2] + ((bits >> 2) & 1) as i64,
        ]
    };
    let values: [f32; 8] = std::array::from_fn(|bits| sample(corner(bits)));
    let mut sum = [0.0_f32; 3];
    let mut crossings = 0;
    let mut gradient = [0.0_f32; 3];
    for axis in 0..3 {
        let step = 1 << axis;
        for bits in (0..8).filter(|bits| bits & step == 0) {
            let (low, high) = (values[bits], values[bits | step]);
            gradient[axis] += (high - low) / (4.0 * voxel_size[axis]);
            if (low > 0.0) != (high > 0.0) {
                let t = low / (low - high);
                for (other, total) in sum.iter_mut().enumerate() {
                    let along = if other == axis {
                        t
                    } else {
                        ((bits >> other) & 1) as f32
                    };
                    *total += along;
                }
                crossings += 1;
            }
        }
    }
    let offset = sum.map(|total| total / crossings as f32);
    let position = [
        (cube[0] as f32 + offset[0] + 0.5) * voxel_size[0],
        (bottom + cube[1] as f32 + offset[1] + 0.5) * voxel_size[1],
        (cube[2] as f32 + offset[2] + 0.5) * voxel_size[2],
    ];
    let length = gradient.iter().map(|g| g * g).sum::<f32>().sqrt();
    // Values that rise as fast one way as they fall the other cancel in the gradient; such a cube
    // has no direction of its own, and faces up.
    let normal = if length > 0.0 {
        gradient.map(|g| -g / length)
    } else {
        [0.0, 1.0, 0.0]
    };
    (position, normal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_folded_quad_is_split_along_the_diagonal_that_keeps_it_facing_its_way() {
        // Facing +y seen from above, with its second corner pushed across the diagonal from the
        // first corner to the third.
        let positions = [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, -0.8],
            [1.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
        ];

        let triangles = split([0, 1, 2, 3], &positions, |normal| normal[1]);

        for triangle in triangles.chunks(3) {
            let normal = unit_normal([0, 1, 2].map(|i| positions[triangle[i] as usize]));
            assert!(normal[1] > 0.0, "{triangle:?} faces {normal:?}");
        }
    }
}
