//! A chunk's ground: a mesh and a height-field collider built from a height field stage.
//!
//! A field holds one height per cell column. The ground puts a vertex on every column's centre and
//! adds the first column of the chunks beyond its +x and +y edges, so two neighbouring chunks share
//! the vertices along their common edge and meet without a seam. Normals come from the heights on
//! either side of a vertex, which at an edge are the neighbour's, so the shading is continuous across
//! chunks too. A chunk's ground therefore reads the fields of its eight neighbours as well as its
//! own, and [`ground`] builds it only once all nine have arrived; [`ground_readers`] says which
//! chunks' ground may have become buildable when one field arrives.
//!
//! Everything is in a Y-up engine's axes, as [`crate::YUpSpace`] maps them: the lattice's x is the
//! engine's x, its y the engine's z, and a field's height the engine's y.

use crate::stages::Field;
use wfc_core::ChunkCoord;

/// One chunk's ground, ready for an engine: a triangle mesh and the same heights as a grid for a
/// height-field collider.
#[derive(Clone, Debug, PartialEq)]
pub struct GroundMesh {
    pub chunk: ChunkCoord,
    /// Vertices along the engine's x and z: one more than the chunk's columns each way.
    pub size: [u32; 2],
    /// Relative to the chunk's corner on the ground plane (the engine's x and z), with the height
    /// absolute. Row by row along z, x fastest: vertex `(i, j)` is at index `j * size[0] + i`, above
    /// the centre of column `(i, j)` of the chunk.
    pub positions: Vec<[f32; 3]>,
    /// Unit normals, one per vertex.
    pub normals: Vec<[f32; 3]>,
    /// Triangles, counter-clockwise seen from above (from +y), two per square of four vertices.
    pub indices: Vec<u32>,
    /// Every vertex's height in engine units, in the same order as `positions`: what a height-field
    /// collider takes.
    pub heights: Vec<f32>,
}

/// The ground of `chunk` from a height field whose chunks `field` looks up, with cells `cell_size`
/// along the engine's x, y and z; a field's value is a height in cells.
///
/// Returns `None` until the field of `chunk` and of all eight chunks around it have arrived.
///
/// # Panics
/// If the fields do not all have the size of `chunk`'s field: a stage's chunks are one size.
#[must_use]
pub fn ground<'a>(
    chunk: ChunkCoord,
    field: impl Fn(ChunkCoord) -> Option<&'a Field>,
    cell_size: [f32; 3],
) -> Option<GroundMesh> {
    let own = field(chunk)?;
    let [sx, sy] = own.size;
    let mut around = [[None; 3]; 3];
    for (dy, row) in around.iter_mut().enumerate() {
        for (dx, slot) in row.iter_mut().enumerate() {
            let at = ChunkCoord::new(chunk.x + dx as i32 - 1, chunk.y + dy as i32 - 1, chunk.z);
            let neighbour = field(at)?;
            assert_eq!(
                neighbour.size, own.size,
                "the fields of one stage share a size"
            );
            *slot = Some(neighbour);
        }
    }
    // The height of a column relative to this chunk's, from -1 to one past the far edge.
    let height = |x: i64, y: i64| -> f32 {
        let (cx, cy) = (x.div_euclid(i64::from(sx)), y.div_euclid(i64::from(sy)));
        let neighbour = around[(cy + 1) as usize][(cx + 1) as usize]
            .expect("every neighbour was looked up above");
        neighbour.get(
            x.rem_euclid(i64::from(sx)) as u32,
            y.rem_euclid(i64::from(sy)) as u32,
        )
    };
    let [cell_x, cell_up, cell_z] = cell_size;
    let size = [sx + 1, sy + 1];
    let mut positions = Vec::with_capacity((size[0] * size[1]) as usize);
    let mut normals = Vec::with_capacity(positions.capacity());
    for j in 0..i64::from(size[1]) {
        for i in 0..i64::from(size[0]) {
            let up = height(i, j) * cell_up;
            positions.push([(i as f32 + 0.5) * cell_x, up, (j as f32 + 0.5) * cell_z]);
            let slope_x = (height(i + 1, j) - height(i - 1, j)) * cell_up / (2.0 * cell_x);
            let slope_z = (height(i, j + 1) - height(i, j - 1)) * cell_up / (2.0 * cell_z);
            let length = (slope_x * slope_x + 1.0 + slope_z * slope_z).sqrt();
            normals.push([-slope_x / length, 1.0 / length, -slope_z / length]);
        }
    }
    let mut indices = Vec::with_capacity((sx * sy * 6) as usize);
    for j in 0..sy {
        for i in 0..sx {
            let corner = j * size[0] + i;
            let (right, below) = (corner + 1, corner + size[0]);
            indices.extend([corner, below, right, right, below, below + 1]);
        }
    }
    let heights = positions.iter().map(|position| position[1]).collect();
    Some(GroundMesh {
        chunk,
        size,
        positions,
        normals,
        indices,
        heights,
    })
}

/// The chunks whose ground reads the field of `chunk`: itself and the eight around it. When that
/// field arrives, these are the chunks whose ground may have become buildable.
#[must_use]
pub fn ground_readers(chunk: ChunkCoord) -> [ChunkCoord; 9] {
    std::array::from_fn(|index| {
        let (dx, dy) = (index as i32 % 3 - 1, index as i32 / 3 - 1);
        ChunkCoord::new(chunk.x + dx, chunk.y + dy, chunk.z)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    const SIZE: [u32; 2] = [4, 3];
    const CELL: [f32; 3] = [2.0, 0.5, 3.0];

    /// A height field that is a function of the world column, as every pure field stage is.
    fn fields(height: impl Fn(i64, i64) -> f32) -> BTreeMap<ChunkCoord, Field> {
        let mut out = BTreeMap::new();
        for cy in -2..=2 {
            for cx in -2..=2 {
                let chunk = ChunkCoord::new(cx, cy, 0);
                let values = (0..SIZE[1])
                    .flat_map(|y| (0..SIZE[0]).map(move |x| (x, y)))
                    .map(|(x, y)| {
                        height(
                            i64::from(cx) * i64::from(SIZE[0]) + i64::from(x),
                            i64::from(cy) * i64::from(SIZE[1]) + i64::from(y),
                        )
                    })
                    .collect();
                out.insert(
                    chunk,
                    Field {
                        chunk,
                        size: SIZE,
                        values,
                    },
                );
            }
        }
        out
    }

    fn hilly(x: i64, y: i64) -> f32 {
        ((x * 7 + y * 13) % 5) as f32 + 0.25 * y as f32
    }

    fn build(fields: &BTreeMap<ChunkCoord, Field>, chunk: ChunkCoord) -> GroundMesh {
        ground(chunk, |at| fields.get(&at), CELL).expect("every neighbour is there")
    }

    #[test]
    fn neighbouring_chunks_share_the_vertices_and_normals_of_their_common_edge() {
        let fields = fields(hilly);
        let here = build(&fields, ChunkCoord::new(0, 0, 0));
        let east = build(&fields, ChunkCoord::new(1, 0, 0));
        let north = build(&fields, ChunkCoord::new(0, 1, 0));
        let [w, h] = here.size;
        let chunk_x = SIZE[0] as f32 * CELL[0];
        let chunk_z = SIZE[1] as f32 * CELL[2];

        for j in 0..h {
            let (mine, theirs) = ((j * w + w - 1) as usize, (j * w) as usize);
            let [x, y, z] = here.positions[mine];
            assert_eq!(
                [x - chunk_x, y, z],
                east.positions[theirs],
                "east edge, row {j}"
            );
            assert_eq!(
                here.normals[mine], east.normals[theirs],
                "east normal, row {j}"
            );
        }
        for i in 0..w {
            let (mine, theirs) = (((h - 1) * w + i) as usize, i as usize);
            let [x, y, z] = here.positions[mine];
            assert_eq!(
                [x, y, z - chunk_z],
                north.positions[theirs],
                "north edge, column {i}"
            );
            assert_eq!(
                here.normals[mine], north.normals[theirs],
                "north normal, column {i}"
            );
        }
    }

    #[test]
    fn every_triangle_faces_up() {
        let fields = fields(hilly);
        let mesh = build(&fields, ChunkCoord::new(-1, 1, 0));

        for triangle in mesh.indices.chunks(3) {
            let [a, b, c] = [0, 1, 2].map(|k| mesh.positions[triangle[k] as usize]);
            let (u, v) = (
                [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
                [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
            );
            let up = u[2] * v[0] - u[0] * v[2];
            assert!(up > 0.0, "triangle {triangle:?} faces down");
        }
        assert_eq!(mesh.indices.len() as u32, SIZE[0] * SIZE[1] * 6);
    }

    #[test]
    fn a_vertex_stands_on_its_columns_height_at_the_columns_centre() {
        let fields = fields(hilly);
        let chunk = ChunkCoord::new(1, -1, 0);
        let mesh = build(&fields, chunk);

        let [i, j] = [2_u32, 1];
        let vertex = mesh.positions[(j * mesh.size[0] + i) as usize];
        let column = fields[&chunk].get(i, j);
        assert_eq!(vertex, [2.5 * CELL[0], column * CELL[1], 1.5 * CELL[2]]);
        assert_eq!(mesh.heights[(j * mesh.size[0] + i) as usize], vertex[1]);
    }

    #[test]
    fn a_plane_has_the_normal_of_its_slope() {
        // Rising one cell of height per cell along x: in engine units 0.5 up per 2 across.
        let fields = fields(|x, _| x as f32);
        let mesh = build(&fields, ChunkCoord::new(0, 0, 0));

        let expected = {
            let length = (0.25_f32 * 0.25 + 1.0).sqrt();
            [-0.25 / length, 1.0 / length, 0.0]
        };
        for normal in &mesh.normals {
            for (got, want) in normal.iter().zip(expected) {
                assert!((got - want).abs() < 1e-6, "{normal:?} against {expected:?}");
            }
        }
    }

    #[test]
    fn nothing_is_built_until_every_neighbour_has_arrived() {
        let mut fields = fields(hilly);
        fields.remove(&ChunkCoord::new(1, 1, 0));

        assert!(ground(ChunkCoord::new(0, 0, 0), |at| fields.get(&at), CELL).is_none());
        assert!(ground(ChunkCoord::new(-1, -1, 0), |at| fields.get(&at), CELL).is_some());
    }

    #[test]
    fn the_readers_of_a_field_are_the_chunks_whose_ground_needs_it() {
        let fields = fields(hilly);
        let arrived = ChunkCoord::new(0, 0, 0);
        let readers = ground_readers(arrived);

        for cy in -1..=1 {
            for cx in -1..=1 {
                let chunk = ChunkCoord::new(cx, cy, 0);
                let without = ground(chunk, |at| (at != arrived).then(|| &fields[&at]), CELL);
                assert_eq!(
                    without.is_none(),
                    readers.contains(&chunk),
                    "chunk {chunk:?}"
                );
            }
        }
    }
}
