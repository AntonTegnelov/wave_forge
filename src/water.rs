//! Lakes and rivers drawn as water: a mesh per chunk from a field of the water's level and the
//! ground the engine draws (docs/reference/packs.md, "Water surfaces").
//!
//! The mesh stands on the ground's grid ([`fn@crate::ground`]): a vertex above the centre of every
//! column, the +x and +y edges from the chunks beyond, so neighbouring chunks' water meets exactly.
//! A vertex is wet where the water's level stands above the ground there, and takes that level; a
//! dry one sits just under the ground. Every square with a wet corner is drawn, so the surface runs
//! from the water out to where it dips under the shore: the ground hides its edge, and no gap opens
//! between the water and the bank whatever the bank's slope.

use crate::ground::ground_values;
use crate::stages::Field;
use wfc_core::ChunkCoord;

/// How far, in cells of height, the water's level has to stand above the ground for a vertex to be
/// wet: a film thinner than this would flicker against the ground it lies on.
pub const WET: f32 = 0.05;

/// How far, in cells of height, a dry vertex of the water sits under the ground, so the surface
/// meets the bank below the bank's own surface rather than on it.
const UNDER: f32 = 0.05;

/// One chunk's water, in a Y-up engine's axes as [`crate::YUpSpace`] maps them.
#[derive(Clone, Debug, PartialEq)]
pub struct WaterMesh {
    /// The chunk.
    pub chunk: ChunkCoord,
    /// Vertices along the engine's x and z: one more than the chunk's columns each way.
    pub size: [u32; 2],
    /// Relative to the chunk's corner on the ground plane, with the height absolute, row by row
    /// along z, x fastest, as [`crate::GroundMesh::positions`]'s grid.
    pub positions: Vec<[f32; 3]>,
    /// Unit normals, one per vertex.
    pub normals: Vec<[f32; 3]>,
    /// Triangles into `positions`, counter-clockwise seen from above: two for every square with a
    /// wet corner. Empty where the chunk has no water.
    pub indices: Vec<u32>,
}

/// The water of `chunk`: from the field of the water's level whose chunks `water` looks up and the
/// height field of the ground the engine draws, whose chunks `ground` looks up, both at the ground's
/// scale and in cells, with cells `cell_size` along the engine's x, y and z. A pack makes the
/// water's field from its Lakes stage and its rivers ([`WET`] says how far above the ground counts
/// as water).
///
/// Returns `None` until both fields of `chunk` and of the chunks beyond its +x edge, its +y edge and
/// its +x+y corner have arrived.
///
/// # Panics
/// If the fields' chunks differ in size.
#[must_use]
pub fn water_surface<'a>(
    chunk: ChunkCoord,
    water: impl Fn(ChunkCoord) -> Option<&'a Field>,
    ground: impl Fn(ChunkCoord) -> Option<&'a Field>,
    cell_size: [f32; 3],
) -> Option<WaterMesh> {
    let [columns_x, columns_y] = water(chunk)?.size;
    let levels = ground_values(chunk, &water)?;
    let floor = ground_values(chunk, &ground)?;
    assert_eq!(
        levels.len(),
        floor.len(),
        "the water's and the ground's chunks share a size"
    );
    let size = [columns_x + 1, columns_y + 1];
    let [w, h] = size;
    let wet: Vec<bool> = levels
        .iter()
        .zip(&floor)
        .map(|(level, ground)| level - ground > WET)
        .collect();
    let [cell_x, cell_up, cell_z] = cell_size;
    let heights: Vec<f32> = (0..levels.len())
        .map(|v| {
            if wet[v] {
                levels[v] * cell_up
            } else {
                (floor[v] - UNDER) * cell_up
            }
        })
        .collect();
    let at = |i: i64, j: i64| {
        let (i, j) = (i.clamp(0, i64::from(w) - 1), j.clamp(0, i64::from(h) - 1));
        heights[(j * i64::from(w) + i) as usize]
    };
    let mut positions = Vec::with_capacity(levels.len());
    let mut normals = Vec::with_capacity(levels.len());
    for j in 0..i64::from(h) {
        for i in 0..i64::from(w) {
            positions.push([
                (i as f32 + 0.5) * cell_x,
                at(i, j),
                (j as f32 + 0.5) * cell_z,
            ]);
            let slope_x = (at(i + 1, j) - at(i - 1, j)) / (2.0 * cell_x);
            let slope_z = (at(i, j + 1) - at(i, j - 1)) / (2.0 * cell_z);
            let length = (slope_x * slope_x + 1.0 + slope_z * slope_z).sqrt();
            normals.push([-slope_x / length, 1.0 / length, -slope_z / length]);
        }
    }
    let mut indices = Vec::new();
    for j in 0..h - 1 {
        for i in 0..w - 1 {
            let corner = j * w + i;
            let (right, below) = (corner + 1, corner + w);
            if [corner, right, below, below + 1]
                .iter()
                .any(|&v| wet[v as usize])
            {
                indices.extend([corner, below, right, right, below, below + 1]);
            }
        }
    }
    Some(WaterMesh {
        chunk,
        size,
        positions,
        normals,
        indices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    const SIZE: [u32; 2] = [8, 8];
    const CELL: [f32; 3] = [2.0, 1.0, 2.0];

    fn fields(value: impl Fn(i64, i64) -> f32) -> BTreeMap<ChunkCoord, Field> {
        let mut out = BTreeMap::new();
        for cy in -2..=2 {
            for cx in -2..=2 {
                let chunk = ChunkCoord::new(cx, cy, 0);
                let values = (0..SIZE[1])
                    .flat_map(|y| (0..SIZE[0]).map(move |x| (x, y)))
                    .map(|(x, y)| {
                        value(
                            i64::from(cx) * 8 + i64::from(x),
                            i64::from(cy) * 8 + i64::from(y),
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

    /// A bowl around the origin, and a lake in it at a level of 3.
    fn bowl(x: i64, y: i64) -> f32 {
        ((x * x + y * y) as f32).sqrt() * 0.5
    }

    fn lake(x: i64, y: i64) -> f32 {
        bowl(x, y).max(3.0)
    }

    #[test]
    fn wet_vertices_stand_at_the_waters_level_and_dry_ones_under_the_ground() {
        let (water, ground) = (fields(lake), fields(bowl));
        let chunk = ChunkCoord::new(0, 0, 0);

        let mesh = water_surface(chunk, |at| water.get(&at), |at| ground.get(&at), CELL)
            .expect("all arrived");

        for j in 0..9 {
            for i in 0..9 {
                let height = mesh.positions[(j * 9 + i) as usize][1];
                let floor = bowl(i, j);
                if lake(i, j) - floor > WET {
                    assert_eq!(height, 3.0, "vertex ({i}, {j}) is under the lake");
                } else {
                    assert!(
                        height < floor,
                        "vertex ({i}, {j}) at {height}, its ground {floor}"
                    );
                }
            }
        }
        assert!(!mesh.indices.is_empty());
    }

    #[test]
    fn a_chunk_without_water_has_no_triangles() {
        let (water, ground) = (fields(lake), fields(bowl));

        // A chunk out, the bowl stands above the lake everywhere.
        let mesh = water_surface(
            ChunkCoord::new(1, 1, 0),
            |at| water.get(&at),
            |at| ground.get(&at),
            CELL,
        )
        .expect("all arrived");

        assert!(mesh.indices.is_empty());
    }

    #[test]
    fn every_drawn_square_touches_the_water() {
        let (water, ground) = (fields(lake), fields(bowl));
        let chunk = ChunkCoord::new(-1, 0, 0);

        let mesh = water_surface(chunk, |at| water.get(&at), |at| ground.get(&at), CELL)
            .expect("all arrived");

        for triangle in mesh.indices.chunks(6) {
            let wet = triangle.iter().any(|&v| {
                let (i, j) = (i64::from(v % 9) - 8, i64::from(v / 9));
                lake(i, j) - bowl(i, j) > WET
            });
            assert!(wet, "a square with no wet corner: {triangle:?}");
        }
    }
}
