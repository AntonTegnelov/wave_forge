//! Boxes that cover chosen cells of a chunk, for the products made of whole cells: interiors
//! ([`mod@crate::region_tags`]) and occluders ([`mod@crate::occluders`]).

use crate::space::YUpSpace;
use wfc_core::ChunkCoord;

/// A box of whole cells in a Y-up engine's world space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellBox {
    /// The lowest corner along the engine's x, y and z.
    pub min: [f32; 3],
    /// The highest corner.
    pub max: [f32; 3],
}

/// Boxes that cover the cells of the chunk at `coord` that `chosen` marks, each cell once and no
/// other; `chosen` holds a flag per cell in the order the chunk stores its tiles.
///
/// Each box grows from its first uncovered chosen cell along x, then y, then z, as far as every
/// cell it would take is chosen and uncovered: not the fewest boxes, but few, in one pass.
///
/// # Panics
/// If `chosen` does not hold one flag per cell of the space's chunks.
pub(crate) fn cell_boxes(coord: ChunkCoord, chosen: &[bool], space: &YUpSpace) -> Vec<CellBox> {
    let shape = space.chunk_shape();
    assert_eq!(chosen.len(), shape.cells() as usize, "a flag per cell");
    let [cell_x, cell_up, cell_z] = space.cell_size();
    let origin = space.chunk_origin(coord);
    let index = |x: u32, y: u32, z: u32| (x + shape.x * (y + shape.y * z)) as usize;
    let mut covered = vec![false; chosen.len()];
    let mut boxes = Vec::new();
    for z in 0..shape.z {
        for y in 0..shape.y {
            for x in 0..shape.x {
                if !chosen[index(x, y, z)] || covered[index(x, y, z)] {
                    continue;
                }
                let free =
                    |x: u32, y: u32, z: u32| chosen[index(x, y, z)] && !covered[index(x, y, z)];
                let mut high_x = x + 1;
                while high_x < shape.x && free(high_x, y, z) {
                    high_x += 1;
                }
                let mut high_y = y + 1;
                while high_y < shape.y && (x..high_x).all(|x| free(x, high_y, z)) {
                    high_y += 1;
                }
                let mut high_z = z + 1;
                while high_z < shape.z
                    && (y..high_y).all(|y| (x..high_x).all(|x| free(x, y, high_z)))
                {
                    high_z += 1;
                }
                for cz in z..high_z {
                    for cy in y..high_y {
                        for cx in x..high_x {
                            covered[index(cx, cy, cz)] = true;
                        }
                    }
                }
                boxes.push(CellBox {
                    min: [
                        origin[0] + x as f32 * cell_x,
                        origin[1] + z as f32 * cell_up,
                        origin[2] + y as f32 * cell_z,
                    ],
                    max: [
                        origin[0] + high_x as f32 * cell_x,
                        origin[1] + high_z as f32 * cell_up,
                        origin[2] + high_y as f32 * cell_z,
                    ],
                });
            }
        }
    }
    boxes
}
