//! Occluders for the chunks near the player (docs/reference/godot.md, "Occlusion").
//!
//! Each chunk within the occluder radius gets one `OccluderInstance3D` holding an `ArrayOccluder3D`
//! of the boxes of its solid cells ([`wave_forge::occluders`]), so Godot's occlusion culling skips
//! drawing what they hide. One occluder mesh per chunk keeps the culler's structure to a handful of
//! meshes however many boxes a chunk has.

use godot::classes::{ArrayOccluder3D, Node, OccluderInstance3D};
use godot::prelude::*;
use std::collections::HashMap;
use wave_forge::{CellBox, ChunkCoord};

/// A box's corners are numbered by bits: bit 0 picks the high x, bit 1 the high y, bit 2 the high
/// z. Each face is two triangles, clockwise seen from outside the box, Godot's front.
const TRIANGLES: [[i32; 3]; 12] = [
    [0, 2, 6],
    [0, 6, 4],
    [1, 5, 7],
    [1, 7, 3],
    [0, 4, 5],
    [0, 5, 1],
    [2, 3, 7],
    [2, 7, 6],
    [0, 1, 3],
    [0, 3, 2],
    [4, 6, 7],
    [4, 7, 5],
];

/// The occluder of each chunk within the radius; `None` for a chunk with no solid cell.
#[derive(Default)]
pub(crate) struct Occluders {
    chunks: HashMap<ChunkCoord, Option<Gd<OccluderInstance3D>>>,
}

impl Occluders {
    /// The chunks that have been given their occluder.
    pub(crate) fn chunks(&self) -> impl Iterator<Item = ChunkCoord> + '_ {
        self.chunks.keys().copied()
    }

    /// Gives `chunk` an occluder of `boxes`, a child of `owner`, replacing the one it had.
    pub(crate) fn build(&mut self, owner: &mut Gd<Node>, chunk: ChunkCoord, boxes: &[CellBox]) {
        self.drop_chunk(chunk);
        if boxes.is_empty() {
            self.chunks.insert(chunk, None);
            return;
        }
        let mut vertices = PackedVector3Array::new();
        let mut indices = PackedInt32Array::new();
        for (index, cell_box) in boxes.iter().enumerate() {
            let first = i32::try_from(index * 8).expect("fewer than 2^28 boxes a chunk");
            for corner in 0..8 {
                let pick = |axis: usize| {
                    if corner & (1 << axis) == 0 {
                        cell_box.min[axis]
                    } else {
                        cell_box.max[axis]
                    }
                };
                vertices.push(Vector3::new(pick(0), pick(1), pick(2)));
            }
            for triangle in TRIANGLES {
                for corner in triangle {
                    indices.push(first + corner);
                }
            }
        }
        let mut mesh = ArrayOccluder3D::new_gd();
        mesh.set_arrays(&vertices, &indices);
        let mut instance = OccluderInstance3D::new_alloc();
        instance.set_occluder(&mesh);
        owner.add_child(&instance);
        self.chunks.insert(chunk, Some(instance));
    }

    /// Frees `chunk`'s occluder.
    pub(crate) fn drop_chunk(&mut self, chunk: ChunkCoord) {
        if let Some(Some(mut instance)) = self.chunks.remove(&chunk) {
            instance.queue_free();
        }
    }
}
