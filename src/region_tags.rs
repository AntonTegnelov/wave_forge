//! What a chunk says about its places, for sound and walking: the rooms inside it, the sounds its
//! modules make, and what walkers stand on anywhere.
//!
//! A module set says it per module ([`crate::modules::ModulePrototype`]'s `surface`, `indoor` and
//! `sounds`). [`region_tags`] turns a solved chunk into interiors, boxes of indoor cells an engine
//! gives a room's acoustics, and emitters, sounds at points in the engine's world space.
//! [`surface_at`] answers what the module under a point says walkers stand on.

use crate::cell_boxes::{CellBox, cell_boxes};
use crate::loader::RuleFile;
use crate::space::YUpSpace;
use wfc_core::{Chunk, ChunkCoord};

/// A chunk's rooms and sounds, in a Y-up engine's world space.
#[derive(Clone, Debug, PartialEq)]
pub struct RegionTags {
    pub chunk: ChunkCoord,
    /// Boxes that together cover the chunk's indoor cells, each cell once.
    pub interiors: Vec<CellBox>,
    /// Every sound the chunk's modules make.
    pub emitters: Vec<Emitter>,
}

/// A sound playing at a point.
#[derive(Clone, Debug, PartialEq)]
pub struct Emitter {
    /// Where, in the engine's world space.
    pub at: [f32; 3],
    /// What plays: a name the game maps to a sound.
    pub key: String,
}

/// The interiors and emitters of a solved `chunk` of a module set.
#[must_use]
pub fn region_tags(chunk: &Chunk, rules: &RuleFile, space: &YUpSpace) -> RegionTags {
    let shape = space.chunk_shape();
    let [cell_x, cell_up, cell_z] = space.cell_size();
    let origin = space.chunk_origin(chunk.coord);
    let indoor: Vec<bool> = chunk
        .tiles
        .iter()
        .map(|&tile| rules.indoor(usize::from(tile)))
        .collect();
    let interiors = cell_boxes(chunk.coord, &indoor, space);

    let mut emitters = Vec::new();
    for (cell, &tile) in chunk.tiles.iter().enumerate() {
        let cell = u32::try_from(cell).expect("a chunk has fewer than 2^32 cells");
        let (x, y, z) = (
            cell % shape.x,
            (cell / shape.x) % shape.y,
            cell / (shape.x * shape.y),
        );
        for sound in rules.sounds(usize::from(tile)) {
            // A sound's point is in the lattice's axes, z up; the engine's y is the lattice's z.
            let [ax, ay, az] = sound.at;
            emitters.push(Emitter {
                at: [
                    origin[0] + (x as f32 + ax) * cell_x,
                    origin[1] + (z as f32 + az) * cell_up,
                    origin[2] + (y as f32 + ay) * cell_z,
                ],
                key: sound.key,
            });
        }
    }

    RegionTags {
        chunk: chunk.coord,
        interiors,
        emitters,
    }
}

/// What walkers stand on at `point` in the engine's world space: the surface of the module in the
/// cell holding it, if the chunk `chunks` looks up is generated and its module says.
#[must_use]
pub fn surface_at<'a>(
    point: [f32; 3],
    chunks: impl Fn(ChunkCoord) -> Option<&'a Chunk>,
    rules: &'a RuleFile,
    space: &YUpSpace,
) -> Option<&'a str> {
    let (coord, cell) = space.cell_at(point);
    let chunk = chunks(coord)?;
    rules.surface(usize::from(chunk.tiles[cell as usize]))
}
