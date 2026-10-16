//! Decals along the curves of the chunks near the player (docs/reference/godot.md, "Road
//! decals"): one instance of a curve stage's decal scene per piece the library cuts
//! ([`wave_forge::decals`]), turned along its curve and sized to it.

use godot::classes::{Decal, Node, PackedScene};
use godot::prelude::*;
use std::collections::{HashMap, HashSet};
use wave_forge::ChunkCoord;
use wave_forge::decals::DecalPiece;

/// How much longer than its piece each decal is drawn, so neighbours overlap by a hair: a texture
/// filtered at a decal's very end fades toward its border, which would leave a hairline between
/// pieces that only meet.
pub(crate) const OVERLAP: f32 = 1.02;

/// The decals of the chunks within the decal radius.
#[derive(Default)]
pub(crate) struct RegionDecals {
    chunks: HashMap<ChunkCoord, Vec<Gd<Decal>>>,
    /// Stages whose scene's root is not a `Decal`, each reported once.
    refused: HashSet<StringName>,
}

impl RegionDecals {
    /// The chunks that have decals.
    pub(crate) fn chunks(&self) -> impl Iterator<Item = ChunkCoord> + '_ {
        self.chunks.keys().copied()
    }

    /// Gives `chunk` an instance of each stage's scene for each of its pieces, as children of
    /// `owner`, replacing what the chunk had. A scene's root has to be a `Decal`: each instance
    /// keeps its textures, fades and projection depth (`size.y`), and takes the piece's place,
    /// direction, length times [`OVERLAP`] (`size.x`) and width (`size.z`).
    pub(crate) fn build(
        &mut self,
        owner: &mut Gd<Node>,
        chunk: ChunkCoord,
        pieces: &[(StringName, Gd<PackedScene>, Vec<DecalPiece>)],
    ) {
        self.drop_chunk(chunk);
        let mut decals = Vec::new();
        for (stage, scene, stage_pieces) in pieces {
            for piece in stage_pieces {
                let Some(mut decal) = scene
                    .instantiate()
                    .and_then(|node| node.try_cast::<Decal>().ok())
                else {
                    if self.refused.insert(stage.clone()) {
                        godot_warn!(
                            "wave forge: the decal scene of {stage:?} in `curve_decals` has no \
                             Decal as its root; it draws nothing"
                        );
                    }
                    break;
                };
                let along = Vector3::new(piece.along[0], 0.0, piece.along[1]);
                let basis = Basis::from_cols(along, Vector3::UP, along.cross(Vector3::UP));
                decal.set_transform(Transform3D::new(basis, Vector3::from_array(piece.at)));
                let depth = decal.get_size().y;
                decal.set_size(Vector3::new(piece.length * OVERLAP, depth, piece.width));
                owner.add_child(&decal);
                decals.push(decal);
            }
        }
        self.chunks.insert(chunk, decals);
    }

    /// Frees `chunk`'s decals.
    pub(crate) fn drop_chunk(&mut self, chunk: ChunkCoord) {
        for mut decal in self.chunks.remove(&chunk).unwrap_or_default() {
            decal.queue_free();
        }
    }
}
