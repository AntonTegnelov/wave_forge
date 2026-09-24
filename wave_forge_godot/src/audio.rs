//! Sound for the chunks near the player (docs/reference/godot.md, "Sound and surfaces").
//!
//! Each chunk within the audio radius gets an `Area3D` per interior, whose reverb bus gives sounds
//! inside a room's acoustics, and an `AudioStreamPlayer3D` per emitter whose key the game mapped to
//! a stream. Players come from a pool, so a chunk leaving and another arriving reuse them rather
//! than freeing and allocating nodes.

use godot::classes::{
    Area3D, AudioStream, AudioStreamPlayer3D, BoxShape3D, CollisionShape3D, Node,
};
use godot::prelude::*;
use std::collections::{HashMap, HashSet};
use wave_forge::{ChunkCoord, RegionTags};

/// What a chunk has in the scene: its interiors' areas and the players of its emitters.
struct ChunkSound {
    areas: Vec<Gd<Area3D>>,
    players: Vec<Gd<AudioStreamPlayer3D>>,
}

/// The areas and players of the chunks within the audio radius.
#[derive(Default)]
pub(crate) struct RegionAudio {
    chunks: HashMap<ChunkCoord, ChunkSound>,
    /// Stopped players, children of the node, ready to play again.
    pool: Vec<Gd<AudioStreamPlayer3D>>,
    /// Keys a module set uses that `sounds` does not map, each reported once.
    unmapped: HashSet<String>,
}

impl RegionAudio {
    /// The chunks that have areas and players.
    pub(crate) fn chunks(&self) -> impl Iterator<Item = ChunkCoord> + '_ {
        self.chunks.keys().copied()
    }

    /// Gives `tags`' chunk areas for its interiors, with reverb on `reverb_bus` unless it is empty,
    /// and a player for each emitter whose key `sounds` maps to a stream, all children of `owner`;
    /// replacing what the chunk had.
    pub(crate) fn build(
        &mut self,
        owner: &mut Gd<Node>,
        tags: &RegionTags,
        sounds: &VarDictionary,
        reverb_bus: &StringName,
    ) {
        self.drop_chunk(tags.chunk);
        let mut areas = Vec::new();
        if !reverb_bus.is_empty() {
            for interior in &tags.interiors {
                let [min, max] = [interior.min, interior.max].map(Vector3::from_array);
                let mut shape = BoxShape3D::new_gd();
                shape.set_size(max - min);
                let mut collision = CollisionShape3D::new_alloc();
                collision.set_shape(&shape);
                let mut area = Area3D::new_alloc();
                // Areas only give their reverb to sounds inside them; they detect nothing.
                area.set_monitoring(false);
                area.set_collision_mask(0);
                area.set_use_reverb_bus(true);
                area.set_reverb_bus_name(reverb_bus);
                area.set_reverb_amount(1.0);
                area.add_child(&collision);
                area.set_position((min + max) / 2.0);
                owner.add_child(&area);
                areas.push(area);
            }
        }
        let mut players = Vec::new();
        for emitter in &tags.emitters {
            let Some(stream) = sounds
                .get(emitter.key.as_str())
                .and_then(|stream| stream.try_to::<Gd<AudioStream>>().ok())
            else {
                if self.unmapped.insert(emitter.key.clone()) {
                    godot_warn!(
                        "wave forge: no stream in `sounds` for the key {:?}; it plays nothing",
                        emitter.key
                    );
                }
                continue;
            };
            let mut player = self.pool.pop().unwrap_or_else(|| {
                let player = AudioStreamPlayer3D::new_alloc();
                owner.add_child(&player);
                player
            });
            player.set_stream(&stream);
            player.set_position(Vector3::from_array(emitter.at));
            player.play();
            players.push(player);
        }
        self.chunks
            .insert(tags.chunk, ChunkSound { areas, players });
    }

    /// Frees every chunk's areas and stops every player, as the node leaves the tree: a player
    /// still playing when the game quits leaves its playback to the audio server, which leaks it.
    pub(crate) fn stop(&mut self) {
        let chunks: Vec<ChunkCoord> = self.chunks().collect();
        for chunk in chunks {
            self.drop_chunk(chunk);
        }
    }

    /// Frees `chunk`'s areas and returns its players to the pool, stopped.
    pub(crate) fn drop_chunk(&mut self, chunk: ChunkCoord) {
        let Some(sound) = self.chunks.remove(&chunk) else {
            return;
        };
        for mut area in sound.areas {
            area.queue_free();
        }
        for mut player in sound.players {
            player.stop();
            self.pool.push(player);
        }
    }
}
