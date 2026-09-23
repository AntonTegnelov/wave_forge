//! A save: what a generated world needs to come back as the player left it.
//!
//! A world is a function of the pack, the seed, the facts and the edits, so a save holds the
//! edits and, for the stages a pack freezes, each chunk as it was first generated. It records the
//! Wave Forge version that wrote it and a digest of the pack, so a game can tell when either has
//! changed since. The facts are the game's own and it saves them itself.

use super::edits::Edits;
use super::runtime::{Product, StageError};
use serde::{Deserialize, Serialize};
use wfc_core::ChunkCoord;

/// A frozen stage's chunk as it was first generated, before the edits of it.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct FrozenChunk {
    pub stage: String,
    pub chunk: ChunkCoord,
    pub product: Product,
}

/// What a save keeps of a world ([`crate::stages::Runtime::save`]).
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct Save {
    /// The Wave Forge version that wrote it.
    pub generator: String,
    /// The digest of the pack it was played with ([`crate::stages::Pack::digest`]).
    pub pack: u64,
    /// The player's edits, less those of ephemeral stages.
    pub edits: Edits,
    /// Every chunk of a frozen stage generated so far, in the order of stage and chunk.
    pub frozen: Vec<FrozenChunk>,
}

impl Save {
    /// The save as RON text.
    #[must_use]
    pub fn to_ron(&self) -> String {
        ron::to_string(self).expect("a save is plain data")
    }

    /// A save from the RON text [`Save::to_ron`] gave.
    ///
    /// # Errors
    /// [`StageError::Edit`] if the text is not a save.
    pub fn from_ron(text: &str) -> Result<Self, StageError> {
        ron::from_str(text).map_err(|error| StageError::Edit(format!("not a save: {error}")))
    }
}
