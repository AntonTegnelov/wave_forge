//! Edits: what a player changed in a generated world, as a log the game keeps and saves.
//!
//! A world is a function of the pack, the seed, the facts and the edits. An edit names what it
//! changes by position, never by an ordinal: a point by its positional id, a field's value by its
//! world column. The runtime applies the log to every product as it is generated, so an edit
//! survives eviction and regeneration, and giving the runtime a new log regenerates only the chunks
//! whose edits changed and what reads them ([`crate::stages::Runtime::set_edits`]).

use super::runtime::StageError;
use crate::products::InstanceId;
use serde::{Deserialize, Serialize};
use wfc_core::ChunkCoord;

/// A point's positional id, as a log saves it: the chunk its candidate stood in and its local id.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize, Serialize)]
pub struct PointId {
    pub chunk: (i32, i32),
    pub local: u64,
}

impl From<InstanceId> for PointId {
    fn from(id: InstanceId) -> Self {
        Self {
            chunk: (id.chunk.x, id.chunk.y),
            local: id.local,
        }
    }
}

impl From<PointId> for InstanceId {
    fn from(id: PointId) -> Self {
        Self {
            chunk: ChunkCoord::new(id.chunk.0, id.chunk.1, 0),
            local: id.local,
        }
    }
}

/// One change a player made.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub enum Edit {
    /// A point taken away, a felled tree say, with where it stood, in cells.
    Remove { point: PointId, at: [f32; 2] },
    /// A point moved from where it stood to `to` and turned to `turn`. It stays in the chunk it
    /// was generated in, so the engine finds it there wherever it now stands.
    Move {
        point: PointId,
        from: [f32; 2],
        to: [f32; 3],
        turn: f32,
    },
    /// `by` added to a field stage's value at a world column of that stage: ground raised or dug.
    Raise {
        stage: String,
        column: (i64, i64),
        by: f32,
    },
}

/// The log of every edit, in the order they were made. Later edits of the same point replace
/// earlier ones, and raises of the same column add up.
#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
pub struct Edits {
    pub log: Vec<Edit>,
}

impl Edits {
    /// Adds `edit` at the end of the log.
    pub fn push(&mut self, edit: Edit) {
        self.log.push(edit);
    }

    /// The log as RON text, for a save.
    #[must_use]
    pub fn to_ron(&self) -> String {
        ron::to_string(self).expect("edits are plain data")
    }

    /// A log from the RON text a save holds.
    ///
    /// # Errors
    /// [`StageError::Edit`] if the text is not an edits log.
    pub fn from_ron(text: &str) -> Result<Self, StageError> {
        ron::from_str(text).map_err(|error| StageError::Edit(format!("not an edits log: {error}")))
    }
}
