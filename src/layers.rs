//! The tiles each chunk in memory had after each phase of the schedule that wrote it, so an
//! operation run again after eviction reads its neighbours as a fresh world had them
//! (docs/architecture/world.md, "Regenerating exactly").

use crate::scheduler;
use std::collections::HashMap;
use wfc_core::{ChunkCoord, ChunkShape, ChunkStore, WorldCell};

/// Where an operation falls in the order a fresh world runs the operations around one chunk in:
/// first-parity first attempts, then first-parity repairs by class, then second-parity first
/// attempts, then second-parity repairs by class. The scheduler's waits hold every operation to
/// that order, so each reads every neighbour as it was before its own phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Phase {
    parity: u8,
    repair: bool,
    class: u8,
}

impl Phase {
    /// The phase of `chunk`'s first attempt.
    pub(crate) const fn first_attempt(chunk: ChunkCoord) -> Self {
        Self {
            parity: chunk.parity(),
            repair: false,
            class: 0,
        }
    }

    /// The phase of a repair of `chunk`.
    pub(crate) fn repair(chunk: ChunkCoord) -> Self {
        Self {
            parity: chunk.parity(),
            repair: true,
            class: scheduler::repair_class(chunk),
        }
    }
}

/// A chunk's versions, oldest first: the phase that wrote each, `None` for a chunk imported as it
/// was, which every phase sees; and the tiles of every version but the last, which is the store's.
#[derive(Debug)]
struct Past {
    phases: Vec<Option<Phase>>,
    earlier: Vec<Box<[u16]>>,
}

/// Every chunk in memory's past, kept beside the store.
#[derive(Debug, Default)]
pub(crate) struct Layers {
    pasts: HashMap<ChunkCoord, Past>,
}

impl Layers {
    /// A chunk entered the store, written by `phase`, or imported as it was when `phase` is
    /// `None`.
    pub(crate) fn born(&mut self, chunk: ChunkCoord, phase: Option<Phase>) {
        self.pasts.insert(
            chunk,
            Past {
                phases: vec![phase],
                earlier: Vec::new(),
            },
        );
    }

    /// `phase` rewrote a chunk whose tiles were `before`.
    ///
    /// # Panics
    /// If the chunk has no past, which every chunk in the store has.
    pub(crate) fn rewritten(&mut self, chunk: ChunkCoord, phase: Phase, before: Box<[u16]>) {
        let past = self
            .pasts
            .get_mut(&chunk)
            .expect("a chunk in the store has a past");
        past.phases.push(Some(phase));
        past.earlier.push(before);
    }

    /// The chunk left memory.
    pub(crate) fn forget(&mut self, chunk: ChunkCoord) {
        self.pasts.remove(&chunk);
    }

    /// The phase that last wrote a chunk in the store: `None` inside for one imported as it was.
    ///
    /// # Panics
    /// If the chunk has no past, which every chunk in the store has.
    pub(crate) fn latest(&self, chunk: ChunkCoord) -> Option<Phase> {
        *self.pasts[&chunk]
            .phases
            .last()
            .expect("a past has a version")
    }

    /// The tile at `at` as it was before `phase`, if its chunk existed then.
    pub(crate) fn tile_before(
        &self,
        store: &ChunkStore,
        at: WorldCell,
        phase: Phase,
    ) -> Option<u16> {
        let shape = store.shape();
        let chunk = store.get(ChunkCoord::of_cell(at, shape))?;
        let past = &self.pasts[&chunk.coord];
        let version = past
            .phases
            .iter()
            .rposition(|written| *written < Some(phase))?;
        let cell = cell_index(chunk.coord, shape, at);
        Some(if version + 1 == past.phases.len() {
            chunk.tiles[cell]
        } else {
            past.earlier[version][cell]
        })
    }

    /// The tiles `phase` left a chunk in the store with, if `phase` wrote it; `None` if the chunk
    /// already existed before `phase` and `phase` left it alone, which a fresh world never does
    /// to a chunk in a repair's region.
    ///
    /// # Panics
    /// If the chunk has no past, which every chunk in the store has.
    pub(crate) fn written_by<'a>(
        &'a self,
        store: &'a ChunkStore,
        chunk: ChunkCoord,
        phase: Phase,
    ) -> Option<&'a [u16]> {
        let past = &self.pasts[&chunk];
        let version = past
            .phases
            .iter()
            .position(|written| *written == Some(phase))?;
        Some(if version + 1 == past.phases.len() {
            &store
                .get(chunk)
                .expect("a chunk with a past is in the store")
                .tiles
        } else {
            &past.earlier[version]
        })
    }

    /// Whether a chunk in the store existed before `phase`.
    ///
    /// # Panics
    /// If the chunk has no past, which every chunk in the store has.
    pub(crate) fn existed_before(&self, chunk: ChunkCoord, phase: Phase) -> bool {
        self.pasts[&chunk].phases[0] < Some(phase)
    }
}

/// Where the cell `at` of `chunk` sits in the chunk's tiles.
pub(crate) fn cell_index(chunk: ChunkCoord, shape: ChunkShape, at: WorldCell) -> usize {
    let origin = chunk.origin(shape);
    let (x, y, z) = (at[0] - origin[0], at[1] - origin[1], at[2] - origin[2]);
    ((z as u32 * shape.y + y as u32) * shape.x + x as u32) as usize
}
