//! A whole finite world generated ahead of time as one run (docs/product/user-stories.md, M1).
//!
//! A maximal world is not generated while it is played: it is generated whole, before play, and
//! played from what that wrote. [`Runtime::run_world`] generates the target stages over every
//! chunk of the pack's bound, one chunk at a time, and hands each chunk's products to a store as
//! soon as the chunk is done, so the runtime holds only what one chunk reads while the world is as
//! large as the store. A chunk whose products the store already holds is skipped, so a run that
//! was stopped, or cut short by a crash, picks up where it left off; and since a product is a
//! function of the pack, the seed, the facts and the edits, the store ends up holding the same
//! bytes whether the run went at once or in pieces.

use super::runtime::{Runtime, StageError};
use crate::FocusPoint;
use crate::frozen::{FrozenStore, StoreError};
use std::ops::ControlFlow;
use wfc_core::ChunkCoord;

/// How far a run has got, which it reports after each chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunProgress {
    /// The chunks done so far, those the store already held included.
    pub done: usize,
    /// Every chunk of the world.
    pub total: usize,
    /// Of those done, the chunks the store already held, which this run did not generate.
    pub skipped: usize,
    /// How many chunks the runtime holds now, over all stages.
    pub held: usize,
}

impl Runtime {
    /// Generates `targets` over every chunk of the pack's bound ([`Runtime::bound_chunks`]), row
    /// by row, and keeps each chunk's product of each target in `store`, as the product's RON
    /// text under the stage's name, as soon as the chunk is done. A chunk whose every target the
    /// store already holds is skipped. After each chunk, `progress` is told how far the run has
    /// got, and the run stops if it breaks.
    ///
    /// Returns the progress when the run ended: `done == total` if it finished.
    ///
    /// # Errors
    /// [`StageError::Unbounded`] for a pack without a bound, what generating gives, and a
    /// [`StoreError`] from the store.
    pub fn run_world(
        &mut self,
        targets: &[&str],
        store: &mut dyn FrozenStore,
        mut progress: impl FnMut(RunProgress) -> ControlFlow<()>,
    ) -> Result<RunProgress, StageError> {
        let chunks = self.bound_chunks()?;
        let mut state = RunProgress {
            done: 0,
            total: chunks.len(),
            skipped: 0,
            held: self.held(),
        };
        for chunk in chunks {
            if !stored(store, targets, chunk)? {
                self.request(&[FocusPoint::new(chunk, 0)], targets)?;
                self.run_until_idle()?;
                for &target in targets {
                    let product = self
                        .product(target, chunk)
                        .expect("a target is generated over the chunk it was asked for");
                    let text = ron::to_string(product).expect("a product is plain data");
                    store.keep(target, chunk, text.into_bytes())?;
                }
            } else {
                state.skipped += 1;
            }
            state.done += 1;
            state.held = self.held();
            if progress(state).is_break() {
                break;
            }
        }
        Ok(state)
    }
}

/// Whether `store` holds every one of `targets` at `chunk`.
fn stored(
    store: &mut dyn FrozenStore,
    targets: &[&str],
    chunk: ChunkCoord,
) -> Result<bool, StoreError> {
    for target in targets {
        if store.fetch(target, chunk)?.is_none() {
            return Ok(false);
        }
    }
    Ok(true)
}
