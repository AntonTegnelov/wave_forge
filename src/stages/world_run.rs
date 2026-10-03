//! A whole finite world generated ahead of time as one run (docs/product/user-stories.md, M1).
//!
//! A maximal world is not generated while it is played: it is generated whole, before play, and
//! played from what that wrote. [`Runtime::run_world`] generates the target stages over every
//! chunk of the pack's bound, a block of chunks as wide as its largest regions at a time, and
//! hands each chunk's products to a store as soon as its block is done, so the runtime holds only
//! what one block reads while the world is as large as the store. A chunk whose products the store already holds is skipped, so a run that
//! was stopped, or cut short by a crash, picks up where it left off; and since a product is a
//! function of the pack, the seed, the facts and the edits, the store ends up holding the same
//! bytes whether the run went at once or in pieces.

use super::codec;
use super::runtime::{Runtime, StageError, StageTiming};
use crate::FocusPoint;
use crate::frozen::{FrozenStore, StoreError};
use std::collections::BTreeMap;
use std::ops::ControlFlow;
use std::time::{Duration, Instant};
use wfc_core::ChunkCoord;

/// The layer of a store a world run keeps each chunk's targets under, together, so a world is a
/// file per chunk rather than one per chunk and target; no stage may take its name.
pub(crate) const RUN_LAYER: &str = "world run";

/// Products a block generates between looks at the time.
const BLOCK_STEP: usize = 64;
/// How often a block being generated reports, so a run can be stopped part way through one.
const REPORT_EVERY: Duration = Duration::from_secs(1);

/// How far a run has got, which it reports after each chunk and while a block generates.
#[derive(Clone, Debug, PartialEq)]
pub struct RunProgress {
    /// The chunks done so far, those the store already held included.
    pub done: usize,
    /// Every chunk of the world.
    pub total: usize,
    /// Of those done, the chunks the store already held, which this run did not generate.
    pub skipped: usize,
    /// How many chunks the runtime holds now, over all stages.
    pub held: usize,
    /// What each stage of the pack has generated so far and what that cost, by name in the pack's
    /// order, as [`Runtime::timings`] gives it: the targets, and the stages they read.
    pub stages: Vec<(String, StageTiming)>,
}

impl Runtime {
    /// Generates `targets` over every chunk of the pack's bound ([`Runtime::bound_chunks`]), a
    /// block at a time: square blocks as wide as the pack's largest regions and aligned with them,
    /// row by row, each asked for whole, so a region's inputs are generated once. It keeps each
    /// chunk's product of each target in `store`, as a frozen chunk is kept, under the stage's
    /// name, as soon as its block is done. A chunk whose every target the store already holds is
    /// skipped. After each chunk, and every second while a block generates, `progress` is told
    /// how far the run has got, and the run stops if it breaks: a block of a large world takes
    /// minutes.
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
        if let Some(&first) = chunks.first() {
            refuse_older_layout(store, targets, first)?;
        }
        let digest = self.content_digest();
        let mut state = RunProgress {
            done: 0,
            total: chunks.len(),
            skipped: 0,
            held: self.held(),
            stages: self.timings(),
        };
        // The world is generated a block at a time, as wide as its largest regions and aligned
        // with them, so each region's inputs are generated once rather than once for each row of
        // chunks that crosses it.
        let side = i32::try_from(self.pack().largest_region()).expect("a region fits i32");
        let block_of = |chunk: &ChunkCoord| (chunk.y.div_euclid(side), chunk.x.div_euclid(side));
        let mut blocks: BTreeMap<(i32, i32), Vec<ChunkCoord>> = BTreeMap::new();
        for chunk in chunks {
            blocks.entry(block_of(&chunk)).or_default().push(chunk);
        }
        for block in blocks.into_values() {
            let mut due = Vec::with_capacity(block.len());
            // The other targets of a chunk a run of the same digest kept, read once, to keep.
            let mut others: BTreeMap<ChunkCoord, Vec<(String, Vec<u8>)>> = BTreeMap::new();
            for &chunk in &block {
                match held(store, targets, chunk, digest)? {
                    Held::All => {}
                    Held::Others(products) => {
                        others.insert(chunk, products);
                        due.push(chunk);
                    }
                    Held::Nothing => due.push(chunk),
                }
            }
            if !due.is_empty() {
                let focus: Vec<FocusPoint> =
                    due.iter().map(|&chunk| FocusPoint::new(chunk, 0)).collect();
                self.request(&focus, targets)?;
                let mut reported = Instant::now();
                while !self.is_idle() {
                    // Nothing generated means what is left waits for towns.
                    if self.step(BLOCK_STEP)?.is_empty() {
                        self.wait_for_towns(REPORT_EVERY)?;
                    }
                    if reported.elapsed() >= REPORT_EVERY {
                        reported = Instant::now();
                        state.held = self.held();
                        state.stages = self.timings();
                        if progress(state.clone()).is_break() {
                            return Ok(state);
                        }
                    }
                }
            }
            for chunk in block {
                if due.contains(&chunk) {
                    let mut products = others.remove(&chunk).unwrap_or_default();
                    for &target in targets {
                        let product = self
                            .product(target, chunk)
                            .expect("a target is generated over the chunk it was asked for");
                        products.push((target.to_owned(), codec::encode(product)));
                    }
                    // By name, so the bytes do not depend on the order the targets were listed in.
                    products.sort_by(|a, b| a.0.cmp(&b.0));
                    let entry = codec::Entry { digest, products };
                    store.keep(RUN_LAYER, chunk, codec::encode_chunk(&entry))?;
                } else {
                    state.skipped += 1;
                }
                state.done += 1;
                state.held = self.held();
                state.stages = self.timings();
                if progress(state.clone()).is_break() {
                    return Ok(state);
                }
            }
        }
        Ok(state)
    }
}

/// What a store holds of a world run's targets at one chunk ([`held`]).
enum Held {
    /// Every target, kept by a run of the same digest.
    All,
    /// Not every target, but the other targets a run of the same digest kept there, to keep; it
    /// may hold none.
    Others(Vec<(String, Vec<u8>)>),
    /// No entry, or one a run of another digest kept, whose products other inputs decided.
    Nothing,
}

/// What `store` holds of `targets` at `chunk`, for a run of `digest`.
fn held(
    store: &mut dyn FrozenStore,
    targets: &[&str],
    chunk: ChunkCoord,
    digest: u64,
) -> Result<Held, StoreError> {
    if !store.holds(RUN_LAYER, chunk)? {
        return Ok(Held::Nothing);
    }
    Ok(match kept(store, chunk)? {
        Some(entry) if entry.digest == digest => {
            if targets
                .iter()
                .all(|target| entry.products.iter().any(|(name, _)| name == target))
            {
                Held::All
            } else {
                Held::Others(
                    entry
                        .products
                        .into_iter()
                        .filter(|(name, _)| !targets.contains(&name.as_str()))
                        .collect(),
                )
            }
        }
        _ => Held::Nothing,
    })
}

/// What `store` holds for `chunk` as a run kept it; none if it holds no entry.
///
/// # Errors
/// If the store fails, or its entry for the chunk is not one a run kept.
pub(crate) fn kept(
    store: &mut dyn FrozenStore,
    chunk: ChunkCoord,
) -> Result<Option<codec::Entry>, StoreError> {
    let Some(bytes) = store.fetch(RUN_LAYER, chunk)? else {
        return Ok(None);
    };
    codec::decode_chunk(&bytes).map(Some).map_err(|error| {
        StoreError(format!(
            "the store's chunk {chunk:?} is not one a world run kept: {error}"
        ))
    })
}

/// Refuses a store an older Wave Forge wrote a world run into, a file per target, which would
/// otherwise look empty: its chunks would be generated again beside the old files.
///
/// # Errors
/// A [`StoreError`] saying so if `chunk` has a target of its own but no world run's entry.
pub(crate) fn refuse_older_layout(
    store: &mut dyn FrozenStore,
    targets: &[&str],
    chunk: ChunkCoord,
) -> Result<(), StoreError> {
    if store.holds(RUN_LAYER, chunk)? {
        return Ok(());
    }
    for target in targets {
        if store.holds(target, chunk)? {
            return Err(StoreError(format!(
                "the store holds a world run an older Wave Forge kept, a file per target \
                 ({target:?} of chunk {chunk:?}); run the world into an empty store"
            )));
        }
    }
    Ok(())
}
