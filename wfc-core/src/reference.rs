//! A sequential CPU solver, as ground truth and as the yardstick GPU numbers are printed against.
//!
//! It is deliberately plain: a stack of changed cells for propagation, a full scan for the
//! fewest-possibilities cell, and marian42's undo-doubling on contradiction. It is not a product
//! and not a fallback, which is why it is behind the `reference` feature and off by default. It
//! answers two questions: what fixpoint a correct propagator must reach, and what one CPU thread
//! achieves on a workload.

use crate::chunk::RegionShape;
use crate::domains::Domains;
use crate::hash::{choice_hash, choose_tile};
use crate::rules::{AXES, MAX_WORDS, Ruleset, TILES_PER_WORD, TileMask};
use crate::solver::{
    BatchResult, JobId, RegionBatch, RegionStats, RegionStatus, Solver, SolverError,
};
use std::collections::VecDeque;
use std::sync::Arc;

/// Limits that keep a pathological region from running forever.
#[derive(Clone, Copy, Debug)]
pub struct ReferenceConfig {
    /// Snapshots kept for undo.
    pub max_history: usize,
    /// Contradictions to recover from before giving up on a region.
    pub max_backtracks: u32,
    /// The deepest undo a repeated failure escalates to.
    pub max_undo: usize,
}

impl Default for ReferenceConfig {
    fn default() -> Self {
        Self {
            max_history: 2048,
            max_backtracks: 20_000,
            max_undo: 64,
        }
    }
}

/// A CPU solver over one rule set.
#[derive(Clone, Debug)]
pub struct ReferenceSolver {
    ruleset: Arc<Ruleset>,
    config: ReferenceConfig,
    next_job: u64,
    finished: Option<(JobId, BatchResult)>,
}

impl ReferenceSolver {
    /// A solver for `ruleset`.
    #[must_use]
    pub fn new(ruleset: Arc<Ruleset>) -> Self {
        Self {
            ruleset,
            config: ReferenceConfig::default(),
            next_job: 1,
            finished: None,
        }
    }

    /// A solver with other limits.
    #[must_use]
    pub fn with_config(mut self, config: ReferenceConfig) -> Self {
        self.config = config;
        self
    }

    /// The rule set it solves.
    #[must_use]
    pub fn ruleset(&self) -> &Ruleset {
        &self.ruleset
    }

    /// Narrows every cell of `domains` until no neighbour can narrow it further, the arc-consistent
    /// fixpoint. Only cells on `stack` and what they reach are visited.
    ///
    /// It works on the cells' words rather than on [`TileMask`], which is a fixed eight words wide:
    /// a rule set of 81 tiles needs three, and touching the other five costs more than the
    /// propagation itself.
    ///
    /// # Errors
    /// The index of a cell left with no tile.
    pub fn propagate(
        &self,
        shape: RegionShape,
        domains: &mut Domains,
        stack: &mut Vec<u32>,
    ) -> Result<(), u32> {
        let table = self.ruleset.table();
        let words = domains.words_per_cell() as usize;
        let mut current = [0u32; MAX_WORDS];
        let mut allowed = [0u32; MAX_WORDS];
        while let Some(cell) = stack.pop() {
            current[..words].copy_from_slice(domains.cell(cell));
            for axis in 0..AXES {
                let Some(neighbour) = neighbour(shape, cell, axis) else {
                    continue;
                };
                allowed[..words].fill(0);
                for (word, &bits) in current[..words].iter().enumerate() {
                    let mut bits = bits;
                    while bits != 0 {
                        let tile = word as u32 * TILES_PER_WORD + bits.trailing_zeros();
                        bits &= bits - 1;
                        for (slot, word) in allowed[..words].iter_mut().zip(table.row(axis, tile)) {
                            *slot |= word;
                        }
                    }
                }
                let mut changed = false;
                let mut remaining = 0;
                for (word, mask) in domains
                    .cell_mut(neighbour)
                    .iter_mut()
                    .zip(&allowed[..words])
                {
                    let narrowed = *word & mask;
                    changed |= narrowed != *word;
                    *word = narrowed;
                    remaining |= narrowed;
                }
                if changed {
                    if remaining == 0 {
                        return Err(neighbour);
                    }
                    stack.push(neighbour);
                }
            }
        }
        Ok(())
    }

    /// Solves one region, choosing tiles as the GPU kernel does: the fewest-possibilities cell
    /// first, lowest index on ties, and a weighted pick from a hash of `(seed, chunk, attempt,
    /// step)`.
    #[must_use]
    pub fn solve_region(
        &self,
        shape: RegionShape,
        init: &Domains,
        chunk_id: u32,
        seed: u32,
    ) -> (Domains, RegionStatus, RegionStats) {
        let mut domains = init.clone();
        let mut stats = RegionStats::default();
        let mut stack: Vec<u32> = (0..domains.cells()).collect();
        if let Err(cell) = self.propagate(shape, &mut domains, &mut stack) {
            stats.contradiction_cell = Some(cell);
            stats.tries = 1;
            return (domains, RegionStatus::BorderContradiction, stats);
        }
        let mut history: VecDeque<(Domains, u32, u32)> = VecDeque::new();
        let mut undo = 1usize;
        while let Some(cell) = fewest_possibilities(&domains) {
            let hash = choice_hash(seed, chunk_id, stats.tries, stats.collapses);
            let tile = choose_tile(self.ruleset.weights(), domains.mask(cell), hash)
                .expect("an undecided cell has a tile to choose");
            if history.len() == self.config.max_history {
                history.pop_front();
            }
            history.push_back((domains.clone(), cell, tile));
            domains.set(cell, TileMask::single(tile));
            stats.collapses += 1;
            stats.steps += 1;
            stack.push(cell);
            let mut failure = self.propagate(shape, &mut domains, &mut stack).err();
            while let Some(cell) = failure {
                stats.tries += 1;
                stats.backtracks += 1;
                stats.contradiction_cell = Some(cell);
                if stats.backtracks >= self.config.max_backtracks {
                    return (domains, RegionStatus::Exhausted, stats);
                }
                let mut restored = None;
                for _ in 0..undo {
                    restored = history.pop_back().or(restored);
                }
                let Some((snapshot, banned_cell, banned_tile)) = restored else {
                    return (domains, RegionStatus::Exhausted, stats);
                };
                undo = (undo * 2).min(self.config.max_undo);
                domains = snapshot;
                let narrowed = {
                    let mut mask = domains.mask(banned_cell);
                    mask.remove(banned_tile);
                    domains.set(banned_cell, mask);
                    mask
                };
                stack.clear();
                stats.steps += 1;
                // Every tile of that cell is now ruled out, so the mistake lies further back.
                if narrowed.is_empty() {
                    failure = Some(banned_cell);
                    continue;
                }
                stack.push(banned_cell);
                failure = self.propagate(shape, &mut domains, &mut stack).err();
            }
            undo = 1;
        }
        (domains, RegionStatus::Solved, stats)
    }
}

impl Solver for ReferenceSolver {
    fn max_batch(&self) -> u32 {
        u32::MAX
    }

    fn start(&mut self, batch: RegionBatch) -> Result<JobId, SolverError> {
        if self.finished.is_some() {
            return Err(SolverError::Busy);
        }
        if !batch.is_well_formed() {
            return Err(SolverError::Malformed(format!(
                "{} ids, {} seeds, {} cells for {} regions of {} cells",
                batch.ids.len(),
                batch.seeds.len(),
                batch.init.cells(),
                batch.ids.len(),
                batch.region.cells()
            )));
        }
        let cells = batch.region.cells();
        let mut statuses = Vec::with_capacity(batch.len());
        let mut stats = Vec::with_capacity(batch.len());
        let mut domains = Domains::from_words(0, batch.init.words_per_cell(), Vec::new())
            .expect("an empty batch result");
        for (region, (&id, &seed)) in batch.ids.iter().zip(&batch.seeds).enumerate() {
            let init = batch.init.chunk(region as u32, cells);
            let (solved, status, region_stats) = self.solve_region(batch.region, &init, id, seed);
            domains.append(&solved);
            statuses.push(status);
            stats.push(region_stats);
        }
        let job = JobId(self.next_job);
        self.next_job += 1;
        self.finished = Some((
            job,
            BatchResult {
                statuses,
                stats,
                domains,
            },
        ));
        Ok(job)
    }

    fn poll(&mut self, job: JobId) -> Result<Option<BatchResult>, SolverError> {
        match self.finished.take() {
            Some((finished, result)) if finished == job => Ok(Some(result)),
            Some((finished, result)) => {
                self.finished = Some((finished, result));
                Err(SolverError::UnknownJob(job))
            }
            None => Err(SolverError::UnknownJob(job)),
        }
    }

    fn wait(&mut self, job: JobId) -> Result<BatchResult, SolverError> {
        self.poll(job)?.ok_or(SolverError::UnknownJob(job))
    }
}

/// The cell with the fewest remaining tiles, lowest index on ties, or `None` when every cell is
/// decided.
fn fewest_possibilities(domains: &Domains) -> Option<u32> {
    (0..domains.cells())
        .map(|cell| (domains.count(cell), cell))
        .filter(|(count, _)| *count > 1)
        .min()
        .map(|(_, cell)| cell)
}

/// The cell along `axis` of `cell`, or `None` at the region's edge.
fn neighbour(shape: RegionShape, cell: u32, axis: usize) -> Option<u32> {
    let layer = shape.x * shape.y;
    let (x, y, z) = (cell % shape.x, (cell / shape.x) % shape.y, cell / layer);
    match axis {
        0 if x + 1 < shape.x => Some(cell + 1),
        1 if x > 0 => Some(cell - 1),
        2 if y + 1 < shape.y => Some(cell + shape.x),
        3 if y > 0 => Some(cell - shape.x),
        4 if z + 1 < shape.z => Some(cell + layer),
        5 if z > 0 => Some(cell - layer),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkShape;
    use wfc_rules::AdjacencyRules;

    const SHAPE: ChunkShape = ChunkShape { x: 4, y: 4, z: 2 };

    /// Two tiles that may only touch themselves, so a region is all one tile or the other.
    fn stripes() -> Arc<Ruleset> {
        let tuples: Vec<(usize, usize, usize)> = (0..AXES)
            .flat_map(|axis| (0..2).map(move |tile| (axis, tile, tile)))
            .collect();
        Arc::new(
            Ruleset::new(
                &AdjacencyRules::from_allowed_tuples(2, AXES, tuples),
                &[1.0, 1.0],
            )
            .expect("stripe rules"),
        )
    }

    fn region() -> RegionShape {
        SHAPE.region([0, 0, 0])
    }

    #[test]
    fn propagation_reaches_the_fixpoint_a_decided_cell_forces() {
        let solver = ReferenceSolver::new(stripes());
        let mut domains = Domains::filled(region().cells(), 2);
        domains.set(0, TileMask::single(1));
        let mut stack = vec![0];

        solver
            .propagate(region(), &mut domains, &mut stack)
            .expect("consistent");

        assert!((0..domains.cells()).all(|cell| domains.decided(cell) == Some(1)));
    }

    #[test]
    fn propagation_reports_the_cell_it_empties() {
        let solver = ReferenceSolver::new(stripes());
        let mut domains = Domains::filled(region().cells(), 2);
        domains.set(0, TileMask::single(0));
        domains.set(1, TileMask::single(1));
        let mut stack = vec![0, 1];

        let emptied = solver.propagate(region(), &mut domains, &mut stack);

        assert!(
            emptied.is_err(),
            "neighbouring cells cannot hold different stripes"
        );
    }

    #[test]
    fn a_region_solves_and_the_same_seed_repeats_it() {
        let solver = ReferenceSolver::new(stripes());
        let init = Domains::filled(region().cells(), 2);

        let (first, status, stats) = solver.solve_region(region(), &init, 3, 7);
        let (again, _, _) = solver.solve_region(region(), &init, 3, 7);
        let (elsewhere, _, _) = solver.solve_region(region(), &init, 4, 7);

        assert_eq!(status, RegionStatus::Solved);
        assert!(first.all_decided());
        assert_eq!(stats.collapses, 1, "one choice decides the whole region");
        assert_eq!(first, again);
        assert_eq!(
            [first.decided(0), elsewhere.decided(0)]
                .iter()
                .filter(|t| **t == Some(1))
                .count(),
            1,
            "these two chunk ids pick different stripes"
        );
    }

    #[test]
    fn an_impossible_border_is_reported_rather_than_searched() {
        let solver = ReferenceSolver::new(stripes());
        let mut init = Domains::filled(region().cells(), 2);
        init.set(0, TileMask::single(0));
        init.set(1, TileMask::single(1));

        let (_, status, stats) = solver.solve_region(region(), &init, 1, 1);

        assert_eq!(status, RegionStatus::BorderContradiction);
        assert!(stats.contradiction_cell.is_some());
        assert_eq!(stats.collapses, 0);
    }

    #[test]
    fn a_batch_solves_every_region_in_order() {
        let mut solver = ReferenceSolver::new(stripes());
        let cells = region().cells();
        let mut init = Domains::filled(cells, 2);
        init.append(&Domains::filled(cells, 2));
        let batch = RegionBatch {
            region: region(),
            ids: vec![3, 4],
            seeds: vec![7, 7],
            init,
        };

        let job = solver.start(batch).expect("a well-formed batch");
        let result = solver
            .wait(job)
            .expect("the reference finishes immediately");

        assert_eq!(result.statuses, vec![RegionStatus::Solved; 2]);
        assert_eq!(result.domains.cells(), cells * 2);
        assert!(result.region(0, cells).all_decided());
        assert!(matches!(solver.poll(job), Err(SolverError::UnknownJob(_))));
    }

    #[test]
    fn a_malformed_batch_is_rejected() {
        let mut solver = ReferenceSolver::new(stripes());
        let batch = RegionBatch {
            region: region(),
            ids: vec![1, 2],
            seeds: vec![1],
            init: Domains::filled(region().cells(), 2),
        };

        assert!(matches!(
            solver.start(batch),
            Err(SolverError::Malformed(_))
        ));
    }
}
