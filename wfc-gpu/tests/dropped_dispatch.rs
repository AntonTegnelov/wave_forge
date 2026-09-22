//! A dispatch the device reports as finished but never ran must not read as solved regions.
//!
//! A driver can end a workgroup early without reporting an error (Mesa 22.3's lavapipe does, see
//! docs/testing.md). What the host then reads back is whatever the buffers held before: zeros on a
//! fresh kernel, the previous batch on a reused one. These tests run the solver on a backend that
//! performs the copies but never the kernel, with buffers that start out holding a plausible
//! result, which is both cases at once. No GPU is needed.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use wfc_core::rules::AXES;
use wfc_core::{ChunkShape, Domains, RegionBatch, Ruleset, Solver, SolverError};
use wfc_gpu::block_solver::BlockSolver;
use wfc_gpu::kernel::{STATS_WORDS, SolverConfig};
use wfc_gpu::{BackendError, BackendLimits, BufferUsage, ComputeBackend};
use wfc_rules::AdjacencyRules;

/// What a region's statistics record looks like after a solve that succeeded in one step: the
/// words a stale buffer could plausibly hold.
const SOLVED_RECORD: [u32; STATS_WORDS as usize] = [0, 1, 1, 0, u32::MAX, 1, 0, 0];

/// A backend whose device finishes every dispatch without running the kernel.
struct NothingRuns;

type Buffer = Rc<RefCell<Vec<u32>>>;

impl ComputeBackend for NothingRuns {
    type Pipeline = ();
    type Binding = ();
    type Buffer = Buffer;
    type Submission = ();

    fn limits(&self) -> BackendLimits {
        BackendLimits {
            workgroup_storage: 32_768,
            invocations_per_workgroup: 256,
            workgroups: 65_535,
        }
    }

    fn create_pipeline(&self, _wgsl: &str, _entry: &str) -> Result<(), BackendError> {
        Ok(())
    }

    fn create_buffer(&self, bytes: u64, _usage: BufferUsage) -> Result<Buffer, BackendError> {
        let words = usize::try_from(bytes / 4).expect("test buffers are small");
        let stale = SOLVED_RECORD.iter().copied().cycle().take(words).collect();
        Ok(Rc::new(RefCell::new(stale)))
    }

    fn create_binding(&self, _pipeline: &(), _buffers: &[&Buffer]) -> Result<(), BackendError> {
        Ok(())
    }

    fn write(&self, buffer: &Buffer, offset: u64, words: &[u32]) {
        let start = usize::try_from(offset / 4).expect("test offsets are small");
        buffer.borrow_mut()[start..start + words.len()].copy_from_slice(words);
    }

    fn dispatch(
        &self,
        _pipeline: &(),
        _binding: &(),
        _workgroups: u32,
        readbacks: &[(&Buffer, &Buffer)],
    ) -> Result<(), BackendError> {
        for (from, to) in readbacks {
            to.borrow_mut().copy_from_slice(&from.borrow());
        }
        Ok(())
    }

    fn is_done(&self, _submission: &()) -> Result<bool, BackendError> {
        Ok(true)
    }

    fn wait(&self, _submission: &()) -> Result<(), BackendError> {
        Ok(())
    }

    fn read(&self, readback: &Buffer, words: &mut [u32]) -> Result<(), BackendError> {
        words.copy_from_slice(&readback.borrow());
        Ok(())
    }
}

/// Two tiles that may sit anywhere, on a 2x2x1 region.
fn solver_and_batch() -> (BlockSolver<NothingRuns>, RegionBatch) {
    let rules = AdjacencyRules::from_allowed_tuples(
        2,
        AXES,
        (0..AXES).flat_map(|axis| (0..2).flat_map(move |a| (0..2).map(move |b| (axis, a, b)))),
    );
    let ruleset = Ruleset::new(&rules, &[1.0, 1.0]).expect("valid weights");
    let region = ChunkShape { x: 2, y: 2, z: 1 }.region([0, 0, 0]);
    let batch = RegionBatch {
        region,
        ids: vec![0, 1],
        seeds: vec![7, 7],
        init: Domains::filled(region.cells() * 2, 2),
        budget: None,
    };
    let solver = BlockSolver::new(NothingRuns, Arc::new(ruleset), SolverConfig::default())
        .expect("the fake device is large enough");
    (solver, batch)
}

#[test]
fn a_dispatch_that_never_ran_is_an_error_when_waited_on() {
    let (mut solver, batch) = solver_and_batch();
    let job = solver.start(batch).expect("a well-formed batch");

    let result = solver.wait(job);

    assert!(
        matches!(result, Err(SolverError::NoReport { region: 0 })),
        "a region the kernel never reported on must not be read as a result: {result:?}"
    );
}

#[test]
fn a_dispatch_that_never_ran_is_an_error_when_polled() {
    let (mut solver, batch) = solver_and_batch();
    let job = solver.start(batch).expect("a well-formed batch");

    let result = solver.poll(job);

    assert!(
        matches!(result, Err(SolverError::NoReport { region: 0 })),
        "a region the kernel never reported on must not be read as a result: {result:?}"
    );
}
