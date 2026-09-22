//! The block solver: one workgroup per region, one dispatch per batch.

use crate::backend::{BufferUsage, ComputeBackend};
use crate::error::GpuError;
use crate::kernel::{ENTRY, KernelSpec, Params, STATS_WORDS, SolverConfig};
use std::collections::HashMap;
use std::sync::Arc;
use wfc_core::{
    BatchResult, Domains, JobId, RegionBatch, RegionShape, RegionStats, RegionStatus, Ruleset,
    Solver, SolverError,
};

/// Where a region's statistics record keeps its step count.
const STEPS: usize = 5;

/// Words of a `u32` buffer, as bytes.
const fn bytes(words: u32) -> u64 {
    words as u64 * 4
}

/// Solves regions on a compute backend, one workgroup each.
pub struct BlockSolver<B: ComputeBackend> {
    backend: B,
    ruleset: Arc<Ruleset>,
    config: SolverConfig,
    /// One kernel per batch capacity and region shape. Compiling one is far more expensive than a
    /// dispatch, so they are kept; [`BlockSolver::warm`] builds them before a run needs them.
    kernels: HashMap<(u32, RegionShape), Kernel<B>>,
    running: Option<InFlight<B>>,
    next_job: u64,
}

/// One compiled specialisation and the buffers it binds.
struct Kernel<B: ComputeBackend> {
    spec: KernelSpec,
    capacity: u32,
    pipeline: B::Pipeline,
    binding: B::Binding,
    init: B::Buffer,
    params: B::Buffer,
    ids: B::Buffer,
    seeds: B::Buffer,
    out: B::Buffer,
    stats: B::Buffer,
    out_readback: B::Buffer,
    stats_readback: B::Buffer,
    /// Held because the binding refers to them.
    _rules: B::Buffer,
    _weights: B::Buffer,
    _snaps: B::Buffer,
}

/// A dispatch waiting for the device.
struct InFlight<B: ComputeBackend> {
    job: JobId,
    key: (u32, RegionShape),
    submission: B::Submission,
    regions: u32,
}

impl<B: ComputeBackend> BlockSolver<B> {
    /// A solver for `ruleset` on `backend`.
    ///
    /// # Errors
    /// If the configuration asks for more invocations than the device allows.
    pub fn new(backend: B, ruleset: Arc<Ruleset>, config: SolverConfig) -> Result<Self, GpuError> {
        let limits = backend.limits();
        if config.invocations > limits.invocations_per_workgroup {
            return Err(GpuError::Invocations {
                wanted: config.invocations,
                available: limits.invocations_per_workgroup,
            });
        }
        Ok(Self {
            backend,
            ruleset,
            config,
            kernels: HashMap::new(),
            running: None,
            next_job: 1,
        })
    }

    /// The backend it runs on.
    #[must_use]
    pub const fn backend(&self) -> &B {
        &self.backend
    }

    /// How it runs a region.
    #[must_use]
    pub const fn config(&self) -> &SolverConfig {
        &self.config
    }

    /// The rule set it solves.
    #[must_use]
    pub fn ruleset(&self) -> &Arc<Ruleset> {
        &self.ruleset
    }

    /// Compiles the kernels a run will need, so the first batch does not pay for it.
    ///
    /// # Errors
    /// If a kernel does not compile or does not fit the device.
    pub fn warm(&mut self, shapes: &[(u32, RegionShape)]) -> Result<(), GpuError> {
        for &(regions, region) in shapes {
            self.ensure_kernel(capacity_for(regions), region)?;
        }
        Ok(())
    }

    /// Starts a batch with parameters of your own, for example propagation without any collapse.
    ///
    /// # Errors
    /// As [`Solver::start`].
    pub fn start_with(&mut self, batch: RegionBatch, params: Params) -> Result<JobId, SolverError> {
        if self.running.is_some() {
            return Err(SolverError::Busy);
        }
        if !batch.is_well_formed() {
            return Err(SolverError::Malformed(format!(
                "{} ids, {} seeds and {} cells for regions of {} cells",
                batch.ids.len(),
                batch.seeds.len(),
                batch.init.cells(),
                batch.region.cells()
            )));
        }
        if batch.is_empty() {
            return Err(SolverError::Malformed("an empty batch".to_owned()));
        }
        let regions = batch.len() as u32;
        if regions > self.max_batch() {
            return Err(SolverError::BatchTooLarge {
                regions: batch.len(),
                max: self.max_batch(),
            });
        }
        if batch.init.words_per_cell() != self.ruleset.words_per_cell() {
            return Err(SolverError::Malformed(format!(
                "domains of {} words per cell for a rule set of {}",
                batch.init.words_per_cell(),
                self.ruleset.words_per_cell()
            )));
        }
        let key = (capacity_for(regions), batch.region);
        self.ensure_kernel(key.0, key.1)
            .map_err(GpuError::into_solver)?;
        let submission = {
            let kernel = &self.kernels[&key];
            let backend = &self.backend;
            backend.write(&kernel.init, 0, batch.init.as_words());
            backend.write(&kernel.ids, 0, &batch.ids);
            backend.write(&kernel.seeds, 0, &batch.seeds);
            backend.write(&kernel.params, 0, bytemuck::cast_slice(&[params]));
            // A kernel's buffers outlive its batches, so a record left over from the last one would
            // pass for this one's if the device dropped the work. Every region writes a record with
            // at least one step, so a cleared record that stays cleared is a region never reported.
            backend.write(&kernel.stats, 0, &vec![0; (STATS_WORDS * regions) as usize]);
            backend
                .dispatch(
                    &kernel.pipeline,
                    &kernel.binding,
                    regions,
                    &[
                        (&kernel.out, &kernel.out_readback),
                        (&kernel.stats, &kernel.stats_readback),
                    ],
                )
                .map_err(|error| SolverError::Backend(error.to_string()))?
        };
        let job = JobId(self.next_job);
        self.next_job += 1;
        self.running = Some(InFlight {
            job,
            key,
            submission,
            regions,
        });
        Ok(job)
    }

    /// Whether a region of this shape fits the device: how much workgroup memory a region needs
    /// grows with its cells and its rule set, so a caller widening a halo has to ask.
    #[must_use]
    pub fn fits(&self, region: RegionShape) -> bool {
        KernelSpec::new(region, &self.ruleset, &self.config)
            .check(&self.backend.limits())
            .is_ok()
    }

    /// Whether the backend can answer [`Solver::poll`] without blocking. A caller that must stay
    /// responsive and gets `false` should drive the solver from a worker thread.
    #[must_use]
    pub fn can_poll(&self) -> bool {
        self.backend.can_poll()
    }

    /// Compiles the kernel for this capacity and shape unless it is already there.
    fn ensure_kernel(&mut self, capacity: u32, region: RegionShape) -> Result<(), GpuError> {
        if !self.kernels.contains_key(&(capacity, region)) {
            let kernel = self.build(capacity, region)?;
            self.kernels.insert((capacity, region), kernel);
        }
        Ok(())
    }

    fn build(&self, capacity: u32, region: RegionShape) -> Result<Kernel<B>, GpuError> {
        let spec = KernelSpec::new(region, &self.ruleset, &self.config);
        spec.check(&self.backend.limits())
            .map_err(|(needed, available)| GpuError::WorkgroupStorage { needed, available })?;
        let pipeline = self.backend.create_pipeline(&spec.wgsl(), ENTRY)?;
        let domains = spec.domain_words() * capacity;
        let rules = self
            .backend
            .create_buffer(bytes(spec.rule_words()), BufferUsage::Input)?;
        self.backend
            .write(&rules, 0, self.ruleset.table().as_words());
        let weights = self
            .backend
            .create_buffer(bytes(self.ruleset.num_tiles()), BufferUsage::Input)?;
        self.backend.write(&weights, 0, self.ruleset.weights());
        let init = self
            .backend
            .create_buffer(bytes(domains), BufferUsage::Input)?;
        let out = self
            .backend
            .create_buffer(bytes(domains), BufferUsage::Output)?;
        let stats = self
            .backend
            .create_buffer(bytes(STATS_WORDS * capacity), BufferUsage::Output)?;
        let params = self
            .backend
            .create_buffer(size_of::<Params>() as u64, BufferUsage::Uniform)?;
        let snaps = self.backend.create_buffer(
            bytes(spec.snapshot_words() * capacity),
            BufferUsage::Scratch,
        )?;
        let ids = self
            .backend
            .create_buffer(bytes(capacity), BufferUsage::Input)?;
        let seeds = self
            .backend
            .create_buffer(bytes(capacity), BufferUsage::Input)?;
        let out_readback = self
            .backend
            .create_buffer(bytes(domains), BufferUsage::Readback)?;
        let stats_readback = self
            .backend
            .create_buffer(bytes(STATS_WORDS * capacity), BufferUsage::Readback)?;
        // Binding order is the kernel's binding order.
        let binding = self.backend.create_binding(
            &pipeline,
            &[
                &rules, &init, &out, &stats, &params, &weights, &snaps, &ids, &seeds,
            ],
        )?;
        Ok(Kernel {
            spec,
            capacity,
            pipeline,
            binding,
            init,
            params,
            ids,
            seeds,
            out,
            stats,
            out_readback,
            stats_readback,
            _rules: rules,
            _weights: weights,
            _snaps: snaps,
        })
    }

    /// Reads a finished dispatch back.
    fn collect(&mut self, running: InFlight<B>) -> Result<BatchResult, SolverError> {
        let kernel = &self.kernels[&running.key];
        let mut stats = vec![0u32; (STATS_WORDS * kernel.capacity) as usize];
        let mut words = vec![0u32; (kernel.spec.domain_words() * kernel.capacity) as usize];
        let backend = &self.backend;
        backend
            .read(&kernel.stats_readback, &mut stats)
            .map_err(|error| SolverError::Backend(error.to_string()))?;
        backend
            .read(&kernel.out_readback, &mut words)
            .map_err(|error| SolverError::Backend(error.to_string()))?;
        let mut statuses = Vec::with_capacity(running.regions as usize);
        let mut per_region = Vec::with_capacity(running.regions as usize);
        for region in 0..running.regions {
            let record = &stats[(region * STATS_WORDS) as usize..][..STATS_WORDS as usize];
            if record[STEPS] == 0 {
                return Err(SolverError::NoReport { region });
            }
            statuses.push(match record[0] {
                0 => RegionStatus::Solved,
                1 => RegionStatus::Exhausted,
                2 => RegionStatus::StepCap,
                3 => RegionStatus::BorderContradiction,
                // The kernel restored a checkpoint holding an empty cell, which the rules cannot
                // cause: it is a bug in the checkpoint ring, so it fails the batch loudly.
                4 => return Err(SolverError::BadCheckpoint { region }),
                status => {
                    return Err(SolverError::Backend(format!(
                        "region {region} reported status {status}, which the kernel never writes"
                    )));
                }
            });
            per_region.push(RegionStats {
                sweeps: record[1],
                collapses: record[2],
                restarts: record[3],
                steps: record[STEPS],
                backtracks: record[6],
                tries: record[7],
                contradiction_cell: (record[4] != u32::MAX).then_some(record[4]),
            });
        }
        words.truncate((kernel.spec.domain_words() * running.regions) as usize);
        let domains = Domains::from_words(
            kernel.spec.region().cells() * running.regions,
            self.ruleset.words_per_cell(),
            words,
        )
        .map_err(|error| SolverError::Backend(error.to_string()))?;
        Ok(BatchResult {
            statuses,
            stats: per_region,
            domains,
        })
    }
}

impl<B: ComputeBackend> Solver for BlockSolver<B> {
    fn max_batch(&self) -> u32 {
        self.config.max_batch.min(self.backend.limits().workgroups)
    }

    fn accepts(&self, region: RegionShape) -> bool {
        self.fits(region)
    }

    fn start(&mut self, batch: RegionBatch) -> Result<JobId, SolverError> {
        let mut params = Params::solve(&self.config);
        if let Some(budget) = batch.budget {
            params.max_attempts = budget.max_attempts;
            params.max_steps = budget.max_steps;
        }
        self.start_with(batch, params)
    }

    fn poll(&mut self, job: JobId) -> Result<Option<BatchResult>, SolverError> {
        let running = match self.running.take() {
            Some(running) if running.job == job => running,
            Some(other) => {
                self.running = Some(other);
                return Err(SolverError::UnknownJob(job));
            }
            None => return Err(SolverError::UnknownJob(job)),
        };
        let done = self
            .backend
            .is_done(&running.submission)
            .map_err(|error| SolverError::Backend(error.to_string()))?;
        if !done {
            self.running = Some(running);
            return Ok(None);
        }
        self.collect(running).map(Some)
    }

    fn wait(&mut self, job: JobId) -> Result<BatchResult, SolverError> {
        let running = match self.running.take() {
            Some(running) if running.job == job => running,
            Some(other) => {
                self.running = Some(other);
                return Err(SolverError::UnknownJob(job));
            }
            None => return Err(SolverError::UnknownJob(job)),
        };
        self.backend
            .wait(&running.submission)
            .map_err(|error| SolverError::Backend(error.to_string()))?;
        self.collect(running)
    }
}

/// Batch capacities go in powers of two, so a run of varying batch sizes reuses a handful of
/// kernels instead of compiling one per size.
fn capacity_for(regions: u32) -> u32 {
    regions.max(1).next_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacities_are_powers_of_two() {
        assert_eq!(capacity_for(0), 1);
        assert_eq!(capacity_for(1), 1);
        assert_eq!(capacity_for(5), 8);
        assert_eq!(capacity_for(64), 64);
        assert_eq!(capacity_for(65), 128);
    }
}
