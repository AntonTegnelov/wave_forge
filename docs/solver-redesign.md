# Solver redesign (#7): what the measurements say, and the plan

This is the reasoning behind the performance work in
[#7](https://github.com/AntonTegnelov/wave_forge/issues/7). It records what we measured, what the
literature says, and which changes follow from both — in that order, because the measurement
overturned the assumption we started with.

See [performance.md](performance.md) for the priorities and method, and [constraints.md](constraints.md)
for the constraint machinery the solver has to keep supporting.

## What we measured

Baseline on the realistic workload (RTX 3070 through the container's translation layer, release):

| Workload | Cells | Tiles | Run | Cells/s |
|---|---|---|---|---|
| Permissive 2-tile 24³ | 13824 | 2 | 96.0 s | 144 |
| City 24×24×8 | 4608 | 81 | 45.3 s | 102 |
| City 48×48×10 | 23040 | 81 | 403.8 s | 57 |

Span breakdown of the city run: **propagation is 76%** (36.9 s over 15284 passes, 2.36 ms each),
grid transfers 6.7 s, entropy and selection 3.1 s together.

The obvious reading is "propagation is expensive, make the shader faster". That is wrong. A
micro-benchmark (`wfc-gpu/tests/propagation_bench.rs`) dispatches the same propagation kernel with
the per-pass readbacks removed:

| Variant | Per pass |
|---|---|
| 1-cell worklist, one submit per pass, no readbacks | 2.69 ms |
| 1-cell worklist, all 200 passes in one submit | 3.81 ms |
| **4608-cell worklist** (the whole grid), no readbacks | **3.13 ms** |

Three conclusions, in order of importance:

1. **A pass costs the same whether it processes 1 cell or 4608.** Sixteen percent more time for
   4608× the work. The cost is a fixed per-dispatch overhead, not the propagation itself.
2. **It is not the readbacks, and not submit overhead.** Removing both readbacks changed nothing
   (2.69 ms vs the 2.36 ms traced mean), and batching every dispatch into one command buffer made it
   *worse*. Kernel launches cost microseconds elsewhere; through dozen→D3D12 each compute pass costs
   milliseconds.
3. **We pay that fixed cost on nearly empty work.** In the traced run, 70% of passes carried ≤8
   cells and 23% were full-grid sweeps forced by the non-atomic shader's fixpoint confirmation.

So the solver spends its time setting up work rather than doing it. The transfer overhead that
dominates the toy two-tile benchmark is a second-order problem by comparison.

## What the literature says

Researched before designing (see [performance.md](performance.md) for why that is the rule here).
The sources agree with the measurement and warn about the obvious fixes:

- **Fixed costs dominate small frontiers.** Atos ([arXiv:2112.00132](https://arxiv.org/abs/2112.00132)):
  "fixed costs (the cost of the global synchronization barrier plus the kernel launch cost) dominate
  the overall processing cost; the GPU is spending a significant amount of time setting up or waiting
  for computation rather than performing it." Their fix — persistent kernels pulling from a device
  queue — gives 3.44× geomean on BFS.
- **A 3D grid is the worst case for frontier parallelism.** Gunrock reports 122516 MTEPS on a scale-free
  graph but 85 MTEPS on a road network; a voxel grid is mesh-like and high-diameter, so frontiers stay
  tiny and iterations stay many.
- **Device-resident CP solvers deliberately drop event-based propagation.** Turbo, a fully GPU-resident
  constraint solver (AAAI-26), uses "a propagation loop similar to AC1" and full recomputation instead
  of a worklist, because view-based propagators cause "uncoalesced memory accesses, load imbalance,
  thread divergence". RTAC ([arXiv:2407.11388](https://arxiv.org/abs/2407.11388)) measures the fixpoint
  needing only ~3.5–4.8 whole-network sweeps even at density 1.0, against tens of thousands of AC-3
  revisions.
- **Every published GPU WFC lost to the CPU.** cuWaveFunctionCollapse is 26–104× *slower* than its CPU
  baseline; a CMU parallel WFC found "the sequential queue-based algorithm outperformed all parallel
  implementations". Neither is a reason to avoid the GPU, but both are reasons not to expect a free win.
- **Temper the expectation.** Turbo, on an H100, is worse than OR-Tools on 58% of instances. Gent et al.:
  "GPUs are not a silver bullet, and direct ports of existing algorithms to a GPU architecture often
  perform poorly."
- **WGSL cannot express a persistent kernel.** The spec has only workgroup-scoped barriers and no
  device-side enqueue, and offers no forward-progress guarantee between workgroups. Hand-rolled global
  barriers are fragile even in CUDA (Xiao & Feng regress past ~18 blocks). So the achievable design is a
  *pre-recorded chain of dispatches* per submit, sized by `dispatch_workgroups_indirect` from a
  device-side counter — not one resident kernel.
- **Failures may matter as much as throughput.** Merrell measures WFC failing 98–100% of attempts at
  200², with lowest-entropy ordering implicated as the cause, while model synthesis succeeds in seconds.
  Our own constrained city ranges from 5 s to 136 s for the same input, and one 48×48×10 run burned
  ~11000 undos. Search cost is part of the performance problem, not separate from it.

## The plan

Ranked by expected win per unit of risk. Each step is measured against the stress suite before the
next is started; nothing here is committed to on theory alone.

1. **Stop paying the fixed cost per tiny pass.** Make propagation dispatch the whole grid (AC-1 style)
   or a large batch, and let a device-side worklist counter drive `dispatch_workgroups_indirect` so the
   host does not decide how big the next pass is. Expected: most of the 36.9 s, since a full-grid pass
   costs 16% more than a one-cell pass today. Risk: low; the kernel already handles a full sweep, and
   propagation is a monotone fixpoint, so extra work per pass cannot change the result.
2. **Keep the loop on the device between collapses.** Record K iterations of
   (entropy → select → collapse → propagate) into one command buffer, with an early-out flag so
   finished dispatches become no-ops, and read back only every K iterations or on a contradiction. This
   needs on-device selection and a stateless hash RNG (pcg3d) for the weighted choice. Expected: removes
   most of the remaining per-collapse round-trips. Risk: medium; needs the indirect-args ordering rules
   verified against our wgpu version.
3. **Make the shader's work cheaper only once the above lands.** Today's kernel recomputes
   `allowed_neighbour_mask` by looping over all tile pairs; AC-4 support counters (fast-wfc, marian42,
   DeBroglie) or bitplane/SoA possibility layout would cut that. The evidence is that SoA matters for
   memory-bound neighbour sweeps (5× on LBM propagate), but our passes are not yet memory-bound — they
   are dispatch-bound. Deferred deliberately.
4. **Attack search cost with the same rigour.** Measure how much of a run is redone work, then try, in
   order: nogood recording across restarts (Lecoutre et al.), weighting cells that repeatedly fail
   (wdeg), and Merrell's finding that scanline ordering can beat lowest-entropy at scale. Our
   conflict-directed backjumping already matches DeBroglie's `PatienceBackjumpPolicy` in spirit.
5. **Chunked streaming for the infinite-city milestone.** N-WFC's overlapping sub-grids with diagonal
   order, or Boris the Brave's "infinite modifying in blocks", give constant work per chunk and a
   fallback to known-good tiles on failure — which is how marian42 shipped an infinite city after
   abandoning exactly our history-undo scheme. The catch, stated by every block-based source: a
   constraint larger than a block (our connectivity constraint) cannot be enforced across blocks. That
   tension needs a decision before the streaming work, not during it.

**Explicitly not doing yet:** persistent kernels (not expressible in WGSL), Morton/space-filling
layouts (the one study that searched the space found no win on stencils), and SIMD micro-optimisation
(layout and dispatch dominate; the CPU-side bitset scan already captures most of the available win).

## How we will know it worked

The stress suite is the yardstick, run in release with the same three workloads. A change is kept when
it moves `cells_per_s` on the 81-variant city, not on the toy two-tile grid, and when the E2E and
constrained-city tests still pass. Every number in this document came from
`wfc-devtools/tests/stress.rs`, `WFC_TRACE_CHROME`, or `wfc-gpu/tests/propagation_bench.rs`, so each
claim can be re-measured after any change.
