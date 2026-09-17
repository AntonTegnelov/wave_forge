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

1. ~~**Stop paying the fixed cost per tiny pass** by sweeping the whole grid instead.~~ **Tried, and it
   made the realistic workload slower** — see "What sweeping actually did" below. The dispatch cost is
   real, but enlarging each pass is the wrong way to amortise it.
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

## What sweeping actually did

The first change followed straight from the micro-benchmark: if a pass costs the same for 1 cell as
for 4608, then dispatch the whole grid whenever the worklist is small. Measured on the stress suite:

| Sweep rule | City 24×24×8 | Permissive 24³ |
|---|---|---|
| None (baseline) | 45.3 s (102 cells/s) | 96.0 s (144 cells/s) |
| Worklist < ¼ of the grid | 53.0 s (87) | 81.3 s (170) |
| Tile-visit budget (sweep under ~400 cells at 81 variants) | 57.6 s (80) | 76.9 s (180) |

**It helps the two-tile grid by 15–20% and costs the 81-variant city 17–27%.** Correctness was
unaffected throughout. The reason the micro-benchmark misled us: it measured a freshly constrained
grid, where most cells hold few possibilities. A swept cell is not free once tile count is high,
because the shader unions the allowed neighbours of *every tile still possible* in that cell — a
4608-cell sweep is about 373000 tile-visits at 81 variants against 9000 at two. Lowering the threshold
did not rescue it, because the city's cheap passes (70% carry ≤8 cells) are exactly the ones that then
trigger a sweep.

Reverted. The lesson is kept: **the fixed cost per dispatch is real, but it must be amortised over more
*collapses*, not over more cells per pass.** That is step 2, and it is now the first thing to build.
Note also that the city baseline itself varies (45.3 s and 48.7 s on two runs) because backtracking
counts differ, so any future change on this workload needs repeated runs, not one sample.

## Correction: it is the shader's per-cell work, not the dispatch

The section above (and an earlier commit) claimed the cost was a fixed ~2.5 ms per dispatch. **That was
a measurement artifact.** The GPU boosts its clocks under load, and the first measurement in each
benchmark ran cold. Re-measured with a warm-up, interleaved variants and medians:

| Measurement (warm) | Cost |
|---|---|
| Trivial dispatch (one workgroup, no real work) | 0.099 ms |
| 256 trivial steps as 256 dispatches vs one looping dispatch | 18x cheaper in one dispatch |
| Same, with storage reads per step | 13x cheaper in one dispatch |
| Propagation pass, **1-cell** worklist | 2.271 ms |
| Propagation pass, **4608-cell** worklist (fresh grid) | 3.094 ms |
| Propagation pass, **4608-cell** worklist, every cell collapsed | **0.605 ms** |

Read together these say something quite different from "dispatches are expensive":

1. **A dispatch costs 0.099 ms**, so the 22332 dispatches of a city solve are ~2 s of the ~48 s run,
   not all of it.
2. **A pass costs what its cells cost.** The same full-grid pass is 5x cheaper when every cell is
   collapsed (0.605 ms vs 3.094 ms). The work, not the launch, is the price.
3. **One cell can cost as much as the whole grid** (2.271 ms vs 3.094 ms) because the grid's cells run
   in parallel while a single cell is one thread. So tiny worklists are slow not because the dispatch
   is expensive but because they leave the GPU idle while one thread grinds.

The grinding is `compute_allowed_neighbor_mask`: for every tile still possible in a cell it loops over
*every* tile testing `check_rule`, i.e. about `num_tiles^2` bit tests per axis — 81 x 81 x 6 ~ 39000
for one uncollapsed cell of the city set, each with a division and a modulo. The adjacency table is
already a bitset; it is simply not stored so that a row can be read as words.

**The fix is algorithmic and small:** store the table row-aligned, one `ceil(num_tiles/32)`-word mask
per `(axis, tile)`, and union those masks for the tiles still possible. That replaces `num_tiles` bit
tests per possible tile with 3 word-ORs — roughly 27x less inner-loop work at 81 variants, more at 256.

Block-local solving (one workgroup looping over a block in workgroup memory) remains interesting for
*streaming*, and the 13-18x result shows it is viable, but it is no longer the performance fix.

Two consequences worth stating, because they reverse earlier conclusions:

- **There is no meaningful "dispatch floor".** At 0.099 ms, the three dispatches per collapse across
  4608 cells cost about 1.4 s of a ~48 s run. Restructuring the loop to fit more collapses per
  dispatch is a streaming feature, not a performance fix.
- **The atomics on the possibility array turned out to be load-bearing, and this passage was wrong.**
  It argued that because propagation only clears bits, racing threads converge and the atomics could be
  dropped. That is true of an atomic AND, but the shader was doing a *load, modify, store*: between the
  load and the store another thread's restriction can be read, overwritten and lost. That is precisely
  why the host re-ran propagation over every cell after each collapse. Replacing the sequence with
  `atomicAnd` made those sweeps unnecessary and took the city run from 37.0 s to 25.9 s. Kept here as a
  correction rather than deleted, because the faulty step was "monotone writes converge" — true for the
  values, false for a read-modify-write.

It is also worth recording the architecture that the literature actually proves out, as a named
alternative rather than an assumption: the one system in this survey that beat a competition-winning
parallel SAT baseline (GPUShareSat) keeps **every dependent decision on the CPU** and uses the GPU as
an asynchronous bulk service the CPU never blocks on. Our solver instead blocks on the device at every
dependent step. Given that every published GPU WFC lost to its CPU baseline, "CPU owns the search, GPU
does batched off-critical-path work" deserves to be costed rather than dismissed.

## Where the round-trips go, and the leads for a GPU-shaped solver

### The round-trip audit

Read from the code at commit e28ab9d (the run loop in `wfc-gpu/src/gpu/accelerator.rs`, with the
default coordinator and `DirectPropagationStrategy`). Every blocking point is a
`device.poll(wait_indefinitely)`, which drains the whole queue rather than waiting for one submission.

| Site | Blocking drains | Read back | Needed on every collapse? |
|---|---|---|---|
| Entropy pass (`entropy/calculator.rs`) | 0 (submit only) | nothing | no: it could be fused with the last propagation sweep |
| Cell selection (`calculator.rs`, `buffers/mod.rs`) | 1 | 8 bytes (packed min key) | no: the choice can be made on the device |
| Each propagation pass (`propagator/direct_strategy.rs`) | 2, plus 2 new staging buffers | contradiction flag and worklist count, 4 bytes each | no: `worklist_count_buf` already has `INDIRECT` usage but nothing dispatches from it |
| Grid download (`gpu/sync.rs`) | 1 | the whole grid, unpacked one bit at a time | no: only needed at the end or on a contradiction |
| History, progress and download copies of `PossibilityGrid` | CPU | one heap allocation per cell, three times | no: a decision log with periodic checkpoints carries the same information |

That is **2 + 2·P drains per collapse**, where P is the number of propagation passes (about 1.8 on the
city, so about 5.7 drains), and three full grid clones. None of them is required by the algorithm: the
CPU only needs to hear "finished" or "contradiction at cell c". A 24x24x8 city trace at the same
commit (seed 1, release, RTX 3070 through dozen, one run, cold device, so indicative rather than a
median) spent 16.2 s propagating, 4.3 s downloading and 3.0 s in entropy and selection out of 25.8 s,
about 7.4 ms per collapse.

### The CPU reference

`wfc_devtools::reference` (timed by `wfc-devtools/tests/cpu_reference.rs`) is a deliberately plain single-threaded solver on the same rules:
two `u64` words per cell, a stack for propagation, a full scan for selection, marian42's undo-doubling.
It is a yardstick, not a product: a GPU design that cannot beat one CPU thread at chunk latency is not
worth shipping. At e28ab9d on a Ryzen 9 5900X (release, eight seeds, one run each, no warm-up beyond
the previous seed) it solves an 8x8x8 chunk in 2.8 to 4.6 ms and a 24x24x8 grid in 0.14 to 0.16 s on
the six seeds that finish; see [solver-fit.md](solver-fit.md) for the table. Its naive undo thrashes on
two of eight 24x24x8 seeds and four of eight 48x48x10 seeds, which is a statement about that undo
policy, not about CPUs.

So on this build the GPU loop is roughly 150 times slower than one CPU thread on the same grid. The
audit says why: the GPU spends its time waiting on per-collapse synchronisation, not computing.

### The leads

Ranked by how directly each makes the work GPU-shaped, and by the least work to live chunk generation.

1. **Block-local chunk solver (chosen first).** One workgroup solves one whole chunk in workgroup
   memory within one dispatch: min-count reduction, hash-RNG weighted choice, sweep propagation to a
   fixpoint, restart on contradiction. The parallelism is spent across chunks (and seeds), where WFC
   is embarrassingly parallel, instead of inside one propagation, where it is not (arc consistency is
   P-complete). An 8x8x8 chunk at 81 tiles is 6 KiB of domains plus a 5.8 KiB copy of the rule table,
   inside the 16 KiB WebGPU default; dozen offers 32 KiB. The support for it is our own measurement
   that 256 dependent steps are 13 to 18 times cheaper inside one dispatch than as separate
   dispatches. Falsified if one 512-cell chunk takes more than about 150 ms, if 64 chunks in one
   dispatch take more than about 8 times one chunk, or if restarts explode.
2. **Device-resident loop for the monolithic grid (fallback).** Keep one big grid but record K
   collapses per submit: on-device selection and choice, propagation as a chain of indirect dispatches
   with an early-out flag, a decision log instead of grid clones, readback every K collapses or on a
   contradiction. It removes every drain listed above but stays sequential in collapses.
3. **Sweep propagation with change epochs.** Each cell intersects the unions of its six neighbours'
   rows, skipping neighbours unchanged since the previous sweep; a shared changed flag ends the loop.
   No worklist append, no per-pass readback, and the fixpoint does not depend on thread order because
   propagation is confluent. It is the propagation inside lead 1 and applies to lead 2.
4. **Speculation.** (a) Portfolio restarts: the same chunk under several seeds in several workgroups,
   keeping the first to finish; parallel Luby restarts report super-linear speedups on heavy-tailed
   instances for solvers without nogood learning, which describes ours. (b) Lookahead probing:
   propagate every candidate tile of the chosen cell in parallel and discard those that contradict.
   Probably low value on adjacency-only rules, where the city backtracks rarely, so it gets a cheap
   CPU-side count of avoidable backtracks before any shader work.
   **First measurement** (one build, RTX 3070 through dozen, see [solver-fit.md](solver-fit.md)):
   none of the falsifiers hit, but the shape differs from the prediction. One chunk alone takes
   29.7 ms, ten times one CPU thread, at about 13 µs per workgroup step. 64 chunks cost 3.7 times one
   chunk rather than 1 to 2 times, and 256 chunks reach 0.91 ms per chunk, three times one CPU thread
   but below a 12-core CPU. Varying the invocations per workgroup then refuted the first explanation
   (that every step pays a barrier across 256 invocations): one invocation is 30 times *slower*, so
   the sweep's per-cell work dominates. Chunks do run in parallel, and a dispatch lasts as long as
   its slowest chunk, which the restart tail makes 3.8 times the mean. The next levers are fewer steps
   per chunk and less work per step (guess 12 in [solver-fit.md](solver-fit.md)).
   Two changes then paid off together. Collapsing every local minimum within a radius per round
   (the selection rule of Luby's parallel maximal independent set, applied to (count, index) keys)
   cuts sweeps per collapse by up to five times but multiplies contradictions; restoring a
   checkpoint from before the failing round, instead of restarting the chunk, makes each
   contradiction cost a few rounds. Combined at radius 1, 256 chunks take 0.17 ms each, about 15
   times one CPU thread in the same run, with no chunk failing.
   Stitching those chunks into a world is where the approach meets its known weakness, the one every
   block-based source warns about. With faces fixed to already solved neighbours, 28 of 63 chunks
   under N-WFC's diagonal order have unsatisfiable borders (29 of 32 under a checkerboard).
   Solving each chunk with a one-cell halo that is discarded afterwards brings the diagonal order down
   to 3 of 63, and never produced a seam violation. The rest need a repair that may change committed
   cells, which is what modifying in blocks and marian42's clearing both do: re-solving a failed chunk
   alone with its halo released completed the world under both orders. The checkerboard then needs
   only two dispatches of 32 chunks plus about ten single-chunk repairs.
   Put together, that is live generation on this build: walking a player across a 192×64×8 world at
   1.4 m/s with a four-chunk view radius costs a median of 43 ms of dispatch time per half-second
   tick, and the world comes out complete and seamless. Only the first tick, which fills the whole
   view at once, exceeds the budget. What that measures is a benchmark kernel, not the solver: the
   shipped `GpuAccelerator` still runs the per-collapse loop this document opened with.
5. **CPU threads own the search (yardstick).** The reference above, times the number of cores.

## How we will know it worked

The stress suite is the yardstick, run in release with the same three workloads. A change is kept when
it moves `cells_per_s` on the 81-variant city, not on the toy two-tile grid, and when the E2E and
constrained-city tests still pass. Every number in this document came from
`wfc-devtools/tests/stress.rs`, `WFC_TRACE_CHROME`, `wfc-gpu/tests/propagation_bench.rs` or
`wfc-devtools/tests/cpu_reference.rs`, so each claim can be re-measured after any change. A number is
evidence about one build on one machine and stack; it is quoted with that context, and it becomes a
design conclusion only after it has been reproduced with a warm-up and medians over interleaved samples.
