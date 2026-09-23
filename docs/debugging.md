# Debugging and observability

Practices and tools for understanding what a parallel, GPU-resident Wave Forge run is doing. Testing is covered in [testing.md](testing.md).

## Why this program is hard to debug

- **The solve happens inside one dispatch.** A region's domains live in workgroup memory from the first propagation to the last collapse, so there is nothing to step through, nothing to print from, and no intermediate state on the host at all. What comes back is tiles, a status and a handful of counters.
- **Hundreds of invocations run at once.** A kernel bug is usually a lost update or a barrier in non-uniform control flow, which shows up as *slightly wrong output*, not as a crash.
- **GPU errors surface late and far from their cause.** A wrong binding is reported when the command buffer is validated or executed, not where the mistake is in the code.
- **A world is made of many solves.** A seam between two chunks can be wrong because of the chunk on either side of it, or because of the order they were solved in.

Stepping through code is therefore not the tool. The approach is: make runs reproducible, check invariants automatically, get counters out with the result, and turn state into images that can be inspected afterwards.

## Practices

### 1. Reproduce before you debug

Generation is deterministic, so a failure is reproducible by construction: the same rule set, prior, configuration and requests give the same world on any machine ([architecture.md §6.3](architecture.md#63-what-determinism-means-here)). To pin a case down:

- Save the inputs: rule file, chunk shape, extent, halo, world seed and the sequence of requests.
- Save the output and render it: `wave-forge --output grid.txt` then `wfc-render` ([testing.md](testing.md#rendering-tools)).
- Shrink it: one chunk with the smallest rule set that still fails is worth more than any amount of logging. A single-chunk world is one dispatch, which is the smallest unit that can be wrong.

### 2. Let invariants find the bug

`wfc_devtools::adjacency_violations` lists every broken adjacency with coordinates, tiles and direction. A handful of violations scattered through a chunk points to a lost update inside the kernel. Violations along one axis point to a direction or indexing mistake. Violations everywhere point to a layout or binding mismatch. Violations only on chunk faces point at the schedule or at `region_init`, not at the kernel.

Two oracles sit behind that:

- **The CPU reference solver** (`wfc-core`, feature `reference`). `ReferenceSolver::propagate` is the fixpoint a correct propagator must reach, and the kernel's propagation is compared against it cell for cell. When the kernel reports a border contradiction, the reference propagates the same starting domains and must empty a cell too; the benchmarks assert that, so a kernel bug cannot hide behind "those borders were impossible".
- **Whole-world comparison.** `tests/facade.rs` generates the same world twice, and in different request orders, and compares every chunk's tiles. Anything that makes a result depend on scheduling shows up there rather than in a rendered image.

### 3. Read the counters the solver returns

There are no spans and no logging inside a solve; a shader cannot log, and a host-side span would only time the wait. What a solve cost comes back as data instead:

| Where | What it says |
|---|---|
| `RegionStats`, one per region | sweeps, collapses, restarts, backtracks, steps, attempts, and the cell index where the last contradiction emptied a cell |
| `RegionStatus`, one per region | `Solved`, `Exhausted`, `StepCap` or `BorderContradiction`, or `Superseded` for a seed of a repair that stopped because a lower seed had already solved |
| `GeneratorStats`, per world | batches dispatched, chunks solved, repaired, rewritten by repairs, given up on, milliseconds waiting for the solver, and how many of the batches and milliseconds were repairs |

How to read them:

- **Sweeps per collapse** is propagation efficiency. Around 3 is normal for the city set; a jump means a change made the fixpoint harder to reach.
- **Restarts and backtracks** say whether the region is being searched or solved. High restarts with a low radius means recovery is not keeping up; that is how the selection radius was chosen ([solver-fit.md](solver-fit.md)).
- **`steps` of the slowest region** divided by the dispatch's wall time gives microseconds per step, which is the number to compare kernels with. A dispatch lasts as long as its slowest region, so the maximum matters more than the mean.
- **`contradiction_cell`** is an index into the region, halo included, so it maps back to a world cell through `Region::cells`. Whether the failure sat inside the chunk or in its halo is usually the whole answer.
- **`repair_ms` against `solver_ms`** says how much of generation is spent recovering rather than generating. A repair dispatch lasts as long as the seeds below its winner, so a rising average per repair means repairs are finding their chunk later in the portfolio.
- **`rewritten_by_repair`** above zero means the rule set is not streaming-clean, so a chunk's tiles depend on generation order ([architecture.md §6.3](architecture.md#63-what-determinism-means-here)).

### 4. Time the host side, coarsely

The host sees wall clock around a dispatch and nothing else: there are no GPU timestamp queries yet (A-15). That is enough for the two questions that usually matter, because the benchmarks print both:

- Milliseconds per chunk against the chunks per dispatch, which is what says whether a batch is big enough.
- The CPU reference on one thread and on every thread, which is what says whether the device is worth using at all for a given shape.

A first dispatch of a new specialisation also pays for translating the shader (about 4 s on this stack), so anything that is not warmed first measures the compiler. `BlockSolver::warm` exists for that, and every benchmark calls it.

### 5. GPU-specific checks

- **Validation errors are fatal by design.** wgpu panics with the failing call and resource label, so keep every buffer, bind group, pipeline and pass labelled.
- **Keep a dispatch bounded.** A shader that does not terminate takes the host's display driver with it (Windows resets a device after about two seconds, and the reset kills the session). The kernel has a hard step cap for that reason, and a benchmark should only grow its batch while four times the last dispatch still fits well inside the limit.
- **Suspect a barrier in non-uniform control flow** when a region hangs or comes back inconsistent. Every invocation of a workgroup must reach the same `workgroupBarrier`, which is why the kernel's state machine is driven by `workgroupUniformLoad`.
- **Suspect a lost update when results are nearly right.** Propagation gathers into an invocation's own cells so that it cannot happen; anything that writes a neighbour's mask is a bug by construction.
- **Check which adapter you got.** `WgpuBackend::describe` names it. In the dev container it should be `Microsoft Direct3D12 (NVIDIA GeForce RTX 3070)`; if it is `llvmpipe`, the instance was not built from the environment, so wgpu hid the non-conformant dozen adapter (`RUST_LOG=wgpu_hal=warn` shows "hiding adapter").
- **A `SIGSEGV` after all tests passed is the dozen unload bug, not our code.** A backtrace ends in `__nptl_deallocate_tsd` calling an unmapped address. See the known issue in [development.md](development.md#toolchain-and-environment) for the preload workaround.
- **Compare adapters when a result looks wrong.** dozen is a non-conformant translation layer. Rerun with `WGPU_ADAPTER_NAME=llvmpipe`: if the software device behaves correctly, suspect the driver stack before the shader.
- **Real hardware for performance.** Software Vulkan (llvmpipe) timings say nothing about GPU performance, and dozen timings include translation overhead.

### 6. Look at the world

A rendered world answers questions that no counter does. `wfc-render` draws a text grid, and the streaming tests write an isometric voxel model of the whole world they generated. A seam shows up as a straight line of wrong modules at a chunk boundary; a repair shows up as a patch that does not match its surroundings; an unplaceable chunk shows up as a hole. All three are visible in a second and invisible in a log.

## When something goes wrong: a checklist

1. Does it reproduce with one chunk, or does it need neighbours? That splits kernel bugs from schedule bugs.
2. What does `adjacency_violations` report, and is there a pattern (scattered, one axis, everywhere, only on faces)?
3. Does the CPU reference reach the same fixpoint on the same starting domains?
4. What do the statuses and counters say: which region, how many restarts, which cell emptied?
5. Do two runs of the same request give the same world?
6. What does the rendered world look like?
7. Only then: narrow the kernel down by reading back a region's domains after propagation and comparing them by hand.
