# Debugging and observability

Practices and tools for understanding what a parallel, GPU-resident Wave Forge run is doing. Testing
is covered in [testing.md](testing.md), and the GPU drivers of the dev container in
[environment.md](environment.md).

## Why this program is hard to debug

- **The solve happens inside one dispatch.** A region's domains live in workgroup memory from the
  first propagation to the last collapse, so there is nothing to step through, nothing to print
  from, and no intermediate state on the host at all. What comes back is tiles, a status and a
  handful of counters.
- **Hundreds of invocations run at once.** A kernel bug is usually a lost update or a barrier in
  non-uniform control flow, which shows up as *slightly wrong output*, not as a crash.
- **GPU errors surface late and far from their cause.** A wrong binding is reported when the command
  buffer is validated or executed, not where the mistake is in the code.
- **A world is made of many solves.** A seam between two chunks can be wrong because of the chunk on
  either side of it, or because of the order they were solved in.
- **A pack is made of many stages.** A wrong tree can come from the Scatter stage that placed it or
  from any stage it reads, and a stage that reads further than it declared can depend on the order
  chunks were generated in.

Stepping through code is therefore not the tool. The approach is: make runs reproducible, check
invariants automatically, get counters out with the result, and turn state into images that can be
inspected afterwards.

## Practices

### 1. Reproduce before you debug

Generation is deterministic, so a failure is reproducible by construction: the same rule set, prior,
configuration and requests give the same world on any machine, within the limits
[world.md](../architecture/world.md) states. To pin a case down:

- Save the inputs: rule file or pack, chunk shape, extent, halo, world seed and the sequence of
  requests.
- Save the output and render it: `wave-forge --output grid.txt` then `wfc-render`
  ([testing.md](testing.md#rendering-tools)).
- Shrink it: one chunk with the smallest rule set that still fails is worth more than any amount of
  logging. A single-chunk world is one dispatch, which is the smallest unit that can be wrong. For a
  pack, drop every stage the failing one does not read.

### 2. Let invariants find the bug

`wfc_devtools::adjacency_violations` lists every broken adjacency with coordinates, tiles and
direction. A handful of violations scattered through a chunk points to a lost update inside the
kernel. Violations along one axis point to a direction or indexing mistake. Violations everywhere
point to a layout or binding mismatch. Violations only on chunk faces point at the schedule or at
`region_init`, not at the kernel.

Four oracles sit behind that:

- **The CPU reference solver** (`wfc-core`, feature `reference`). `ReferenceSolver::propagate` is
  the fixpoint a correct propagator must reach, and the kernel's propagation is compared against it
  cell for cell. When the kernel reports a border contradiction, the reference propagates the same
  starting domains and must empty a cell too; the benchmarks assert that, so a kernel bug cannot hide
  behind "those borders were impossible".
- **Whole-world comparison on the CPU.** `tests/facade.rs` generates the same world twice, and in
  different request orders, and compares every chunk's tiles. The stage tests (`tests/stages.rs`,
  `tests/scatter.rs`, `tests/towns.rs`) do the same for a pack. Anything that makes a result depend
  on scheduling shows up there rather than in a rendered image.
- **Order independence on a device.** `wfc-devtools/tests/order_diff.rs` generates a 4×4-chunk city
  all at once, chunk by chunk in raster order and in reverse, with repairs off and on, and names how
  many chunks differ and the first differing cell. When it fails with repairs on but passes with
  them off, the repair schedule is at fault, not the kernel.
- **The golden world.** `wfc-devtools/tests/golden_world.rs` compares a city tile for tile with one
  recorded on another device. The host code is the same on both, so when it fails on one device
  and passes on another, suspect the driver or the kernel before the host. When a change is meant to
  change worlds, record the fixture again ([testing.md](testing.md)).

### 3. Read the counters the solver returns

There are no spans and no logging inside a solve; a shader cannot log, and a host-side span would
only time the wait. What a solve cost comes back as data instead:

| Where | What it says |
|---|---|
| `RegionStats`, one per region | sweeps, collapses, restarts, backtracks, steps, tries (contradictions met, recovered ones included), and the cell index where the last contradiction emptied a cell |
| `RegionStatus`, one per region | `Solved`, `Exhausted`, `StepCap` or `BorderContradiction`, or `Superseded` for a seed of a repair that stopped because a lower seed had already solved |
| `GeneratorStats`, per world | batches dispatched, chunks solved, repaired, rewritten by repairs, given up on, milliseconds waiting for the solver, and how many of the batches and milliseconds were repairs |

How to read them:

- **Sweeps per collapse** is propagation efficiency. Compare it with the city's value in
  [measurements.md](../research/measurements.md); a jump means a change made the fixpoint harder to
  reach.
- **Restarts and backtracks** say whether the region is being searched or solved. High restarts with
  a low radius means recovery is not keeping up; that is how the selection radius was chosen
  ([measurements.md](../research/measurements.md)).
- **`steps` of the slowest region** divided by the dispatch's wall time gives microseconds per step,
  which is the number to compare kernels with. A dispatch lasts as long as its slowest region, so the
  maximum matters more than the mean.
- **`contradiction_cell`** is an index into the region, halo included, so it maps back to a world
  cell through `Region::cells`. Whether the failure sat inside the chunk or in its halo is usually
  the whole answer.
- **`repair_ms` against `solver_ms`** says how much of generation is spent recovering rather than
  generating. A repair dispatch lasts as long as the seeds below its winner, so a rising average per
  repair means repairs are finding their chunk later in the portfolio.
- **`repaired` above zero** means the rule set is not streaming-clean. Repairs keep the world a
  function of its configuration, but a chunk evicted and generated again comes back as its first
  attempt, without the repairs of neighbours that had rewritten it ([world.md](../architecture/world.md)).
  `rewritten_by_repair` counts the neighbours those repairs touched.

### 4. Debug a pack's stages

A stage runtime reports what went wrong by name, so the first step is reading the error.

- **`PackError`** comes from loading a pack and names the stage at fault: an input no stage is named
  (`UnknownInput`), a parameter out of range (`Invalid`, with the stage and a message), two stages of
  one name (`DuplicateName`), stages that read each other (`Cycle`, listing them), a pack of another
  version (`Version`), or text that is not a pack at all (`Syntax`).
- **`StageError::OutOfReach`** means a stage read an input further from its chunk than its declared
  reach, and names the stage, the input, the reach and how far it read. That is a bug in the stage
  kind's code, not in the pack: the reach a stage kind declares for its inputs has to cover what it
  reads. A stage reads its inputs through a `FieldView`, which holds only the columns within reach,
  so the read fails loudly instead of returning whatever a neighbour chunk happened to hold. A stage
  that silently read beyond its reach would make its output depend on generation order.
- **`Pack::reach`** says how far beyond a column of the target stage each stage it depends on is
  generated: the largest sum of reaches along any path of inputs. When a request generates far more
  than expected, this is where the area comes from.
- **The other `StageError`s** name their stage too: an unknown stage asked for, a Solve stage with no
  town solver, a town solver whose chunk size differs from the runtime's, and a town that could not
  be solved, with its region.
- **Order is the first suspect.** If a stage's output looks wrong only sometimes, generate the same
  area all at once and one chunk at a time, as `tests/stages.rs` does, and compare. A difference
  means some input was read outside what the runtime guarantees.

### 5. Time the host side, coarsely

The host sees wall clock around a dispatch and nothing else: there are no GPU timestamp queries.
That is enough for the two questions that usually matter, because the benchmarks print both:

- Milliseconds per chunk against the chunks per dispatch, which is what says whether a batch is big
  enough.
- The CPU reference on one thread and on every thread, which is what says whether the device is
  worth using at all for a given shape.

A first dispatch of a new specialisation also pays for translating the shader, which takes seconds
through dozen ([environment.md](environment.md)), so anything that is not warmed first measures the
compiler. `BlockSolver::warm` exists for that, and every benchmark calls it.

### 6. Read the Godot node's frame costs

In Godot, what matters is the time the extension takes from Godot's own thread. `WaveForgeWorld`'s
`stats()` ([godot.md](../reference/godot.md)) reports it two ways:

- **Recent frames:** `process_ms_median`, `process_ms_p99` and `process_ms_max`, the node's own time
  per frame, and the same for navigation bakes (`navigation_bake_ms_*`, `navigation_start_ms_*`,
  `navigation_finish_ms_*`). `WaveForgeStages.stats()` reports `process_ms_*` for its node.
- **The slowest frame since the start, broken down:** `slowest_frame_ms` in all,
  `slowest_frame_events` drained and `slowest_frame_signals_ms` emitting them (the handlers connected
  to the signals included), `slowest_frame_colliders` chunks given bodies in
  `slowest_frame_colliders_ms`, and `slowest_frame_navigation_ms`.

Godot's own `Performance.TIME_PROCESS` is the slowest frame of the last second, not each frame's
time, so it says that a frame was slow but not why. When `verify.gd` finds Godot's slowest frame
over its bar, its message quotes the node's slowest-frame breakdown. A slow frame with many events
and a large signal time points at the handlers a game connected; a large collider or navigation
time points at the work the node does on Godot's thread for chunks near the focus. The
generation counters (`solved`, `repaired`, `failed`, `solver_ms`, `repair_ms`) are in the same
dictionary.

### 7. GPU-specific checks

- **Validation errors are fatal by design.** wgpu panics with the failing call and resource label,
  so keep every buffer, bind group, pipeline and pass labelled.
- **Keep a dispatch bounded.** A shader that does not terminate can take the host's graphics driver
  with it ([environment.md](environment.md)). The kernel has a hard step cap for that reason, and a
  benchmark should only grow its batch while four times the last dispatch still fits well inside the
  limit.
- **Suspect a barrier in non-uniform control flow** when a region hangs or comes back inconsistent.
  Every invocation of a workgroup must reach the same `workgroupBarrier`, which is why the kernel's
  state machine is driven by `workgroupUniformLoad`.
- **Suspect a lost update when results are nearly right.** Propagation gathers into an invocation's
  own cells so that it cannot happen; anything that writes a neighbour's mask is a bug by
  construction.
- **`SolverError::NoReport` means the dispatch never ran to the end.** A region's statistics record
  stayed empty, which is what a driver that drops a workgroup leaves behind.
- **Check which adapter you got.** `WgpuBackend::describe` names it. An unexpected adapter, a
  `SIGSEGV` after all tests passed, and `NoReport` on a software device are driver matters, covered
  in [environment.md](environment.md).
- **Compare adapters when a result looks wrong.** Rerun on another device with `WGPU_ADAPTER_NAME`:
  if the other device behaves correctly, suspect the driver stack before the shader. The golden world
  is that comparison, made permanent.

### 8. Look at the world

A rendered world answers questions that no counter does. `wfc-render` draws a text grid, the
streaming tests write an isometric voxel model of the whole world they generated, the city towns
test draws each town, and `render_city.sh` draws a city in Godot. A seam shows up as a straight line
of wrong modules at a chunk boundary; a repair shows up as a patch that does not match its
surroundings; an unplaceable chunk shows up as a hole. All three are visible in a second and
invisible in a log.

## When something goes wrong: a checklist

1. Does it reproduce with one chunk, or does it need neighbours? That splits kernel bugs from
   schedule bugs.
2. What does `adjacency_violations` report, and is there a pattern (scattered, one axis, everywhere,
   only on faces)?
3. Does the CPU reference reach the same fixpoint on the same starting domains?
4. What do the statuses and counters say: which region, how many restarts, which cell emptied?
5. Do two runs of the same request give the same world, and do two request orders
   (`order_diff.rs`)? Does another device (`golden_world.rs`)?
6. For a pack: which stage does the error name, and does the stage read within its reach?
7. What does the rendered world look like?
8. Only then: narrow the kernel down by reading back a region's domains after propagation and
   comparing them by hand.
