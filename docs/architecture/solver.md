# The solver

How one region is solved: the model the solver works on, the block kernel that solves a batch of
regions in one GPU dispatch, and the seams around it. How regions become a streamed world is in
[world.md](world.md). The numbers each decision rests on are in
[measurements.md](../research/measurements.md), and the story of the solver this one replaced is in
[per-collapse-solver.md](../research/per-collapse-solver.md).

## The model

### Topology

The model is a cubic lattice with six directions, and 2D is a world one cell deep. Rules are
expressed per direction.

This is not free: a one-deep world still carries the two vertical directions, so a rule set for a
2D tile set has to say something about them. There is no topology abstraction, and axis names in
rule files are fixed to `±x/±y/±z`; hexagonal or irregular grids wait for a game that needs one
([roadmap.md](../plan/roadmap.md#deferred)).

### Tiles and compiled rules

Rule files come in two forms, and one loader (`wfc_rules::loader::parse_rule_file`) tells them
apart. A **tile set** lists tiles with weights and every allowed adjacency by name; it suits a
handful of tiles and has no rotations. A **module set** describes each module by the connectors on
its six faces, and rotated variants, adjacency and weights are derived (`wfc_rules::modules`); the
city (`examples/city.ron`) is written this way. A module set can also be built in Rust.

Before solving, rules are **compiled** into a dense form: `RuleTable` holds, for each direction and
tile, a bitmask of compatible neighbour tiles. Propagation needs "which neighbour tiles does *any*
of my remaining tiles allow in this direction?", which is then a union of precomputed masks:
branch-free, and suited to both SIMD and a GPU. Named, symmetric rules exist for authors; the hot
path never sees them.

Weights are **quantised to integers** when a `Ruleset` is built (at most 65535, and at least 1 for
any positive weight). A choice is `hash % total` followed by a walk over the weights. A float walk
would not be reproducible across devices: drivers may fuse a multiply and an add, which changes the
last bit, and one bit is enough to pick another tile and send a region down another path.

### Possibility storage

A cell's remaining tiles are a bitmask. `Domains` holds every cell's mask in **one contiguous
`Vec<u32>`** (cells × words per cell), in the layout the shader reads, so a batch is uploaded and
read back without repacking. One array is cache-friendly, suits SIMD and crosses to the device as
it is; per-cell allocations would mean pointer chasing and a pack step on every transfer.

`TileMask` is the value type for one cell's mask, eight words wide (256 tiles) so it needs no
allocation. Hot code works on the words of `Domains` instead, because the city's 81 tiles need
three of those eight words and touching the other five costs more than the propagation.

### The prior

A cell's starting mask comes from a `Prior`: a mask per layer, a ban per world face, and overrides
for single cells. That is how "street level only on the bottom layer" or "no path pointing out of
the world" is said without the solver knowing what a street is, and it is the door through which
other stages drive WFC ([stages.md](stages.md)).

### Seeds and hashing

Every choice is a **stateless hash**: `pcg3d` over the world seed mixed with the region's id, the
number of contradictions the solve has met so far, and where the solve has got to (the round, and
the cell). The id is a hash of the chunk's coordinate, not its index in a batch. No random state is
carried anywhere, on the host or on the device.

A region is solved by hundreds of invocations at once and restored from checkpoints, so any carried
random state would make the result depend on scheduling. A hash of where and when a choice is made
is the same whatever order the lanes run in, which is what lets the same request give the same world
on any backend and at any invocation count ([world.md](world.md#what-determinism-means-here)).
Salting with the contradiction count is what makes a retried round choose differently.

## The block kernel

One workgroup solves one region, and a dispatch solves a batch of equally shaped regions. The
region's domains stay in workgroup memory for the whole solve, so propagation, selection, collapse
and recovery never leave the device. The host builds starting domains, dispatches once, and reads
results back once.

### The algorithm

A region is solved by repeating four steps until every cell is decided or the budget runs out.

1. **Propagate** as a *gather* sweep: every invocation recomputes its own cells' masks from their
   six neighbours, until a sweep changes nothing. A change bumps an epoch counter, so a sweep that
   finds no change ends the fixpoint.
2. **Select** every *local minimum* of the possibility count within a Chebyshev radius. A cell's
   key is its count with its index in the low bits, so ties go to the lowest cell index. Two local
   minima are always more than the radius apart, and the global minimum is always one.
3. **Collapse** each selected cell to one tile by the integer weight walk, each with its own hash.
4. **Recover** from a contradiction by restoring a checkpoint from a ring buffer, doubling how far
   back it goes when the same round fails again, and restarting the region with a fresh attempt
   when the ring cannot reach far enough.

**Why gather rather than scatter:** a scatter update writes a neighbour's mask, so two invocations
can lose each other's restriction, and making that safe means an atomic on every word. A gather
writes only the invocation's own cells, so a sweep is race-free by construction and needs one
barrier per sweep.

**Why local minima rather than the global one:** a global minimum means one collapse per round, and
a round's sweeps are the expensive part. Collapsing every local minimum keeps the workgroup busy,
and it pays only because recovery is cheap: with restart-only recovery a radius of 1 made 73 of 256
chunks fail, and with checkpoint undo the same radius is the fastest setting and none fail.

**Why the plain count rather than Shannon entropy:** the count is what the kernel already has, and
the measured difference did not justify the extra work. The tie-break is fixed by cell index rather
than scan order within a lane, so it does not depend on how cells are spread over invocations.

### Contradictions

A contradiction is normal for WFC, not an exceptional error. Inside the kernel it is recovered from
as above. When the recovery budget runs out, the region comes back with a **status** rather than an
error:

- `Exhausted`: attempts used up;
- `StepCap`: the hard step limit, which exists so a shader cannot hang a device;
- `BorderContradiction`: propagating the starting domains alone empties a cell, so no arrangement
  satisfies the borders;
- `Superseded`: in a portfolio of seeds, a lower seed solved first, so this one stopped.

A game cannot show a dialog because a chunk failed. A status says which region and why, and the
world decides what to do: it repairs the chunk ([world.md](world.md#repairs)).

### Where each step runs

Measurement settled it ([#7](https://github.com/AntonTegnelov/wave_forge/issues/7)): **everything
runs on the device, inside one dispatch per batch.** In the order the measurements forced it:

- A per-collapse loop is dominated by what surrounds each collapse, not by the collapse. Every
  round-trip drained the queue and copied the grid. With the whole region in workgroup memory there
  is no round-trip to pay for.
- Per-step cost tracks per-cell sweep work, not the number of barriers, so a kernel that sweeps a
  whole region per step is not wasteful.
- A dispatch lasts as long as its slowest region, so many regions per dispatch is what makes the
  device pay off.

The CPU reference solver (`wfc_core::reference`) stays, behind a feature, as the fixpoint oracle and
the yardstick GPU numbers are printed against. It is not a fallback
([vision.md](../product/vision.md#non-goals)).

## Dispatch and the seams

### The solver seam

```rust
pub trait Solver {
    fn max_batch(&self) -> u32;
    fn accepts(&self, region: RegionShape) -> bool;
    fn start(&mut self, batch: RegionBatch) -> Result<JobId, SolverError>;
    fn poll(&mut self, job: JobId) -> Result<Option<BatchResult>, SolverError>;
    fn wait(&mut self, job: JobId) -> Result<BatchResult, SolverError>;
}
```

A batch of equally shaped regions goes in with their starting domains, ids, seeds and an optional
budget; per region, a status, statistics and the domains come back. A batch can be a **portfolio**:
one problem with different seeds, where a region stops once a lower one has solved.

The seam is job-based rather than async because the work is one dispatch, and what differs between
engines is only where the waiting happens ([overview.md](overview.md#threads)). `accepts` exists
because a region's workgroup memory is a hard device limit, so the world can ask before it builds a
batch that would be refused.

### Hot paths are generic

`BlockSolver<B: ComputeBackend>` and `WorldGenerator<S: Solver>` are generic, and the kernel is
specialised per region shape and word count by substituting constants into its WGSL. There is no
`dyn` on any path that runs per cell, per step or per batch.

The specialisation is not cosmetic. The same kernel with its mask words in a loop instead of written
out is 2.5× slower, because a mask indexed by a loop variable lands in scratch memory where one
written out stays in registers.

### The backend seam

`ComputeBackend` is everything the kernel needs from a device: limits, a pipeline from WGSL text,
buffers, writes, a dispatch with its readbacks, and either polling or waiting on a submission.
`can_poll` is part of it because some devices cannot be asked whether they are done without
blocking.

`WgpuBackend::from_env` builds a device of its own; `WgpuBackend::from_device` takes one an engine
already owns, which is how the Bevy plugin shares Bevy's. Sharing requires the plugin and the engine
to agree on a wgpu version, which is why `wave_forge_bevy` tracks the Bevy release that uses the
library's. Which adapters the dev container and CI use is in
[environment.md](../guides/environment.md).

## GPU specifics

- **wgpu** is the GPU layer because it is the only mature Rust option that covers Vulkan, Metal,
  DirectX 12 and WebGPU from one code base, which shipping inside Godot and Bevy on every desktop
  platform requires.
- **The shader is embedded in the binary** with `include_str!`, never read from disk at run time,
  because the library ships inside a game where the source tree does not exist. Shape constants and
  the mask helpers are substituted into that text per specialisation, which is also what a SPIR-V
  backend needs, since naga bakes overrides into a module per shape anyway.
- **Workgroup memory is the binding limit.** A region's domains, its selection scratch and its
  epochs must fit the device's per-workgroup allowance (32 KiB on the reference devices), so the
  solver refuses a shape that does not fit and names the numbers. The city's 81 tiles in an 8×8×8
  chunk with a halo of 3 need 38 288 B, so a repair's halo stops at 2 there. Bigger chunks or more
  tiles would need domains in a storage buffer, which nothing needs yet.
- **A dispatch is capped in steps.** A shader that does not terminate takes the display driver with
  it (Windows resets a device after about two seconds), so a region has a hard step budget and
  reports `StepCap` rather than running on.
- **Kernels are compiled once per region shape, ahead of time where possible, and cached across
  runs.** One pipeline takes seconds to compile on some drivers, which would otherwise look like a
  seconds-long solve. A pipeline depends on the region shape alone, so every batch size of one
  shape shares it; `BlockSolver::warm` compiles a list of shapes up front; and
  `WgpuBackend::cache_pipelines_in` keeps compiled pipelines in a file named for the adapter and
  driver, which roughly halves a later start's compile on the dev container's stack
  ([measurements.md](../research/measurements.md) E32 to E34).

## Alternatives that were measured or read and not taken

Each has its evidence in [measurements.md](../research/measurements.md) or
[per-collapse-solver.md](../research/per-collapse-solver.md).

- **A host loop that dispatches per collapse** (the original design): round-trips and grid copies
  dominated it, and no kernel speed-up could close the gap to a live budget. Deleted.
- **Scatter propagation with atomics:** correct, but it needs an atomic per word and a confirmation
  sweep; the gather sweep needs neither.
- **SAT or CDCL encodings of adjacency, and clause-sharing portfolios:** no competitive GPU
  implementation exists, and measured CPU ports were far slower than propagation with undo.
- **AC-4 support counters:** cells × 6 × tiles counters against a few words of bitset per cell, and
  they combine poorly with backtracking.
- **Nogood recording:** it helped the old solver on constraint-heavy rule sets, but the block kernel
  with checkpoint undo and a 32-seed repair portfolio places every chunk of the reference worlds
  without it.
- **Morton or other space-filling layouts:** the region lives in workgroup memory, where layout
  does not decide the cost.

## Open questions

- **Native numbers.** Every timing is from the dev container's RTX 3070 through Mesa's dozen
  (Vulkan on Direct3D 12) or from lavapipe. Translation distorts dispatch and submission costs most.
  The desktop measurement ([#39](https://github.com/AntonTegnelov/wave_forge/issues/39)) answers
  this.
- **A safe parallel-collapse radius.** Choices are not confluent, and no published algorithm says
  when two collapses cannot interact. A conservative bound for a specific rule set may exist; today
  the radius is a measured setting and undo absorbs the conflicts.
- **Sub-complete rule sets.** A rule set in which every partial arrangement extends would need no
  backtracking at all. Whether the city could be made one is a module-design question nobody has
  tried.
- **Per-cell cost as a curve.** Cost per step fits a floor plus a per-cell term on five points of one
  stack; what the floor is made of is unmeasured.
