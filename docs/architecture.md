# Architecture

This document describes the architecture of Wave Forge and, above all, *why* it is shaped this way. It follows from the goals in [vision.md](vision.md). Where the code does not match it yet you will find a callout like this:

> **Misaligned today (A-n):** what the code does instead. See [status.md](status.md#alignment-tasks).

Details that are local to one module belong in code comments next to that code. This document covers the big picture: the pieces, how they fit, and the reasoning that would otherwise be invisible from reading any single file.

## 1. Guiding principles

Each principle is a consequence of the vision's priority list (performance first, then correctness, integration, binary size, ergonomics).

1. **One engine-agnostic core, thin integrations.** Engines change their APIs often; generation logic should not. Keeping the solver free of engine types means it can be tested, benchmarked and profiled without an engine, and the Bevy plugin and Godot extension stay small adapters that are cheap to keep up to date.
2. **Measure, then place work.** GPU, CPU threads, SIMD and plain code each have different overheads. Where work runs is decided by benchmarks, not by guesses, and the seams are drawn so that moving it does not mean restructuring.
3. **Static dispatch on hot paths.** Anything executed per cell, per tile or per propagation step must be monomorphized (generics, enums), so the compiler can inline and vectorize it. Dynamic dispatch is only acceptable at coarse boundaries such as choosing a backend once per generator, where its cost is paid once rather than millions of times.
4. **Data lives where it is processed.** Moving data between CPU and GPU, or between threads, costs more than most computations on it. State stays on the device that is working on it for as long as possible, and crosses the boundary in bulk and rarely.
5. **Bounded, deterministic regions.** Worlds are generated as bounded chunks with deterministic seeds, so memory stays bounded, work can be parallelized per chunk, and any chunk can be regenerated identically.
6. **Constraints in, data out.** The solver accepts constraints (a prior on initial domains, borders from neighbouring chunks or from other layers) and returns plain data. This is what makes chunk stitching and Phase 2 layering possible without special cases.

## 2. Component overview

```
┌────────────────────────────────────────────────────────────────────────┐
│ Integrations (thin)          wave_forge_bevy     wave_forge_godot      │
├────────────────────────────────────────────────────────────────────────┤
│ wave_forge                   Builder, WorldGenerator (store +          │
│ (the library facade)         scheduler + solver + events), Worker      │
├────────────────────────────────────────────────────────────────────────┤
│ wfc-core                     the model: chunks and regions, rules,     │
│                              domains, prior, chunk store;              │
│                              the Solver seam; the CPU reference        │
├────────────────────────────────────────────────────────────────────────┤
│ wfc-gpu                      the block solver: one workgroup solves    │
│                              one region per dispatch, over a           │
│                              ComputeBackend (wgpu, or an engine's)     │
├────────────────────────────────────────────────────────────────────────┤
│ wfc-rules                    RON/vox authoring and connector-based     │
│                              module sets → compiled adjacency          │
└────────────────────────────────────────────────────────────────────────┘
 Dev-only (never shipped in the library): wfc-devtools, which holds the
 CLI, the invariant checker, the reference rule sets, the PNG renderers
 and the benchmarks
```

Dependencies only point downwards. The model and the seams can be used, fuzzed, benchmarked and profiled with nothing above them, and an engine integration cannot leak engine types into the core: it supplies either a [`Solver`](#51-the-solver-seam) or a [`ComputeBackend`](#82-the-backend-seam) and gets plain data back.

There are two seams rather than one because engines differ in what they own. Bevy owns a wgpu device and shares it, so `wave_forge_bevy` uses the wgpu backend and the kernel unchanged. Godot owns a `RenderingDevice` that takes SPIR-V and blocks in `sync()`, and `wave_forge_godot` does not use it: it runs the generator on a [`Worker`](#53-a-thread-when-an-engine-needs-one) with a device of its own, which is what keeps Godot's frame loop free without translating a kernel. A backend over `RenderingDevice` would save the second device; the seam is there for it (see the Godot section of [roadmap.md](roadmap.md#engine-integrations)).

## 3. Model

### 3.1 Topology: 2D and 3D

The model is a cubic lattice with six directions, and 2D is a world one cell deep. Rules are expressed per direction.

**Why this is not free:** a one-deep world still carries the two vertical directions, so a rule set for a 2D tile set has to say something about them, and a periodic border along z would wrap a cell onto itself. The generator has no periodic borders, so only the first cost is real today.

> **Misaligned today (A-2):** there is no topology abstraction. 2D is `z = 1` with six axes, and axis names in rule files are hard-coded to `±x/±y/±z`.

### 3.2 Tiles and compiled rules

Authors describe tiles (with weights and allowed symmetries) and adjacency by name in rule files, or as connector-based module prototypes in Rust (`wfc_rules::modules`), which derive rotated variants, adjacency and weights. Before solving, rules are **compiled** into a dense form: `RuleTable` holds, for each (direction, tile), a bitmask of compatible neighbour tiles, as `words_per_cell` words at row `(axis * num_tiles + tile) * words_per_cell`.

**Why compile:** propagation needs "which neighbour tiles does *any* of my remaining tiles allow in this direction?" That is a union of precomputed masks, which is branch-free and suits both SIMD and a GPU. Named, symmetric rules exist for authors; the hot path never sees them.

Weights are **quantised to integers** when a `Ruleset` is built (to at most 65535, and to at least 1 for any positive weight). A choice is then `hash % total` followed by a walk over the weights.

**Why integers:** a float walk is not reproducible across devices. Drivers are free to contract a multiply and an add into one fused instruction, which changes the last bit, and one bit is enough to pick a different tile and send the whole region down another path.

> **Misaligned today (A-3):** the RON loader still creates identity variants only, so symmetry in rule *files* is unsupported; module sets in Rust cover it. `generate_transformed_rules` also pairs a transformed tile only with the same transformation of its neighbour, so a rotated tile can never border an identity-only tile.

### 3.3 Possibility storage

A cell's remaining tiles are a bitmask. `Domains` holds every cell's mask in **one contiguous `Vec<u32>`** (`cells × words_per_cell`), in the same layout the shader reads, so a batch is uploaded and read back without repacking. A `TileMask` is the value type for one cell's mask, eight words wide (256 tiles) so that it needs no allocation; code on a hot path works on `Domains`' words instead, because a rule set of 81 tiles needs three of those eight and touching the other five costs more than the propagation.

**Why one array:** it is cache-friendly, it can be processed with SIMD, and it crosses to the device as it is. Per-cell heap allocations cause pointer chasing and force a pack step on every transfer.

A cell's starting mask comes from a `Prior`: a mask per layer, a ban per world face, and overrides for single cells. That is how "street level only on the bottom layer" or "no path pointing out of the world" is expressed without the solver knowing what a street is.

### 3.4 Seeds and randomness

Every choice is a **stateless hash**: `pcg3d(seed, chunk id, attempt, step)`, where the chunk id is a hash of its coordinate rather than its index in a batch. No RNG state is carried anywhere, on the host or on the device.

**Why stateless:** a region is solved by hundreds of invocations at once and restarted from checkpoints, so any carried RNG state would make the result depend on scheduling. A hash of where and when a choice is made is the same whatever order the lanes run in, which is what makes the same request give the same world on any backend and at any invocation count (see [§6.3](#63-what-determinism-means-here)).

## 4. Solver

### 4.1 The algorithm

A region is solved by repeating: propagate to a fixpoint, pick cells, collapse them, and recover when that fails. What is unusual here is that all of it happens inside one dispatch, with the region's domains in workgroup memory for the whole solve. The reasoning is recorded in [solver-redesign.md](solver-redesign.md); the measurements are in [solver-fit.md](solver-fit.md).

1. **Propagate** as a *gather* sweep: every invocation recomputes its own cells' masks from their six neighbours until a sweep changes nothing. A change bumps an epoch counter, so a sweep that finds no change ends the fixpoint.
2. **Select** every *local minimum* of the possibility count within a Chebyshev radius, breaking ties by the choice hash. Cells that far apart rarely interact, so many collapse per round.
3. **Collapse** each selected cell to one tile, weighted, by the integer walk of [§3.2](#32-tiles-and-compiled-rules).
4. **Recover** from a contradiction by restoring a checkpoint from a ring buffer, doubling how far it goes back each time the same region fails again, and restarting the region with the next attempt number when the ring is exhausted.

**Why gather rather than scatter:** a scatter update writes a neighbour's mask, so two invocations can lose each other's restriction, and making it safe means atomics on every word. A gather writes only the invocation's own cells, so the sweep is race-free by construction and needs a barrier per sweep rather than an atomic per word.

**Why local minima instead of the global one:** a global minimum means one collapse per dispatch, and a dispatch is the expensive part. Collapsing every local minimum keeps the device busy, and it only pays when recovery is cheap: with restart-only recovery a radius of 1 made 73 of 256 chunks fail, and with checkpoint undo the same radius is the fastest setting and none fail.

**Why weighted counts rather than Shannon entropy:** the count is what the kernel already has, and the measured difference did not justify the extra work. Ties are broken by the choice hash, not by scan order, so there is no directional artefact.

> **Superseded (A-7):** weighted Shannon entropy with a noise term was the plan for selection. The block kernel selects on the possibility count with hash tie-breaking instead; the artefact the noise term existed to prevent is not reachable this way.

### 4.2 Contradictions

A contradiction is normal for WFC, not an exceptional error. Inside the kernel it is recovered from as above. When the recovery budget runs out, the region comes back with a **status** rather than an error: `Exhausted` (attempts used up), `StepCap` (the hard step limit, which exists so a shader cannot hang a device), or `BorderContradiction` (propagating the starting domains alone empties a cell, so no arrangement satisfies the borders).

The world's answer to a status is a **repair**: solve the chunk alone with its halo released, which lets it rewrite the neighbouring cells the halo covers, and widen the halo if that is not enough. A chunk that still will not solve is reported once and left alone, because solving it again would fail the same way.

**Why not report an error:** a game cannot show a dialog because a chunk failed. A status says which chunk and why, and the world decides.

### 4.3 Where each step runs

Measurement settled this ([#7](https://github.com/AntonTegnelov/wave_forge/issues/7)): **everything runs on the device, inside one dispatch per batch of regions.** The host builds starting domains, dispatches, and reads results back once.

The reasoning, in the order the measurements forced it:

- A per-collapse loop is dominated by what surrounds the collapse, not by the collapse. Every round-trip drained the queue and cloned the grid; with the whole region resident in workgroup memory, there is no round-trip to pay for.
- Per-step cost tracks per-cell sweep work, not the number of barriers, so a kernel that sweeps a whole region per step is not wasteful: at 256 invocations one step of an 8×8×8 chunk with a halo costs about 20 µs.
- A dispatch lasts as long as its slowest region, so many regions per dispatch is what makes the device pay off: 0.18 ms per chunk at 256 chunks against 3.9 ms for the same chunk on one CPU thread.

The CPU reference solver stays, behind a feature, as the fixpoint oracle and the yardstick every GPU number is printed against. It is not a fallback ([vision.md](vision.md#non-goals)).

## 5. Dispatch, async and threading

### 5.1 The solver seam

```rust
pub trait Solver {
    fn max_batch(&self) -> u32;
    fn accepts(&self, region: RegionShape) -> bool;
    fn start(&mut self, batch: RegionBatch) -> Result<JobId, SolverError>;
    fn poll(&mut self, job: JobId) -> Result<Option<BatchResult>, SolverError>;
    fn wait(&mut self, job: JobId) -> Result<BatchResult, SolverError>;
}
```

A batch of equally shaped regions goes in with their starting domains, their chunk ids and seeds, and an optional budget; one status, one set of statistics and one region of domains per region comes back.

**Why job-based rather than async:** the work is one dispatch, and what differs between engines is only *where the waiting happens*. Bevy polls once per frame on its own schedule. Godot's `RenderingDevice` cannot be asked whether it is finished without blocking, so a [`Worker`](#53-a-thread-when-an-engine-needs-one) owns the device on a thread and the engine drains events. Either way, nothing in the library awaits, and no async runtime is forced into a game that did not ask for one. `accepts` exists because a region's workgroup memory is a hard device limit, so the world has to be able to ask before it builds a batch that would only be refused.

### 5.2 Hot paths are generic

`BlockSolver<B: ComputeBackend>` and `WorldGenerator<S: Solver>` are generic, and the kernel is specialised per region shape and word count by substituting constants into its WGSL. There is no `dyn` on any path that runs per cell, per step or per batch.

The specialisation is not cosmetic: the same kernel with its mask words in a loop instead of written out is 2.5× slower, because a mask indexed by a loop variable lands in scratch memory where one written out stays in registers.

### 5.3 A thread when an engine needs one

`Worker` runs a `WorldGenerator` on a std thread behind two channels: commands in, events and statistics out. It builds the generator *inside* the thread, because a device may belong to the thread that created it.

CPU parallelism beyond that is the integration's choice: a game already has a thread budget, so the generator shares it rather than spawning its own pool.

## 6. Scaling to large worlds: chunks and streaming

### 6.1 The lattice

A world is a lattice of chunks of one shape, inside a `WorldExtent` that may be unbounded along any axis. A chunk is solved as a **region**: itself widened by a *halo* that is solved and then thrown away.

**Why a halo:** a chunk solved with free faces can leave border tiles that no row of neighbours can complete. Solving one cell beyond the chunk and discarding it took the chunks that could not be placed from 29 of 32 to 10 of 32 under a checkerboard order, and to 3 of 63 under a diagonal one.

### 6.2 The schedule

Two rules shape every batch, and between them they are the whole scheduler:

- **Chunks that share a face never ride in one dispatch.** Each would read the other's cells, so a batch takes one parity of the lattice: `(x + y + z) & 1`.
- **A chunk waits for the face neighbours it reads.** Parity 0 goes first and reads nothing; parity 1 follows and reads all four or six of its neighbours.

The set of wanted chunks is **closed** under the face neighbours of its parity-1 members. Without that, a chunk at the edge of the view would be solved against fewer fixed faces than the same chunk in the middle of it, and its tiles would depend on where the player happened to be. The closure is what makes the next section true, and it has a measured cost: it leaves 3.2% of city chunks unplaceable against 1.6% when the frontier is left free.

### 6.3 What determinism means here

For a fixed rule set, prior and configuration:

- The same sequence of requests gives an identical world on any backend, on any number of threads, and whatever a solver's invocation count is. Every choice is a hash of the world seed, the chunk's coordinate and where the solve had got to ([§3.4](#34-seeds-and-randomness)).
- A chunk is solved against its face neighbours and nothing else, so without repairs its tiles are a function of its coordinate: a neighbourhood evicted and asked for again comes back the same. Evicting *part* of one does not. A chunk regenerated beside a neighbour that stayed is solved against that neighbour, which is what keeps the seam invisible, and need not give the tiles it had.
- A repair rewrites cells of the neighbours its halo covers, which makes those chunks depend on the order the world was generated in. Every chunk a repair rewrote is reported as an `Updated` event and counted. A rule set is **streaming-clean** when that count stays zero; the city module set is not, and until a set is, the generated tiles are the source of truth rather than a promise about them.

### 6.4 Streaming

`WorldGenerator::request` takes focus points (a chunk and a radius), and `tick` starts the next batch: the nearest eligible chunks of one parity, then a repair if there is nothing left to start. `poll` commits whatever has finished and returns events. `evict_outside` hands far chunks back so a game can persist them, and `import` puts one back.

That is enough to stay ahead of a player: a 24×8-chunk city around a focus walking at 1.4 m/s generates in a median of 47 ms per 0.5 s tick.

## 7. Phase 2: layered generation (design constraints to keep in mind now)

Phase 2 organises generation into **layers** (for example landscape noise → coastlines via Fractal Jittered Voronoi Partitions → WFC cities), in the spirit of LayerProcGen: each layer generates bounded regions and may read a padded neighbourhood of the layers below it.

Nothing of this exists yet, and it should not be built before Phase 1 is solid. But Phase 1 decisions must keep it possible:

- The solver accepts external constraints as a `Prior` on starting domains (principle 6), so a lower layer can decide "this area is water" before WFC runs.
- Chunks, seeds and the scheduler are about a lattice and a solver seam, not about WFC, so another kind of layer can use the same machinery.
- Outputs are plain tiles that other layers can consume, which supports blending between techniques and several passes over one area.

## 8. GPU specifics

### 8.1 The kernel

- **wgpu** is the reference GPU layer because it is the only mature Rust option that covers Vulkan, Metal, DirectX 12 and WebGPU from one code base, which is required for shipping inside Godot and Bevy on all desktop platforms.
- **The shader is embedded in the binary** with `include_str!`, never read from disk at runtime, because the library ships as a prebuilt extension or inside a game where the source tree does not exist. Shape constants and the mask helpers are substituted into that text per specialisation, which is also what a SPIR-V backend needs: naga bakes overrides into a module per shape anyway.
- **Workgroup storage is the binding limit.** A region's domains, its selection scratch and its epochs must fit the device's per-workgroup allowance (32 KiB here), so the solver refuses a shape that does not and says which numbers it was: 81 tiles in an 8×8×8 chunk with a halo of 3 needs 38 288 B.
- **A dispatch is capped in steps.** A shader that does not terminate takes the host's display driver with it (Windows resets a device after about two seconds), so a region has a hard step budget and reports `StepCap` rather than running on.
- **Kernels are compiled when a game loads,** not at the first dispatch: one specialisation takes about 4 s to translate on this stack, which otherwise looks like a 4.8 s solve. `BlockSolver::warm` compiles a list of shapes and invocation counts up front.

### 8.2 The backend seam

`ComputeBackend` is everything the kernel needs from a device: limits, a pipeline from WGSL text, buffers, writes, a dispatch with its readbacks, and either polling or waiting on a submission. `can_poll` is part of it because a Godot local device cannot be asked without blocking.

`WgpuBackend::from_env` builds a device of its own (and, in this dev container, allows Mesa's non-conformant dozen adapter); `WgpuBackend::from_device` takes one an engine already owns, which is how the Bevy plugin shares Bevy's. Sharing requires the plugin and the engine to agree on a wgpu version, which is why `wave_forge_bevy` tracks the Bevy release that uses ours.

## 9. Errors and observability

- Each crate exposes a small typed error enum (`thiserror`) describing what the *caller* can act on: `ModelError` for a rule set, world or chunk that is wrong, `SolverError` for a batch a solver refused or could not run, `BackendError` and `GpuError` for a device, and `wave_forge::Error` over them for the facade. A region that simply did not solve is a status, not an error ([§4.2](#42-contradictions)).
- What a solve *cost* is reported as data rather than logged: `RegionStats` per region (sweeps, collapses, restarts, backtracks, steps, attempts, and where the last contradiction emptied a cell) and `GeneratorStats` per world (batches, chunks solved, repaired, rewritten by repairs, given up on, and time spent waiting for the solver). **Why:** a parallel, GPU-resident solver cannot be understood by stepping through a debugger, and per-step logging from a shader is not possible at all; a counter read back with the result is.
- Benchmarks and the streaming tests print those numbers, and the invariant checker in `wfc-devtools` decides whether what was generated is valid. Practices and tools are described in [debugging.md](debugging.md).

> **Misaligned today (A-15):** there are no `tracing` spans left (they described the deleted per-collapse loop) and no GPU timestamp queries, so wall-clock timings around a dispatch are all the host sees of the device's time.

## 10. Testing and developer tooling

Covered in depth by [testing.md](testing.md). The architectural requirements are:

- The model, the seams and the scheduler are testable without a device: the facade's contract is checked on the CPU reference solver, and the reference is also the oracle the kernel's propagation is compared against.
- A GPU is required for the solver itself, since there is deliberately no CPU fallback ([vision.md](vision.md#non-goals)).
- End-to-end tests exist for 2D (with PNG rendering of a simple tile set) and 3D (a small city in the style of the reference implementation), and a streaming suite generates whole worlds around a moving focus.
- Dev-only tools render results as images (2D tiles, four orthographic views and voxel models for 3D) so humans *and* LLM-assisted development can inspect output quickly. These tools live outside the shipped library.
- Invariants are checked automatically, not by eye: every adjacency between decided cells, and whole worlds compared cell for cell between runs.

> **Misaligned today (A-16):** unit coverage of the kernel's internals is thin (its behaviour is checked through whole-region results), and there are no golden-image tests although generation is now reproducible.
