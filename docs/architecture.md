# Architecture

This document describes the **target** architecture of Wave Forge and, above all, *why* it is shaped this way. It is derived from the goals in [vision.md](vision.md), not from the current code: parts of the code base were written before these goals were set and still contradict them. Wherever that is the case you will find a callout like this:

> **Misaligned today (A-n):** what the code does instead. See [status.md](status.md#alignment-tasks).

Details that are local to one module belong in code comments next to that code. This document covers the big picture: the pieces, how they fit, and the reasoning that would otherwise be invisible from reading any single file.

## 1. Guiding principles

Each principle is a consequence of the vision's priority list (performance first, then correctness, integration, binary size, ergonomics).

1. **One engine-agnostic core, thin integrations.** Engines change their APIs often; generation logic should not. Keeping the solver free of engine types means it can be tested, benchmarked and profiled without an engine, and the Bevy plugin and Godot extension stay small adapters that are cheap to keep up to date.
2. **Measure, then place work.** GPU, CPU threads, SIMD and plain code each have different overheads. The architecture keeps these as interchangeable *backends* behind one solver interface so that benchmarks, not guesses, decide what runs where, and moving work between tiers doesn't require restructuring.
3. **Static dispatch on hot paths.** Anything executed per cell, per tile or per propagation step must be monomorphized (generics, enums), so the compiler can inline and vectorize it. Dynamic dispatch (`dyn Trait`, `async_trait`) is only acceptable at coarse boundaries such as choosing a backend once per generator, where its cost is paid once rather than millions of times.
4. **Data lives where it is processed.** Moving data between CPU and GPU, or between threads, costs more than most computations on it. State stays on the device that is working on it for as long as possible, and crosses the boundary in bulk and rarely.
5. **Bounded, deterministic regions.** Worlds are generated as bounded chunks with deterministic seeds, so memory stays bounded, work can be parallelized per chunk, and any chunk can be regenerated identically.
6. **Constraints in, data out.** The solver accepts constraints (pre-set cells, masks, borders from neighbouring chunks or other layers) and returns plain data. This is what makes chunk stitching and Phase 2 layering possible without special cases.

## 2. Component overview

```
┌────────────────────────────────────────────────────────────────────────┐
│ Integrations (thin)          wave_forge_bevy     wave_forge_godot       │
├────────────────────────────────────────────────────────────────────────┤
│ Public library API           wave_forge  (builder, generate region,     │
│                              streaming/chunk scheduler, results)        │
├────────────────────────────────────────────────────────────────────────┤
│ Solver                       observe → collapse → propagate, generic    │
│                              over a Backend (static dispatch)           │
├───────────────────────┬───────────────────────┬────────────────────────┤
│ Backend: GPU (wgpu)   │ Backend: CPU threads  │ Kernels: SIMD bitsets  │
├───────────────────────┴───────────────────────┴────────────────────────┤
│ Model                        topology (2D/3D), tiles, compiled rules,   │
│                              possibility storage, seeds/RNG             │
├────────────────────────────────────────────────────────────────────────┤
│ Authoring formats            RON/JSON rule files → compiled rules       │
└────────────────────────────────────────────────────────────────────────┘
 Dev-only (never shipped in the library): CLI, PNG/orthographic renderers,
 benchmarks, profiling and inspection tools
```

**Why this layering:** dependencies only point downwards, so the model and solver can be used (and fuzzed, benchmarked, profiled) with nothing above them, and engine integrations can never leak engine types into the core.

> **Misaligned today (A-1):** the workspace is `wfc-core` (model + an unused CPU runner), `wfc-rules` (formats + rule types), `wfc-gpu` (GPU solver, which is also the de-facto public API via `GpuAccelerator`), and the root `wave_forge` package, which is a **binary** CLI rather than the library facade. There is no backend abstraction, no public generation API, and no integration crates.

## 3. Model

### 3.1 Topology: 2D and 3D as first-class

The solver works on a *topology*: a set of cells and, for each cell, a fixed, ordered list of neighbour directions. A square 2D grid has 4 directions; a cubic 3D grid has 6. Rules are expressed per direction.

**Why not "3D with depth 1" for 2D:** it wastes memory and work on two dead directions, makes rule files for 2D tile sets mention axes that don't exist, and produces wrong results with periodic borders (a depth-1 grid wraps onto itself along Z, silently adding a "tile must be compatible with itself" constraint). Making the direction count part of the topology keeps both cases exact and lets kernels specialise on it at compile time.

> **Misaligned today (A-2):** grids are always 3D and rules always have 6 axes; 2D is only possible as depth 1, with the periodic-border problem above. Axis names in rule files are hard-coded to `±x/±y/±z`.

### 3.2 Tiles and compiled rules

Authors describe tiles (with weights and allowed symmetries) and adjacency by name in rule files. Before solving, rules are **compiled** into a dense form: for each (direction, tile) a bitset of compatible neighbour tiles. Symmetry variants (rotations, flips) are expanded during compilation.

**Why compile:** propagation needs "which neighbour tiles does *any* of my remaining tiles allow in this direction?" This becomes a union of precomputed bitsets, which is branch-free and SIMD/GPU friendly. Named, symmetric rules exist only for authors; the hot path never sees them.

> **Misaligned today (A-3):** tile weights bias collapse on the GPU (`GpuAccelerator::with_tile_weights`) but not entropy, which still counts tiles (A-7). Rotated variants can be generated from connector-based module sets in Rust (`wfc_rules::modules`), but the RON loader still only creates identity variants.

### 3.3 Possibility storage

Each cell holds a bitset of still-possible tiles. All cells' bitsets are stored in **one contiguous array of machine words** (`cells × words_per_cell`), using the same layout on CPU and GPU.

**Why:** a contiguous array is cache-friendly, can be processed with SIMD, and can be uploaded to or mapped from the GPU without repacking. Per-cell heap allocations cause pointer chasing and force a pack/unpack step on every transfer.

> **Misaligned today (A-4):** `PossibilityGrid` stores a `Vec<BitVec>` (one allocation per cell) and is repacked into `u32` words on every GPU upload and unpacked on every download.

> **Resolved (A-5):** propagation now handles every 32-bit word of a cell. WGSL needs constant-size function-local arrays, so masks hold up at most 8 words (256 tile variants) and the host rejects larger rule sets. Realistic tile sets exceed 32 quickly: the city E2E set has 81 variants.

### 3.4 Seeds and randomness

Every random decision derives from `(world seed, region coordinate, layer, step)` through a counter-based RNG, never from a global or thread-local RNG. Parallel work must break ties by deterministic keys (for example cell index mixed with the seed), never by which thread finished first.

**Why:** determinism across machines, thread counts and generation order is what makes on-demand chunk regeneration and multiplayer consistent (see [vision.md](vision.md#determinism)).

> **Misaligned today (A-6):** the GPU run loop picks tiles with an unseeded thread RNG, so `--seed` has no effect and runs are not reproducible.

## 4. Solver

### 4.1 The algorithm

Wave function collapse repeats three steps until every cell has one tile or a contradiction occurs:

1. **Observe:** pick the uncollapsed cell with the lowest entropy. We use *weighted* Shannon entropy with a small deterministic noise term.
2. **Collapse:** choose one of its remaining tiles at random, **weighted** by tile weight.
3. **Propagate:** remove neighbour possibilities that are no longer supported, transitively, until nothing changes (arc consistency).

**Why weights:** weights are the main art-direction control ("mostly grass, some rocks"). Without them output is uniformly noisy.
**Why noise on entropy:** without it, ties are broken by scan order, which produces visible directional artefacts (generation sweeping from the origin corner).

> **Misaligned today (A-7):** ties are broken by lowest cell index (scanline order), collapse ignores weights, and the "weighted count" heuristic silently falls back to plain count.

### 4.2 Contradictions

A contradiction (a cell with no possible tiles) is normal for WFC, not an exceptional error. Within a bounded region the solver recovers by **bounded backtracking** or, failing that, **restarting the region with a derived seed**. Only when a budget is exhausted is failure reported to the caller.

**Why:** games cannot show an error dialog because a chunk failed. Keeping regions bounded (section 6) keeps restarts cheap.

> **Misaligned today (A-9):** a contradiction aborts the whole run. There is a large, generic "error recovery" framework in `wfc-gpu/src/utils/error_recovery` and recovery hooks on `GpuAccelerator`, but nothing in the run path uses them.

### 4.3 Where each step should run

This is the central performance question and must be answered by measurement ([#7](https://github.com/AntonTegnelov/wave_forge/issues/7)). The current working hypotheses:

- **Propagation** dominates run time. On the GPU it runs as bulk-synchronous passes over a frontier of changed cells. That pays off when frontiers are large; a single collapse usually produces a small frontier, where a CPU queue-based propagator (AC-3/AC-4 style, SIMD bitset unions) is likely faster than a GPU round-trip.
- **Observation** (a global minimum search) is embarrassingly parallel and a good GPU or multi-threaded CPU candidate *if the data is already there*.
- **Batching is the multiplier:** many cells far enough apart can be observed and collapsed in the same iteration because their propagation fronts don't interact, turning one-cell-per-round-trip into many-cells-per-dispatch.

**Why keep the solver generic over backends:** the answer will differ by grid size, tile count and hardware, and may be "hybrid" (CPU for small fronts, GPU for large ones). A statically-dispatched `Backend` trait lets each combination be benchmarked without rewriting the solver.

> **Misaligned today (A-10):** every iteration collapses one cell, uploads the **entire grid** to the GPU, propagates, and downloads the **entire grid** again. Transfer cost therefore grows with grid size on every single collapse; this is currently the dominant cost and the main blocker for scale.

> **Misaligned today (A-11):** the GPU propagation shader updates neighbours with a non-atomic load-AND-store, so concurrent writes can lose a restriction. The host compensates with a final full-grid pass per collapse. Correct, but expensive; the kernel should be made race-free instead.

## 5. Dispatch, async and threading model

- **Hot paths are generic.** The solver is generic over its backend and topology; kernels are specialised per word size and direction count. **Why:** inlining and auto-vectorisation are only possible when the compiler sees concrete types.
- **Async only at I/O and GPU boundaries.** GPU readback is naturally asynchronous, and engines schedule work on their own task systems. The library exposes non-blocking "start region / poll region" style operations and does **not** depend on a particular async runtime. **Why:** Bevy and Godot each bring their own executors; a library that requires Tokio forces a second runtime into every game.
- **CPU parallelism via a work-stealing pool (e.g. rayon) or the engine's task pool,** chosen by the integration. **Why:** games already have a thread budget, so the generator must be able to share it rather than spawn its own threads.

> **Misaligned today (A-12):** strategies are `Box<dyn …>` trait objects called through `async_trait` (a heap allocation per call) on per-iteration paths; `wfc-gpu` depends on Tokio; async functions block internally with `pollster::block_on` and `device.poll(Wait)`.

## 6. Scaling to large worlds: regions and streaming

- The world is partitioned into **regions (chunks)** of a bounded size chosen per rule set.
- A region is solved with its **borders constrained** by already-generated neighbours (or by overlap that is discarded afterwards), so adjacent regions agree.
- A **scheduler** in the public API keeps regions around one or more focus points generated, prioritised by distance, and evicts far ones. Evicted regions regenerate identically thanks to deterministic seeds.

**Why regions instead of one big grid:** memory and failure probability stay bounded, regions are natural units of parallelism (many regions on many threads, or many regions batched into one GPU dispatch), and the approach matches how games stream worlds. This is also how the infinite-city reference implementation works.

> **Misaligned today (A-13):** the solver only knows one monolithic grid. There is a "subgrid" strategy intended for splitting a large grid for parallel propagation, but it does not produce consistent borders, relies on a CPU-side grid the coordinator no longer has, and is disabled.

## 7. Phase 2: layered generation (design constraints to keep in mind now)

Phase 2 organises generation into **layers** (for example landscape noise → coastlines via Fractal Jittered Voronoi Partitions → WFC cities), in the spirit of LayerProcGen: each layer generates bounded regions and may read a padded neighbourhood of the layers below it.

Nothing of this exists yet, and it should not be built before Phase 1 is solid. But Phase 1 decisions must keep it possible:

- The solver must accept **external constraints** (section 1, principle 6), so a lower layer can decide "this area is water" before WFC runs.
- Regions, seeds and scheduling (section 6) must be generic enough to serve non-WFC layers too, not be WFC-specific.
- Outputs must be plain data that other layers can consume, supporting **blending** between techniques and **multiple passes** over the same area.

## 8. GPU specifics

- **wgpu** is the GPU layer because it is the only mature Rust option that covers Vulkan, Metal, DirectX 12 (and WebGPU) from one code base, which is required for shipping inside Godot and Bevy on all desktop platforms.
- **Shaders are embedded in the binary** (for example with `include_str!` plus compile-time specialisation), never read from disk at runtime. **Why:** the library ships as a prebuilt GDExtension or inside a game where the source tree does not exist.
- Buffers are allocated per region-size class and reused across regions, avoiding allocation churn during streaming.

> **Misaligned today (A-14):** `ShaderManager` reads shader "variants" and a component registry from the build directory at runtime (compile-time paths via `env!`), so a built binary only works on the machine that built it. The "variants" are copies of the base shaders; the component/registry system around them is scaffolding that does not affect the pipelines.

## 9. Errors and observability

- Each crate exposes a small typed error enum (`thiserror`) describing what the *caller* can act on. **Why:** callers need "retry / give up / configuration is wrong", not a generic framework.
- Instrumentation uses structured spans (the `tracing` ecosystem) around every stage and region, so the same data can feed logs, a timeline profiler and dev tools. **Why:** a multi-threaded, GPU-async generator cannot be understood by stepping through a debugger; you need timelines of what ran where and when. Practices and tools are described in [debugging.md](debugging.md).
- GPU work is timed with timestamp queries where supported.

> **Misaligned today (A-15):** there are two parallel `GpuError` hierarchies (`utils::error` and `utils::error_recovery`) with string-based conversions between them, and a `DebugVisualizer` that is never enabled. Spans cover the GPU run loop and propagation passes, but most of the code still uses plain `log` calls, and there are no GPU timestamp queries.

## 10. Testing and developer tooling

Covered in depth by [testing.md](testing.md). The architectural requirements are:

- The model and solver are testable without an engine. A GPU is required, since there is deliberately no CPU fallback ([vision.md](vision.md#non-goals)); logic that does not touch the GPU (rule compilation, packing, invariants) is unit-tested without one.
- End-to-end tests exist for 2D (with PNG rendering of a simple tile set) and 3D (a small city in the style of the reference implementation).
- Dev-only tools render results as images (2D tiles, four orthographic views for 3D) so humans *and* LLM-assisted development can inspect output quickly. These tools live outside the shipped library.
- Invariants (every adjacency satisfied, determinism for a fixed seed) are checked automatically, not by eye.

> **Misaligned today (A-16):** end-to-end tests, invariant checks and image tools exist for the reference rule sets, but unit coverage of the solver internals (entropy selection, propagation strategy, synchronisation) is thin, and without determinism (A-6) there are no golden-image tests.
