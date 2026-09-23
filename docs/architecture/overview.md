# Architecture overview

How Wave Forge is put together and why. The goals it serves are in
[vision.md](../product/vision.md); this page gives the principles, the crates and their seams, and
how data flows from a rule file or a pack to an engine. The detail is in four companions:

- [solver.md](solver.md): the model (rules, domains, priors, hashing) and the block kernel that
  solves a region on the GPU.
- [world.md](world.md): chunks, the schedule, repairs, determinism and streaming.
- [stages.md](stages.md): generation as a pack of stages, of which WFC is one kind.
- [engine-integration.md](engine-integration.md): what the library hands Godot and Bevy, and how
  each engine takes it.

These pages describe the design and its reasons. What is built is in the [reference](../reference/),
where things stand is in [status.md](../plan/status.md), and the numbers behind each decision are
in [measurements.md](../research/measurements.md).

## Guiding principles

Each principle follows from the vision's priority list (performance first, then correctness,
integration, binary size, ergonomics).

1. **One engine-agnostic core, thin integrations.** Engines change their APIs often; generation
   logic should not. Keeping the core free of engine types means it can be tested, benchmarked and
   profiled without an engine, and the Bevy plugin and Godot extension stay small adapters that are
   cheap to keep up to date.
2. **Measure, then place work.** GPU, CPU threads, SIMD and plain code each have different
   overheads. Where work runs is decided by benchmarks, not by guesses, and the seams are drawn so
   that moving it does not mean restructuring.
3. **Static dispatch on hot paths.** Anything executed per cell, per tile or per propagation step is
   monomorphized (generics, enums), so the compiler can inline and vectorize it. Dynamic dispatch is
   acceptable only at coarse boundaries, such as choosing a backend once per generator.
4. **Data lives where it is processed.** Moving data between CPU and GPU, or between threads, costs
   more than most computations on it. State stays on the device working on it for as long as
   possible and crosses in bulk, rarely.
5. **Bounded, deterministic regions.** Worlds are generated as bounded chunks with deterministic
   seeds, so memory stays bounded, work parallelises per chunk, and any chunk can be regenerated
   identically.
6. **Constraints in, data out.** The solver accepts constraints (a prior on starting domains,
   borders from neighbouring chunks) and returns plain data. That is what lets chunks stitch and
   lets other stages drive WFC without special cases.

## Crates

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Integrations (thin)     wave_forge_godot: WaveForgeWorld, WaveForgeStages  │
│                         wave_forge_bevy:  WaveForgePlugin,                 │
│                                           WaveForgeStagesPlugin            │
├────────────────────────────────────────────────────────────────────────────┤
│ wave_forge              Builder, WorldGenerator (store + scheduler +       │
│ (the library facade)    solver + repairs), Worker; products (InstanceSet,  │
│                         InstanceId, NavSource), YUpSpace; stages (Pack,    │
│                         Runtime, StageWorker); towns (TownSolver, WfcTowns)│
├────────────────────────────────────────────────────────────────────────────┤
│ wfc-core                the model: chunks and extents, compiled rules,     │
│                         domains, Prior, chunk store, hashing; the Solver   │
│                         seam; the CPU reference solver                     │
├────────────────────────────────────────────────────────────────────────────┤
│ wfc-gpu                 BlockSolver: one workgroup solves one region per   │
│                         dispatch, over a ComputeBackend (wgpu, or another) │
├────────────────────────────────────────────────────────────────────────────┤
│ wfc-rules               RON tile sets and module sets (connectors,         │
│                         derived rotations) → compiled adjacency            │
└────────────────────────────────────────────────────────────────────────────┘
 Dev-only, never shipped: wfc-devtools (the CLI, the invariant checker, the
 reference rule sets, the renderers, model export and the streaming and
 order tests)
```

Dependencies only point downwards. The model and the seams can be used, fuzzed, benchmarked and
profiled with nothing above them, and an engine integration cannot leak engine types into the core.

## Seams

A seam is a trait the core owns, placed where something outside it (a device, an engine, another
algorithm) has to be swappable. There are four.

| Seam | Crate | What it hides | Why it exists |
|---|---|---|---|
| `Solver` | wfc-core | how a batch of regions is solved | a GPU solver, the CPU reference and test fakes behind one job-based interface ([solver.md](solver.md#the-solver-seam)) |
| `ComputeBackend` | wfc-gpu | the device the kernel runs on | engines own devices differently ([solver.md](solver.md#the-backend-seam)) |
| `TownSolver` | wave_forge::towns | how a Solve stage turns a site into a town | the stage runtime runs on the CPU while towns need a GPU solver on a thread of its own ([stages.md](stages.md#how-wfc-joins-one-bounded-world-per-site)) |
| products | wave_forge::products | nothing; it is the boundary itself | engines receive typed, engine-neutral data rather than tile ids ([engine-integration.md](engine-integration.md#the-boundary-products-not-rendering)) |

There are two device seams rather than one because engines differ in what they own. Bevy owns a
wgpu device and shares it, so `wave_forge_bevy` hands its device to `Builder::build_on` and the
kernel runs unchanged. Godot owns a `RenderingDevice` that takes SPIR-V and blocks in `sync()`, so
`wave_forge_godot` runs the generator on a `Worker` with a wgpu device of its own, which keeps
Godot's frame loop free without translating a kernel. A backend over Godot's `RenderingDevice`
would save the second device; the seam is there for it, and whether it is worth building is a
measurement on a desktop ([engine-integration.md](engine-integration.md#godot-the-solver-stays-on-its-own-device)).

## Two ways to generate

The library generates a world in one of two shapes, and both reach an engine the same way.

**A streamed WFC world.** A rule file (a tile set or a module set) compiles to a `Ruleset`, and a
`Prior` says what each cell may start as. `Builder` builds a `WorldGenerator`, which a game drives
with focus points: `request`, `tick`, `poll`. It solves chunks in batches on the GPU, stitches them,
repairs the ones borders made unsolvable, and reports `ChunkEvent`s. This is the infinite city of
the MVP. [world.md](world.md) covers it.

```
rule file ──compile──▶ Ruleset ─┐
                Prior ──────────┼─▶ WorldGenerator ──batches──▶ Solver (BlockSolver on a device)
       focus points ────────────┘         │
                                          ▼
                 ChunkEvent, tiles ─▶ products (InstanceSet, NavSource) ─▶ engine
```

**A pack of stages.** A pack (`*.world.ron`) names stages that read each other: height fields,
settlement sites, levelled ground, towns solved with WFC, scattered points. `Runtime` generates the
chunks of each stage a request needs, providers first, each through a view bounded by the stage's
declared reach, so the result is the same in any order. WFC joins as the Solve stage, one bounded
world per site, through the `TownSolver` seam. [stages.md](stages.md) covers the design and
[reference/packs.md](../reference/packs.md) the format.

```
pack ──load, check──▶ Pack ─▶ Runtime ──per stage, providers first──▶ Field, Sites, Tiles, Points
                                 │                                          │
                        TownSolver (WfcTowns over a Solver)                 ▼
                                                                  products ─▶ engine
```

## Threads

Nothing in the library awaits and no crate depends on an async runtime. What differs between
engines is where the waiting happens, so the library offers two ways to wait:

- **Polling.** `WorldGenerator::poll` never blocks when the backend can poll. Bevy polls once per
  frame from its own systems, on its own device.
- **A thread of the library's.** `Worker` runs a `WorldGenerator` on a std thread behind two
  channels, and `StageWorker` does the same for a stage `Runtime`, stepping eight products at a
  time. Each builds its generator or runtime *inside* the thread, because a device may belong to the
  thread that created it. The engine drains events on its own thread. The Godot node uses both
  workers, and the Bevy stages plugin uses `StageWorker`.

CPU parallelism beyond that is the integration's choice: a game already has a thread budget, so the
library shares it rather than spawning a pool of its own.

## Errors and observability

- Each crate exposes small typed errors for what a *caller* can act on: `ModelError` for a rule set,
  world or chunk that is wrong; `LoadError`, `ModuleError` and `TileSetError` for rule files;
  `SolverError` for a batch a solver refused or could not run; `BackendError` and `GpuError` for a
  device; `PackError` for a pack that does not load, naming the stage; `StageError` and `TownError`
  while stages run; `NavSourceError` for navigation source that cannot be built; and
  `wave_forge::Error` over the facade. A region that did not solve is a status, not an error
  ([solver.md](solver.md#contradictions)).
- What work *cost* is reported as data, not logged: `RegionStats` per region, `GeneratorStats` per
  world, and the Godot node's `stats()` per frame. A GPU-resident solver cannot be stepped through
  in a debugger and a shader cannot log, but a counter read back with the result can be printed,
  asserted on and compared. [debugging.md](../guides/debugging.md) describes the counters and the
  tools.
- There are no GPU timestamp queries and no `tracing` spans yet, so wall-clock time around a
  dispatch is all the host sees of the device's time ([status.md](../plan/status.md#known-limits)).

## Testing requirements

What the architecture needs from its tests; the tests themselves are in
[testing.md](../guides/testing.md).

- The model, the seams and the scheduler are testable without a device: the facade's contract runs
  on the CPU reference solver, which is also the oracle the kernel's propagation is compared
  against.
- The GPU solver needs a device, deliberately, because there is no CPU fallback
  ([vision.md](../product/vision.md#non-goals)). CI provides one in software (lavapipe).
- Invariants are checked by code, not by eye: every adjacency between decided cells, whole worlds
  compared cell for cell between runs and devices, and the same world generated in different
  orders.
- Dev-only tools render results (2D tiles, orthographic views, voxel models) so people and
  assistants can inspect output quickly. They live outside the shipped library.
