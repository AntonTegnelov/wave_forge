# Performance

Wave Forge is meant to generate content *while a game runs*, so runtime speed is the priority that
decides trade-offs here ([vision.md](../product/vision.md)). Compile time and developer ergonomics
come second, binary size a distant third but not ignored. Where a fast design hurts readability, the
fix is documentation that explains **why**, not a slower design.

This page is how we work on performance: the method, where work should run, the build profiles and
the profiling tools. The numbers themselves, each with the build, machine, driver stack and command
that produced it, are in [measurements.md](../research/measurements.md); where performance work
stands and what comes next is in [status.md](../plan/status.md) and [roadmap.md](../plan/roadmap.md).

## Method

1. **Research before optimising.** Before replacing or tuning an algorithm or data structure, find
   out what the current literature says: is there a faster modern algorithm, a parallel or SIMD
   variant, a better data structure for this access pattern? WFC has an active literature (parallel
   propagation, arc-consistency variants, GPU constraint solving), and the answer changes what we
   should build ([literature.md](../research/literature.md)). Write down what was found and why the
   chosen approach fits.
2. **Profile the real workload.** Optimise what the profile shows, not what looks slow. The
   realistic benchmark is the city rule set, in the chunk benchmark and the streaming suite
   ([testing.md](testing.md)), not the toy fixtures: 81 structured variants exercise propagation and
   recovery the way a real set does, and a handful of tiles exercises neither.
3. **Measure the compiled artefact.** Some optimisations only exist in release builds, and a
   profiling build needs debug info without losing optimisation, so [build profiles](#build-profiles)
   are part of the method rather than an afterthought. What a solve cost comes back with its result,
   as counters ([debugging.md](debugging.md)).
4. **Change one thing, measure again, keep the number.** Every performance claim in this repo is
   traceable to a measurement on a named workload and machine, recorded in
   [measurements.md](../research/measurements.md).

### What the per-collapse solver taught

The first solver collapsed one cell per round trip to the device. Replacing it produced the lessons
below; the measurements behind each are in [measurements.md](../research/measurements.md), and the
story of that solver is in [per-collapse-solver.md](../research/per-collapse-solver.md).

- **Profile before moving work.** Propagation was most of the old solver's time, not the transfers,
  so the answer was never "move the transfers" but "stop having a per-collapse loop at all". A
  region now stays in workgroup memory for its whole solve, and a collapse no longer costs more on a
  bigger grid.
- **Batching is the multiplier.** A dispatch lasts as long as its slowest region, so many regions in
  one dispatch cost little more than one.
- **Recovery decides how aggressive selection can be.** Collapsing every local minimum within a
  radius is faster than collapsing one, but only undo from checkpoints makes it safe; with restarts
  alone, the same radius fails many chunks.
- **Measure what bounds a step.** One step's cost at 256 invocations against one invocation showed
  that the kernel is bound by the work it does per cell, not by synchronisation.

## Where work should run

The order of preference is a hypothesis to be tested, never assumed:

- **GPU** wherever parallelism pays for the transfer and dispatch overhead.
- **CPU threads** where a task is parallel but GPU overhead is too high for its size.
- **SIMD** where even thread overhead is too much. SIMD and threading are *orthogonal*: the same code
  can and often should use both.
- **Single-threaded** only where nothing else wins.

Each placement has to be justified by real measurements, because the crossover points depend on
grid size, tile count and hardware. The dev container has the host GPU, so GPU measurements are
available locally, but through a translation layer that distorts exactly the transfer and dispatch
overhead a placement hinges on ([environment.md](environment.md)). Confirm such conclusions on
native hardware.

## Memory hierarchy

Keeping data in the CPU caches and staying inside RAM bandwidth matters as much as instruction count
for a solver that sweeps large grids:

- Possibility data is one contiguous word array in the layout the shader reads, not a per-cell
  allocation, so a sweep is a linear scan and a batch crosses to the device as it is.
- Prefer layouts where propagation touches neighbouring memory, and sizes where the working set of a
  region fits in cache.
- Profile to find where bandwidth, not compute, is the limit; callgrind simulates the caches
  ([below](#profiling-in-the-dev-container)).

## Code-level priorities

- **Static dispatch.** The solver and the generator are generic over their seams, and the kernel is
  specialised per region shape; nothing on a hot path pays for a virtual call. Specialisation is not
  only about calls: the same kernel with its mask words in a loop rather than written out is
  measurably slower, because an indexed mask lands in scratch memory.
- **Nothing crosses the boundary per collapse.** A batch of regions costs one upload, one dispatch
  and one readback; a region's domains stay in workgroup memory for the whole solve.
- **Fewer per-iteration allocations:** bind groups, encoders and staging buffers are created once
  and reused.

## Scalability, with realistic bounds

The generator should build very large worlds, but bounded ones: WFC will never be practical on a
grid the size of a galaxy, and no game needs one at once. The design target is **continuous solving
of the region around the player**, streaming chunks in and out with constrained borders, rather than
one enormous grid. Keep that in mind when choosing data structures: the working set is a region and
its neighbours, not the world.

## Build profiles

Compile time is traded away freely, but every setting that affects speed must be *measured*, and
measurements are only meaningful in the right profile. The settings are in the root `Cargo.toml`.

| Profile | Command | Use it for | Never use it for |
|---|---|---|---|
| `dev` | `cargo build`, `cargo run` | Everyday editing and debugging | Any timing: unoptimised code has a completely different performance profile |
| `test` | `cargo test` | Correctness tests | Performance assertions |
| `release` | `cargo build --release` | Anything shipped, and final benchmark numbers | Profiling with sampling or call-graph tools, because it has no symbols |
| `profiling` | `cargo build --profile profiling` | Profilers (callgrind, perf, Tracy, flame graphs) | Shipping: it carries full debug info |

Binaries land in `$CARGO_TARGET_DIR/<profile>/`, for example `target/profiling/wave-forge`.

### `release`

- **`lto = true` (fat link-time optimisation).** Lets LLVM inline and specialise across crate
  boundaries: our crates, `wgpu` and the rest. That matters most for the small, hot, generic
  functions that static dispatch is supposed to make cheap. Thin LTO is typically 10 to 20% faster
  than none; fat LTO can go further but costs the most compile time, which is the trade-off this
  project accepts.
- **`codegen-units = 1`.** Compiling each crate as one unit gives the optimiser the whole crate at
  once, for better inlining and dead-code removal, at the cost of parallel compilation.
- **`strip = true`.** Removes symbols from shipped binaries. Size matters for games and the Godot
  Asset Store, and nobody profiles a shipped build.

### `profiling`

- **Inherits `release`,** so the optimiser makes the same decisions as in a shipped build. Profiling
  an unoptimised or differently optimised build points at bottlenecks that do not exist in release.
- **`debug = true`, `strip = false`.** Profilers need symbols and line tables to attribute cost to
  functions, including inlined ones. Debug info does not change generated code.

### Options not enabled, and why

| Option | Expected effect | Why it is not on |
|---|---|---|
| `panic = "abort"` | Slightly faster, smaller binaries (no unwinding tables) | Must be measured first. It also interacts with FFI: an engine embedding the library (the GDExtension) should never see a Rust unwind cross the boundary, which argues *for* it, so decide it together with the engine integrations. |
| `-C target-cpu=native` | Enables the newest SIMD instructions of the build machine | Produces binaries that crash on older CPUs, so it can never be used for anything shipped. Wide SIMD should instead use runtime feature detection (for example `std::arch` with `is_x86_feature_detected!`) so one binary serves every player. Acceptable only for local experiments, labelled as such. |
| Profile-guided optimisation (PGO) and BOLT | Often 10% or more | Needs representative workloads (the streaming suite) and a more complex build. Revisit once hot paths have stabilised. |
| Another allocator (mimalloc, jemalloc) | Can be large for allocation-heavy code | The hot path should not allocate per step at all; measure after allocation hot spots are removed rather than masking them. |
| `lto = "thin"` in `profiling`, for faster builds | A shorter profiling build | Profiles would describe a different binary than `release`. |

## Profiling in the dev container

- **GPU work** is measured by the counters a solve returns and wall clock around the dispatch
  ([debugging.md](debugging.md)); CPU profilers only see the CPU side waiting on the driver. Timings
  through the container's driver stack are not native ([environment.md](environment.md)).
- **callgrind** (`valgrind --tool=callgrind`) works without extra permissions and can also simulate
  the L1 and last-level caches (`--cache-sim=yes`), which answers "is this loop cache-friendly"
  questions deterministically. It is slow (tens of times slower than native), so run it on small
  inputs or on CPU-only benchmarks.
- **perf** is installed but cannot open performance counters inside the container
  (`perf_event_paranoid` and the container's permissions). Enabling it requires changing the
  container's capabilities or the WSL kernel setting, which widens the sandbox. That is a deliberate
  decision, not a default.
- **In Godot**, the node's own time per frame and its slowest frame's breakdown come from `stats()`
  ([debugging.md](debugging.md)).

## Rules of thumb

1. Time `release`; profile `profiling`; never time `dev`.
2. Change one setting at a time and record the before and after numbers with the command that
   produced them, in [measurements.md](../research/measurements.md).
3. Keep benchmark inputs fixed and large enough that per-run startup (device creation, shader
   compilation) does not dominate; call `BlockSolver::warm` first, and report setup and steady state
   separately.
