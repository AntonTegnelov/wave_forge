# Performance: priorities, method and placement

Wave Forge is meant to generate content *while a game runs*, so runtime speed is the priority that
decides trade-offs here. Compile time and developer ergonomics come second, binary size a distant
third but not ignored. Where a fast design hurts readability, the fix is documentation that explains
**why**, not a slower design.

This page records how we work on performance. The concrete plan and its tasks live in
[roadmap.md](roadmap.md) and [status.md](status.md) (A-4, A-10, A-11, A-12); the current work is
issue [#7](https://github.com/AntonTegnelov/wave_forge/issues/7).

## Method

1. **Research before optimising.** Before replacing or tuning an algorithm or data structure, find
   out what the current literature says: is there a faster modern algorithm, a parallel or SIMD
   variant, a better data structure for this access pattern? WFC has an active literature (parallel
   propagation, arc-consistency variants, GPU constraint solving), and the answer changes what we
   should build. Write down what was found and why the chosen approach fits.
2. **Profile the real workload.** Optimise what the profile shows, not what looks slow. The realistic
   benchmark is the city rule set and the opt-in stress suite ([testing.md](testing.md)), not the toy
   fixtures: a two-tile grid measures per-collapse overhead, a structured 81-variant set measures
   propagation.
3. **Measure the compiled artefact.** Some optimisations only exist in release builds, and a
   profiling build needs debug info without losing optimisation, so build profiles are part of the
   method rather than an afterthought. Timelines come from `--trace-chrome`
   ([debugging.md](debugging.md)).
4. **Change one thing, measure again, keep the number.** Every performance claim in this repo should
   be traceable to a measurement on a named workload and machine.

## Where work should run

The order of preference is a hypothesis to be tested, never assumed:

- **GPU** wherever parallelism pays for the transfer and dispatch overhead.
- **CPU threads** where a task is parallel but GPU overhead is too high for its size.
- **SIMD** where even thread overhead is too much. SIMD and threading are *orthogonal*: the same code
  can and often should use both.
- **Single-threaded** only where nothing else wins.

Each placement has to be justified by real measurements, because the crossover points depend on grid
size, tile count and hardware. The dev container has the host GPU
([development.md](development.md)), so GPU measurements are available locally; anything that hinges
on transfer or dispatch overhead should be confirmed on native hardware, since translation layers
distort exactly those numbers.

## Memory hierarchy

Keeping data in L1–L3 and staying inside RAM bandwidth matters as much as instruction count for a
solver that sweeps large grids:

- Possibility data should be one contiguous word array shared by CPU and GPU layouts (A-4), not a
  per-cell allocation, so a sweep is a linear scan.
- Prefer layouts where propagation touches neighbouring memory, and sizes where the working set of a
  region fits in cache.
- Profile to find where bandwidth, not compute, is the limit; cache simulation is available through
  callgrind ([debugging.md](debugging.md)).

## Code-level priorities

- **Static dispatch** in place of `Box<dyn …>` and `async_trait` wherever it measurably helps; the
  solver's hot paths should not pay for virtual calls or boxed futures (A-12).
- **Less cloning.** Grid states and buffers are the big ones; passing the whole grid through the
  CPU/GPU boundary every collapse is the current bottleneck (A-10).
- **Fewer per-iteration allocations:** bind groups, encoders and staging buffers should be created
  once and reused.

## Scalability, with realistic bounds

The generator should build very large worlds, but bounded ones: WFC will never be practical on a
grid the size of a galaxy, and no game needs one at once. The design target is **continuous solving
of the region around the player**, streaming regions in and out with constrained borders (A-13),
rather than one enormous grid. Keep that in mind when choosing data structures: the working set is a
region and its neighbours, not the world.

## Baseline

Measured on an RTX 3070 through the container's translation layer, release build, before the #7 work.
From the stress suite ([testing.md](testing.md)):

| Workload | Cells | Tiles | Run | Cells/s |
|---|---|---|---|---|
| Permissive 2-tile 24³ | 13824 | 2 | 96.0 s | 144 |
| City 24×24×8 | 4608 | 81 | 45.3 s | 102 |
| City 48×48×10 | 23040 | 81 | 403.8 s | 57 |

Throughput falls as the grid grows even though the rules do not change, which points at per-collapse
overhead rather than propagation cost: the whole grid crosses the CPU/GPU boundary on every collapse,
so each one costs more on a larger grid (A-10). A trace of a smaller run splits roughly half into
propagation and the rest between downloading the grid, selecting a cell, entropy and upload, with
about six synchronisation points per collapse.

Two more numbers are worth keeping in view: backtracking-heavy runs are far slower than the median
(one 48×48×10 run exhausted its iteration budget after ~11000 undos), and the connectivity-constrained
8×8×5 city ranges from 5 to 136 seconds. Search cost, not just throughput, is part of the problem.
