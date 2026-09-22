# Performance: priorities, method and placement

Wave Forge is meant to generate content *while a game runs*, so runtime speed is the priority that
decides trade-offs here. Compile time and developer ergonomics come second, binary size a distant
third but not ignored. Where a fast design hurts readability, the fix is documentation that explains
**why**, not a slower design.

This page records how we work on performance. The concrete plan and its tasks live in
[roadmap.md](roadmap.md) and [status.md](status.md); the solver redesign was issue
[#7](https://github.com/AntonTegnelov/wave_forge/issues/7).

## Method

1. **Research before optimising.** Before replacing or tuning an algorithm or data structure, find
   out what the current literature says: is there a faster modern algorithm, a parallel or SIMD
   variant, a better data structure for this access pattern? WFC has an active literature (parallel
   propagation, arc-consistency variants, GPU constraint solving), and the answer changes what we
   should build. Write down what was found and why the chosen approach fits.
2. **Profile the real workload.** Optimise what the profile shows, not what looks slow. The realistic
   benchmark is the city rule set, in the chunk benchmark and the streaming suite
   ([testing.md](testing.md)), not the toy fixtures: 81 structured variants exercise propagation and
   recovery the way a real set does, and a handful of tiles exercises neither.
3. **Measure the compiled artefact.** Some optimisations only exist in release builds, and a
   profiling build needs debug info without losing optimisation, so build profiles are part of the
   method rather than an afterthought. What a solve cost comes back with its result, as counters
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

- Possibility data is one contiguous word array in the layout the shader reads, not a per-cell
  allocation, so a sweep is a linear scan and a batch crosses to the device as it is.
- Prefer layouts where propagation touches neighbouring memory, and sizes where the working set of a
  region fits in cache.
- Profile to find where bandwidth, not compute, is the limit; cache simulation is available through
  callgrind ([debugging.md](debugging.md)).

## Code-level priorities

- **Static dispatch.** The solver and the generator are generic over their seams, and the kernel is
  specialised per region shape; nothing on a hot path pays for a virtual call. Specialisation is not
  only about calls: the same kernel with its mask words in a loop rather than written out is 2.5×
  slower, because an indexed mask lands in scratch memory.
- **Nothing crosses the boundary per collapse.** A batch of regions costs one upload, one dispatch
  and one readback; a region's domains stay in workgroup memory for the whole solve.
- **Fewer per-iteration allocations:** bind groups, encoders and staging buffers should be created
  once and reused.

## Scalability, with realistic bounds

The generator should build very large worlds, but bounded ones: WFC will never be practical on a
grid the size of a galaxy, and no game needs one at once. The design target is **continuous solving
of the region around the player**, streaming chunks in and out with constrained borders, rather than
one enormous grid. Keep that in mind when choosing data structures: the working set is a
region and its neighbours, not the world.

## Where it stands

Every number here describes one build on one machine and driver stack (an RTX 3070 through the
container's dozen translation layer, release), and [solver-fit.md](solver-fit.md) records each one
with its protocol. Two are the ones to know:

| | |
|---|---|
| One 8×8×8 city chunk, 256 chunks in one dispatch | **0.18 ms per chunk**, against 3.9 ms on one CPU thread and 0.79 ms spread over 24 |
| A 24×8-chunk city around a walking focus | **median 47 ms** per 0.5 s tick, p90 61 ms |
| A player walking and running an unbounded city in real time | **no chunk in view ever late**, at most 0.18 ms of a frame on the main thread |

What the redesign that produced them changed, and what each change was worth:

- **The region stays on the device.** The old loop uploaded the whole grid, collapsed one cell,
  propagated and downloaded the whole grid again, so a collapse cost more on a bigger grid: the
  24×24×8 city ran at 102 cells/s and the 48×48×10 city at 57. Keeping a region in workgroup memory
  for the whole solve removed the round-trip entirely; the streaming world generates at about 86 000
  cells/s while it is generating.
- **Many regions per dispatch.** A dispatch lasts as long as its slowest region, so batching is the
  multiplier: the same kernel costs 29.7 ms for one chunk and 46.8 ms for 256.
- **Recovery decides how aggressive selection can be.** Collapsing every local minimum within a
  radius is 1.8× faster than collapsing one, but with restart-only recovery it also failed 73 of 256
  chunks. With checkpoint undo the same radius fails none.
- **Per-step cost is per-cell sweep work, not barriers.** One step of an 8×8×8 chunk with a halo
  costs about 21 µs at 256 invocations, and 445 µs at one, which is what says the kernel is bound by
  the work it does per cell rather than by synchronisation.

The old loop's profile is kept in [solver-fit.md](solver-fit.md) because it is the evidence the
redesign rests on: propagation was 76% of that run, so the answer was never "move the transfers", it
was "stop having a per-collapse loop at all".
