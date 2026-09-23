# Wave Forge documentation

Read in this order:

1. **[Vision](vision.md):** what we are building, for whom, and the priorities that decide every trade-off.
2. **[Architecture](architecture.md):** the design and the reasoning behind it, with callouts where today's code differs.
3. **[Status](status.md):** what works now, known limitations, and the numbered alignment tasks.
4. **[Roadmap](roadmap.md):** phases, suggested order, exit criteria, and the release gate.
5. **[User stories](user-stories.md):** what people want to do with Wave Forge, from replicating a game's world generation to a newcomer's first world and the performance a game needs; verifying all of them is the gate before anything is done or published.
6. **[Generation model](generation-model.md):** how Phase 2 layers fields, scatter, sites, WFC and region-scale passes as a pack of stages, and the research on engines, libraries and games behind it.
7. **[Development guide](development.md):** building, workflow and documentation conventions.
8. **[Testing and inspection](testing.md):** test layers, end-to-end tests, artifacts and rendering tools.
9. **[Debugging and observability](debugging.md):** reproducing failures, the invariant oracles, the counters a solve returns, and GPU debugging.
10. **[Build profiles](build-profiles.md):** which Cargo profile to use for development, shipping and profiling, and why each setting is chosen.

Reference:

- **[Engine integration](engine-integration.md):** what the library emits for Godot and Bevy (instance sets, meshes, colliders, navigation, spawn points), where the solver runs in each engine, which engine features it feeds, and how users configure it.
- **[Constraints](constraints.md):** what adjacency rules can and cannot express, designing module sets for properties like connectivity, and the global and statistical rules kept as future capabilities.
- **[Performance](performance.md):** priorities, how we measure, where work should run (GPU, threads, SIMD), and the baseline.
- **[What fits](solver-fit.md):** whether to adopt CDCL/SAT, ghost cells and block decomposition, what changes the parallelism story, and the resulting recommendation.
- **[Solver redesign](solver-redesign.md):** what the profile and micro-benchmarks show, what the literature says, and the ranked plan for [#7](https://github.com/AntonTegnelov/wave_forge/issues/7).

The dev container is documented separately in [`.devcontainer/README.md`](../.devcontainer/README.md).
