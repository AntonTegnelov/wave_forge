# Wave Forge documentation

Read in this order:

1. **[Vision](vision.md):** what we are building, for whom, and the priorities that decide every trade-off.
2. **[Architecture](architecture.md):** the design and the reasoning behind it, with callouts where today's code differs.
3. **[Status](status.md):** what works now, known limitations, and the numbered alignment tasks.
4. **[Roadmap](roadmap.md):** phases, suggested order and exit criteria.
5. **[Development guide](development.md):** building, workflow and documentation conventions.
6. **[Testing and inspection](testing.md):** test layers, end-to-end tests, artifacts and rendering tools.
7. **[Debugging and observability](debugging.md):** reproducing failures, the invariant oracles, the counters a solve returns, and GPU debugging.
8. **[Build profiles](build-profiles.md):** which Cargo profile to use for development, shipping and profiling, and why each setting is chosen.

Reference:

- **[Constraints](constraints.md):** what adjacency rules can and cannot express, designing module sets for properties like connectivity, and the global and statistical rules kept as future capabilities.
- **[Performance](performance.md):** priorities, how we measure, where work should run (GPU, threads, SIMD), and the baseline.
- **[What fits](solver-fit.md):** whether to adopt CDCL/SAT, ghost cells and block decomposition, what changes the parallelism story, and the resulting recommendation.
- **[Solver redesign](solver-redesign.md):** what the profile and micro-benchmarks show, what the literature says, and the ranked plan for [#7](https://github.com/AntonTegnelov/wave_forge/issues/7).

The dev container is documented separately in [`.devcontainer/README.md`](../.devcontainer/README.md).
