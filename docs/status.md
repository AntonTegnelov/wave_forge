# Current status

Where the project actually is, as of September 2026, and what has to change to match [vision.md](vision.md) and [architecture.md](architecture.md). Keep this file honest: update it in the same PR that changes the status.

## What works

- The workspace builds on Rust 1.98.1 (edition 2024) with up-to-date dependencies (wgpu 30). No library crate needs an async runtime.
- **The library generates a world in chunks, on the GPU, around moving focus points.** `wave_forge::Builder` builds a `WorldGenerator` on its own device, on a device an engine owns, or on a solver of your own; `request`, `tick` and `poll` drive it without blocking, and `Worker` runs it on a thread for an engine that cannot poll its compute API ([architecture.md §5](architecture.md#5-dispatch-async-and-threading)).
- **One workgroup solves one region per dispatch**, with the region's domains in workgroup memory for the whole solve: propagation, selection, collapse and recovery never leave the device ([solver-redesign.md](solver-redesign.md)). 0.18 ms per chunk at 256 chunks in one dispatch, against 3.9 ms for the same chunk on one CPU thread.
- **Live streaming works.** A 24×8-chunk city (192×64×8 cells) generated in front of a focus walking at 1.4 m/s: median 47 ms per 0.5 s tick, p90 61 ms, no seam violations ([solver-fit.md](solver-fit.md)).
- **Generation is reproducible.** Every choice is a hash of the world seed, the chunk's coordinate and where the solve had got to, so the same requests give the same world on any backend, thread count or invocation count, and the facade's tests compare whole worlds cell for cell ([architecture.md §6.3](architecture.md#63-what-determinism-means-here)).
- Chunks are stitched with a halo that is solved and discarded, and a chunk that fixed borders left unsolvable is repaired with its halo released, which is reported.
- Constraints enter as a `Prior` on starting domains: masks per layer, bans per world face, overrides per cell. The city's boundary conditions are expressed that way.
- Up to 256 tile variants, weights quantised to integers, and rule sets wider than one possibility word (the city's 81 variants need three).
- End-to-end tests generate a 2D coastline and a small marian42-style 3D city through the library, check them against their rules and render them to PNG; `wave-forge` generates one chunk from a rule file and `wfc-render` draws it ([testing.md](testing.md)).
- In the dev container it runs on the host's NVIDIA RTX 3070 through Mesa's dozen driver (Vulkan on Direct3D 12); timings carry translation overhead. Without a GPU, a software Vulkan device (Mesa llvmpipe) is enough for correctness tests but not for performance work.

## Known limitations

- **No engine integrations yet.** The seams are drawn for them and the library API is what they will bind to, but there is no Bevy plugin and no Godot extension, and the Godot backend's threading has not been prototyped ([architecture.md §5.1](architecture.md#51-the-solver-seam)).
- **3D only**, with 6 fixed axes; 2D means a world one cell deep (A-2).
- **A region must fit the device's workgroup memory.** At 81 tiles, an 8×8×8 chunk fits with a halo of 1 or 2 but not 3 (38 288 B against 32 768 B), so a repair ladder stops there. Bigger chunks or more tiles need a kernel that keeps domains in a storage buffer instead, which nothing needs yet.
- **Some chunks cannot be placed.** 3.2% of city chunks in the streaming test, after repairs. A chunk whose borders no arrangement satisfies is a property of the module set rather than of the solver: a set is *streaming-clean* when that count is zero, and the city's is not. It is reported, not retried.
- **Rule files carry no symmetry.** Rotated variants come from connector-based module sets in Rust; the RON loader creates identity variants only (A-3).
- **No GPU timestamp queries and no spans.** Host wall-clock around a dispatch, plus the statistics a solve reports, are all the observability there is (A-15).
- **No golden-image tests**, although generation is now reproducible (A-16).

## Alignment tasks

Each task is referenced from the matching callout in [architecture.md](architecture.md). Order within a phase is decided in [roadmap.md](roadmap.md), informed by measurements where noted.

| ID | Task | Architecture |
|---|---|---|
| A-1 | ~~Restructure the workspace: model, solver, backends, public `wave_forge` library facade; move the CLI to a dev-tool binary crate.~~ **Done** ([#7](https://github.com/AntonTegnelov/wave_forge/issues/7)). | [§2](architecture.md#2-component-overview) |
| A-2 | Introduce a topology abstraction with 4-direction 2D and 6-direction 3D grids; make rule-file direction names depend on topology. | [§3.1](architecture.md#31-topology-2d-and-3d) |
| A-3 | Support symmetry variants in rule files. Connector-based module sets in Rust (`wfc_rules::modules`) already generate rotated variants, and weights are used in collapse. Note: `generate_transformed_rules` only pairs a transformed tile with the *same* transformation of its neighbour, so a rotated tile can never border a tile that has only the identity variant. | [§3.2](architecture.md#32-tiles-and-compiled-rules) |
| A-4 | ~~Store possibilities as one contiguous word array shared by CPU and GPU layouts.~~ **Done** ([#7](https://github.com/AntonTegnelov/wave_forge/issues/7)): `Domains`, in the layout the shader reads. | [§3.3](architecture.md#33-possibility-storage) |
| A-5 | ~~Support more than 32 tiles per cell in all GPU kernels.~~ **Done** ([#13](https://github.com/AntonTegnelov/wave_forge/issues/13)), up to 256 variants; the city E2E test covers it. | [§3.3](architecture.md#33-possibility-storage) |
| A-6 | ~~Seeded RNG for the collapse choice; deterministic tie-breaking.~~ **Done** ([#7](https://github.com/AntonTegnelov/wave_forge/issues/7)): every choice is a stateless hash of the world seed, the chunk id, the attempt and the step, so nothing depends on scheduling. | [§3.4](architecture.md#34-seeds-and-randomness) |
| A-7 | ~~Weighted Shannon entropy with deterministic noise; stop scanline-order tie-breaking.~~ **Superseded**: the kernel selects every local minimum of the possibility count and breaks ties by the choice hash, so scan order plays no part. Shannon entropy was not worth the extra work at the measured difference. | [§4.1](architecture.md#41-the-algorithm) |
| A-8 | ~~Propagate pre-constrained cells before the first observation.~~ **Done** ([#6](https://github.com/AntonTegnelov/wave_forge/issues/6)). | [§4.1](architecture.md#41-the-algorithm) |
| A-9 | ~~Bounded backtracking.~~ **Done** ([#7](https://github.com/AntonTegnelov/wave_forge/issues/7)): the kernel restores a checkpoint from a ring buffer, doubles how far back it goes on a repeated failure, restarts the region when the ring is exhausted, and reports a status when the budget is. The generic recovery framework is deleted. | [§4.2](architecture.md#42-contradictions) |
| A-10 | ~~Keep solver state on the device; batch collapses of non-interacting cells; read back only results.~~ **Done** ([#7](https://github.com/AntonTegnelov/wave_forge/issues/7)): a region's domains stay in workgroup memory for the whole solve, every local minimum within a radius collapses per round, and a batch costs one dispatch and one readback. | [§4.3](architecture.md#43-where-each-step-runs) |
| A-11 | ~~Make the propagation kernel race-free and drop the per-collapse full-grid verification pass.~~ **Done** ([#7](https://github.com/AntonTegnelov/wave_forge/issues/7)): propagation gathers into an invocation's own cells, so no invocation writes a neighbour's mask and there is nothing to verify. | [§4.3](architecture.md#43-where-each-step-runs) |
| A-12 | ~~Replace `Box<dyn …>`/`async_trait` strategies with static dispatch; remove the Tokio dependency from library crates; expose runtime-agnostic non-blocking APIs.~~ **Done** ([#7](https://github.com/AntonTegnelov/wave_forge/issues/7)): the solver and the generator are generic, the `Solver` seam is job-based, and no library crate depends on an async runtime. | [§5](architecture.md#5-dispatch-async-and-threading) |
| A-13 | ~~Region/chunk solving with constrained borders and a streaming scheduler; remove the disabled subgrid strategy.~~ **Done** ([#7](https://github.com/AntonTegnelov/wave_forge/issues/7)): a chunk lattice with parity batches, a halo that is discarded, repairs for what fixed borders leave unsolvable, and a scheduler around focus points. | [§6](architecture.md#6-scaling-to-large-worlds-chunks-and-streaming) |
| A-14 | ~~Embed shaders in the binary; remove the runtime shader registry/variant scaffolding.~~ **Done** ([#7](https://github.com/AntonTegnelov/wave_forge/issues/7)): one WGSL file with `include_str!` and constants substituted per specialisation. | [§8](architecture.md#8-gpu-specifics) |
| A-15 | GPU timestamp queries, and spans around the host's side of a batch if they turn out to be worth it. One typed error enum per crate, the unused debug visualiser and the duplicate error hierarchies are done ([#7](https://github.com/AntonTegnelov/wave_forge/issues/7)). | [§9](architecture.md#9-errors-and-observability) |
| A-16 | Unit tests for the kernel's internals; golden-image tests, now that runs are deterministic. (E2E tests, invariant checks and image tools were added in [#6](https://github.com/AntonTegnelov/wave_forge/issues/6); the realistic city in [#13](https://github.com/AntonTegnelov/wave_forge/issues/13); the facade's determinism tests and the streaming suite in [#7](https://github.com/AntonTegnelov/wave_forge/issues/7).) | [§10](architecture.md#10-testing-and-developer-tooling) |
