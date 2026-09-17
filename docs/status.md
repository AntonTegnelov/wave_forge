# Current status

Where the project actually is, as of September 2026, and what has to change to match [vision.md](vision.md) and [architecture.md](architecture.md). Keep this file honest: update it in the same PR that changes the status.

## What works

After the project was revived ([#3](https://github.com/AntonTegnelov/wave_forge/issues/3), [#4](https://github.com/AntonTegnelov/wave_forge/issues/4)):

- The workspace builds on Rust 1.98.1 (edition 2024) with up-to-date dependencies (wgpu 30).
- The CLI generates a 3D grid on the GPU from a RON rule file and writes it as text:

  ```bash
  cargo run --release -- --rule-file examples/simple-pattern.ron --width 8 --height 8 --depth 8
  ```

- Constraint propagation on the GPU is enforced and verified: generated grids satisfy the adjacency rules, and unsatisfiable rule sets report the contradiction location.
- Periodic and clamped borders both work (for full 3D grids).
- Cells constrained before a run (for example pinned layers) propagate before the first collapse.
- End-to-end tests generate a 2D coastline and a small marian42-style 3D city (81 connector-based module variants, walkable paths, stairs and passages), check them against their rules and render them to PNG; `wfc-render` renders any CLI output ([testing.md](testing.md)).
- `--trace-chrome` writes a timeline of the GPU run loop for Perfetto ([debugging.md](debugging.md)).
- In the dev container it runs on the host's NVIDIA RTX 3070 through Mesa's dozen driver (Vulkan on Direct3D 12); timings carry translation overhead. Without a GPU, a software Vulkan device (Mesa llvmpipe) is enough for correctness tests but not for performance work.

## Known limitations

- **3D only**, with 6 fixed axes; 2D means depth 1 (see A-2).
- **At most 256 tile variants** after rotation expansion, the size of the propagation shader's fixed per-cell mask (A-5).
- **Reproducible under a seed** (A-6, partly done): `WFC_SEED` fixes the collapse choice and the GPU's min-entropy selection is a deterministic reduction, so the same seed replays a run exactly, including a pathological one. The CLI's `--seed` is still not wired to it.
- **Slow for larger grids:** the whole grid crosses the CPU/GPU boundary twice per collapsed cell (A-10).
- **Contradictions backtrack** (A-9): the run jumps back to the most recent choice next to where the failure surfaced (or undoes a doubling number of steps when there is none) and forbids the choice it returns to. The undo depth also escalates with how often a conflict cell has failed, which took a pathological seed from 197 backtracks to 8 and removed the timeouts on the 4608-cell city. Escalation is deliberately *not* applied to global-constraint failures, where it was measured and made things strictly worse; that path still has no working recovery (see [thrashing.md](thrashing.md)). There is still no seeded restart, and the history is capped.
- **No library API:** the only entry points are the CLI binary and the internal `GpuAccelerator` type (A-1).
- The CLI's benchmark, progress and visualization options have not been re-verified since the revive.

## Alignment tasks

Each task is referenced from the matching callout in [architecture.md](architecture.md). Order within a phase is decided in [roadmap.md](roadmap.md), informed by measurements where noted.

| ID | Task | Architecture |
|---|---|---|
| A-1 | Restructure the workspace: model, solver, backends, public `wave_forge` library facade; move the CLI to a dev-tool binary crate. | [§2](architecture.md#2-component-overview) |
| A-2 | Introduce a topology abstraction with 4-direction 2D and 6-direction 3D grids; make rule-file direction names depend on topology. | [§3.1](architecture.md#31-topology-2d-and-3d-as-first-class) |
| A-3 | ~~Use tile weights in collapse~~ (done in [#13](https://github.com/AntonTegnelov/wave_forge/issues/13) through `GpuAccelerator::with_tile_weights`); use them in entropy (with A-7); support symmetry variants in rule files. Connector-based module sets in Rust (`wfc_rules::modules`) already generate rotated variants. Note: `generate_transformed_rules` only pairs a transformed tile with the *same* transformation of its neighbour, so a rotated tile can never border a tile that has only the identity variant. | [§3.2](architecture.md#32-tiles-and-compiled-rules) |
| A-4 | Store possibilities as one contiguous word array shared by CPU and GPU layouts. | [§3.3](architecture.md#33-possibility-storage) |
| A-5 | ~~Support more than 32 tiles per cell in all GPU kernels.~~ **Done** ([#13](https://github.com/AntonTegnelov/wave_forge/issues/13)), up to 256 variants; the city E2E test covers it. | [§3.3](architecture.md#33-possibility-storage) |
| A-6 | ~~Seeded RNG for the collapse choice; deterministic tie-breaking~~ **Done for the solver** ([#7](https://github.com/AntonTegnelov/wave_forge/issues/7)): `GpuAccelerator::with_seed`, and the min-entropy reduction now packs entropy and index into one key reduced with `atomicMin`, so selection no longer depends on which workgroup arrives first. Still to do: wire the CLI's `--seed`, and a counter-based RNG if per-cell independence is ever needed. | [§3.4](architecture.md#34-seeds-and-randomness) |
| A-7 | Weighted Shannon entropy with deterministic noise; stop scanline-order tie-breaking. | [§4.1](architecture.md#41-the-algorithm) |
| A-8 | ~~Propagate pre-constrained cells before the first observation.~~ **Done** ([#6](https://github.com/AntonTegnelov/wave_forge/issues/6)). | [§4.1](architecture.md#41-the-algorithm) |
| A-9 | ~~Bounded backtracking~~ **Done** ([#13](https://github.com/AntonTegnelov/wave_forge/issues/13)): conflict-directed, jumping back to the most recent choice next to the failure, with a doubling fallback, forbidding the reverted choice. Still to do: seeded restart per region, and delete the unused generic recovery framework. | [§4.2](architecture.md#42-contradictions) |
| A-10 | Keep solver state on the device; batch collapses of non-interacting cells; read back only results. **Profile first** ([#7](https://github.com/AntonTegnelov/wave_forge/issues/7)). Profiled, and a prototype settles the design: `wfc-gpu/tests/block_solver_bench.rs` keeps a whole chunk in workgroup memory and solves it in one dispatch, with no readback until it finishes ([solver-redesign.md](solver-redesign.md)). The solver itself still round-trips per collapse. | [§4.3](architecture.md#43-where-each-step-should-run) |
| A-11 | Make the propagation kernel race-free and drop the per-collapse full-grid verification pass. | [§4.3](architecture.md#43-where-each-step-should-run) |
| A-12 | Replace `Box<dyn …>`/`async_trait` strategies with static dispatch; remove the Tokio dependency from library crates; expose runtime-agnostic non-blocking APIs. | [§5](architecture.md#5-dispatch-async-and-threading-model) |
| A-13 | Region/chunk solving with constrained borders and a streaming scheduler; remove the disabled subgrid strategy. The prototype benchmark streams a 192×64×8 world around a walking focus with seamless borders, a halo that is discarded, and a repair pass for chunks whose borders cannot be satisfied; none of it is in the library yet. | [§6](architecture.md#6-scaling-to-large-worlds-regions-and-streaming) |
| A-14 | Embed shaders in the binary; remove the runtime shader registry/variant scaffolding. | [§8](architecture.md#8-gpu-specifics) |
| A-15 | One typed error enum per crate; extend `tracing` spans beyond the GPU run loop; GPU timestamp queries; remove unused debug visualizer. | [§9](architecture.md#9-errors-and-observability) |
| A-16 | Unit tests for solver internals; golden-image tests once runs are deterministic. (E2E tests, invariant checks and image tools were added in [#6](https://github.com/AntonTegnelov/wave_forge/issues/6); the realistic city and the opt-in stress suite in [#13](https://github.com/AntonTegnelov/wave_forge/issues/13).) | [§10](architecture.md#10-testing-and-developer-tooling) |
