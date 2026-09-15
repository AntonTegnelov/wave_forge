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
- It runs headless on a software Vulkan device (Mesa llvmpipe), so GPU code can be tested in containers without a GPU.

## Known limitations

- **3D only**, with 6 fixed axes; 2D means depth 1 (see A-2).
- **At most 32 tiles** after symmetry expansion (A-5).
- **Not reproducible:** `--seed` is ignored (A-6).
- **Slow for larger grids:** the whole grid crosses the CPU/GPU boundary twice per collapsed cell (A-10).
- **No contradiction recovery:** a contradiction ends the run (A-9).
- **No library API:** the only entry points are the CLI binary and the internal `GpuAccelerator` type (A-1).
- The CLI's benchmark, progress and visualization options have not been re-verified since the revive.

## Alignment tasks

Each task is referenced from the matching callout in [architecture.md](architecture.md). Order within a phase is decided in [roadmap.md](roadmap.md), informed by measurements where noted.

| ID | Task | Architecture |
|---|---|---|
| A-1 | Restructure the workspace: model, solver, backends, public `wave_forge` library facade; move the CLI to a dev-tool binary crate. | [§2](architecture.md#2-component-overview) |
| A-2 | Introduce a topology abstraction with 4-direction 2D and 6-direction 3D grids; make rule-file direction names depend on topology. | [§3.1](architecture.md#31-topology-2d-and-3d-as-first-class) |
| A-3 | Use tile weights in collapse and entropy; support symmetry variants in rule files. | [§3.2](architecture.md#32-tiles-and-compiled-rules) |
| A-4 | Store possibilities as one contiguous word array shared by CPU and GPU layouts. | [§3.3](architecture.md#33-possibility-storage) |
| A-5 | Support more than 32 tiles per cell in all GPU kernels. | [§3.3](architecture.md#33-possibility-storage) |
| A-6 | Seeded, counter-based RNG for every random decision; deterministic tie-breaking. | [§3.4](architecture.md#34-seeds-and-randomness) |
| A-7 | Weighted Shannon entropy with deterministic noise; stop scanline-order tie-breaking. | [§4.1](architecture.md#41-the-algorithm) |
| A-8 | Propagate pre-constrained cells before the first observation. | [§4.1](architecture.md#41-the-algorithm) |
| A-9 | Bounded backtracking / seeded restart per region; delete the unused generic recovery framework. | [§4.2](architecture.md#42-contradictions) |
| A-10 | Keep solver state on the device; batch collapses of non-interacting cells; read back only results. **Profile first** ([#7](https://github.com/AntonTegnelov/wave_forge/issues/7)). | [§4.3](architecture.md#43-where-each-step-should-run) |
| A-11 | Make the propagation kernel race-free and drop the per-collapse full-grid verification pass. | [§4.3](architecture.md#43-where-each-step-should-run) |
| A-12 | Replace `Box<dyn …>`/`async_trait` strategies with static dispatch; remove the Tokio dependency from library crates; expose runtime-agnostic non-blocking APIs. | [§5](architecture.md#5-dispatch-async-and-threading-model) |
| A-13 | Region/chunk solving with constrained borders and a streaming scheduler; remove the disabled subgrid strategy. | [§6](architecture.md#6-scaling-to-large-worlds-regions-and-streaming) |
| A-14 | Embed shaders in the binary; remove the runtime shader registry/variant scaffolding. | [§8](architecture.md#8-gpu-specifics) |
| A-15 | One typed error enum per crate; `tracing` spans; GPU timestamp queries; remove unused debug visualizer. | [§9](architecture.md#9-errors-and-observability) |
| A-16 | CPU backend for GPU-free testing; E2E tests for 2D (PNG) and 3D (city); image-based inspection tools; replace placeholder tests ([#6](https://github.com/AntonTegnelov/wave_forge/issues/6)). | [§10](architecture.md#10-testing-and-developer-tooling) |
