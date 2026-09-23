# Status

Where Wave Forge stands, area by area, as of 2026-09-23. Where it is heading is in
[roadmap.md](roadmap.md), and how far each user story has got is in
[story-coverage.md](story-coverage.md). Update this page in the pull request that changes what it
says.

Nothing is published. Phase 0 (reviving the project) and Phase 1 (the standalone WFC library) are
done, the MVP's engine integrations work and are verified in real engines, and Phase 2 (packs of
stages) has its runtime and a first slice.

## By area

**Build.** The workspace builds on Rust 1.98.1 (edition 2024) with wgpu 30. No library crate needs
an async runtime. Continuous integration on GitHub Actions runs formatting, clippy with warnings as
errors, every test including the GPU tests on Mesa's lavapipe, the per-crate feature builds, the
Godot checks in a real Godot and the Bevy tests, on every pull request
([environment.md](../guides/environment.md)).

**Solver.** One workgroup solves one region per dispatch with the region resident in workgroup
memory: gather propagation, every local minimum collapsed per round, checkpoint undo and restarts,
all on the device ([solver.md](../architecture/solver.md)). Up to 256 tiles, integer weights, rule
sets wider than one mask word (the city's 81 tiles need three). The CPU reference solver is the
oracle and the yardstick.

**World.** `WorldGenerator` streams chunks around focus points with a parity schedule, a halo on the
first parity, and repairs of 32 seeds side by side that go by class, so the world is the same in any
generation order and on any device ([world.md](../architecture/world.md)). Every chunk of the
reference worlds is placed. `Worker` runs it on a thread.

**Rules.** Tile sets and module sets in RON; module sets derive rotations and adjacency from
connectors, and the city (`examples/city.ron`) is one. A `Prior` carries masks per layer, bans per
world face and per-cell overrides.

**Stages.** `wave_forge::stages` loads a pack (Field, Rules, Blur, Sites, Flatten, Solve, Scatter), checks
it, and generates any stage around focus points, providers first, through reads bounded by each
stage's reach, the same in any order. Towns are bounded WFC worlds per site behind the `TownSolver`
seam, solved on the GPU. `StageWorker` runs it on a thread. The valley test pack
(`examples/valley.world.ron`) generates rolling ground, towns on levelled sites and trees kept apart
([reference/packs.md](../reference/packs.md)). The stage runtime runs on the CPU. A chunk's ground
comes as a mesh and a height grid from any height field stage, seamless across chunks.

**Products.** Instance sets in Godot's MultiMesh layout with positional `InstanceId`s, for streamed
worlds and towns; navigation source geometry with a border into the neighbours; `YUpSpace` for
where a cell is in a Y-up engine ([engine-integration.md](../architecture/engine-integration.md#products)).

**Godot.** `WaveForgeWorld` streams the city with multimesh buffers, colliders on a ring near the
player and navigation baked off Godot's thread; `WaveForgeStages` serves a pack's fields, sites,
towns and points, and builds the ground's meshes and, near the player, bodies for the ground and
the towns, which a walker crosses without falling through. Both are checked in headless Godot 4.7.2 by `verify.sh`
([reference/godot.md](../reference/godot.md)).

**Bevy.** `WaveForgePlugin` generates on Bevy's own device; `WaveForgeStagesPlugin` serves a pack
and its ground as meshes and height grids.
Headless apps with the real `DefaultPlugins` test both ([reference/bevy.md](../reference/bevy.md)).

**Tooling.** The `wave-forge` CLI, PNG and isometric renderers, glTF model export for the city's
modules, an invariant checker, the streaming, game-session, order-diff and golden-world tests
([testing.md](../guides/testing.md)).

## Headline numbers

All from the dev container's NVIDIA RTX 3070 through Mesa's dozen (Vulkan on Direct3D 12), release
builds, the city rule set unless a row says otherwise; protocols and history in [measurements.md](../research/measurements.md).

| What | Result | Test |
|---|---|---|
| A batch of 256 city chunks (8×8×8, halo 1) | 0.18 ms per chunk, against 3.9 ms on one CPU thread | `wfc-gpu/tests/block_solver_bench.rs` |
| Streaming 24×8 chunks in front of a focus walking 1.4 m/s, 0.5 s ticks | median tick 53 to 54 ms, p90 128 to 130 ms, 0 of 192 chunks unplaced | `wfc-devtools/tests/streaming.rs` |
| A 230 s walk and run through an unbounded city, in wall-clock time | 0 frames with a chunk near the player missing; main thread p99 0.005 ms; at most 94 chunks held; 42 chunks walked back to, all identical; 0 violations in 2 454 seam checks | `wfc-devtools/tests/game_session.rs` |
| Five census worlds | 0 of 1 772 chunks given up on, 181 repairs, 4.0 s in the solver | `wfc-devtools/tests/hole_census.rs` |
| Order independence, 4×4-chunk city, seeds 8 and 11 | tile for tile the same all at once and chunk by chunk in either raster order, repairs included | `wfc-devtools/tests/order_diff.rs` |
| The same city on two GPUs | tile for tile the same on the RTX 3070 and on lavapipe | `wfc-devtools/tests/golden_world.rs` |
| Godot, a focus running at 4.2 units/s over a small band rule set, with colliders and navigation | Godot's slowest frame 3.4 ms, the node's own time 0.56 ms per frame at p99 | `wave_forge_godot/godot/verify.gd` |
| Godot, the valley pack's 49 chunks around a town from a cold start | about 16 s, towns included; the node's own time 0.02 ms per frame at p99 | `wave_forge_godot/godot/verify_stages.gd` |

## Known limits

Each is a gap between the code and the design or the stories, with where it is tracked.

- **Godot generates on a device of its own**, not on Godot's `RenderingDevice`. It works and keeps
  Godot's frame loop free; a backend over Godot's device needs a desktop measurement first
  ([engine-integration.md](../architecture/engine-integration.md#godot-the-solver-stays-on-its-own-device),
  [#39](https://github.com/AntonTegnelov/wave_forge/issues/39)).
- **No desktop numbers.** Every timing comes from the dev container's translated driver or from
  lavapipe ([#39](https://github.com/AntonTegnelov/wave_forge/issues/39)).
- **Order independence has two limits:** worlds more than one chunk tall, and partial eviction of a
  repaired neighbourhood ([world.md](../architecture/world.md#what-determinism-means-here)). The city
  needs a repair in about one chunk in ten.
- **A region must fit the device's workgroup memory.** At 81 tiles an 8×8×8 chunk fits a halo of 1
  or 2, not 3 (38 288 B against 32 768 B), so a repair's halo stops at 2. Bigger regions would need
  domains in a storage buffer, which nothing needs yet.
- **3D only, six fixed axes.** 2D is a world one cell deep; other topologies wait for a game that
  needs one.
- **No GPU timestamp queries and no `tracing` spans.** Host wall-clock time and the counters a solve
  reports are all the observability there is.
- **Kernel internals have no unit tests.** The kernel is checked through whole-region results, the
  CPU oracle and the golden world; its generated source and workgroup budget are unit tested.
- **Golden worlds cover tiles, not images or products.** Instance sets and chunk hashes are not in
  them yet.
- **The stage runtime is a first slice.** Fields are one `f32` per column, at a scale per stage;
  region jobs produce curves but nothing rasterises them yet; tables of facts reach stages through
  a focused row, sites and towns, not yet as roads
  ([#72](https://github.com/AntonTegnelov/wave_forge/issues/72)); there are no edits, persistence modes, world
  bounds or per-stage request radii; noise is value noise only; Scatter has four tests and one kind
  per stage. Each is an issue under
  [#104](https://github.com/AntonTegnelov/wave_forge/issues/104) ([story-coverage.md](story-coverage.md)).
- **The ground is one mesh per chunk at the field's resolution**, untextured and without levels of
  detail; materials, ground cover and far levels are
  [#46](https://github.com/AntonTegnelov/wave_forge/issues/46) and
  [#47](https://github.com/AntonTegnelov/wave_forge/issues/47). Scattered points stand at their
  column's height, which can differ from the mesh between column centres by up to half a column's
  slope.
- **No levels of detail** for the exported module meshes; they wait for authored models
  ([#38](https://github.com/AntonTegnelov/wave_forge/issues/38)).
- **Scenes are not yet bound to points**, and there are no region tags, far proxies, occluders,
  noise parity with Godot, or authoring tools
  ([#44](https://github.com/AntonTegnelov/wave_forge/issues/44),
  [#43](https://github.com/AntonTegnelov/wave_forge/issues/43),
  [#47](https://github.com/AntonTegnelov/wave_forge/issues/47),
  [#45](https://github.com/AntonTegnelov/wave_forge/issues/45),
  [#48](https://github.com/AntonTegnelov/wave_forge/issues/48)).
- **Licences for a distributed build.** Everything in both integrations' dependency trees is MIT,
  Apache-2.0, Zlib or Unlicense except godot-rust (0.5.5), which is MPL-2.0: compatible, but a
  distributed extension binary has to say where the MPL-covered source can be obtained.
