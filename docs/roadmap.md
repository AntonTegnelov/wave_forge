# Roadmap

Phases come from [vision.md](vision.md). Tasks are the alignment tasks (A-n) in [status.md](status.md#alignment-tasks) plus new work. A phase is done when its exit criteria hold, not when its task list is empty.

## Phase 0: revive the project (done)

Make the abandoned code base build and actually produce valid output again, and bring dependencies and toolchain up to date. Done in [#3](https://github.com/AntonTegnelov/wave_forge/issues/3) and [#4](https://github.com/AntonTegnelov/wave_forge/issues/4).

## Phase 1: standalone 2D and 3D WFC generator

**Goal:** a correct, deterministic, fast WFC library for 2D and 3D grids that other code can call.

Order, and where it stands:

1. **Tooling and tests first** ([#6](https://github.com/AntonTegnelov/wave_forge/issues/6)). Done. The later steps rewrote the core data structures and moved all the work onto the device; without end-to-end tests, image-based inspection and invariant checks, regressions in a parallel GPU program are nearly impossible to find.
2. **Profile, then redesign the solver** ([#7](https://github.com/AntonTegnelov/wave_forge/issues/7); A-10, A-11, A-12, A-4, A-6, A-9). Done. Measurement chose the design: one workgroup solves one region per dispatch, with everything resident on the device ([solver-redesign.md](solver-redesign.md), [solver-fit.md](solver-fit.md)).
3. **Chunks and streaming** (A-13). Done, together with step 2: the streaming suite generates a city in front of a walking player and the world comes out seamless.
4. **Public API and workspace structure** (A-1, A-14). Done: the root package is the `wave_forge` library, the model and the two seams are separate crates, and the CLI is a developer tool.
5. **Model and correctness gaps that remain.** A-3 (symmetry in rule files) turned out to block the MVP and moved there; A-16 (golden images) moved to hardening; A-2 (topology) is deferred until a game needs another grid.

**Exit criteria:**

- ~~2D and 3D generation through a documented library API, with no CLI or engine required.~~ Done.
- ~~Same seed and inputs produce identical output across runs and thread counts.~~ Done, and tested by comparing whole worlds.
- ~~E2E tests: a 2D tile set rendered to PNG, and a small 3D city comparable in spirit to [marian42's WFC city](https://marian42.de/article/wfc/).~~ Done, both through the library.
- ~~Benchmarks on reference rule sets with results recorded in the repository, and each CPU/GPU/SIMD placement decision backed by a measurement.~~ Done ([solver-fit.md](solver-fit.md)).
- ~~Streaming generation of regions around a moving focus point with seamless borders.~~ Done, within a 0.5 s tick budget.

Phase 1 is done. 2D works as a world one cell deep, so the topology abstraction (A-2) is a cost
question rather than a capability one, and it waits for a consumer.

## MVP: engine integrations and a walkable city

**Goal:** a Bevy plugin and a Godot GDExtension, both thin wrappers around the library, and the
success criterion that decides whether the MVP is done: **walking around in a Godot game inside an
infinite city that is generated continuously, on demand, at runtime**, in the spirit of marian42's
infinite city. Still WFC only.

### The integrations (done)

Both exist and are verified against a real engine; neither is published. The library API was
designed against what each engine actually owns, and that is what each integration turned out to
need ([architecture.md §5.1](architecture.md#51-the-solver-seam)):

- **Bevy** (`wave_forge_bevy`): the plugin builds a generator on Bevy's own device with `build_on`,
  holds it as a resource, and asks, starts and collects from two systems per frame. It tracks Bevy
  0.20 because that is the first release on wgpu 30, which is what makes sharing the device a matter
  of handing two resources over. A kernel specialisation takes seconds to compile, so
  `WaveForgePlugin::warm` exists to do it while a game loads.
- **Godot** (`wave_forge_godot`): a `Worker` owns the generator on a thread with a wgpu device of its
  own, and `_process` drains its events into signals. Nothing on Godot's side waits for the device,
  and nothing on the generating thread touches a Godot object, so the extension needs none of
  godot-rust's thread-safety features. `verify.sh` drives it in a real Godot and checks what it
  produced.

### What stands between the integrations and the criterion

The extension produces tile data, but nothing yet turns a city into something a player walks
through. In order:

1. ~~**The city has to reach Godot as data**~~ ([#30](https://github.com/AntonTegnelov/wave_forge/issues/30)). Done: a rule file can describe modules
   by their connectors, with rotated variants derived, and the city is `examples/city.ron`, which the
   tests, the CLI and the Godot extension load.
2. **A game has to know what to draw** ([#33](https://github.com/AntonTegnelov/wave_forge/issues/33)). Each tile is a variant of a prototype at a rotation; the
   facade and both integrations expose that mapping, so a game can place one model per prototype.
3. **No chunk may be left unplaced** ([#31](https://github.com/AntonTegnelov/wave_forge/issues/31)). 3.2% of city chunks cannot be placed in the streaming test,
   after repairs, and a walker would see every one of them as a hole. The target is zero. The
   approach is to measure first (which border patterns fail, from the contradiction cell each solve
   reports), then fix it in the module set, so that every border it can produce has a completion, or
   with a larger repair. A larger repair is bounded by workgroup memory: a halo of 3 needs 38 288 B
   against 32 768 B at 81 tiles. A third candidate, from the alternate architecture linked in
   [#5](https://github.com/AntonTegnelov/wave_forge/issues/5), is a coarse pass that decides chunk
   boundary faces before any interior is solved.
   The game session's `no_chunk_in_view_is_a_hole` holds this bar and fails until it is met: 13
   holes in view on a 776 m walk ([testing.md](testing.md#the-game-session)).
4. **Models to draw** ([#34](https://github.com/AntonTegnelov/wave_forge/issues/34)). A devtools command exports each prototype's voxel model as a mesh (glTF, which
   both engines import). Authored models can replace them later without touching the library.
5. **The walk itself**, with the products it needs from the library: instance sets, a collider
   library and mesh levels of detail ([#38](https://github.com/AntonTegnelov/wave_forge/issues/38),
   [engine-integration.md](engine-integration.md)). A game that places the models per chunk, builds
   colliders from them, and lets a first-person player walk an unbounded city. Verified by a
   scripted walk: frame rate while walking, no holes, the player never falls through, and a chunk
   walked back to comes back identical. The same city then gets a Bevy walk.

Items 1 to 4 are library work and happen here. The games in item 5 live in **their own
repository**: they are consumers of the library, they carry art that has no place in a library
repository, and building them against the public API alone is the test of whether that API is
enough. The extension's own verification project (`wave_forge_godot/godot`) stays here, because it
tests the extension rather than showing it off.

Godot's compatibility renderer runs in this dev container on the host GPU through Mesa's D3D12
OpenGL driver under `xvfb` ("OpenGL API 4.2 (Core Profile) Mesa 22.3.6 - Compatibility - Using
Device: Microsoft - D3D12 (NVIDIA GeForce RTX 3070)", Godot 4.7.2), so a scripted walk can be
screenshotted and checked here. Forward+ and Mobile need a Vulkan swapchain and cannot run here
([testing.md](testing.md#the-engine-integrations)).

### Still open: generating on Godot's own `RenderingDevice`

It would avoid a second device and its allocator, which is the reason the `ComputeBackend` seam
exists. It needs the kernel's WGSL translated to SPIR-V (naga can, either at runtime or as a
build-time matrix) and Godot's compute API driven from a worker thread, with `is_done` as
`is_task_completed` on a `WorkerThreadPool` task. What it *cannot* have here is a measurement: this
dev container reaches its GPU through Mesa's dozen, which does not expose `VK_KHR_swapchain`, and
Godot refuses to create any `RenderingDevice` without it. Godot does run on the software Vulkan
device (lavapipe), which is enough to check that such a backend produces correct worlds but says
nothing about whether it is faster than a device of its own. The comparison belongs on a desktop
Godot, and nothing gets built for it before that measurement says it is worth it.

## Hardening

Work that publishing would require, and that pays for itself before then:

- **Continuous integration on free, standard GitHub-hosted runners** ([#32](https://github.com/AntonTegnelov/wave_forge/issues/32); the repository is public).
  Formatting, clippy with warnings as errors, the tests that need no GPU (units, the facade on the
  CPU reference, the Bevy wiring tests), and the per-crate feature builds from
  [testing.md](testing.md#test-layers). GPU tests run on Mesa's software Vulkan device if they stay
  fast there; that is measured before it is added. Heavy engine builds run only when their
  directories change.
- **Golden worlds (A-16, [#35](https://github.com/AntonTegnelov/wave_forge/issues/35)).** A hash of a small generated world, recorded on the RTX 3070 and
  compared on the software device in CI. That is the first test of the promise that a world is the
  same on any GPU vendor, which so far has only been observed on one.
- **Licences.** Everything in both integrations' dependency trees is MIT, Apache-2.0, Zlib or
  Unlicense, except godot-rust (`godot` and its `godot-*` crates, 0.5.5), which is MPL-2.0. That is
  compatible with shipping our MIT code, but a distributed extension binary has to say where the
  MPL-covered source can be obtained.
- **Builds for every desktop platform** the Godot Asset Store expects (Windows, macOS, Linux). Whether
  and when anything is published is the owner's decision.

## Deeper engine integration

The integrations should do more than hand out tile ids: the library emits typed products per chunk
(instance sets, meshes with levels of detail, colliders, navigation source geometry, region tags,
spawn points), and each integration maps them onto its engine's own systems and ships reference
shaders and authoring tools. The design, with the reasoning and the evidence for each engine, is in
[engine-integration.md](engine-integration.md). The work is staged around the rest of this roadmap:

- with the MVP walk: the products the walk needs ([#38](https://github.com/AntonTegnelov/wave_forge/issues/38));
- with hardening: the measurement that decides where the solver runs in both engines ([#39](https://github.com/AntonTegnelov/wave_forge/issues/39)),
  Godot editor polish and the Godot 4.7 minimum ([#40](https://github.com/AntonTegnelov/wave_forge/issues/40)), and a spike on Rust resources loaded off
  the main thread ([#41](https://github.com/AntonTegnelov/wave_forge/issues/41));
- after the MVP: navigation ([#42](https://github.com/AntonTegnelov/wave_forge/issues/42)), audio and localisation tags ([#43](https://github.com/AntonTegnelov/wave_forge/issues/43)), and placing users'
  own scenes by rule ([#44](https://github.com/AntonTegnelov/wave_forge/issues/44));
- with Phase 2: noise that means the same in both engines ([#45](https://github.com/AntonTegnelov/wave_forge/issues/45)), ground, grass and wind
  ([#46](https://github.com/AntonTegnelov/wave_forge/issues/46)), far proxies and occluders ([#47](https://github.com/AntonTegnelov/wave_forge/issues/47));
- then authoring: recipes, brushes and bake, in Godot first because Bevy has no editor ([#48](https://github.com/AntonTegnelov/wave_forge/issues/48)).

Products that stay on the GPU are built only where a measurement shows the read-back path is too
slow.

## Phase 2: layered world generation

**Goal:** LayerProcGen-style layers combining techniques: noise landscapes, Fractal Jittered Voronoi
Partition coastlines and WFC cities, with layering, blending and multiple passes. The first step is
two layers: a landscape from noise, with villages, towns and cities placed in it and built with WFC.
See [architecture.md §7](architecture.md#7-phase-2-layered-generation-design-constraints-to-keep-in-mind-now)
for the constraints Phase 1 had to respect.

Detailed planning waits for the walk, which will show what a game actually asks for. The
big-picture shape is decided now, because it is what later work must not contradict:

- **A layer is a pure function of the world seed and a chunk coordinate, with a declared reach into
  the layers below it**, on the same chunk lattice and the same per-chunk seeds the WFC generator
  uses. Layers form a graph with no cycles, and a layer only reads layers at its own scale or a
  coarser one. That is what makes any chunk generable in any order with the same result, which the
  generator already guarantees for WFC.
- **WFC is one layer, driven by the layers below it.** Its `Prior` (layer masks, face bans, per-cell
  overrides) is computed from lower layers instead of being set by hand; that is "driven WFC", and
  the `Prior` was built to carry it.
- **The first graph:** a noise heightfield; settlement sites as a Poisson process per region, with
  footprints that cannot overlap (Boris the Brave's Poisson-rect process); a final height that
  flattens the terrain under each footprint, which keeps sites and height free of a cycle; and WFC
  inside the footprints.
- **The hard part is a city on uneven ground.** The prior pins street level to the terrain height
  per column, so the module set needs fill below the street and has to meet the landscape at a
  footprint's edge.
- **The integrations change little.** They hand out "tiles of a chunk" today and will hand out "the
  outputs of the layers a game asked for"; both are thin, so that change is cheap when it comes.

## Deferred

| Item | Why it waits |
|---|---|
| A-2: other topologies (hex, triangle, irregular) | No consumer needs one; 2D works as a world one cell deep. |
| A-15: GPU timestamp queries | Worth it when a performance question needs it, not before. |
| Godot `RenderingDevice` backend | Needs a measurement on a desktop Godot first (above). |
| `GlobalConstraint`, `TileWeighting`, a path constraint | Documented as future capabilities in [constraints.md](constraints.md). Revisited if the walk shows disconnected street networks. |
| A SAT (CDCL) solver, sparse voxel DAGs, 64-bit world coordinates with a floating origin | None of them solves a problem the MVP or the first two layers have. The chunk lattice uses 32-bit chunk coordinates, and `f32` positions stay millimetre-precise within several kilometres of the origin, far more than a walk covers. |
