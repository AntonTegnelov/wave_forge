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
2. ~~**A game has to know what to draw**~~ ([#33](https://github.com/AntonTegnelov/wave_forge/issues/33)). Done: a loaded rule file names every tile,
   its rotation and its tags; `YUpSpace` in the library says where a cell is and how a model turns
   in a Y-up engine; the Godot node and the Bevy `WaveForgeTiles` resource expose both.
3. ~~**No chunk may be left unplaced**~~ ([#31](https://github.com/AntonTegnelov/wave_forge/issues/31)). Done: the chunks the city gave up on were
   placeable, and a repair now tries 32 seeds side by side; with the second parity solved without a
   halo, five worlds place all 1 605 chunks and the game session shows no holes
   ([solver-fit.md](solver-fit.md)).
4. ~~**Models to draw**~~ ([#34](https://github.com/AntonTegnelov/wave_forge/issues/34)). Done: `wfc-export-models` writes each module's voxel model as
   binary glTF in a Y-up engine's frame, coloured by a palette texture; Godot loads all of them,
   and `render_city.sh` draws a generated city with them. Authored models can replace them without
   touching the library.
5. **The walk itself**, with the products it needs from the library: instance sets, a collider
   library and mesh levels of detail ([#38](https://github.com/AntonTegnelov/wave_forge/issues/38),
   [engine-integration.md](engine-integration.md)). A game that places the models per chunk, builds
   colliders from them, and lets a first-person player walk an unbounded city. Verified by a
   scripted walk: frame rate while walking, no holes, the player never falls through, and a chunk
   walked back to comes back identical. The same city then gets a Bevy walk.

Items 1 to 4 are library work and happen here. The walk in item 5 is the first of the
proof-of-concept games, which live in repositories of their own
([below](#games-packs-and-repositories)). The extension's own verification project
(`wave_forge_godot/godot`) stays here, because it tests the extension rather than showing it off.

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

## Games, packs and repositories

Three proof-of-concept games consume Wave Forge, each kept simple at first; any of them that
proves fun can grow into more:

| Game | What it needs from Wave Forge | Order |
|---|---|---|
| Parkour hide-and-seek and tag in an infinite WFC city, in the style of marian42's city | the city, walk products, colliders and navigation: the MVP walk ([#38](https://github.com/AntonTegnelov/wave_forge/issues/38)) | first |
| A classic survival and crafting game in a Valheim-like surface world with WFC inside masks | Phase 2's first slice ([#68](https://github.com/AntonTegnelov/wave_forge/issues/68)) | second |
| A grand strategy game inspired by Europa Universalis, on a world and a history generated once per new game in the manner of Dwarf Fortress | region jobs ([#69](https://github.com/AntonTegnelov/wave_forge/issues/69)) and records ([#72](https://github.com/AntonTegnelov/wave_forge/issues/72)); the history simulation belongs to the game | third |

**Each game lives in its own private repository.** They are not open source. They depend on Wave
Forge by a pinned commit, and moving the pin is a deliberate change in the game's repository, which
is also the best test of whether the public API is enough.

**This repository keeps** the library, the integrations, their verification projects, tests and
manual experiments in Godot and Bevy, and two kinds of packs:

| Kind | Where | Purpose |
|---|---|---|
| **Game packs** | each game's repository | the game's world, tuned for play; smoke tests for Wave Forge |
| **Test packs** | here | edge cases, stress tests, and one pack per user story exercising its techniques; the evidence for the [verification gate](user-stories.md#the-verification-gate) |
| **Presets** | here | the product's out-of-the-box worlds (N1, N2, N10) |

**Smoke tests run downstream.** Each game's continuous integration builds it against Wave Forge's
latest `develop` every night and on every change of its pin, and runs its packs, so nothing here
needs access to a private repository and no private pack reaches a public log. A failure comes back
as an issue here with a public reproduction ([template](../.github/ISSUE_TEMPLATE/from-a-game.md)):
the game's pack is reduced to a test pack in this repository, which then stays as a regression
test.

**Access.** This repository is public, so the games read it without a token. A game's environment
may open issues here (a token with issue access only), and this repository's environment may read
the games' repositories (read-only). Nothing from a private game (code, assets, design text, pack
contents) is ever copied into this repository, its issues or its commits; a report reproduces the
problem in Wave Forge's own terms.

## Hardening

Work that publishing would require, and that pays for itself before then:

- ~~**Continuous integration on free, standard GitHub-hosted runners** ([#32](https://github.com/AntonTegnelov/wave_forge/issues/32)).~~ Done: every pull
  request runs formatting, clippy with warnings as errors, the workspace tests including the GPU
  tests on Mesa's lavapipe, the per-crate feature builds, the Godot extension's check in a real
  Godot and the Bevy plugin's tests ([testing.md](testing.md)).
- ~~**Golden worlds (A-16, [#35](https://github.com/AntonTegnelov/wave_forge/issues/35)).**~~ Done: a 4×4-chunk city recorded on the RTX 3070 is tile for
  tile the same on Mesa's lavapipe, which CI checks on every pull request.
- **Licences.** Everything in both integrations' dependency trees is MIT, Apache-2.0, Zlib or
  Unlicense, except godot-rust (`godot` and its `godot-*` crates, 0.5.5), which is MPL-2.0. That is
  compatible with shipping our MIT code, but a distributed extension binary has to say where the
  MPL-covered source can be obtained.
- **Builds for every desktop platform** the Godot Asset Store expects (Windows, macOS, Linux),
  prepared for the release gate below.

## Release gate and publishing

- **The user stories are the release gate.** Nothing is done or published until every story in
  [user-stories.md](user-stories.md) is verified by repeated, recorded checks, with the evidence
  linked from the story ([user-stories.md, the verification gate](user-stories.md#the-verification-gate)).
- **Publishing is a human-only task,** and so is anything like it: releasing to the Godot Asset
  Store or crates.io, creating a release or a release tag, announcing, and promoting `develop` to
  `main`. Agents prepare builds, notes and checklists; the owner decides and does the rest.

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
- after the MVP: navigation ([#42](https://github.com/AntonTegnelov/wave_forge/issues/42), done), audio and localisation tags ([#43](https://github.com/AntonTegnelov/wave_forge/issues/43)), and placing users'
  own scenes by rule ([#44](https://github.com/AntonTegnelov/wave_forge/issues/44)), which becomes the Scatter and Emit stages of Phase 2's first slice;
- with Phase 2: noise that means the same in both engines ([#45](https://github.com/AntonTegnelov/wave_forge/issues/45)), ground, grass and wind
  ([#46](https://github.com/AntonTegnelov/wave_forge/issues/46)), far proxies and occluders ([#47](https://github.com/AntonTegnelov/wave_forge/issues/47));
- then authoring: recipes, brushes and bake, in Godot first because Bevy has no editor ([#48](https://github.com/AntonTegnelov/wave_forge/issues/48)).

Products that stay on the GPU are built only where a measurement shows the read-back path is too
slow.

## Phase 2: generation as a pack of stages

**Goal:** a developer can build the kind of world the games in [user-stories.md](user-stories.md)
generate (Minecraft, Dwarf Fortress, No Man's Sky, Noita, Caves of Qud, Elite Dangerous, Valheim,
Deep Rock Galactic), combining fields, scatter, sites, WFC and region-scale passes, and a newcomer
still reaches a walkable world of their own in minutes. The design, and the research behind it, is
[generation-model.md](generation-model.md).

The shape, which later work must not contradict:

- **A stage is a pure function of the seed, its id and a key, with a declared reach into its
  inputs**, generated provider-first in any order with the same result. Reads outside the declared
  reach are errors; cycles are refused when a pack loads. This generalises what the WFC generator
  already guarantees.
- **Typed data flows between stages**: fields, point sets, curve sets, stamps, records, Priors and
  tile grids, and an edits log. Engines receive products derived from them.
- **WFC is one stage kind among several**, driven by a Prior that Rules stages compute; fields and
  scatter are first-class, because most of the studied games spend their generation there.
- **Region jobs** cover what one chunk cannot see: rivers, zones, cave levels and finite worlds, on
  a coarse lattice, with hashed retries.
- **One pack format, four tiers**: presets, a stack, the pack as a graph, and Rust stages, with the
  debugging views as part of the product.

Order, each step verified before the next ([generation-model.md §8](generation-model.md#8-order-of-work)):

1. Positional placement ids.
2. The stage runtime on the CPU with the order-diff test, wrapping today's WFC.
3. ~~Making repairs a pure level of the WFC stage~~ ([#66](https://github.com/AntonTegnelov/wave_forge/issues/66)). Done: repairs wait for their
   neighbourhood and go by class, and the city is the same in any generation order.
4. The first slice, a Valheim-like surface world with WFC inside masks (G7), then the GPU Field stage.
5. Godot authoring and viewers, the Bevy loader.
6. One slice per stage kind: region jobs, Assemble, density volumes, records and hierarchy.

The integrations change little: they hand out the products of the stages a game asked for, as they
hand out a chunk's tiles today.

## Deferred

| Item | Why it waits |
|---|---|
| A-2: other topologies (hex, triangle, irregular) | No consumer needs one; 2D works as a world one cell deep. |
| A-15: GPU timestamp queries | Worth it when a performance question needs it, not before. |
| Godot `RenderingDevice` backend | Needs a measurement on a desktop Godot first (above). |
| `GlobalConstraint`, `TileWeighting`, a path constraint | Documented as future capabilities in [constraints.md](constraints.md). Revisited if the walk shows disconnected street networks. |
| A SAT (CDCL) solver, sparse voxel DAGs, 64-bit world coordinates with a floating origin | None of them solves a problem the MVP or the first two layers have. The chunk lattice uses 32-bit chunk coordinates, and `f32` positions stay millimetre-precise within several kilometres of the origin, far more than a walk covers. |
