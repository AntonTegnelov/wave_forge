# Roadmap

Where Wave Forge is heading, and in what order. This page is the only place the order of work is
kept; design pages say what and why, and link here for when. Phases come from
[vision.md](../product/vision.md), where things stand now is in [status.md](status.md), and a phase
is done when its exit criteria hold, not when its task list is empty.

## Phases

**Phase 0, revive the project: done.** The abandoned code base builds, produces valid output, and
runs on current dependencies ([#3](https://github.com/AntonTegnelov/wave_forge/issues/3),
[#4](https://github.com/AntonTegnelov/wave_forge/issues/4)).

**Phase 1, a standalone 2D and 3D WFC library: done.** Tooling and tests came first
([#6](https://github.com/AntonTegnelov/wave_forge/issues/6)); profiling then chose the block kernel
and the chunked, streamed world ([#7](https://github.com/AntonTegnelov/wave_forge/issues/7)). Its
exit criteria hold: a documented library API, identical output across runs and thread counts, 2D and
3D end-to-end tests, benchmarks behind every placement decision, and streaming with seamless
borders inside a 0.5 s tick.

**MVP, engine integrations and a walkable city: integrations done, the walk moves to its game.** The
success criterion is walking around in a Godot game inside an infinite city generated continuously
at run time, in the spirit of marian42's city. The library side is done: module sets, tile
metadata, every chunk placed, exported models, instance sets, colliders and navigation, in a Godot
extension and a Bevy plugin that are verified in real engines. The walk itself is the first
proof-of-concept game, in its own repository
([#76](https://github.com/AntonTegnelov/wave_forge/issues/76)). What remains here is levels of
detail for module meshes, once authored models exist
([#38](https://github.com/AntonTegnelov/wave_forge/issues/38)).

**Phase 2, generation as a pack of stages: in progress.** A developer can build the kind of world
the games in [user-stories.md](../product/user-stories.md) generate, and a newcomer still reaches a
walkable world of their own in minutes. The design is [stages.md](../architecture/stages.md). Done
so far:

- positional placement ids ([#64](https://github.com/AntonTegnelov/wave_forge/issues/64));
- the stage runtime on the CPU and the order-diff test
  ([#67](https://github.com/AntonTegnelov/wave_forge/issues/67),
  [#65](https://github.com/AntonTegnelov/wave_forge/issues/65));
- repairs as a pure level of the WFC stage, so a city is the same in any order
  ([#66](https://github.com/AntonTegnelov/wave_forge/issues/66));
- a first slice towards G7's island world: height, towns on levelled sites, trees kept apart, and
  a Godot node and a Bevy plugin serving it ([#68](https://github.com/AntonTegnelov/wave_forge/issues/68),
  [#85](https://github.com/AntonTegnelov/wave_forge/pull/85),
  [#86](https://github.com/AntonTegnelov/wave_forge/pull/86)).

The first slice did not reach G7: biomes, per-biome height, rivers, a location table, dungeons
and edits are still missing; ground meshes and colliders came after it
([#88](https://github.com/AntonTegnelov/wave_forge/issues/88)). [story-coverage.md](story-coverage.md)
has the full gap, and [#104](https://github.com/AntonTegnelov/wave_forge/issues/104) tracks it.

## Open work, in order

Each step is a verified pull request that updates the stories it moves. The order puts G7's world
first, which the survival game grows into, then what the strategy game needs.

1. **Frame bounds and town kernels:** collider bodies bounded per frame in both Godot nodes
   ([#118](https://github.com/AntonTegnelov/wave_forge/issues/118)), and town kernels compiled
   before play and cached ([#111](https://github.com/AntonTegnelov/wave_forge/issues/111)).
   Per-stage timings ([#87](https://github.com/AntonTegnelov/wave_forge/issues/87)) showed fields
   cost 0.02 ms per chunk, so Field stages stay on the CPU.
2. **Tables of facts** ([#72](https://github.com/AntonTegnelov/wave_forge/issues/72)): rows the game
   gives, a simulated history say, or rows generated from a parent table, read by stages; with N11's
   example project, a toy history in GDScript. Given and generated tables and a focused row are
   built; next are the engines' access, Sites and Solve reading tables, roads with
   [#98](https://github.com/AntonTegnelov/wave_forge/issues/98), and the example. With region jobs
   ([#69](https://github.com/AntonTegnelov/wave_forge/issues/69)) and these, the strategy game
   becomes possible, and an issue opens for its repository.
3. **The rest of G7**, in the order of [#104](https://github.com/AntonTegnelov/wave_forge/issues/104):
   FastNoiseLite-compatible noise ([#45](https://github.com/AntonTegnelov/wave_forge/issues/45)),
   neighbourhood filters ([#94](https://github.com/AntonTegnelov/wave_forge/issues/94)), a Scatter
   modifier chain ([#95](https://github.com/AntonTegnelov/wave_forge/issues/95)), blocking across
   Scatter stages ([#96](https://github.com/AntonTegnelov/wave_forge/issues/96)), a location table
   ([#97](https://github.com/AntonTegnelov/wave_forge/issues/97)), rivers that carve the ground
   ([#98](https://github.com/AntonTegnelov/wave_forge/issues/98)), a world bound
   ([#99](https://github.com/AntonTegnelov/wave_forge/issues/99)), dungeons through Assemble
   ([#70](https://github.com/AntonTegnelov/wave_forge/issues/70)), edits and persistence modes
   ([#101](https://github.com/AntonTegnelov/wave_forge/issues/101),
   [#102](https://github.com/AntonTegnelov/wave_forge/issues/102)), scenes bound to points
   ([#44](https://github.com/AntonTegnelov/wave_forge/issues/44)), and a radius per stage
   ([#103](https://github.com/AntonTegnelov/wave_forge/issues/103)). A test pack of G7's world then
   verifies G7.
4. **Engine depth, alongside:** ground cover, grass and wind
   ([#46](https://github.com/AntonTegnelov/wave_forge/issues/46)), region tags for audio and
   localisation ([#43](https://github.com/AntonTegnelov/wave_forge/issues/43)), far proxies and
   occluders ([#47](https://github.com/AntonTegnelov/wave_forge/issues/47)), density volumes for
   G1, G3 and G8 ([#71](https://github.com/AntonTegnelov/wave_forge/issues/71)).
5. **Authoring:** presets, the stage stack, viewers, brushes and bake, in Godot first because Bevy
   has no editor ([#48](https://github.com/AntonTegnelov/wave_forge/issues/48)).

**Waiting on the owner's hardware:** the frame-time measurement of own against shared devices on
desktops with native drivers ([#39](https://github.com/AntonTegnelov/wave_forge/issues/39)). It
decides the device policy in both engines, whether a backend over Godot's `RenderingDevice` is worth
building, and whether the P stories' targets hold.

**Before any release:** builds for every desktop platform the Godot Asset Store expects (Windows,
macOS, Linux), and a note in the extension on where godot-rust's MPL-2.0 source can be obtained.

## Games, packs and repositories

Three proof-of-concept games consume Wave Forge, each kept simple at first; any that proves fun can
grow.

| Game | Engine | What it needs from Wave Forge | Repository |
|---|---|---|---|
| Parkour hide-and-seek and tag in an infinite WFC city, in the style of marian42's city | Godot | the streamed city with colliders and navigation | possible now ([#76](https://github.com/AntonTegnelov/wave_forge/issues/76)) |
| Survival and crafting on a generated island with biome rings (G7) | Godot | the stages, with ground meshes and colliders; then G7 | possible for a first version ([#89](https://github.com/AntonTegnelov/wave_forge/issues/89)) |
| Grand strategy inspired by Europa Universalis, on a world and history generated once per new game in the manner of Dwarf Fortress | Bevy | region jobs, levels, the atlas and tables of facts ([#69](https://github.com/AntonTegnelov/wave_forge/issues/69), [#93](https://github.com/AntonTegnelov/wave_forge/issues/93), [#100](https://github.com/AntonTegnelov/wave_forge/issues/100), [#72](https://github.com/AntonTegnelov/wave_forge/issues/72)); the history simulation is the game's, given as facts | after step 2 above |

**Each game lives in its own private repository.** They are not open source. They depend on Wave
Forge by a pinned commit, and moving the pin is a deliberate change in the game's repository, which
is also the best test of whether the public API is enough.

**This repository keeps** the library, the integrations, their verification projects, tests and
manual experiments in Godot and Bevy, and two kinds of packs:

| Kind | Where | Purpose |
|---|---|---|
| **Game packs** | each game's repository | the game's world, tuned for play; smoke tests for Wave Forge |
| **Test packs** | here | edge cases, stress tests, and one pack per user story exercising its techniques; the evidence for the [verification gate](../product/user-stories.md#the-verification-gate) |
| **Presets** | here | the product's out-of-the-box worlds (N1, N2, N10) |

**Smoke tests run downstream.** Each game's CI builds it against Wave Forge's latest `develop` every
night and on every change of its pin, and runs its packs, so nothing here needs access to a private
repository and no private pack reaches a public log. A failure comes back as an issue here with a
public reproduction ([template](../../.github/ISSUE_TEMPLATE/from-a-game.md)): the game's pack is
reduced to a test pack in this repository, which then stays as a regression test.

**Access.** This repository is public, so the games read it without a token. A game's environment
may open issues here (a token with issue access only), and this repository's environment may read
the games' repositories (read-only). Nothing from a private game (code, assets, design text, pack
contents) is ever copied into this repository, its issues or its commits; a report reproduces the
problem in Wave Forge's own terms.

## Release gate

Nothing is done or published until every story in [user-stories.md](../product/user-stories.md) is
verified by repeated, recorded checks, with the evidence linked from the story. Publishing, and
anything like it, is the owner's alone ([contributing.md](../guides/contributing.md#publishing-is-human-only)).

## Deferred

| Item | Why it waits |
|---|---|
| Other topologies (hex, triangle, irregular) | No consumer needs one; 2D works as a world one cell deep. |
| GPU timestamp queries and `tracing` spans | Worth it when a performance question needs them. |
| A backend over Godot's `RenderingDevice` | Needs the desktop measurement ([#39](https://github.com/AntonTegnelov/wave_forge/issues/39)) first. |
| Global and statistical constraints (`GlobalConstraint`, `TileWeighting`, a path constraint) | Documented as future capabilities in [constraints.md](../architecture/constraints.md); revisited if a game shows disconnected street networks. |
| A SAT (CDCL) solver, sparse voxel DAGs, 64-bit world coordinates with a floating origin | None solves a problem the current games have. Chunk coordinates are 32-bit and products are chunk-local, and `f32` positions stay millimetre-precise within several kilometres of the origin. |
| Spheres, galaxies, `f64` | wgpu has no portable `f64`; the types allow them later ([stages.md](../architecture/stages.md#scope)). |
| Kernel unit tests | The kernel is checked through whole regions, the CPU oracle and golden worlds; unit tests of its internals wait for a bug they would have caught. |
