# Story coverage

For each user story in [user-stories.md](../product/user-stories.md), what the library can express
today and what is missing, with the issues that close the gap. A story's status and evidence live in
the story itself; this page is the gap analysis behind the order in [roadmap.md](roadmap.md). Update
it when a pull request closes part of a gap.

## G7, a Valheim-like island world

Checked on 2026-09-24 against `src/stages/` after the Assemble stage
([#70](https://github.com/AntonTegnelov/wave_forge/issues/70)). The ring world meets every acceptance
criterion of G7, which is verified ([user-stories.md](../product/user-stories.md#g7-a-valheim-like-island-world)).
The table compares the library with every feature Valheim's generator uses, some beyond G7's
criteria: 22 are available and 4 partial. The umbrella issue was
[#104](https://github.com/AntonTegnelov/wave_forge/issues/104). Valheim's algorithms come from a
community port and are unverified from Iron Gate (see the story's sources).

| Valheim feature | What it needs | Today | Issue |
|---|---|---|---|
| Seed-derived offsets | named hash streams | available: per stage, and per named noise | |
| Base height from products of Perlin octaves, with its own lacunarity and gain | Perlin or simplex noise, fBm parameters, a seed per noise, subtraction and absolute values | available: Godot's FastNoiseLite noises, Perlin and simplex among them, with every fractal parameter and a seed per noise (`tests/fastnoise.rs`); the formula itself is expressible (`tests/expressions.rs`) | |
| Radial falloff to the edge, a flattened spawn area | distance to the origin, smoothstep, remap, clamp, min and max, curves | available (`tests/expressions.rs`) | |
| Biomes by an ordered first-match list over distance, noise and height | categorical fields, a Rules stage | available (`examples/rings.world.ron`, `tests/rules.rs`) | |
| Height per biome, blended where biomes meet | select by category, blending | available (`examples/rings.world.ron`, `tests/blend.rs`) | |
| Cellular noise, quantised height in some biomes | cellular noise, floor | available: FastNoiseLite's cellular noise and `Floor` | |
| Lakes from a world scan | a region job on a coarse lattice | available: a Lakes stage fills each region's hollows with a priority flood, on a fine or a coarse lattice, the same in any order, and gives the water's surface as a field (`tests/lakes.rs`); the ring world has lakes; a hollow across a region border stays dry | |
| Rivers and streams that carve the ground | curves, a network between sites, rasterising curves into height | available: a Rivers stage runs rivers downhill to the sea in every region, and Apply carves them into the ground without seams (`tests/rivers.rs`, `tests/apply.rs`); a Network stage finds paths between sites (`tests/network.rs`); rivers end where they reach a lake (`tests/lakes.rs`); a river stops at its region's edge | |
| A water level, ocean depth, snapping to water | a world water level, depth as a field | available: a pack declares one water, a sea level and its lakes, which Scatter's depth tests, rivers and an engine read, and a Scatter point can stand on the water's surface (`tests/scatter_chain.rs`, `tests/lakes.rs`) | |
| Zones of 64 m, terrain at a finer resolution | field resolution independent of the WFC cell, coarse levels | available: a scale per stage, coarse feeding fine (`tests/levels.rs`) | |
| A location table: priority, quotas, unique, minimum distance from similar, centre first | several kinds per Sites stage, quotas on a region job | available: a Locations stage places kinds in priority order per region with quotas, distances from their own kind and conditions, and logs its refusals (`tests/locations.rs`); 'unique' in an infinite world is a quota of one per region, and 'centre first' is not built | |
| Location filters: biome, biome area, altitude, forest, terrain delta | filters reading fields and categories within a reach | available: `Delta` gives terrain delta and `Area` biome area (`tests/filters.rs`), and biome, altitude and a forest noise are conditions over fields, which a location table's kinds apply | |
| Levelling the ground under locations, clearing around them | base, sites, adapted field; a margin for scatter | available: `Flatten`, Scatter's `avoid` | |
| Vegetation rules: counts and groups per zone, tilt, altitude, depth, forest threshold, scale | a Scatter modifier chain, slots, masks, normals | available: counts and groups per block, conditions over any field or category, water depth, scale, tilt and ground alignment (`tests/scatter_chain.rs`); the ring world scatters ore and birch groves | |
| Placements blocked by earlier placements | priority across Scatter stages | available: `block` keeps a clearance from earlier Scatter stages' points, the same in any order (`tests/scatter_chain.rs`) | |
| Dungeons: prefab room graphs, rerolled, placed above the entrance | Assemble into stamps, with retries | available: an Assemble stage grows rooms from connectors on a location table's kind, caps dead ends, rerolls a dungeon that comes out small, and lifts it above its entrance; the ring world grows one above every crypt (`tests/assemble.rs`, `verify_assemble.gd`) | |
| Clutter, never saved | an ephemeral stage; ground cover on the GPU | available in Godot: an `Ephemeral` stage's edits are never saved, and the ring world's grass is one (`tests/save.rs`, `verify_edits.gd`); grass expanded on the GPU from a cover field (`verify_ground.gd`, E45), and in Bevy through a reference WESL material (`grass_render.rs`) | |
| Felled trees and mined ore stay gone | an edits log keyed by `InstanceId` | available: `Edit::Remove` by a point's id, through eviction and regeneration (`tests/edits.rs`, `verify_edits.gd`) | |
| Terrain the player digs and raises | height deltas in the edits log, invalidating dependants | available: `Edit::Raise` of a field by column, dirtying what reads it within its reach (`tests/edits.rs`) | |
| Frozen locations, a world-generator version in saves | persistence modes per stage | available: a `Frozen` stage keeps its chunks as first generated through a pack change, and a save records the generator's version and the pack's digest; the ring world freezes its shrines (`tests/save.rs`) | |
| The ground's mesh and collider | a ground product, seamless across chunks, with material ids | available: a mesh and a height-field collider per chunk in Godot, with a material per category of a Rules stage through a reference ground shader (`verify_ground.gd`); a mesh and height grid in Bevy, with the same materials through a reference WESL material (`ground_material_render.rs`); levels of detail with skirts, which both engines draw by distance without gaps (`render_lods.gd`, E47, `ground_levels_render.rs`) | |
| Generate, load and active rings of different sizes | a radius per target stage | available: `request_each`, the Godot node's `target_radii` and Bevy's `with_radius` (`tests/stages.rs`) | |
| Queries for the minimap and spawners | sampling a stage at a point without chunks | available for Field, Rules, Blur, Delta and Area stages: `Runtime::sample` and `atlas` | |
| A finite disk with ocean outside | a world bound for stages | available: a pack's `bound`, a disk or a rectangle, beyond which no target chunk is generated, and `request_bound` to prepare a finite world before play (`tests/bound.rs`); the ring world is a bounded island | |
| Creatures spawned by biome | spawn points and a biome query | partial: Scatter points, no biome | [#44](https://github.com/AntonTegnelov/wave_forge/issues/44), [#91](https://github.com/AntonTegnelov/wave_forge/issues/91) |
| World-scale generation speed | per-stage timings, GPU Field stages | timings available; fields cost 0.02 ms per chunk on the CPU, and the first town's seconds of kernel compilation dominate | [#111](https://github.com/AntonTegnelov/wave_forge/issues/111) |

Not needed for G7: density volumes ([#71](https://github.com/AntonTegnelov/wave_forge/issues/71);
Valheim is a height field) and tables of facts ([#72](https://github.com/AntonTegnelov/wave_forge/issues/72)).
Far proxies ([#47](https://github.com/AntonTegnelov/wave_forge/issues/47), built) and a far ground
from a coarse height stage ([#183](https://github.com/AntonTegnelov/wave_forge/issues/183)) serve
mountain views (P2) but are not in G7's criteria.

## The other stories

Stories not listed have nothing built towards them yet beyond the shared runtime.

| Story | What exists | What is missing |
|---|---|---|
| G1 Minecraft-like | fields, categories by rules, sites, scatter, jigsaw villages by Assemble, the order-diff test | biomes by nearest point, 3D density volumes, aquifers, carvers ([#71](https://github.com/AntonTegnelov/wave_forge/issues/71)) |
| G2 Dwarf Fortress-like | fields, region jobs with retries and curves, levels, the atlas and point queries, given tables whose rows become sites with towns of each row's rule set and roads levelled into the ground, rivers carved by region jobs, roads between sites by a Network stage | the test pack |
| G3 No Man's Sky-like | fields, sites that flatten the ground, scatter, generated tables of planet parameters read through a focused row | density volumes, `locate` queries ([#71](https://github.com/AntonTegnelov/wave_forge/issues/71), [#100](https://github.com/AntonTegnelov/wave_forge/issues/100)) |
| G4 Noita-like | WFC, positional ids | image import, Wang tiles, region jobs with path checks, stamps |
| G5 Caves of Qud-like | WFC inside bounded regions (towns), frozen stages for zones kept as first generated | map import, region jobs per zone, segmentation filters, connectivity ([#94](https://github.com/AntonTegnelov/wave_forge/issues/94)) |
| G6 Elite-like | positional ids, fields, sites, levels, generated tables of sectors, systems and bodies with shared budgets, a surface stage reading a focused body in both engines | the test pack and its check, authored rows among generated ones, Scatter and Apply reading a body's row ([#72](https://github.com/AntonTegnelov/wave_forge/issues/72)) |
| G8 Deep Rock-like | region jobs, curves drawn into a height field, rooms by Assemble | caves carved into density volumes from a room graph, tunnels through them ([#71](https://github.com/AntonTegnelov/wave_forge/issues/71)) |
| N1 Press play | nodes that start on their own, colliders, navigation, a ground mesh and collider | presets, textured defaults (the reference ground shader colours by palette), navigation over packs ([#48](https://github.com/AntonTegnelov/wave_forge/issues/48)) |
| N3 Place my own scene | Scatter with spacing across seams, scenes bound to kinds in Godot as MultiMeshes or nodes with promotion near the player, and in Bevy as entities | pooling ([#44](https://github.com/AntonTegnelov/wave_forge/issues/44)), rule resources ([#48](https://github.com/AntonTegnelov/wave_forge/issues/48)) |
| N7 My own noise | FastNoiseLite parity, exact against Godot, and `FastNoiseLite` resources assigned in the Godot node | nothing for its criterion; Bevy's `Reflect` and 3D noise are [#45](https://github.com/AntonTegnelov/wave_forge/issues/45) |
| N11 My own history | region jobs, curves, categories, levels, the atlas, tables given and read back from GDScript, villages as sites with towns by culture or ruins, roads levelled into the ground, the example project (`examples/history`) and its check | a person following the README from a clean checkout |
| N9 Share a world by seed | identical tiles across GPUs and generation orders | product and chunk hashes |
| P1 Live generation without hitches | the Godot check meets the bars for a streamed WFC world with colliders and navigation | Phase 2 stages under the same bars; desktop numbers ([#39](https://github.com/AntonTegnelov/wave_forge/issues/39)) |
| P3 Fast into a new world | kernel warming | a kernel cache, and the time recorded on a desktop |
| P4 Bounded memory | the game session's chunk bound | a 30-minute walk measuring process and GPU memory |
