# Story coverage

For each user story in [user-stories.md](../product/user-stories.md), what the library can express
today and what is missing, with the issues that close the gap. A story's status and evidence live in
the story itself; this page is the gap analysis behind the order in [roadmap.md](roadmap.md). Update
it when a pull request closes part of a gap.

## G7, a Valheim-like island world

Checked on 2026-09-23 against `src/stages/` after the first slice
([#68](https://github.com/AntonTegnelov/wave_forge/issues/68)). The valley pack gives rolling ground,
towns on levelled sites and trees kept apart, and the engines build its ground mesh and colliders. Of the features Valheim's generator uses, that covers
about three fully and six partly. The umbrella issue is
[#104](https://github.com/AntonTegnelov/wave_forge/issues/104). Valheim's algorithms come from a
community port and are unverified from Iron Gate (see the story's sources).

| Valheim feature | What it needs | Today | Issue |
|---|---|---|---|
| Seed-derived offsets | named hash streams | available: per stage, and per named noise | |
| Base height from products of Perlin octaves, with its own lacunarity and gain | Perlin or simplex noise, fBm parameters, a seed per noise, subtraction and absolute values | partial: value noise with fixed fBm; the formula itself is expressible (`tests/expressions.rs`) | [#45](https://github.com/AntonTegnelov/wave_forge/issues/45) |
| Radial falloff to the edge, a flattened spawn area | distance to the origin, smoothstep, remap, clamp, min and max, curves | available (`tests/expressions.rs`) | |
| Biomes by an ordered first-match list over distance, noise and height | categorical fields, a Rules stage | available (`examples/rings.world.ron`, `tests/rules.rs`) | |
| Height per biome, blended where biomes meet | select by category, blending | available (`examples/rings.world.ron`, `tests/blend.rs`) | |
| Cellular noise, quantised height in some biomes | cellular noise, floor | partial: `Floor` quantises; cellular noise is missing | [#45](https://github.com/AntonTegnelov/wave_forge/issues/45) |
| Lakes from a world scan | a region job on a coarse lattice | partial: region jobs and coarse levels exist; a lake job does not | |
| Rivers and streams that carve the ground | curves, a network between sites, rasterising curves into height | partial: region jobs make curves and Apply carves them into the height field without seams (`tests/apply.rs`); no Network stage | [#98](https://github.com/AntonTegnelov/wave_forge/issues/98) |
| A water level, ocean depth, snapping to water | a world water level, depth as a field | partial: a height range can stand in for depth | [#95](https://github.com/AntonTegnelov/wave_forge/issues/95) |
| Zones of 64 m, terrain at a finer resolution | field resolution independent of the WFC cell, coarse levels | available: a scale per stage, coarse feeding fine (`tests/levels.rs`) | |
| A location table: priority, quotas, unique, minimum distance from similar, centre first | several kinds per Sites stage, quotas on a region job | missing: one footprint per region, a chance only | [#97](https://github.com/AntonTegnelov/wave_forge/issues/97) |
| Location filters: biome, biome area, altitude, forest, terrain delta | filters reading fields and categories within a reach | partial: `Delta` gives terrain delta and `Area` biome area (`tests/filters.rs`), and biome and altitude are Rules conditions; the location table that applies them is missing | [#97](https://github.com/AntonTegnelov/wave_forge/issues/97) |
| Levelling the ground under locations, clearing around them | base, sites, adapted field; a margin for scatter | available: `Flatten`, Scatter's `avoid` | |
| Vegetation rules: counts and groups per zone, tilt, altitude, depth, forest threshold, scale | a Scatter modifier chain, slots, masks, normals | partial: chance, height, slope, spacing, a margin from sites | [#95](https://github.com/AntonTegnelov/wave_forge/issues/95) |
| Placements blocked by earlier placements | priority across Scatter stages | missing | [#96](https://github.com/AntonTegnelov/wave_forge/issues/96) |
| Dungeons: prefab room graphs, rerolled, placed above the entrance | Assemble into stamps, with retries | missing | [#70](https://github.com/AntonTegnelov/wave_forge/issues/70) |
| Clutter, never saved | an ephemeral stage; ground cover on the GPU | partial: every stage regenerates, so a Scatter stage is ephemeral; no GPU cover | [#46](https://github.com/AntonTegnelov/wave_forge/issues/46) |
| Felled trees and mined ore stay gone | an edits log keyed by `InstanceId` | missing; positional ids exist | [#101](https://github.com/AntonTegnelov/wave_forge/issues/101) |
| Terrain the player digs and raises | height deltas in the edits log, invalidating dependants | missing | [#101](https://github.com/AntonTegnelov/wave_forge/issues/101) |
| Frozen locations, a world-generator version in saves | persistence modes per stage | missing | [#102](https://github.com/AntonTegnelov/wave_forge/issues/102) |
| The ground's mesh and collider | a ground product, seamless across chunks, with material ids | available without materials: a mesh and a height-field collider per chunk in Godot, a mesh and height grid in Bevy | [#46](https://github.com/AntonTegnelov/wave_forge/issues/46) (materials) |
| Generate, load and active rings of different sizes | a radius per target stage | partial: one radius for all | [#103](https://github.com/AntonTegnelov/wave_forge/issues/103) |
| Queries for the minimap and spawners | sampling a stage at a point without chunks | available for Field, Rules, Blur, Delta and Area stages: `Runtime::sample` and `atlas` | |
| A finite disk with ocean outside | a world bound for stages | missing | [#99](https://github.com/AntonTegnelov/wave_forge/issues/99) |
| Creatures spawned by biome | spawn points and a biome query | partial: Scatter points, no biome | [#44](https://github.com/AntonTegnelov/wave_forge/issues/44), [#91](https://github.com/AntonTegnelov/wave_forge/issues/91) |
| World-scale generation speed | per-stage timings, GPU Field stages | timings available; fields cost 0.02 ms per chunk on the CPU, and the first town's seconds of kernel compilation dominate | [#111](https://github.com/AntonTegnelov/wave_forge/issues/111) |

Not needed for G7: density volumes ([#71](https://github.com/AntonTegnelov/wave_forge/issues/71);
Valheim is a height field) and tables of facts ([#72](https://github.com/AntonTegnelov/wave_forge/issues/72)).
Far proxies ([#47](https://github.com/AntonTegnelov/wave_forge/issues/47)) help mountain views
(P2) but are not in G7's criteria.

## The other stories

Stories not listed have nothing built towards them yet beyond the shared runtime.

| Story | What exists | What is missing |
|---|---|---|
| G1 Minecraft-like | fields, categories by rules, sites, scatter, the order-diff test | biomes by nearest point, 3D density volumes, Assemble (jigsaw), aquifers, carvers ([#71](https://github.com/AntonTegnelov/wave_forge/issues/71), [#70](https://github.com/AntonTegnelov/wave_forge/issues/70)) |
| G2 Dwarf Fortress-like | fields, region jobs with retries and curves, levels, the atlas and point queries, given tables whose rows become sites with towns of each row's rule set and roads levelled into the ground, rivers carved by region jobs | a Network stage, a world bound, the test pack ([#98](https://github.com/AntonTegnelov/wave_forge/issues/98), [#99](https://github.com/AntonTegnelov/wave_forge/issues/99)) |
| G3 No Man's Sky-like | fields, sites that flatten the ground, scatter, generated tables of planet parameters read through a focused row | density volumes, `locate` queries, edits ([#71](https://github.com/AntonTegnelov/wave_forge/issues/71), [#100](https://github.com/AntonTegnelov/wave_forge/issues/100), [#101](https://github.com/AntonTegnelov/wave_forge/issues/101)) |
| G4 Noita-like | WFC, positional ids | image import, Wang tiles, region jobs with path checks, stamps |
| G5 Caves of Qud-like | WFC inside bounded regions (towns) | map import, region jobs per zone, segmentation filters, connectivity, persistence modes ([#94](https://github.com/AntonTegnelov/wave_forge/issues/94), [#102](https://github.com/AntonTegnelov/wave_forge/issues/102)) |
| G6 Elite-like | positional ids, fields, sites, levels, generated tables of sectors, systems and bodies with shared budgets, a surface stage reading a focused body in both engines | the test pack and its check, authored rows among generated ones, Scatter and Apply reading a body's row ([#72](https://github.com/AntonTegnelov/wave_forge/issues/72)) |
| G8 Deep Rock-like | region jobs, curves drawn into a height field | Assemble, tunnels through density volumes, edits ([#70](https://github.com/AntonTegnelov/wave_forge/issues/70), [#71](https://github.com/AntonTegnelov/wave_forge/issues/71), [#101](https://github.com/AntonTegnelov/wave_forge/issues/101)) |
| N1 Press play | nodes that start on their own, colliders, navigation, a ground mesh and collider | presets, lit and textured defaults, navigation over packs ([#48](https://github.com/AntonTegnelov/wave_forge/issues/48), [#46](https://github.com/AntonTegnelov/wave_forge/issues/46)) |
| N3 Place my own scene | Scatter with spacing across seams | binding scenes to points ([#44](https://github.com/AntonTegnelov/wave_forge/issues/44)), rule resources ([#48](https://github.com/AntonTegnelov/wave_forge/issues/48)) |
| N7 My own noise | | FastNoiseLite parity ([#45](https://github.com/AntonTegnelov/wave_forge/issues/45)) |
| N11 My own history | region jobs, curves, categories, levels, the atlas, tables given and read back from GDScript, villages as sites with towns by culture or ruins, roads levelled into the ground, the example project (`examples/history`) and its check | a person following the README from a clean checkout |
| N9 Share a world by seed | identical tiles across GPUs and generation orders | product and chunk hashes |
| P1 Live generation without hitches | the Godot check meets the bars for a streamed WFC world with colliders and navigation | Phase 2 stages under the same bars; desktop numbers ([#39](https://github.com/AntonTegnelov/wave_forge/issues/39)) |
| P3 Fast into a new world | kernel warming | a kernel cache, and the time recorded on a desktop |
| P4 Bounded memory | the game session's chunk bound | a 30-minute walk measuring process and GPU memory |
