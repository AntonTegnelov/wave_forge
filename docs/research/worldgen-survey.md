# World generation survey

The research behind [stages.md](../architecture/stages.md): how engines, libraries and eight games
structure world generation, and the failures the design has to rule out. The game-by-game findings,
with their sources, are the context sections of the G stories in
[user-stories.md](../product/user-stories.md#g-replicate-a-games-generation).

## How it was done

Two studies, in September 2026:

- **Engines and libraries:** Unreal PCG, LayerProcGen, Houdini, Gaea and World Machine, MapMagic,
  godot_voxel, ProtonScatter, WFC tools, CityEngine, Minecraft datapacks and FastNoise2, read from
  their documentation and source. An adversarial fact-check of the 43 load-bearing claims confirmed
  32, corrected 9, left 2 unverifiable and refuted none.
- **Case studies:** Minecraft, Dwarf Fortress, No Man's Sky, Noita, Caves of Qud, Elite Dangerous,
  Valheim and Deep Rock Galactic, from developer talks, official wikis and, where nothing official
  exists, community ports and reverse engineering (marked as such in the stories).

## What the systems have in common

**Every system that works is a directed acyclic graph of pure transforms over typed data.** They
differ in what a node is, what flows along an edge, and whether an edge declares how far it reads.

- **LayerProcGen gets execution right.** Each layer has its own chunk size and declares a padding
  per dependency; before a chunk is generated, every provider chunk inside that padding is generated
  first ([Layer Dependencies](https://github.com/runevision/LayerProcGen/blob/main/Documentation/LayerDependencies.md),
  `EnsureChunkProviders` in [ChunkBasedDataLayer.cs](https://github.com/runevision/LayerProcGen/blob/main/Src/LayerProcGen/ChunkBasedDataLayer.cs)).
  A step that modifies data writes a new layer instead, which is what makes the result independent
  of generation order ([Contextual Generation](https://runevision.github.io/LayerProcGen/md_ContextualGeneration.html),
  [Internal Layer Levels](https://github.com/runevision/LayerProcGen/blob/main/Documentation/InternalLayerLevels.md)).
  Its chunk data is opaque to the framework, so there are no generic previews and no data-driven
  graphs; it is a library for programmers.
- **Unreal PCG gets data and tooling right.** Its currency is points with fixed properties and open
  attributes; samplers turn landscapes, splines and volumes into points, and filters and modifiers
  work on them ([data types](https://dev.epicgames.com/documentation/en-us/unreal-engine/procedural-content-generation-framework-data-types-reference-in-unreal-engine)).
  Scale is set per graph branch by grid size, and data only cascades from larger grids to smaller
  ones ([hierarchical generation](https://dev.epicgames.com/documentation/unreal-engine/hierarchical-generation?lang=en-US)).
  No node declares a reach, the documentation leaves duplicate points at cell borders to the author,
  and no source shows that partitioned output equals unpartitioned output (unverified either way).
  PCG moved its point data from an array of structs to a structure of arrays in 5.6, at the cost of a
  breaking change ([Epic roadmap](https://portal.productboard.com/epicgames/1-unreal-engine-public-roadmap/c/1894-point-data-structure-of-array),
  [migration report](https://forums.unrealengine.com/t/pcg-problem-going-from-ue-5-5-to-5-6-get-point-data-not-working-anymore/2651694)).
- **Minecraft gets user extensibility right.** World generation is named JSON resources that
  reference each other (density functions, noise settings, biome sources, placed features,
  structure sets), run as a staged pipeline with bounded neighbour reach
  ([World generation](https://minecraft.wiki/w/World_generation),
  [Density function](https://minecraft.wiki/w/Density_function),
  [Placed feature](https://minecraft.wiki/w/Placed_feature)). The community edits it through live
  visualisers such as [misode's generators](https://misode.github.io/worldgen/).
- **Offline terrain tools are raster graphs on a finite canvas.** Houdini, Gaea and World Machine
  admit that erosion and other simulation nodes give different results when tiled
  ([HeightField Tile Split](https://www.sidefx.com/docs/houdini/nodes/sop/heightfield_tilesplit.html)),
  which is why their output reaches games as baked assets.
- **Layered authoring succeeds where graphs overwhelm.** MapMagic's layered nodes and
  ProtonScatter's modifier stack ([ProtonScatter modifiers](https://github.com/HungryProton/scatter/wiki/Modifiers))
  let people build rich rules as ordered lists, which is the origin of the stack tier.

## What the eight games need

- Every one needs points with attributes and persistent edits, and seven need continuous fields.
- Only Caves of Qud uses WFC (the overlapping model, inside segmented regions). Noita uses
  herringbone Wang tiles, which are constraint-matched but not a solver.
- Six of the eight need a whole-region or whole-world pass: Dwarf Fortress's erosion and history,
  Noita's Wang regions with path checks, Qud's zones, Elite's planets, Valheim's rivers and location
  table, and Deep Rock Galactic's mission level.

## Failures the design has to rule out

Each was seen in a shipped system.

- **A sequential random stream,** so inserting one thing reshuffles everything after it
  (Minecraft's feature seeds, Dwarf Fortress when one token changes, Valheim's draw order).
- **Ids that are ordinals,** so they shift when content is inserted (Elite, when authored bodies
  were added).
- **Generators writing into neighbouring chunks,** so a world depends on the player's travel route
  (Minecraft's features, MC-55596, cited on [World generation](https://minecraft.wiki/w/World_generation);
  godot_voxel's multipass generator, whose [documentation](https://voxel-tools.readthedocs.io/en/latest/api/VoxelGeneratorMultipassCB/)
  says so).
- **One global margin for every stage,** which causes seams
  ([MapMagic, Tile Seams Reasons](https://gitlab.com/api/v4/projects/denispahunov%2Fmapmagic/wikis/Tile_Seams_Reasons)).
- **A graph too large to author by hand:** the Overworld's terrain `offset` alone is a nested
  spline of 253 points ([overworld noise settings](https://raw.githubusercontent.com/misode/mcmeta/data/data/minecraft/worldgen/noise_settings/overworld.json)).

## What followed for the design

- A stage is a pure function with a declared reach, generated providers first (LayerProcGen's
  execution), over typed points and fields (Unreal PCG's data), described in a named-resource pack
  (Minecraft's extensibility).
- Every random decision is a named hash stream and every id is positional.
- Structures adapt terrain by the base, sites, adapted pattern rather than by writing into
  neighbours.
- Whole-region passes are first-class (region jobs), because six of eight games need one.
- WFC is one stage kind, not the centre.

[stages.md](../architecture/stages.md) turns these into the execution contract.
