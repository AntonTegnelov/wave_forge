# Generation model: a pack of stages

How Phase 2 layers many kinds of generation (fields, scatter, sites, WFC, simulation passes) so a
developer can build the kind of world the games in [user-stories.md](user-stories.md) generate, and
still start in minutes. Every design choice here names the user stories it serves (G for replicating a
game, N for newcomers, P for performance).

This refines the Phase 2 shape in [roadmap.md](roadmap.md#phase-2-generation-as-a-pack-of-stages):
a unit of generation is a pure function of the seed and a key, with a declared reach and no cycles,
and WFC is driven by a `Prior`. What changed is the scope. Research into how engines, libraries and
eight games structure generation showed that WFC cannot be the centre of the design, that most games
need whole-region computation, and that the generator's repairs as built today conflict with order
independence (§5).

## 1. What the research found

Two studies fed this document: one of engines and libraries (Unreal PCG, LayerProcGen, Houdini,
Gaea and World Machine, MapMagic, godot_voxel, ProtonScatter, WFC tools, CityEngine, Minecraft
datapacks, FastNoise2), checked by an adversarial fact-check of 43 load-bearing claims (32
confirmed, 9 corrected, 2 unverifiable, none refuted), and case studies of Minecraft, Dwarf
Fortress, No Man's Sky, Noita, Caves of Qud, Elite Dangerous, Valheim and Deep Rock Galactic. The
game-by-game findings are summarised with their sources in [user-stories.md](user-stories.md#g-replicate-a-games-generation).

**Every system that works is a directed acyclic graph of pure transforms over typed data.** They
differ in what a node is, what flows along an edge, and whether an edge declares how far it reads.

- **LayerProcGen gets execution right.** Each layer has its own chunk size and declares a padding
  per dependency; before a chunk is generated, every provider chunk inside that padding is generated
  first ([Layer Dependencies](https://github.com/runevision/LayerProcGen/blob/main/Documentation/LayerDependencies.md),
  `EnsureChunkProviders` in [ChunkBasedDataLayer.cs](https://github.com/runevision/LayerProcGen/blob/main/Src/LayerProcGen/ChunkBasedDataLayer.cs)).
  A step that modifies data writes a new layer instead, which is what makes the result independent of
  generation order ([Contextual Generation](https://runevision.github.io/LayerProcGen/md_ContextualGeneration.html),
  [Internal Layer Levels](https://github.com/runevision/LayerProcGen/blob/main/Documentation/InternalLayerLevels.md)).
  Its chunk data is opaque to the framework, so there are no generic previews and no data-driven
  graphs; it is a library for programmers.
- **Unreal PCG gets data and tooling right.** Its currency is points with fixed properties and open
  attributes; samplers turn landscapes, splines and volumes into points, and filters and modifiers
  work on them ([data types](https://dev.epicgames.com/documentation/en-us/unreal-engine/procedural-content-generation-framework-data-types-reference-in-unreal-engine)).
  Scale is set per graph branch by grid size, and data only cascades from larger grids to smaller
  ones ([hierarchical generation](https://dev.epicgames.com/documentation/unreal-engine/hierarchical-generation?lang=en-US)).
  No node declares a reach, the docs leave duplicate points at cell borders to the author, and no
  source shows that partitioned output equals unpartitioned output (unverified either way).
- **Minecraft gets user extensibility right.** World generation is named JSON resources that
  reference each other (density functions, noise settings, biome sources, placed features, structure
  sets), run as a staged pipeline with bounded neighbour reach
  ([World generation](https://minecraft.wiki/w/World_generation),
  [Density function](https://minecraft.wiki/w/Density_function),
  [Placed feature](https://minecraft.wiki/w/Placed_feature)). The community edits it through live
  visualisers such as [misode's generators](https://misode.github.io/worldgen/).
- **Offline terrain tools are raster graphs on a finite canvas.** Houdini, Gaea and World Machine
  admit that erosion and other simulation nodes give different results when tiled
  ([HeightField Tile Split](https://www.sidefx.com/docs/houdini/nodes/sop/heightfield_tilesplit.html)),
  which is why their output reaches games as baked assets.

**In the eight games,** every one needs points with attributes and persistent edits, seven need
continuous fields, only Caves of Qud uses WFC (overlapping model, inside segmented regions), and
Noita uses herringbone Wang tiles, which are constraint-matched but not a solver. Six of the eight
need a whole-region or whole-world pass: Dwarf Fortress's erosion and history, Noita's Wang regions
with path checks, Qud's zones, Elite's planets, Valheim's rivers and location table, and Deep Rock
Galactic's mission level.

**Failures the design has to rule out,** each seen in a shipped system:

- A sequential random stream, so inserting one thing reshuffles everything after it (Minecraft's
  feature seeds, Dwarf Fortress when one token changes, Valheim's draw order).
- Ids that are ordinals, so they shift when content is inserted (Elite, when authored bodies were
  added).
- Generators writing into neighbouring chunks, so a world depends on the player's travel route
  (Minecraft's features, MC-55596, cited on [World generation](https://minecraft.wiki/w/World_generation);
  godot_voxel's multipass generator, whose [documentation](https://voxel-tools.readthedocs.io/en/latest/api/VoxelGeneratorMultipassCB/)
  says so).
- One global margin for every stage, which causes seams ([MapMagic, Tile Seams Reasons](https://gitlab.com/api/v4/projects/denispahunov%2Fmapmagic/wikis/Tile_Seams_Reasons)).
- A graph too large to author by hand: the Overworld's terrain `offset` alone is a nested spline of
  253 points ([overworld noise settings](https://raw.githubusercontent.com/misode/mcmeta/data/data/minecraft/worldgen/noise_settings/overworld.json)).

## 2. The execution contract

The unit is called a **stage**, not a layer: `Prior` already uses "layer" for its per-height masks
([architecture.md §3.3](architecture.md)).

- **Purity.** A stage's output for a key is a pure function of the world seed, the stage's id, the
  key, and read-only outputs of its inputs within a declared reach. Reach is in world units and 3D
  from the start; where parameters imply it (a blur radius, a spacing, a footprint), it is derived
  from them. (G1 to G8, N9.)
- **Bounded reads.** A stage reads its inputs through a view bounded by its declared reach. A read
  outside it is an error naming the stage and the reach it would need. The transitive reach, the
  area each stage has to generate for a given request, is computed at load and shown to the user.
  (N5, P2.)
- **Gather, never scatter.** A stage writes only its own key. Anything that straddles a border, such
  as a building or a spacing decision, has exactly one owner, chosen by a pure function of its
  anchor; neighbours recompute the same anchor instead of receiving writes. Stages ask for their
  inputs in one of two ways: *owned* (emit once, for things with stable ids) and *overlapping* (apply
  everywhere it touches, for flatten masks and Prior bans).
- **No cycles, checked at load.** Terrain that structures adapt is split the way every studied game
  splits it: a base field, then sites that read only the base field, then an adapted field (Minecraft's
  beardifier, No Man's Sky's flatten points, Valheim's location levelling). (G1, G3, G7.)
- **Named hash streams.** Every random decision is `hash(seed, stage id, key, purpose)`, built on the
  stateless `pcg3d` the solver already uses (`wfc-core/src/hash.rs`), never a sequential generator.
  Existence tests compare integers. Data crossing a stage boundary (site positions, spline control
  points) is integer or fixed point, because equal floats are not guaranteed across GPU vendors.
  (N9, G6.)
- **Levels and scales.** Each stage declares a cell size on a hierarchy of global, region, chunk and
  cell. A stage may read its parent level (Elite's sector, system and body; No Man's Sky's region,
  system and planet). A **region job** runs on a coarse region lattice: any bounded pure computation,
  which may iterate, count and retry with a hashed retry index. A finite world is a single region
  computed before streaming starts. Border agreement between region jobs comes from edge-keyed
  hashes (a river's crossing point hashed from the shared edge), not from reading each other.
  (G2, G4, G5, G6, G7, G8.)
- **Scheduling.** Providers first, eagerly, with reference-counted lifetimes, following LayerProcGen.
  Batches go stage-major and nearest-first so one GPU dispatch covers one stage and level. The
  generator's schedule is already this pattern: parity 0 and parity 1 are two levels of the WFC
  stage, and the closure rule in [architecture.md §6.2](architecture.md) is provider-first
  generation written by hand. (P1, P3.)
- **Persistence per stage.** A *pure* stage regenerates and replays edits; a *freeze on first emit*
  stage is snapshotted (as Minecraft, Noita, Qud and Valheim do with placed objects); an *ephemeral*
  stage (Valheim's clutter) is never saved. Saves record the generator version. (G7, N8, P4.)

## 3. Data between stages

| Type | Contents | Covers |
|---|---|---|
| **Field** | named `f32` channels on a 2D or 3D grid at the stage's cell size | height, climate, masks (a channel in 0..1), density and signed-distance volumes, categorical ids such as a biome |
| **PointSet** | structure of arrays: fixed-point position, rotation, scale, stable id, kind, and attribute columns | sites, anchors, scatter candidates and placements, spawn points |
| **CurveSet** | polylines with per-vertex attributes (radius, flow, profile) and optional connectivity | roads, rivers, tunnels, room and site graphs |
| **Stamps** | an ordered list of carve, fill and prefab primitives, each with bounds | jigsaw pieces, cave rooms, flatten areas |
| **Record** | a typed struct keyed by a hierarchical address | planet or system parameters, a history log the user supplies, a location table |
| **Prior** and **TileGrid** | the WFC stage's input and output, as today | tiles |
| **Edits** | an operation log keyed by stable ids and cells | brushes, removed and moved placements, terrain deltas |

PointSet is a structure of arrays from the start; Unreal PCG moved from an array of structs in 5.6
at the cost of a breaking change ([Epic roadmap](https://portal.productboard.com/epicgames/1-unreal-engine-public-roadmap/c/1894-point-data-structure-of-array),
[migration report](https://forums.unrealengine.com/t/pcg-problem-going-from-ue-5-5-to-5-6-get-point-data-not-working-anymore/2651694)). The products an engine consumes (instance sets, colliders,
navigation source) are not an eighth type: an `Emit` stage derives them from PointSets and TileGrids
([engine-integration.md §2](engine-integration.md#2-products)).

## 4. Stage kinds

| Kind | What it does | Reach | Serves |
|---|---|---|---|
| **Field** | a fused pointwise expression graph: noise, splines, remap, math, first-match classifiers, image lookup; compiles to one kernel | 0 | G1, G3, G6, G7, N2, N7 |
| **Filter** | stencils, blur, cellular automata, slope, distance transforms | declared per pass | G2, G5, G7 |
| **Rules** | first-match rule trees (sequence, condition, result) producing a Prior or a categorical field, like Minecraft's surface rules | that of its conditions | G1, G7, N4 |
| **Solve** | WFC over a Prior; Wang tiling later | the solver's halo | G5, G4, N6 |
| **Sites** | owned region-scale points, one candidate per region cell (Minecraft's `random_spread`), with spacing and separation | a region cell | G1, G3, G7 |
| **Scatter** | a generator and a chain of modifiers producing a PointSet (below) | its largest spacing or footprint | all G, N3, N5 |
| **Network** | bounded paths between owned sites (roads, rivers, tunnels) | declared | G7, G8 |
| **Assemble** | a jigsaw or room graph grown from one site into Stamps, with a bounded extent | the extent | G1, G7, G8 |
| **Apply** | rasterises curves and stamps into fields or Priors in a stable order | the primitives' bounds | G1, G3, G8 |
| **Region job** | any bounded pure computation over a region, with retries | the region | G2, G4, G5, G6, G7, G8 |
| **Record** | computed once per seed or per address | its parent | G2, G3, G6 |
| **Emit** | products for an engine | 0 | N1, N3, P1 |
| **Edits**, **Import** | sources: the edits log, painted images, imported heightmaps | 0 | N4, N8, G4 |

**Placement rules are Scatter stages.** A Scatter stage has:

- generators: tile or face anchors from a TileGrid, a jittered grid hashed per cell, points from
  sites or along curves;
- modifiers, in order: chance, height, slope, mask, levels, jitter and turn, all pointwise, and
  spacing, whose reach is its radius. Spacing reads its neighbours' *candidates*, never their
  results, and breaks ties by priority, then hash, then id, so it agrees across seams;
- overlap with other Scatter stages resolved by priority;
- an `Emit` that binds each point's `kind` to an engine asset: a `PackedScene` in Godot, a scene
  handle in Bevy.

A placement's id is `(chunk coordinate, local)`, where `local` packs the stage, the cell and a slot,
so it is positional and never an ordinal. It replaces the 32-bit chunk hash in today's instance ids,
which two chunks can share. (N3, N5, G7.)

**The Godot side authors packs through GDScript resources** that hand their engine-neutral fields to
the library's serde types and save RON; the Bevy side uses the same types with `Reflect` derived
behind an optional feature. The decision and its reasons are in
[#41](https://github.com/AntonTegnelov/wave_forge/issues/41).

## 5. Where the current generator does not fit yet

**Repairs.** When a chunk cannot be solved, a repair re-solves it together with its neighbours and
may rewrite neighbours that were already solved ([architecture.md §6.3](architecture.md)). That is
a same-level mutation: tiles can depend on the order chunks were generated in, which breaks the
purity rule. About one chunk in nine is repaired in the city ([status.md](status.md)). Either a
repair becomes a pure third level of the WFC stage, whose input is the first two levels and whose
output nothing earlier reads, or a rule set is labelled "not streaming-clean" and its worlds are
only reproducible in the same generation order. The first is prototyped and measured before any
other stage depends on tiles.

**Variable-length GPU output.** Atomic appends make the order of emitted points depend on thread
timing. Points are compacted by prefix sums in a fixed order and checked against a CPU reference.

**Edits invalidation.** An edit dirties the keys it touches and every dependant within the summed
reach. Neither LayerProcGen nor PCG has this, so it is new work.

## 6. Four tiers over one pack

A **pack** (`*.world.ron`) is a set of named stages that reference each other by id, the model
Minecraft's datapacks use. It is validated at load: acyclic, every reach declared, no inherently
global node in an infinite world (normalise, auto-level, whole-map erosion are refused with the
node's name; finite imported heightmaps are how their results come in), and a schema version with
migrations. The same pack runs in Godot and in Bevy.

- **Tier 0, presets.** A pack with three to six parameters exposed in the inspector. Brushes and
  splines write Edits. (N1, N2, N10.)
- **Tier 1, stack.** An ordered list of stages. Each row shows its kind, cell size, reach and inputs,
  "the row above" by default; rows hold inline modifier chains, mask stacks and rule lists, which is
  where MapMagic's layered nodes and ProtonScatter's modifier stack succeed
  ([ProtonScatter modifiers](https://github.com/HungryProton/scatter/wiki/Modifiers)). The stack
  compiles to the graph; it is a view of it, not a second engine. (N2, N3, N4, N7.)
- **Tier 2, graph.** The pack text is the graph. A read-only DAG view ships first; a graph editor
  follows only if the stack proves too limiting, because an editor costs twice (Bevy has no graph
  widget). (G1 to G8.)
- **Tier 3, code.** A Rust `Stage` trait and typed WGSL kernel kinds (field, point generator, point
  processor), for logic that is code in every studied game: Minecraft's feature types, Qud's zone
  builders, Deep Rock Galactic's cave graphs. (G2, G5, G8.)

**Debugging is part of the product.** Per-stage viewers (a field heatmap with a value probe, a
palette for categorical fields, mask overlays, point glyphs that show which modifier rejected each
candidate, curve and graph overlays), a reach pyramid, the WFC stepper with a contradiction heatmap,
per-stage timings, a rejection log for region jobs, `locate(kind)`, a contact sheet over many seeds,
and an **order-diff test** that generates the same world in shuffled chunk orders and compares
product hashes. (N5, P6, and the verification gate in [user-stories.md](user-stories.md#the-verification-gate).)

## 7. Scope

"Build any game's generation" means covering the techniques those games use, not reproducing any of
them bit for bit; Valheim's floating-point quirks and Noita's float random numbers make exact
reproduction impossible anyway.

- **Out of scope:** civilisation or history simulation (a Record slot the user fills), runtime
  simulation (falling sand, destruction, fluids, lighting), which belongs to the engine, blending
  terrain after a generator change, and Dwarf Fortress's whole-world rejection on infinite worlds.
- **Deferred, with types that allow them later:** spheres, galaxies and 64-bit floats (wgpu has no
  portable `f64`).
- **WFC stays the distinctive stage** but not the centre: fields and scatter are where most of the
  studied games spend their generation, so they are first-class.
- A claim that a preset is "like" a game is made only after that game's story in
  [user-stories.md](user-stories.md) is verified.

## 8. Order of work

1. Stable placement ids, `(chunk, local)`.
2. The stage runtime on the CPU: pack loading and validation, bounded views, the provider-first
   scheduler and the order-diff test, wrapping the existing WFC as a Solve stage.
3. The repair-purity prototype and its measurement (§5).
4. The first slice, a Valheim-like surface world with WFC inside masks: Field (height), Sites, Apply
   (flatten), Rules (to a Prior), Solve, Scatter and Emit, verified by the order-diff test before the
   Field stage moves to the GPU.
5. Godot authoring resources and the viewers; the Bevy loader.
6. Later slices, one stage kind each: region jobs (rivers, Qud zones, cave levels), Assemble (jigsaw,
   room graphs), density volumes (Minecraft and Deep Rock caves), Records and hierarchy (Elite, No
   Man's Sky).
