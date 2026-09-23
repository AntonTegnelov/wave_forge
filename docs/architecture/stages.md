# Generation as a pack of stages

How Wave Forge combines many kinds of generation (fields, sites, scatter, WFC, region-scale passes)
so a developer can build the kind of world the games in
[user-stories.md](../product/user-stories.md) generate, and a newcomer can still start in minutes.
Design choices name the stories they serve (G for a game's generation, N for newcomers, P for
performance).

This page is the design, and some of it is not built yet. A statement that describes the target
rather than the code is marked **not built yet**, with its issue. What is built is in
[reference/packs.md](../reference/packs.md), and the order of the remaining work is in
[roadmap.md](../plan/roadmap.md). The research the design rests on (engines, libraries and eight
games) is in [worldgen-survey.md](../research/worldgen-survey.md); its central finding is that every
generation system that works is a directed acyclic graph of pure transforms over typed data, and
that WFC cannot be the centre of such a design, because most of the studied games spend their
generation on fields and scatter and need whole-region passes.

The unit is called a **stage**, not a layer, because `Prior` already uses "layer" for its per-height
masks ([solver.md](solver.md#the-prior)).

## The execution contract

- **Purity.** A stage's output for a key is a pure function of the world seed, the stage's id, the
  key, and read-only outputs of its inputs within a declared reach. Where parameters imply the reach
  (a blur radius, a spacing, a footprint), it is derived from them, and a pack never states one by
  hand. (G1 to G8, N9.) Reach is in cells or whole chunks on a 2D lattice today; reach in world
  units and in 3D is **not built yet** ([#93](https://github.com/AntonTegnelov/wave_forge/issues/93),
  [#71](https://github.com/AntonTegnelov/wave_forge/issues/71)).
- **Bounded reads.** A stage reads its inputs through a view bounded by its declared reach. A read
  outside it is an error naming the stage and the reach it would need. The transitive reach, the
  area each stage has to be generated over for a request, is computed when the pack loads. (N5, P2.)
- **Gather, never scatter.** A stage writes only its own key. Anything that straddles a border, such
  as a building or a spacing decision, has exactly one owner, chosen by a pure function of its
  anchor; neighbours recompute the same anchor instead of receiving writes. A consumer takes its
  inputs in one of two ways: *owned* (emitted once, for things with stable ids, such as a site in
  the chunks it covers) or *overlapping* (applied everywhere it touches, such as a flatten
  footprint).
- **No cycles, checked at load.** Terrain that structures adapt is split the way every studied game
  splits it: a base field, then sites that read only the base field, then an adapted field
  (Minecraft's beardifier, No Man's Sky's flatten points, Valheim's location levelling). The valley
  pack does exactly this with Sites and Flatten. (G1, G3, G7.)
- **Named hash streams.** Every random decision is a hash of the seed, the stage's id, the key and a
  purpose, built on the stateless `pcg3d` the solver uses, never a sequential generator, so adding,
  removing or reordering stages changes no other stage. Existence tests compare integers. (N9, G6.)
  A noise node can name its own stream, which is then the same function in every stage that names
  it, as Minecraft's named noise parameters are.
- **Integer data across stages.** Data crossing a stage boundary that decides existence or position
  (site positions, spline control points) should be integer or fixed point, because equal floats are
  not guaranteed across GPU vendors. Today fields, site heights and point positions are `f32`, which
  is safe while the stages run on the CPU; fixed-point boundaries are **not built yet** and come
  with GPU stages.
- **Levels and scales.** Each stage declares a cell size on a hierarchy of global, region, chunk and
  cell, and may read its parent level (Elite's sector, system and body; No Man's Sky's region,
  system and planet). A **region job** runs on a coarse region lattice: any bounded pure
  computation, which may iterate, count and retry with a hashed retry index. A finite world is a
  single region computed before streaming starts. Region jobs agree at their borders through
  edge-keyed hashes (a river's crossing point hashed from the shared edge), not by reading each
  other. (G2, G4, G5, G6, G7, G8.) Region jobs exist as Region stages, whose job is Rust code a
  game gives the runtime; every stage still runs on the WFC chunk lattice, and levels and records
  are **not built yet** ([#93](https://github.com/AntonTegnelov/wave_forge/issues/93),
  [#72](https://github.com/AntonTegnelov/wave_forge/issues/72)).
- **Scheduling.** Providers first, with lifetimes held by what needs them, following LayerProcGen.
  The WFC generator's schedule is the same pattern written by hand: parity 0, parity 1 and the
  repair classes are levels of one stage, and the closure rule is provider-first generation
  ([world.md](world.md#the-schedule)). (P1, P3.) Batching one stage and level per GPU dispatch comes
  with GPU stages.
- **Persistence per stage.** A *pure* stage regenerates and replays edits; a *freeze on first emit*
  stage is snapshotted (as Minecraft, Noita, Qud and Valheim do with placed objects); an *ephemeral*
  stage (Valheim's clutter) is never saved. Saves record the generator version. (G7, N8, P4.) Every
  stage is pure today; edits and the other modes are **not built yet**
  ([#101](https://github.com/AntonTegnelov/wave_forge/issues/101),
  [#102](https://github.com/AntonTegnelov/wave_forge/issues/102)).

## Data between stages

| Type | Contents | Covers | Today |
|---|---|---|---|
| **Field** | named channels on a 2D or 3D grid at the stage's cell size | height, climate, masks, density and signed distance, categorical ids such as a biome | one `f32` per cell column, or a category per column from a Rules stage |
| **PointSet** | structure of arrays: position, rotation, scale, stable id, kind, attribute columns | sites, anchors, scatter candidates and placements, spawn points | `Sites` and `Points` |
| **CurveSet** | polylines with per-vertex attributes (radius, flow, profile) and optional connectivity | roads, rivers, tunnels, room and site graphs | `Curves` from region jobs: points and one value per point; connectivity and rasterising are [#98](https://github.com/AntonTegnelov/wave_forge/issues/98) |
| **Stamps** | an ordered list of carve, fill and prefab primitives, each with bounds | jigsaw pieces, cave rooms, flatten areas | not built yet ([#70](https://github.com/AntonTegnelov/wave_forge/issues/70)) |
| **Table** | named rows of facts, each with an id, a position or a curve, and typed columns; given by the game or generated from a parent table | a history the game simulated, planet or system parameters, a location table | not built yet ([#72](https://github.com/AntonTegnelov/wave_forge/issues/72)) |
| **Prior** and **TileGrid** | the WFC stage's input and output | tiles | `Tiles`, a town's chunk |
| **Edits** | an operation log keyed by stable ids and cells | brushes, removed and moved placements, terrain deltas | not built yet ([#101](https://github.com/AntonTegnelov/wave_forge/issues/101)) |

PointSet is a structure of arrays from the start: Unreal PCG moved from an array of structs in 5.6
at the cost of a breaking change. The products an engine consumes (instance sets, colliders,
navigation source) are not another type between stages: they are derived from PointSets and
TileGrids for an engine ([engine-integration.md](engine-integration.md#products)).

## Stage kinds

| Kind | What it does | Reach | Serves | Today |
|---|---|---|---|---|
| **Field** | a fused pointwise expression graph: noise, splines, remap, math, first-match classifiers, image lookup | 0 | G1, G3, G6, G7, N2, N7 | value noise with named streams, coordinates and distances, arithmetic, clamps, smoothsteps, remaps, curves, selects, and a blended match per category; FastNoiseLite's noises are [#45](https://github.com/AntonTegnelov/wave_forge/issues/45) |
| **Filter** | stencils, blur, cellular automata, slope, distance transforms | declared per pass | G2, G5, G7 | `Blur`, `Flatten` ([#94](https://github.com/AntonTegnelov/wave_forge/issues/94)) |
| **Rules** | first-match rule trees producing a Prior or a categorical field, like Minecraft's surface rules | that of its conditions | G1, G7, N4 | a categorical field; a Prior is not built yet |
| **Solve** | WFC over a Prior; Wang tiling later | the solver's halo | G4, G5, N6 | one bounded town per site (below) |
| **Sites** | owned region-scale points, one candidate per region cell (Minecraft's `random_spread`), with spacing | a region | G1, G3, G7 | one footprint per region ([#97](https://github.com/AntonTegnelov/wave_forge/issues/97) adds a location table) |
| **Scatter** | a generator and a chain of modifiers producing a PointSet (below) | its largest spacing or footprint | all G, N3, N5 | one kind per stage, four tests ([#95](https://github.com/AntonTegnelov/wave_forge/issues/95), [#96](https://github.com/AntonTegnelov/wave_forge/issues/96)) |
| **Network** | bounded paths between owned sites (roads, rivers, tunnels) | declared | G7, G8 | not built yet ([#98](https://github.com/AntonTegnelov/wave_forge/issues/98)) |
| **Assemble** | a jigsaw or room graph grown from one site into Stamps, with a bounded extent | the extent | G1, G7, G8 | not built yet ([#70](https://github.com/AntonTegnelov/wave_forge/issues/70)) |
| **Apply** | rasterises curves and stamps into fields or Priors in a stable order | the primitives' bounds | G1, G3, G8 | `Flatten` for site footprints ([#98](https://github.com/AntonTegnelov/wave_forge/issues/98)) |
| **Region job** | any bounded pure computation over a region, with retries | the region | G2, G4 to G8 | a Region stage running a `RegionJob` the game registers, producing curves |
| **Table** | rows given by the game, or generated once per parent row by expressions | its parent table | G2, G3, G6 | not built yet ([#72](https://github.com/AntonTegnelov/wave_forge/issues/72)) |
| **Emit** | products for an engine | 0 | N1, N3, P1 | done by the engine integrations today |
| **Edits**, **Import** | sources: the edits log, painted images, imported heightmaps | 0 | N4, N8, G4 | not built yet ([#101](https://github.com/AntonTegnelov/wave_forge/issues/101)) |

### Placement rules are Scatter stages

A Scatter stage has:

- **generators:** tile or face anchors from a TileGrid, a jittered grid hashed per cell, points from
  sites or along curves;
- **modifiers, in order:** chance, height, slope, mask, levels, jitter and turn, all pointwise, and
  spacing, whose reach is its radius. Spacing reads its neighbours' *candidates*, never their
  results, and breaks ties by priority, then hash, then id, so it agrees across seams;
- **overlap with other Scatter stages** resolved by priority;
- **a binding** from each point's `kind` to an engine asset: a `PackedScene` in Godot, a scene
  handle in Bevy.

Today a Scatter stage has the jittered-grid generator with chance, height, slope, a margin from
sites and spacing. The full modifier chain, blocking across stages and the asset binding are **not
built yet** ([#95](https://github.com/AntonTegnelov/wave_forge/issues/95),
[#96](https://github.com/AntonTegnelov/wave_forge/issues/96),
[#44](https://github.com/AntonTegnelov/wave_forge/issues/44)).

A placement's id is `InstanceId { chunk, local }` (`src/products.rs`): the chunk coordinate and a
64-bit local id packing the stage (15 bits), a slot (16 bits) and the cell's index (32 bits). It is
positional, never an ordinal, so inserting content never shifts another id (Elite's authored bodies
shifted procedural ids), and it fits an engine's signed 64-bit integer. The chunk is a coordinate
rather than the 32-bit chunk hash, which two chunks can share. The tile placements of a WFC world
use stage 0. (N3, N5, G7.)

## How WFC joins: one bounded world per site

A surface world's heights vary far more than a one-chunk-tall WFC lattice can span, and repairs are
order-independent only in worlds one chunk tall ([world.md](world.md#what-determinism-means-here)).
So the Solve stage solves each site as its own bounded WFC world, the size of its footprint in
chunks, seeded from the world seed and the site's region, with the module set's boundary rules at its
edges, and an engine places it at the site's levelled height. Sites are two chunks apart, so no WFC
seam ever joins two towns, and a site's town is a pure function of the site: a small region job,
the same in whatever order it is asked for.

The Solve stage reaches the solver through the `TownSolver` seam, because the stage runtime runs on
the CPU while a town needs a GPU solver on the thread that owns its device.

WFC inside masks painted or computed by other stages (a Rules stage writing a Prior) is the general
form, and is **not built yet** ([#91](https://github.com/AntonTegnelov/wave_forge/issues/91)). The
infinite city of the MVP keeps the streamed WFC world of [world.md](world.md).

## History and other facts from the game

Some of what shapes a world is not generation at all. A Dwarf Fortress-like game simulates
centuries of history over its world map, and the towns, ruins and roads it leaves must be in the
world the player walks. That simulation is game logic, iterative and stateful, and Wave Forge does
not host it. Such a world comes together in three phases, and Wave Forge owns the first and last:

1. **Terrain,** pure: a finite world at a coarse level, with its heights, biomes and rivers.
2. **History,** the game's own code, run once before play. It reads the terrain through an atlas and
   point queries ([#100](https://github.com/AntonTegnelov/wave_forge/issues/100)) and writes its
   results as facts.
3. **Realisation,** pure: stages turn the facts into what the player sees where the player goes. A
   site becomes a town of its culture's rule set, or ruins; a road is carved into the ground.

The seam is **tables of facts** ([#72](https://github.com/AntonTegnelov/wave_forge/issues/72)): named tables of rows, each with an id, a position or a
curve, and typed columns. A game gives a table at run time, from Rust or from GDScript. A history is
only data, so it can be written in any language and run anywhere, unlike a stage, which runs on the
stages' thread and must stay pure. Stages read tables as they read fields: Sites from a table, a
Solve choosing its rule set by a column, Apply carving roads from curves. The world is then a
function of the pack, the seed and the facts, and a save holds the facts, never the world.

The same tables describe generated hierarchies. A table can compute its rows from a parent table's
by expressions, a budget shared among a parent's children for instance, which is what Elite's
sectors, systems and bodies and No Man's Sky's planets need. A runtime focused on one row reads
that row's columns, so one surface pack serves every planet. Generated rows have positional ids,
and given rows keep the game's ids, so neither ever shifts the other.

A fact that changes regenerates only what depended on it, through the same invalidation as the edits
log ([#101](https://github.com/AntonTegnelov/wave_forge/issues/101)). A history is an edits log
written before play, and a town the player burns during play is one more fact.

The limits are deliberate. History needs a finite world, as Dwarf Fortress's does. Its cost and its
determinism are the game's, and Wave Forge offers its hash streams to make the second easy. The loop
runs one way, terrain to history to realisation: a history that changes terrain it later reads
keeps track of that in its own state. Whether this is approachable is decided by a worked example,
a continent preset with a toy history of about a hundred lines of GDScript, which story N11 makes a
criterion.

## What stays hard

- **Variable-length GPU output.** Atomic appends make the order of emitted points depend on thread
  timing, so points on the GPU must be compacted by prefix sums in a fixed order and checked against
  the CPU reference, once a stage moves to the GPU.
- **Edits invalidation.** An edit dirties the keys it touches and every dependant within the summed
  reach. Neither LayerProcGen nor Unreal PCG has this, so it is new work
  ([#101](https://github.com/AntonTegnelov/wave_forge/issues/101)).
- **Inherently global nodes.** Normalising, auto-levelling and whole-map erosion cannot run on an
  infinite world. A pack will refuse them there, naming the node, and finite imported heightmaps
  are how such results come in. Today no stage kind is global, so there is nothing to refuse yet.

## Four tiers over one pack

A **pack** (`*.world.ron`) is a set of named stages that reference each other by name, the model
Minecraft's datapacks use. It is validated at load and runs the same in Godot and in Bevy. Migrations
between pack versions are **not built yet**; a pack of another version is refused.

- **Tier 0, presets.** A pack with three to six parameters exposed in the inspector. Brushes and
  splines write Edits. (N1, N2, N10.)
- **Tier 1, stack.** An ordered list of stages. Each row shows its kind, cell size, reach and inputs,
  "the row above" by default; rows hold inline modifier chains, mask stacks and rule lists, which is
  where MapMagic's layered nodes and ProtonScatter's modifier stack succeed. The stack is a view of
  the graph, not a second engine. (N2, N3, N4, N7.)
- **Tier 2, graph.** The pack text is the graph. A read-only DAG view ships first; a graph editor
  follows only if the stack proves too limiting, because an editor costs twice (Bevy has no graph
  widget). (G1 to G8.)
- **Tier 3, code.** A Rust `Stage` trait and typed WGSL kernel kinds (field, point generator, point
  processor), for logic that is code in every studied game: Minecraft's feature types, Qud's zone
  builders, Deep Rock Galactic's cave graphs. (G2, G5, G8.)

Only the pack text exists today. Presets, the stack and the viewers are
[#48](https://github.com/AntonTegnelov/wave_forge/issues/48); the Godot side authors packs through
GDScript resources that hand their engine-neutral fields to the library's serde types and save RON,
and the Bevy side uses the same types with `Reflect` behind an optional feature
([#41](https://github.com/AntonTegnelov/wave_forge/issues/41) records the decision).

**Debugging is part of the product.** Per-stage viewers (a field heatmap with a value probe, a
palette for categorical fields, mask overlays, point glyphs that show which modifier rejected each
candidate, curve and graph overlays), a reach pyramid, the WFC stepper with a contradiction heatmap,
per-stage timings, a rejection log for region jobs, `locate(kind)`, a contact sheet over many seeds,
and an **order-diff test** that generates the same world in different chunk orders and compares the
results. (N5, P6, and the [verification gate](../product/user-stories.md#the-verification-gate).)
The order-diff tests exist (`tests/stages.rs`, `wfc-devtools/tests/order_diff.rs`); per-stage
timings are in the runtime and both engines, and the viewers come with authoring.

## Scope

"Build any game's generation" means covering the techniques those games use, not reproducing any of
them bit for bit; Valheim's floating-point quirks and Noita's float random numbers make exact
reproduction impossible anyway. The project's scope and non-goals are in
[vision.md](../product/vision.md#non-goals); for stages specifically:

- **Out of scope:** civilisation or history simulation (the game's, given as tables of facts), runtime
  simulation (falling sand, destruction, fluids, lighting), which belongs to the engine, blending
  terrain after a generator change, and Dwarf Fortress's whole-world rejection on infinite worlds.
- **Deferred, with types that allow them later:** spheres, galaxies and 64-bit floats (wgpu has no
  portable `f64`).
- **WFC stays the distinctive stage** but not the centre: fields and scatter are where most of the
  studied games spend their generation, so they are first-class.
- A claim that a preset is "like" a game is made only after that game's story is verified.
- **The stage runtime runs on the CPU today.** That is a performance tier, not a fallback: the
  project still requires a GPU ([vision.md](../product/vision.md#non-goals)), towns are solved on
  one, and a Field stage moves to the GPU when per-stage timings show it should. They do not yet:
  a field costs about 0.02 ms per chunk, while a town's first solve takes seconds ([measurements.md](../research/measurements.md) E29).
