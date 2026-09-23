# User stories

What people want to do with Wave Forge, written so a check can decide whether they can. The stories
guide the design ([stages.md](../architecture/stages.md)), which guides the architecture, which
guides the implementation: a design choice names the stories it serves, and a story that no design
serves is a gap.

There are three groups. **G** stories replicate the world generation of a well-known game, **N**
stories are newcomers and beginners who want something simple but their own, or something impressive
without effort, and **P** stories need performance.

## The verification gate

**Extensive verification of every story is a mandatory gate.** Wave Forge is not done, and nothing
of it is published, until every story below is shown to be possible by repeated, recorded
verification:

- an automated check wherever the story allows one (a Godot `verify.gd` check, the order-diff test,
  a benchmark with a bar), run in CI where the hardware allows and on the reference desktops where
  it does not;
- a written walkthrough with evidence wherever the story needs a person (screenshots, a recording,
  numbers with the build, driver and hardware they were measured on), repeated after changes that
  touch the story.

A story counts as **verified** only when every acceptance criterion passes on the reference stacks
and the evidence is linked from the story. The evidence comes from this repository's test packs,
presets and examples, so it can be linked and rerun by anyone; the proof-of-concept games in their
private repositories are downstream smoke tests that find problems, not evidence
([roadmap.md](../plan/roadmap.md#games-packs-and-repositories)). Each story carries a status (not started, in progress,
verified), the date it last changed and a link to its evidence. A story that regresses goes back to
in progress.

**Publishing is a human-only task,** and so is anything like it: releasing to the Godot Asset Store
or crates.io, creating a release or a release tag, announcing, and promoting `develop` to `main`. An
agent prepares all of it and never does any of it
([contributing.md](../guides/contributing.md#publishing-is-human-only)).

Where each story stands against the code, with the issues that close its gaps, is in
[story-coverage.md](../plan/story-coverage.md).

## How a story is written

- **Who, what and why**, in one or two sentences.
- **Context**: for a G story, how the game generates its world, with sources; for the others, what
  the person already knows and expects.
- **Acceptance criteria** a check can decide, and the one that needs a person, if any.
- **Needs**: the stages, tools and measurements it implies, linked to
  [stages.md](../architecture/stages.md).
- **Status**.

A G story asks for the techniques a game uses, not a bit-for-bit copy of its worlds
([stages.md](../architecture/stages.md#scope)).

## G: replicate a game's generation

### G1. A Minecraft-like voxel world

*A developer making a block survival game wants terrain with continents, mountains, caves, rivers,
villages and ores, and wants to tune it the way datapack authors tune Minecraft.*

**Context.** Minecraft's world is infinite horizontally and 384 blocks tall. Six climate noises
(temperature, humidity, continentalness, erosion, weirdness, depth) are sampled per 4×4×4 cell and a
multi-noise biome source picks the nearest biome in that six-dimensional space. Terrain is a DAG of
density functions defined in JSON; caves are more density terms (cheese, spaghetti, noodle),
aquifers decide fluid per jittered cell, surface rules are a first-match decision tree, carvers
replay worm tunnels from neighbouring chunks, and features are placed by a chain of placement
modifiers in eleven ordered steps. Structures start on a jittered grid (`random_spread`) and jigsaw
structures grow from template pools with a bounded extent. Chunks advance through ordered statuses
with bounded neighbour requirements. Everything is exposed as datapack JSON
([World generation](https://minecraft.wiki/w/World_generation), [Noise router](https://minecraft.wiki/w/Noise_router),
[Density function](https://minecraft.wiki/w/Density_function), [Jigsaw structure](https://minecraft.wiki/w/Jigsaw_structure),
[Placed feature](https://minecraft.wiki/w/Placed_feature)). Features writing into neighbours make a
world depend on the travel route, a flaw Wave Forge must not copy.

**Acceptance criteria.**
- A test pack in this repository generates an unbounded voxel world with climate-driven biomes, 3D density terrain
  with overhangs and caves, aquifers, surface materials by rule, ores, trees and a jigsaw village.
- Changing one number in the pack (a noise amplitude, a feature's rarity) changes only what depends
  on it, shown by the order-diff test and a diff of product hashes.
- Two travel orders give the same world (order-diff test).
- Needs a person: a Minecraft player recognises the kinds of landforms in a walkthrough.

**Needs.** Field (climate, density with splines), Rules (biomes by nearest point, surface rules),
3D density volumes, Sites and Assemble (jigsaw), Scatter chains, Apply, the order-diff test.
**Status:** not started (2026-09-23).

### G2. A Dwarf Fortress-like finite world with history

*A developer making a colony simulation wants a finite world of mountains, rivers, biomes and
geology, with sites and a history their own simulation fills in.*

**Context.** Dwarf Fortress generates a finite world up front on a grid of 17 to 257 world tiles per
side. Base fields (elevation, rainfall, temperature, drainage, volcanism, savagery) are seeded on a
mesh and filled fractally; peaks are placed and altitudes fixed up, with whole-world rejection when
parameters are not met; test rivers erode mountainsides; rain shadows are swept; biomes come from
threshold rules and are labelled into connected regions; geology is layered underground; then
civilisations are placed and a history simulation of thousands of agents runs. When the player
embarks, a local 3D map is generated from the world tile's fields. Players tune all of it through
named parameter sets ([World generation](https://dwarffortresswiki.org/index.php/World_generation),
[Advanced world generation](https://dwarffortresswiki.org/index.php/Advanced_world_generation),
[Simulation principles, Game AI Pro 2](https://www.gameaipro.com/GameAIPro2/GameAIPro2_Chapter41_Simulation_Principles_from_Dwarf_Fortress.pdf)).

**Acceptance criteria.**
- A test pack in this repository generates a finite world as one region job on a coarse grid: fields, eroded rivers
  as curves, biomes and connected regions, geology strata, with a rejection log when a retry is
  needed.
- The game's own history, simulated before play over the world's atlas, is given as tables of facts
  (sites with their culture, size and fate; roads as curves), and the local stages realise them:
  towns of each culture's rule set, ruins where a site fell, roads carved into the ground.
- A local area generated from a world tile matches its coarse fields at the edges.
- The same seed and parameters give the same world on every run and GPU (a golden world).

**Needs.** Region jobs with hashed retries, Field and Filter at coarse scale, CurveSet, an atlas and
point queries, tables of facts ([#72](https://github.com/AntonTegnelov/wave_forge/issues/72)), a local level reading its parent. History simulation is the
game's; tables of facts are the seam, and N11 is the check that the seam is approachable.
**Status:** not started (2026-09-23). The history seam this story's second criterion needs is built
and checked by N11's example (`examples/history`): sites with their culture and fate, roads as
curves, towns of each fate's rule set and ruins, and roads levelled into the ground.

### G3. A No Man's Sky-like planet

*A developer making an exploration game wants planets whose parameters come from a seed, with
deformable 3D terrain, buildings that sit on flattened ground, and flora and creatures per biome.*

**Context.** No Man's Sky addresses a universe with 64-bit integers. A positional seed yields the
solar system and each planet's parameters, chosen from data-driven presets. Terrain is continuous
3D density from a stack of "uber noise" layers with analytic derivatives, over a column pass that
blocks out mountains, plains and water, on a cube-sphere. Buildings come from an offset grid across
the planet, so the nearest one can be queried from anywhere, and they flatten the terrain around
them. Props are scattered per detail level, creatures by spawn rules per biome, and player edits are
kept as a capped log in the save
([Continuous World Generation in No Man's Sky, GDC 2017](https://www.gdcvault.com/play/1024265/Continuous_World_Generation_in__No_Man_s_Sky_),
[Building Worlds Using Math(s), GDC 2017](https://www.gdcvault.com/play/1024514/Building-Worlds-Using)).

**Acceptance criteria.**
- A test pack in this repository generates a planet section from a planet's row in a generated table: 3D density with overhangs, sites
  on an offset grid that flatten the ground under them, and props scattered by biome.
- `locate(kind)` finds the nearest site from any position without generating the chunks between.
- Terrain edits are kept in the edits log and survive leaving and returning.

**Needs.** Generated tables, Field with derivatives, density volumes, Sites with point queries, Apply
(flatten), Scatter, Edits. The spherical domain and the universe scale are deferred
([stages.md](../architecture/stages.md#scope)). **Status:** not started (2026-09-23).

### G4. A Noita-like cave world from painted tiles

*A developer making a 2D action game wants hand-painted tiles recombined into caves that are always
passable, inside biomes laid out on a painted map, with hand-made set pieces.*

**Context.** Noita's world is a 2D grid of simulated pixels. A hand-painted biome map assigns a
biome to each 512×512 chunk, with noisy edges. Each biome region is filled with Sean Barrett's
herringbone Wang tiles, painted as images whose colours stand for materials and spawn markers; a
path search checks the region is passable and rerolls it if not; "maybe" pixels resolve by coin
flip. Hand-painted pixel scenes are stamped in, and marker colours call spawn functions
([Exploring the Tech and Design of Noita, GDC 2019](https://www.gdcvault.com/play/1025695/Exploring-the-Tech-and-Design),
[herringbone Wang tiles](https://nothings.org/gamedev/herringbone/herringbone_tiles.html),
[Noita wiki, world generation](https://noita.wiki.gg/wiki/World_generation)). The region-by-region
order and the reroll are known from community reverse engineering (unverified from Nolla).

**Acceptance criteria.**
- A test pack in this repository takes a painted biome map and painted tiles, and generates a 2D world in which every
  biome region is passable from its entry to its exit, checked by a path search in the test.
- Marker colours in tiles become spawn points with stable ids; pixel scenes are stamped at their
  places.
- The simulation of materials stays the engine's; the world is handed over as data.

**Needs.** Import (images), Region jobs with retries, Solve with Wang tiles, Apply (stamps), Scatter
from markers. **Status:** not started (2026-09-23).

### G5. A Caves of Qud-like world of zones

*A developer making a roguelike wants a hand-made world map whose zones are generated on first visit
from layered builders, with WFC ruins, guaranteed connectivity and populations from tables.*

**Context.** Caves of Qud's world map is a fixed 80×25 grid of parasangs, each a 3×3 grid of zones
of 80×25 cells. Sultan histories come from a grammar and a state machine; a world builder places
villages and historic sites on the map. A zone is built on first visit by a chain of builders:
coarse layout, overlapping-model WFC inside segmented regions with their own templates, a pass that
connects every open area, then populations from weighted XML tables and hand-made templates
([End-to-End Procedural Generation in Caves of Qud, GDC 2019](https://media.gdcvault.com/gdc2019/presentations/Grinblat_Jason_End-to-End_Procedural_Generation.pdf),
[Zones and worlds](https://wiki.cavesofqud.com/wiki/Modding:Intro_-_Zones_and_Worlds)).

**Acceptance criteria.**
- A test pack in this repository generates zones from an authored map: segments, WFC per segment with its own
  template, every open cell reachable (checked), populations from weighted tables.
- Stairs and paths between neighbouring zones line up, from edge-keyed hashes.
- A zone visited, left and revisited is identical, or restored from its snapshot when set to freeze
  on first emit.

**Needs.** Import (maps and templates), Region jobs per zone, Filter (segmentation, connectivity),
Solve inside masks, Scatter from tables, per-stage persistence. **Status:** not started (2026-09-23).

### G6. An Elite Dangerous-like galaxy of systems and planets

*A developer making a space game wants star systems whose bodies follow from a seed and a few
authored real systems, and planets whose surfaces follow from each body's parameters.*

**Context.** Elite's Stellar Forge models the Milky Way with about 400 billion systems in an octree
of sectors, each layer handling stars of one mass range from a budget handed down by its parent.
Authored systems from real catalogues are injected. Each system forms by an accretion simulation,
and each body's parameters drive its surface: GPU-generated patches on a cube-sphere, coarse-to-fine
zones since Odyssey, scatter through noise graphs, and hand-built settlements placed on top
([Generating the Universe in Elite: Dangerous](https://80.lv/articles/generating-the-universe-in-elite-dangerous),
[How Frontier rebuilt the planets for Odyssey](https://www.pcgamer.com/heres-how-frontier-rebuilt-a-galaxys-worth-of-planets-for-elite-dangerous-odyssey/)).
Inserting authored bodies shifted procedural ids, which is why ids here are positional.

**Acceptance criteria.**
- A test pack in this repository produces a hierarchy of generated tables (sector, system, body) where each level reads only
  its parent and its budget, with authored rows that do not shift any procedural id.
- A body's row drives a surface section: fields, zones, scatter with spacing, and an authored
  site placed on flattened ground.
- Two runs, and two travel orders, give the same systems and surfaces.

**Needs.** Generated tables with parent-level reads, Field, Scatter, Sites and Apply, positional ids. Galaxy
scale, spheres and `f64` are deferred. **Status:** not started (2026-09-23).

### G7. A Valheim-like island world

*A developer making a survival game wants a world whose biomes get harder away from the centre,
with rivers, points of interest placed by a table with quotas, vegetation by rules, dungeons made of
prefab rooms, and terrain the player can dig and raise.*

**Context.** Valheim's world is a flat disk of radius 10 km. A base height from Perlin octaves falls
off towards the edge; biomes are chosen per point by an ordered first-match list over distance from
the centre and noise; lakes and rivers come from a global pass; each biome has its own height
function. A location table places dungeons, altars and traders by priority with quotas and minimum
distances, levelling the ground under them. Vegetation is placed per zone by rules, dungeons are
room graphs of prefabs, clutter is never saved, and player terrain edits are stored per zone
([Valheim interview](https://www.gaisciochmagazine.com/articles/valheim_interview.html),
[vegvisr, a community port](https://github.com/aritropaul/vegvisr); the algorithms come from the port
and are unverified from Iron Gate).

**Acceptance criteria.**
- A test pack in this repository generates a disk world with biomes by distance and noise, per-biome height, rivers,
  a location table honouring quotas and spacing, vegetation rules and a prefab-room dungeon.
- Placements have stable ids; removing one or editing terrain persists across eviction and
  regeneration.
- Clutter is regenerated, never saved.

**Needs.** Field, Rules, Region jobs (rivers, the location table), Network, Sites, Apply, Scatter,
Assemble, Edits, per-stage persistence. The gap, feature by feature, is in
[story-coverage.md](../plan/story-coverage.md#g7-a-valheim-like-island-world) and tracked by
[#104](https://github.com/AntonTegnelov/wave_forge/issues/104). **Status:** in progress (2026-09-23). The ring world (`examples/rings.world.ron`) is a bounded island with biomes by distance and noise, per-biome height, rivers carved into its ground, a location table of shrines per ring with quotas and spacing, and ore and birch groves placed by rules, each the same in any order; placements have stable ids, and felling one or editing terrain persists across eviction and regeneration; its shrines are frozen in a save and its grass is clutter whose edits a save never holds (`tests/rules.rs`, `tests/blend.rs`, `tests/rivers.rs`, `tests/locations.rs`, `tests/scatter_chain.rs`, `tests/edits.rs`, `tests/save.rs`, `verify_edits.gd`). The valley pack adds towns on levelled sites, served to a game by a Godot node and a Bevy plugin with ground a walker crosses ([testing.md](../guides/testing.md)). A prefab-room dungeon ([#70](https://github.com/AntonTegnelov/wave_forge/issues/70)) is to come.

### G8. A Deep Rock Galactic-like cave level

*A developer making a co-op mining game wants a cave level built from authored room shapes, joined
by tunnels, carved into destructible terrain, with resources and enemies placed by budget.*

**Context.** Each mission in Deep Rock Galactic is a finite 3D cave generated from scratch. A mission
template chooses a cave graph pattern (linear, star or hub) of rooms; each room is expanded from an
authored room generator into carving shapes; a noise-costed pathfinder finds tunnels between room
entrances; everything is carved as CSG into solid rock; resources and encounters are placed by
counts and difficulty budgets per room, and the terrain stays destructible during play
([Procedural level generation, Ghost Ship blog](https://web.archive.org/web/2017/http://www.ghostship.dk/blog/procedural-level-generation);
names from community header dumps, unverified).

**Acceptance criteria.**
- A test pack in this repository generates a cave level as one region job: a room graph from authored patterns,
  rooms expanded into stamps, tunnels as curves, carved into a density volume.
- Resource totals meet the level's quota exactly; enemy budgets hold per room.
- Carving at runtime goes through the edits log and survives reload.

**Needs.** Region jobs, Assemble, Network, Apply into density volumes, Scatter with budgets, Edits.
**Status:** not started (2026-09-23).

## N: newcomers and beginners

These stories are about staying intuitive: defaults that look good, few words to learn, and a tool
that explains itself. Where a story says "without code", adding code must not be the fix.

### N1. Press play and walk a world

*Someone new to Godot drags the Wave Forge node into a scene, presses play and walks a good-looking
world, without code.*

**Acceptance criteria.** A new project with the node and its default preset shows a lit, textured,
walkable world with colliders and navigation on the first play; no error or warning in the output;
the node's inspector shows three to six parameters. Needs a person: five people new to the plugin
reach a walkable world in under five minutes, timed. **Needs.** Tier 0 presets, Emit, the defaults.
**Status:** not started (2026-09-23).

### N2. Pick a preset and tweak it with a live preview

*A hobbyist picks "islands" from a preset list and moves three sliders (land amount, roughness, tree
density) while the editor preview updates.*

**Acceptance criteria.** Each preset's exposed parameters have ranges that never produce a broken
world (swept by a seed and parameter contact sheet); the preview updates within the P6 target;
undo works for every change. **Needs.** Tier 0, preview in the editor, P6. **Status:** not started (2026-09-23).

### N3. Place my own scene by dragging it in

*A beginner drags their tree scene onto a "grass" rule and sets density and spacing; trees appear on
grass, never floating, never overlapping.*

**Acceptance criteria.** Dropping a `PackedScene` onto a rule creates a Scatter stage bound to it;
Auto mode picks MultiMesh for a plain mesh scene and nodes for a scene with scripts or bodies; no
two trees are closer than the spacing, across chunk seams too (checked); every tree stands on its
anchor's surface (checked). **Needs.** Scatter, Emit, the Godot rule resources. **Status:** not
started (2026-09-23).

### N4. Paint where the town goes

*A designer paints a mask in the editor where a town should be, and the generator builds the town
there and nowhere else.*

**Acceptance criteria.** A painted mask becomes Edits that a Rules stage reads; the town's WFC
stage only fills the masked area; repainting regenerates only the affected chunks (checked by
product hashes). **Needs.** Edits, Rules, Solve inside masks, invalidation. **Status:** not started (2026-09-23).

### N5. Understand why nothing spawned

*A beginner's rule places nothing; the viewer shows the candidates and which modifier removed each
one.*

**Acceptance criteria.** Selecting a Scatter stage shows its candidate points in the viewport,
coloured by the modifier that rejected them, with a legend and counts; hovering a point shows its
attributes. **Needs.** Point viewer with rejection reasons. **Status:** not started (2026-09-23).

### N6. Turn my building kit into a city

*An artist has a modular building kit and wants a WFC city from it, without writing a rule file by
hand.*

**Acceptance criteria.** A guided import reads a `MeshLibrary` or a folder of scenes, proposes
connectors from matching face shapes, lets the artist confirm or rename them, and writes a module
set that generates a city with no unplaced chunk in the test world. **Needs.** Import, module-set
authoring tools, Solve. **Status:** not started (2026-09-23).

### N7. Use my own noise

*A developer who already tuned a `FastNoiseLite` resource wants that exact noise as the terrain.*

**Acceptance criteria.** Assigning a `FastNoiseLite` to a Field stage produces heights equal to
Godot's own sampling of it within a stated tolerance (golden test against Godot's C++ output).
**Needs.** Field, the FastNoiseLite port ([#45](https://github.com/AntonTegnelov/wave_forge/issues/45)).
**Status:** verified by automated checks (2026-09-23), with a tolerance of zero. The port gives every
one of 2 560 samples Godot 4.7 computed over 80 configurations exactly (`tests/fastnoise.rs`, in the
library's CI job), and `wave_forge_godot/godot/verify_noise.gd`, in the Godot job of every pull
request's CI, assigns a cellular, ridged, warped `FastNoiseLite` to a pack's noise and finds the
field equal to the resource's `get_noise_2d` at every column.

### N8. Bake an area to hand-edit

*A level designer generates an area, bakes it into an ordinary scene, and edits it by hand for a
story mission.*

**Acceptance criteria.** Baking writes a scene of plain nodes and resources that opens without the
plugin; a linked bake can be regenerated with the designer's edits kept. **Needs.** Bake, Edits
([#48](https://github.com/AntonTegnelov/wave_forge/issues/48)). **Status:** not started (2026-09-23).

### N9. Share a world by seed

*Two friends play the same world by sharing a seed and a pack, alone or together in multiplayer.*

**Acceptance criteria.** The same seed and pack give identical product hashes on two machines with
GPUs from different vendors; a chunk hash lets peers confirm they agree. **Needs.** Determinism
across GPUs (the golden world test), chunk hashes. **Status:** in progress (2026-09-23): the golden world is
identical on NVIDIA through dozen and on Mesa's lavapipe, and a city is the same whatever order its
chunks are generated in, repairs included ([testing.md](../guides/testing.md)), for tiles only; chunk hashes
do not exist yet.

### N10. Something impressive in an hour

*A developer preparing a trailer wants a sample world with wind in the grass, a day and night cycle,
buildings and agents walking between them, running in under an hour.*

**Acceptance criteria.** An example project in this repository, built on a preset, opens and runs
with grass and wind, lighting that changes over a day, a generated town, and navigation agents
walking its streets, at the P1 frame rate on a reference desktop. Needs a person: the trailer
checklist in the example's README is followed from a clean checkout within an hour. **Needs.** Grass
and wind ([#46](https://github.com/AntonTegnelov/wave_forge/issues/46)), navigation, presets.
**Status:** not started (2026-09-23).

### N11. Put my own history into the world

*A developer making a colony or strategy game wants the villages, roads and ruins of a history their
game simulates to be in the world the player walks, without learning a new language or writing Rust.*

**Acceptance criteria.** An example project in this repository takes a continent preset, reads its
world map through the atlas, and runs a toy history of about 100 lines of GDScript: villages founded
along rivers, grown over the years, joined to their nearest neighbours by roads, and some of them
burned. It gives the result to the stages as tables of facts, and the player walks through the
villages, the roads between them and the ruins. The same seed and history give the same world, and
a save that holds only the history restores it. Needs a person: the example's README is followed
from a clean checkout in an afternoon, and extending the toy history (a new kind of site, say) needs
no Rust. **Needs.** Tables of facts ([#72](https://github.com/AntonTegnelov/wave_forge/issues/72)), the atlas and point queries
([#100](https://github.com/AntonTegnelov/wave_forge/issues/100)), levels
([#93](https://github.com/AntonTegnelov/wave_forge/issues/93)), rasterised curves
([#98](https://github.com/AntonTegnelov/wave_forge/issues/98)). **Status:** in progress (2026-09-23): the example project is
`examples/history`: a continent preset, a toy history of 102 lines of GDScript (`history.gd`) with
rivers, villages that grow and burn and roads to nearest neighbours, and a scene to walk through the
result. Its headless check (`check.gd`, run by `wave_forge_godot/verify.sh`) passed on the dev
container: one seed gives one history, which ran in 5 ms and made 112 river stretches, 8 villages and
6 roads; a standing village and a burned one got their sites and towns 5 to 7 s after they were asked
for; a road levels the ground; and a history saved to JSON and read back gives the same ground in a
fresh node. The walk was looked at in rendered pictures only. The afternoon from a clean checkout,
and extending the history without Rust, still need a person.

## P: performance

Targets are proposals until the desktop measurement ([#39](https://github.com/AntonTegnelov/wave_forge/issues/39))
confirms them; each is checked by a benchmark or a Godot check with its build, stack and hardware
recorded.

### P1. Live generation at run speed without hitches

*A player runs through the world and never sees a hitch or an empty chunk.*

**Acceptance criteria.** At 4.2 m/s through the reference world with every enabled stage: 60 frames
per second, Godot's slowest frame under 8 ms, the node's own time under 2 ms at the 99th percentile,
and no frame with a chunk missing within the ready radius. **Status:** in progress (2026-09-23): the Godot check
meets these for a streamed WFC world (a small band rule set) with colliders and navigation (slowest frame 3.4 ms, node p99 0.56 ms,
[measurements.md](../research/measurements.md)), on the dev container's RTX 3070
through dozen; not yet with Phase 2 stages or on a desktop. With the valley pack's stages, the stages node's own time stays at 0.23 to 0.28 ms at the 99th percentile and under 3 ms at worst over ten runs, now that it emits at most 256 signals per frame ([measurements.md](../research/measurements.md) E31); that check does not yet run at 4.2 m/s through a stages world. With collider bodies bounded per frame, the city node's own time stays under 3.7 ms at worst over ten runs, but one run in ten still exceeded Godot's 8 ms bar through a stall outside the node (E37), which [#118](https://github.com/AntonTegnelov/wave_forge/issues/118) tracks.

### P2. Long view distance

*An open-world game shows mountains and towns far away without generating everything at full
detail.*

**Acceptance criteria.** Coarse levels (fields and proxies) reach a view distance of at least 2 km
on a mid-range desktop GPU at 60 frames per second, with the chunks per second each level generates
recorded. **Needs.** Levels and scales, far proxies ([#47](https://github.com/AntonTegnelov/wave_forge/issues/47)).
**Status:** not started (2026-09-23). Measured so far: a field stage costs 0.02 ms per chunk of 64 columns on the CPU in release (valley pack, dozen on the RTX 3070, [measurements.md](../research/measurements.md) E29), so fields are not what limits view distance.

### P3. Fast into a new world

*A player starts a new world and is walking quickly.*

**Acceptance criteria.** Time from pressing play to a playable area around the player under 5 s on a
reference desktop with compiled kernels cached, and the first-run time recorded. Today it is about
14 s in the dev container, most of it compiling kernels. **Status:** not started (2026-09-23). The valley pack's first town arrives 8.5 to 8.7 s after it is asked for on a cold start, almost all of it compiling two kernels, and 4.8 s with the kernels cached from an earlier run; every other stage together takes 72 ms ([measurements.md](../research/measurements.md) E29, E33, E34). Compiling town kernels off the stages' thread is [#111](https://github.com/AntonTegnelov/wave_forge/issues/111).

### P4. Memory that stays bounded

*A long session does not grow in memory.*

**Acceptance criteria.** Over a 30-minute scripted walk, process memory and GPU memory stay within 5%
of their level after the first minute, with every cache and timing window bounded. **Status:** not
started (2026-09-23).

### P5. Usable on an integrated GPU

*A player on a laptop with integrated graphics can still play with reduced settings.*

**Acceptance criteria.** On a reference integrated GPU, a reduced preset (smaller view radius,
fewer stages near the player) keeps 30 frames per second and P1's hitch bars. Mobile is out of
scope. **Status:** not started (2026-09-23).

### P6. Interactive editing

*A designer changes a stage's parameter and sees the result immediately.*

**Acceptance criteria.** Changing a parameter regenerates only the stages downstream of it and
updates a 3×3-chunk preview within 200 ms on a reference desktop, measured per stage.
**Needs.** Cache keyed by stage parameters, invalidation of dependants. **Status:** not started (2026-09-23).
