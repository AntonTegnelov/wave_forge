# Testing and inspection

How Wave Forge is tested, why it is tested this way, and how to look at what the generator produced.
For debugging techniques see [debugging.md](debugging.md); for the machine, the GPU drivers and CI
see [environment.md](environment.md).

## Why testing needs extra structure here

Wave Forge is a parallel program whose main work happens on the GPU. Its bugs rarely crash; they
produce **output that is almost right**: a few cells that break an adjacency rule, a constraint that
silently does not propagate, a shader reading the wrong field, a world that depends on the order its
chunks were asked for. Such bugs are invisible in logs and easy to miss by eye. The test setup is
built around four ideas:

1. **Check invariants mechanically.** Every generated grid is checked against the rules it came
   from, cell by cell, in all directions, and seams between chunks are checked like any other cell.
2. **Compare whole worlds.** A world is a function of its configuration, so the same world is
   generated in different request orders, on different threads and on different devices, and
   compared tile for tile.
3. **Pin the contracts between host and shader.** Layouts that must match on both sides are
   asserted in unit tests, because a mismatch compiles fine and fails silently.
4. **Make results visible.** End-to-end tests render their output to PNG, so humans and
   LLM-assisted development can see what happened in seconds.

## What the architecture requires of the tests

- **Everything but the solver is testable without a device.** The model, the seams, the scheduler,
  the facade's contract and the stage runtime run on the CPU reference solver (`wfc-core`, feature
  `reference`), and the reference is also the oracle the kernel's propagation is compared against.
- **The solver itself needs a GPU.** There is deliberately no CPU fallback
  ([vision.md](../product/vision.md)), so the kernel is tested on a real device, and in CI on a
  software one.
- **End-to-end tests exist for 2D and 3D**, with a simple tile set rendered to PNG and a small city
  in the style of marian42's, and opt-in suites generate whole worlds around a moving focus.
- **Rendering tools are dev-only.** They draw 2D tiles, four orthographic views and voxel models, and
  live in `wfc-devtools`, outside the shipped library.
- **Both engine integrations are tested inside their engine**: the Bevy plugin in a Bevy app, the
  Godot extension in a real Godot binary.

## Test layers

"Needs a GPU" means a Vulkan, Metal or DirectX 12 device; a recent software Vulkan device is enough
for correctness ([environment.md](environment.md)). CI runs every layer that is not `#[ignore]`d.

### Root workspace

| Layer | Where | What it covers | Needs a GPU |
|---|---|---|---|
| Model units | `wfc-core/src/`: `chunk`, `domains`, `hash`, `prior`, `reference`, `rules`, `store` | Chunk and region geometry and parity; the one-array domains; stateless hashes and weighted choice; the prior's layers and face bans; compiled rules and weights; the CPU reference solver; the chunk store pinning, releasing and committing cells | No |
| Rule units | `wfc-rules/src/`: `types`, `generator`, `modules`, `formats/module_format` | Adjacency tuples and axis transformations; rule generation by rotation and flip; module connectors, rotated variants, walkable faces and exclusions; module rule files and their named errors | No |
| Rule files | `wfc-rules/tests/loader_tests.rs` | Loading the valid and invalid files in `tests/rules_data/`, a missing file, module sets against tile sets, names, rotations and tags | No |
| Kit import | `wfc-rules/src/import.rs` (its tests), `wfc-devtools/tests/import.rs` | A block filling its cell fits itself on every side; a wall fits its mirror image and not itself across its asymmetric side; an empty face fits only an empty face; an oriented top fits a bottom turned the same way; the set proposed from the city's voxel models generates a city of many modules with no unplaced chunk on four seeds | No |
| Kernel units | `wfc-gpu/src/`: `kernel`, `block_solver` | The generated kernel source has every constant substituted, a rule set wider than four words uses two vectors, the workgroup budget counts what the kernel declares, a kernel that does not fit says so in numbers | No |
| Library units | `src/`: `scheduler`, `space`, `products`, `stages/pack`, `stages/runtime` | Parities and repair classes, batches that never share a face, nearest chunks first; the Y-up mapping of the lattice; instance sets, navigation sources and instance ids; pack loading, ordering and reach; the stage runtime's reach, blur and dropping | No |
| Devtools units | `wfc-devtools/src/`: `city`, `fixtures`, `invariants`, `models`, `render` | The city module set's structure (roofs on buildings, stairs with headroom, walkways that never end at walls); the fixtures; the invariant checker; voxel models and glTF export; the renderers | No |
| GPU integration | `wfc-gpu/tests/block_solver.rs` | The kernel on a real device: propagation against the reference fixpoint, validity and reproducibility, the same result at 64 and 256 invocations, a portfolio that stops seeds above its winner, weights, rule sets two and five words wide, impossible borders, regions and invocation counts the device cannot take, polling, malformed batches, and one pipeline compiled per region shape whatever the batch sizes | Yes |
| Dropped dispatch | `wfc-gpu/tests/dropped_dispatch.rs` | A backend that copies buffers but never runs the kernel, over buffers holding a plausible stale result: waiting and polling both give `SolverError::NoReport`, never solved regions | No, a fake backend |
| Library contract | `tests/facade.rs` | The same requests give the same world, the order they are asked in does not matter, a chunk evicted with its neighbours comes back the same, no batch holds two chunks that share a face, second-parity chunks solve without a halo, repairs try several seeds and keep the lowest that solves, report every chunk they rewrote and give a chunk up when no seed solves, a worker generates the same world on its own thread | No, the CPU reference |
| Stages | `tests/stages.rs` | A pack's stages come out bit for bit the same over a 4×4-chunk area asked for all at once and one chunk at a time in either order; `Pack::reach` reports how far each stage is generated beyond the target; sites stay inside their region and two chunks apart; the ground inside a site is level at its height | No |
| Tables | `tests/tables.rs` | A share of a parent's budget adds up to it exactly; a generated row's id is its parent's and its index, and its `Index` and `Count` say so; the same seed gives the same tables; adding a given row never changes another row's children; bad rows are refused and change nothing; a stage reads the focused row, and a new focus or new facts drop and regenerate only the stages that read the changed table, and none when the focused row stayed as it was; a row read with none focused, facts of another seed and a missing row are refused; loading refuses an expression that reads what its place cannot, and parents that lead back to a table |
| Table sites | `tests/table_sites.rs` | A table's rows put sites where their positions are, as large as their size says, levelled to their height; rows whose sites would crowd each other or break the size are refused by row; a town's rule set follows its row's names; a burned village's town is solved again with the other rule set, and nothing that reads no table is dropped; a changed row drops only the chunks within reach of its site, and the other village's town is not solved again; scattered trees keep their margin from a table's sites; loading refuses choosing by a column that is not a table's names |
| Apply | `tests/apply.rs` | A road levels the ground across its width to its centre line's height and blends back over its blend, leaving the ground beyond untouched; a carve lowers it by its depth; curves crossing chunks draw the same all at once and chunk by chunk in either order; a road wider than its Apply stage allows is refused by row; moving a road drops only the chunks within reach of where it was and is; rivers from a region job carve both sides of every region border, the same in any order |
| Filters | `tests/filters.rs` | A Delta is the highest value less the lowest within its radius; an Area marks a column edge exactly where a neighbour that far out has another category; an Area names `median` and `edge` and a field tests them with `Is`; both reach as far as their parameters say and sample as their chunks hold; a distance of 0 is refused |
| Nearest | `tests/nearest.rs` | A column takes the biome nearest its climate and a tie the first listed; a field reads the biomes and a sample is what the chunk holds; a point of another length or a biome listed twice is refused |
| Embed | `tests/embed.rs` | An embedded point lies in solid rock, as its surface runs, where its conditions hold; embedded points have ids of their own and come back the same; a mined point stays mined when its chunk is generated again; a spacing of nothing, heights the wrong way or a coarse volume are refused, and so is an Embed stage whose ids would share a Scatter stage's stage number |
| Aquifer | `tests/aquifer.rs` | Fluid fills only empty rock; caves below one pool's level are dry in another pool; fluid lies under its pool's level and pools under the lava line are lava; fluid comes back the same and chunks agree whatever their order; without a barrier pools of different levels meet at faces, fluid beside dry open space, and with one a Carve stage walls off every such face, only ever filling rock and leaving the barrier dry; a barrier of no thickness, or one naming no walled aquifer, is refused; cells of nothing, levels the wrong way or a coarse volume are refused |
| Locate | `tests/locate.rs` | From five positions, the located site of a Sites stage is one generation places and the nearest of them; locating generates no chunk; only a Sites stage can be located |
| Voxel world (G1) | `tests/voxel_world.rs` | The voxel pack (`examples/voxel.world.ron`): every biome takes the point nearest its climate somewhere; the terrain has overhangs and caves under its skin; aquifers pool water and lava in the caves and leave some dry; the surface takes snow, sand and grass by rule; ore lies in the rock and trees on the top; a village grows on ground levelled under its pieces; two travel orders give the same world, product for product; a changed number changes only the stages that depend on it |
| Cave level (G8) | `tests/cave_level.rs` | The cave level pack (`examples/cave_level.world.ron`): the level is one region whose rooms the tunnels join into one graph; rooms and tunnels are carved into the rock and the rock under the surface is untouched; the crystal meets the quota and every room keeps to its budget; the level is the same whatever order its chunks come in; a dig goes through the edits log and survives a save and a reload, moving no crystal |
| Planet (G3) | `tests/planet.rs` | The planet pack (`examples/planet.world.ron`): a planet's row decides its section and the same row gives the same one; the terrain has overhangs; every outpost stands on level ground under open sky; plants grow in jungles and rocks in deserts on the terrain's top; the nearest outpost is located without generating the chunks between; a dig is kept in the edits log and survives leaving and returning |
| Volume | `tests/volume.rs` | A Volume stage holds every level of every column from its bottom; a voxel's value is its expression at its height `Z` with its column's fields, so `height - Z` is exact everywhere; `FastNoise` in a volume is Godot's `get_noise_3d` at the voxel with its height as Godot's y; a coarse voxel is as tall as its column is wide; the top of a volume solid below a height is that height, and of a column solid to its highest voxel the volume's top; a voxel takes the first material whose rules hold at its height, a volume's materials are its categories and one without has none; `Z` outside a volume, a volume without levels, a volume read as a field and the top of a volume of another scale are refused |
| Pack text | `tests/pack_text.rs` | Every pack in the repository written as text reads back the same and loads; as plain data with every number a float, a whole one an integer, it reads back the same | No |
| Volume surface | `tests/volume_mesh.rs`, `src/volume_mesh.rs` (its tests) | A volume solid below a height is a flat surface there, two triangles per column, facing up with its normals, every vertex taking the material of the solid voxel below it; four chunks mesh the same triangles and materials as one chunk twice as wide; where features are wider than a voxel every triangle faces the way its normals point; a chunk has no surface until its neighbours' volumes have arrived; a volume's height (`volume_height`) is the top of the ground over a cave everywhere, across chunk seams too, and there is none until the surfaces within a cell are built; a folded quad is split along the diagonal that keeps it facing its way |
| Surface thread | `tests/surface_worker.rs` | A surface from the surface thread is the one `volume_mesh` builds; only a chunk's newest request is answered, and a cancelled one never is |
| Caves | `tests/caves.rs`, and the path search's tests in `src/stages/caves.rs` | Rooms lie apart inside their region at their depths; each pattern links the rooms as it says (a chain, a star, a hub of three branches); tunnels and rooms are carved empty and the rock far from them is not; a level is the same whatever order its chunks come in; a level whose rooms cannot all be placed is refused with every attempt; caves and tunnels out of range are refused. A tunnel over even ground runs straight between its ends and goes round a costly wall |
| Cave budgets | `tests/cave_budgets.rs` | A level holds exactly its deposits in solid rock around its rooms, apart from each other; each room spends its budget on its floor and no more; a room short of its cheapest kind keeps what is left; deposits and spawns are the same whatever order their chunks come in; a mined deposit and a killed enemy stay gone when their chunks come back; a level without room for its total is refused; deposits and spawns out of range are refused |
| Carve | `tests/volume_carve.rs` | A tunnel from a table's row is empty along its centre line six cells under the ground; a dungeon's rooms are empty inside their boxes, the rock is unchanged far from tunnels and rooms, and a carve keeps its volume's materials; a levelled site is solid under its floor and open above it; a carve is the same whatever order its chunks are asked for in; a curve with heights carves a tube along them; a tunnel wider than its stage allows, tunnels along curves on the ground plane without a height field and carving a volume of another scale are refused |
| Digs and fills | `tests/volume_edits.rs` | A dig is empty inside its ball and unchanged beyond a cell of it; a fill is solid inside its ball; digs and fills apply in the order they were made; a dig stales only its chunks and survives leaving and coming back; a save keeps digs and fills; a dig of a field or a ball without size is refused |
| Scatter chain | `tests/scatter_chain.rs` | A block holds as many candidates as its count allows, each with an id of its own; a group scatters within its radius around its first point, at its size, and is the same all at once and chunk by chunk in either order though members cross seams; points stand only where their conditions hold and as deep in water as their range allows; each leans within its tilt or along the ground's normal; a point's basis carries the vertical to its up at its scale; a chain out of range is refused; the ring world's ore and birch groves stand where their rules say; a stage keeps its clearance from a blocking stage's points across seams, where one without it does not, and comes out the same in any order |
| Golden stages | `tests/golden_stages.rs` | A pack whose field, categories and points depend on a sine, a distance, an angle and point spacing, over 5 by 5 chunks, hashed bit for bit against `tests/fixtures/golden_stages.txt`; CI runs it on Linux and the desktop script on Windows | No |
| Ground height | `tests/ground_height.rs` | `ground_height` is where the ground mesh's surface at full detail stands, at 1 600 points across a chunk | No |
| FastNoiseLite | `tests/fastnoise.rs` | The port gives every one of 2 560 samples in 2D and 2 560 in 3D that Godot 4.7 computed over 80 configurations (every noise type, fractal, cellular function and return type, warp type and warp fractal, offsets, negative and large seeds and positions), exactly; a configuration writes every property and takes Godot's default for one left out; a Field reads a pack's named noise at its columns' centres; an engine's replacement holds whatever the world's seed; a name the pack does not declare is refused. `tests/fixtures/fastnoise/generate.gd` regenerates the fixture in Godot |
| Locations | `tests/locations.rs` | Every kind keeps to its quota in every region and mostly meets it; no two sites come within a chunk, across regions too, and each stays inside its region; a kind keeps its distance from its own kind and meets its conditions at its centre; each region logs every kind's placements and refusals in order of priority; the table is the same in any order; a table that cannot be placed is refused; the ring world's shrines and crypts stand in their rings within their quotas |
| Place names | `tests/place_names.rs` | A location is named by `wf-place-<kind>` with underscores as hyphens and its region and index as arguments; a Sites stage's site has none; a kind whose name could not be a key is refused; a pack lists every name key its sites can be given, one per kind of its location tables, a kind two share once |
| Region tags | `tests/region_tags.rs` | Interiors cover every indoor cell once and no other, for scattered rooms and an L-shaped one; a chunk of rooms is one box; a sound plays at its point of its cell in the engine's axes; `surface_at` gives the module's surface |
| Occluders | `tests/occluders.rs` | Occluders cover every solid cell once and no other, indoor cells that are not solid included; a chunk without solid cells has none |
| Far proxies | `tests/proxies.rs` | A lone coloured cell is a closed box of its cell's size in the engine's axes; the face between two filled cells is left out; every face of every level faces out of its box; a coarser box is filled where half its cells are, in their average colour; levels halve until one box stands for the chunk, each straying further, a full chunk's levels being exactly its outer surfaces; an uncoloured module is left out |
| Lakes | `tests/lakes.rs`, and the flood's own tests in `src/stages/lakes.rs` | The lakes of an area are the same asked for at once, one chunk at a time and backwards; a lake is level, above the sea, never under its ground, never at its region's edge, and its shore stands at least as high as its water; a reed stands as deep as its range below the sea or a lake, some in lakes; a river ends where it reaches a lake and never runs on through one; lakes fill a coarse lattice as a fine one; the ring world has lakes in its hollows, above its sea, and rivers that end in them; lakes without water, with a region or least size of 0, or named by the pack's water when they are no Lakes stage are refused. The flood fills a hollow to the lowest notch in its rim, drains one open to the region's edge, leaves one below the sea to the sea and one under the least size dry |
| World bound | `tests/bound.rs` | No target chunk wholly outside a disk is generated while its inputs beyond the edge are; a rectangle bounds a world with its far edges included and its near edge exclusive; a finite world computes each region job once before play and keeps it, where an unbounded one computes a region again after leaving it; `request_bound` fails without a bound; a bound that holds nothing is refused |
| Assemble | `tests/assemble.rs`, and unit tests in `src/stages/assemble.rs` | A village of streets and houses and a dungeon of rooms and corridors are the same all at once and chunk by chunk in either order, levelled ground and trees included; every piece stays inside its site and no two overlap; every piece joins an earlier one door to door; the ground under each piece is levelled to its floor and trees keep their margin from pieces; a dungeon stands its lift above its entrance on its own kind of site, with dead ends capped; a piece placed by its turn and position opens its door onto what it joined; an assembly under its least size after its rerolls is refused, naming its stage; pieces that cannot grow are refused by stage; the ring world grows a dungeon of at least eight pieces above every crypt | No |
| Edits | `tests/edits.rs` | A felled tree stays felled through eviction and drops only its chunk; a moved tree stands where it was put in the chunk it came from; raised ground reaches what reads it within its reach and nothing further; a sample holds a raise as its chunk does; a log saves and loads as it was; an edit of nothing the pack makes is refused |
| Brushes | `tests/brushes.rs`, `src/stages/brushes.rs` | A raise lifts the path by its strength and fades to nothing at its radius; a negative strength lowers; a smooth brings each column nearer its neighbours; a dig stroke leaves a tunnel along its path; a remove stroke takes every point near its path and none further; a brush of the wrong kind or size is refused. Points along a path are no further apart than the step and reach its end | No |
| Presets | `tests/presets.rs` | Every value in each preset's ranges gives a sound world, over a grid of three values of each parameter and two seeds. Islands: finite heights, some sea and some land at the middle land amount, no tree at a density of 0 and many at 1; more land makes more land and more trees more trees. Hills and forests: heights finite and above zero, relief at every value, every tree in the woods, no tree at a forest of 0 and many at 1, no grass at meadows of 0; more hills make more relief and more forest more trees. Canyon desert: heights finite and above zero, relief, some sand and some rock and some level ground at every value, every cactus on sand, no cactus at cacti of 0 and many at 1; wider canyons make more sand, more strata more level ground and more cacti more cacti. Archipelago: heights finite, some land and never most of it at every value, every palm on land above the waves, no palm at palms of 0 and many at 1 once the islands are of middle size; larger islands make more land, wider reefs more shallow water and more palms more palms. Cave level: finite heights and values, some of the rock hollow, some crystal and trees on the hills at every value, every tree on the hills' surface and never down a sinkhole, no cave open to the sky at openings of 0; more caves hollow out more rock, more openings open more columns to the sky and more crystals make more crystal. Small city, its city solved by the reference solver from the city module set: one city of 3 by 3 chunks beside the world's centre at every value, finite heights, relief in the countryside, no tree in the city, no building at a density of 0 and many at 1, no tree at trees of 0 and many at 1; more density builds more, more hills make more relief and more trees more trees; a changed parameter drops only what reads it; a parameter undeclared or out of range is refused | No |
| Interactive editing (opt-in) | `tests/interactive_edit.rs` | P6's measurement: each of the islands preset's parameters changed five times on a 3×3-chunk preview, printing the time until the preview is generated again and what each stage regenerated | No |
| Rivers | `tests/rivers.rs` | A river runs strictly downhill and ends at the sea, a hollow or its region's edge, never leaving its region; it widens from its source to its mouth; rivers are the same in any order; rivers that cannot run are refused; the ring world's rivers are carved into its ground |
| Network | `tests/network.rs` | A road goes round a ridge through its gap and never onto it, in single steps; over flat ground it runs straight between the villages and stops at their footprints, its values its width; no road crosses ground below `dry`; every town of a region is joined to the others, by one road fewer than there are towns; roads are the same in any order; a network that cannot run is refused | No |
| Frozen world | `wfc-devtools/tests/frozen_world.rs` | A frozen city walked away from and back is as it was, its evicted chunks in the store and brought back from it; a city of changed weights given the same store brings them back as they were and generates none of them, where without the store the change changes them | Yes |
| Frozen store | `tests/frozen_store.rs`, `src/frozen.rs` | With a store, a walk of 60 chunks across an infinite world holds at most the 9 frozen chunks its view needs, the rest in the store; a frozen chunk walked back to is as it was first generated; a runtime of a changed pack given the same store reads a stored chunk as it was; a directory store gives back what it kept and nothing for the rest | No |
| World run | `tests/world_run.rs` | A run writes every target of every chunk of a bounded world to a store, reporting each chunk done, and holds no more products for a world four times as large; after each chunk a run reports what each stage has generated, never less than before, every target at least every chunk by the end; a run stopped part way and resumed by a fresh runtime writes the same bytes as one at once; a world without a bound cannot be run whole; a run generates a region's inputs at most half again as often as asking for the whole world at once; a played world serves what the run wrote, equal to what a runtime generates and holding the same chunks a runtime asked the same holds, and runs no stage; it fails on a stage the run did not write and on an edit | No |
| Continent | `wfc-devtools/tests/continent.rs` | The maximal preset's continent, in chunks of 8 by 8 columns, declares at least 40 biomes by rules and a survey of 128 by 128 points finds at least 40 of them; a ring 1 000 cells out from its centre is all ocean, and 35 to 70 % of it is land; a history simulated over its sampled map (`wfc_devtools::continent::history`) settles at least 40 settlements of all eight cultures, which its table takes; its Assemble stages hold at least 200 pieces; the pack holds what M1 names (at least 100 stages, 30 location kinds each with a quota and spacing, a Solve stage choosing among the eight cultures by the settlements' culture, rivers, lakes, roads, a volume and 40 Scatter stages or more); every vegetation and clutter stage names a biome the survey finds; in release, with `--ignored`, part of it generates to completion with rivers and lakes and prints what each stage cost, the rock under a plateau's rim holds caves, overhanging voxels and each of its four ores, a whole region of 64×64 chunks places at least 15 location kinds, joins its places by roads, one fewer than places or near it, and levels the ground of every settlement in it to the settlement's height, the first settlement of each culture gets a town of its culture's module set on the GPU, standing at its site's height (L39), every larger place of four regions grows at least its assembly's minimum of pieces (L40), every vegetation and clutter stage whose biomes cover 500 of a region's columns places points there (L41), and the whole continent runs ahead of time into a store that counts its bytes (L47) | Yes, for the towns |
| Cultures | `wfc-devtools/tests/cultures.rs` | Each of the continent's eight cultures is a module set of at least 60 modules that solves a town of 3 by 3 chunks of 8 by 8 by 6 cells on the GPU with no adjacency broken, every column on the street under air and one roof on each building, and a share of walkable cells in the largest network at least the city's on the same town less 0.15 | Yes |
| Surface navigation | `tests/surface_navigation.rs` | A stages world's navigation source for a chunk covers the chunk and its border with the ground's own triangles, none wholly beyond, its bottom on a whole cell height; it waits for every neighbour inside the world and skips those outside; a border wider than a chunk is refused | No |
| Saves | `tests/save.rs` | A frozen stage keeps the chunks it first generated after the pack changes, and a chunk never generated before follows the new pack; an ephemeral stage's edits are never saved; a save records the generator's version and the pack's digest, and loads as it was written; the ring world freezes its locations and never saves its grass | No |
| Stage worker | `tests/stage_worker.rs` | A runtime on its own thread hands back exactly what a runtime run directly generates, a request elsewhere drops what it no longer needs, a build failure or an unknown stage is reported, each stage counts the products it generated, as the worker reports them, a save made on the thread brings frozen chunks back on another, and a Scatter stage's report from the thread is the runtime's | No |
| Expressions | `tests/expressions.rs` | Coordinates, distance and angle are the column's; every operation computes its formula at every column; a named noise is the same in every stage and apart from other names; loading refuses parameters an expression cannot use, naming the stage; the ring world's height in one Field stage matches its formula computed from its noises | No |
| Rules | `tests/rules.rs` | The first rule that holds gives a column its category, else the fallback; `Is` reads categories back as a mask; loading refuses a category read as a field, a field read as categories, an unnamed category and more than 256 categories; the ring world's biomes (`examples/rings.world.ron`) are the ones its rules computed directly give, the same in any generation order | No |
| Blend | `tests/blend.rs` | A `Match` blends by its tent on a straight border and steps by at most its share; without a blend each column takes its own category's expression; the categories are generated as far out as it blends; loading refuses a match it cannot evaluate; the ring world's per-biome terrain equals each biome's shape inside it and steps by no more than the blend allows between any two neighbouring columns, across biome borders and chunk seams | No |
| Levels | `tests/levels.rs` | A coarse stage's columns are measured in WFC cells; a fine stage reads a coarse field between its columns and a coarse category from the column it lies in; loading lets data flow only from coarse to fine; reach across levels is counted in WFC cells; a coarse biome feeding a fine terrain is the same in any order, and a coarse stage is generated once for the fine chunks it covers | No |
| Regions | `tests/regions.rs` | A river job's curves meet their neighbours' exactly at every region border through edge hashes, and come out the same in any generation order; a chunk holds the curves through it; a rejected attempt is retried with new hashes within the budget, and a region is given up on with every reason when it runs out; a finite world is one region computed once; a job cannot read beyond its halo; loading checks a Region stage's numbers and its reach | No |
| Sampling | `tests/sample.rs` | A sample is bit for bit what the chunk holds, for fields, categories and blended matches of the ring world and for a coarse world map; an atlas is a stage's columns row by row; a stage that needs chunks is refused by name; an atlas of 256 by 256 world tiles is timed | No |
| Scatter | `tests/scatter.rs` | Points over a 12×12-chunk area come out the same in any order, no two are closer than `apart`, seams included, every one stands on the field and passes its height, slope and site-margin tests, and ids are unique and name their chunk and column | No |
| Scatter report | `tests/scatter_report.rs` | A chunk's kept candidates are exactly its points; every rejection names the modifier that failed (the height outside its range, the slope over its limit, the condition that did not hold, a passing candidate of higher priority within `apart`), and chance, height, slope, condition and spacing all occur; every verdict agrees with what the modifiers read there (the height and the condition's value are the fields', and a candidate passes each test before the one that rejected it); a report of a stage that scatters nothing is refused | No |
| Town kernels | `wfc-devtools/tests/town_kernels.rs` | Only the first of several towns compiles kernels, and a second start on a fresh device finds the first's pipeline cache file where the device can cache; both print their times | Yes |
| Towns | `tests/towns.rs` | A Solve stage puts a town only on its site, street level below and air above, the same in any request order, and names a missing town solver or rule set; every other stage generates while a town is being solved, and no chunk in a site arrives before its town; on the CPU reference with a ground-and-air module set | No |
| Masks (N4) | `tests/masks.rs` | A town fills only the masked columns of its site, painted by a raise brush through a Rules stage, every other column open ground under air; painting again solves that town again with its site kept, and every chunk the eraser does not reach keeps its districts; on the CPU reference with a ground, plaza and air module set | No |
| City towns | `wfc-devtools/tests/towns.rs` | The city as a Solve stage over a 12×12-chunk area: four towns, each valid across its chunk seams, the same all at once and one chunk at a time in reverse, each rendered to PNG | Yes |
| Order independence | `wfc-devtools/tests/order_diff.rs` | A 4×4-chunk city generated all at once, chunk by chunk in raster order and in reverse is the same world, tile for tile, with the same chunks given up on, with repairs off and on (seeds 8 and 11); a mismatch names how many chunks differ and the first differing cell | Yes |
| Partial eviction | `wfc-devtools/tests/partial_eviction.rs` | An 8×4-chunk city whose repairs rewrite neighbours, evicted beyond each cut from its corner and asked for again, is tile for tile the city generated once, with repairs replayed (seeds 8 and 11); a focus walked along a 16×4 city and back, evicting behind it, leaves every settled chunk as the city generated once; each cut prints what generating it again cost | Yes |
| Golden world | `wfc-devtools/tests/golden_world.rs` | A 4×4-chunk city compared tile for tile, repairs included, with `tests/fixtures/golden_city.txt`, recorded on the RTX 3070 through dozen; CI runs it on lavapipe, so it checks that a world is the same on two vendors' Vulkan | Yes |
| End to end | `wfc-devtools/tests/e2e_2d.rs`, `e2e_3d_city.rs` | Whole runs on reference rule sets, with invariants checked and images written ([below](#end-to-end-tests)) | Yes |
| Streaming (opt-in) | `wfc-devtools/tests/streaming.rs` | A whole world asked for at once comes out seamless, and a city generated in front of a walking player keeps ahead of it against a tick budget | Yes |
| Hole census (opt-in) | `wfc-devtools/tests/hole_census.rs` | Five city worlds streamed; every chunk given up on is solved again with more seeds, budgets and halos, with where each failed solve contradicted | Yes |
| Game session (opt-in) | `wfc-devtools/tests/game_session.rs` | A player walks and runs through an unbounded city in wall-clock time, driving a `Worker` the way an engine does ([below](#the-game-session)) | Yes |
| Benchmarks (opt-in) | `wfc-devtools/tests/cpu_reference.rs`, `wfc-gpu/tests/block_solver_bench.rs` | The CPU reference's time on the city, and one chunk's cost on the device against one CPU thread and all of them; every chunk a large dispatch reports as solved is valid ([below](#benchmarks)) | The second, yes |

### Engine workspaces

| Layer | Where | What it covers | Needs a GPU |
|---|---|---|---|
| Bevy plugin | `wave_forge_bevy/tests/wiring.rs` | On the CPU reference: a focus entity generates around itself, messages arrive, a chunk that leaves every focus is dropped and reported, chunks and cells sit where Bevy's Y-up space says, a quarter-turned tile turns its model the way the lattice turns it, the world a Bevy app generates is the library's, a game can drive the generator itself | No |
| Bevy region tags | `wave_forge_bevy/tests/region_tags.rs` | On the CPU reference: in a chunk of rooms and yards a Bevy app generated, every cell's `surface_at` at the centre the settings give it is its module's surface, every room is in exactly one interior and no yard is, and there is an emitter per room | No |
| Bevy sound and names | `wave_forge_bevy/examples/sound_and_names.rs` (its tests) | The example's mapping: a point in a room is indoors and one beside it is not; a sound is heard over its whole radius on its own side of the walls and over a quarter of it through one; a place name is its Fluent translation with its arguments, or nothing without one; every place the example's pack names in a 7×7-chunk area has a translation | No |
| Bevy far proxies | `wave_forge_bevy/tests/proxies.rs` | On the CPU reference: a generated chunk's proxy comes as several levels of coloured meshes, the finest drawn from the given start, each range ending where the next begins and the last with no end; a chunk of uncoloured modules has none | No |
| Bevy stages | `wave_forge_bevy/tests/stages.rs` | A focus entity makes a pack's stages generate around it; what arrives equals what the library's runtime generates directly; a point's place in Bevy's world is the lattice's; moving away drops what was left behind, with messages; each chunk's ground equals the library's for the same fields, is announced once and is dropped with its field; a Rules stage's categories arrive as the runtime generates them; each stage reports its cost; a region job's curves arrive as the runtime makes them; a save made in one app brings a felled tree back felled in another; an assembled house placed by its transform opens its door onto a street; every tree of a bound kind gets one entity at its transform, and entities go with their chunks; a piece overlapping several chunks gets one entity; a cave's rooms and its spawned points get an entity each; the ground's materials are the library's categories of its vertices; the far ground is the library's, leaving out the chunks with ground, and goes with its field; each chunk's volume surface is the library's and goes with its volume, as its fluid surface does with its fluid, and its coloured mesh has each vertex in its material's colour; a dig in a neighbour builds again the surface that reads it; the ground's height is the library's, and without a ground the library's top of the volume; a raise in a neighbour builds again the ground that reads it; the noise configuration is reflected and registered; a world a run wrote plays with its ground and trees the runtime's and no stage run; a chunk's navigation source is the library's of the same ground | No |
| Bevy ground material | `wave_forge_bevy/tests/ground_material_render.rs` (ignored) | Bevy's renderer draws a chunk whose west half is one material and east half another, read back from the GPU unlit and untonemapped, in exactly their palette colours, blended only in a band around the border | Yes, a device |
| Bevy grass | `wave_forge_bevy/tests/grass_render.rs` (ignored) | Bevy's renderer draws grass over a chunk whose west half is covered and east half bare: blades appear only over the covered half, pixels change from frame to frame while the wind blows and none change once it drops | Yes, a device |
| Bevy vegetation | `wave_forge_bevy/tests/vegetation_render.rs` (ignored) | Bevy's renderer draws trees through the vegetation material over a lit ground, with a shadow-casting light and a depth and motion-vector prepass: the trees and their shadows move while the wind blows and nothing moves once it drops, and no pixel is left in the clear colour, which is where the prepass and the main pass would put a tree apart | Yes, a device |
| Bevy ground levels | `wave_forge_bevy/tests/ground_levels_render.rs` (ignored) | Bevy's renderer draws 15 by 15 chunks of ground through `ground_levels` at thresholds of 1, 4 and 16 pixels, ground filling the picture over a magenta background: neighbours at different levels are in view, and no pixel is magenta, a gap between them (without skirts, 32 and 38 at 1 and 4 pixels) | Yes, a device |
| Bevy on a device (opt-in) | `wave_forge_bevy/tests/shared_device.rs`, `real_render_plugin.rs` | A city generates on a device created the way `bevy_render` creates its own, and on the device Bevy's own render plugin hands over; a `solver_config` of one region a batch solves each chunk in a batch of its own; a frozen plugin keeps the chunks it evicts in its store | Yes |
| Godot units | `wave_forge_godot/src/timings.rs` | The median, 99th percentile and maximum that `stats()` reports, and the window of recent samples | No |
| Godot extension | `wave_forge_godot/godot/verify.gd` | Inside a real Godot, headless, on Jolt: the city's tiles are named, turned and tagged the way a game places them, every module model loads as glTF inside its cell, and the inspector groups the node's properties. A focus runs a strip of chunks and back by the clock without waiting for generation: the chunks beside it are there on every frame, chunks behind are dropped, tiles obey the rules across seams, a chunk returned to is unchanged, the chunks near it get colliders a ray hits and navigation meshes an agent paths across two seams, and a path crosses every chunk on its own `navigation_ready`, asked for with no wait. Godot's slowest frame stays under 8 ms, and the node's own time per frame under 2 ms at the 99th percentile | Yes, and a Godot binary |
| Godot tables | `wave_forge_godot/godot/verify_tables.gd` | A small pack (`godot/facts.world.ron`) takes villages from GDScript as plain Dictionaries and reads them back, names included; the houses generated under each village share its population out exactly; rows with an unknown name, a missing column, a negative id or a value of another type, rows for a generated table and a missing row to focus are all refused and change nothing; a focused village's population is what its field reads, its site stands where it is and is named by its row, villages whose sites would crowd each other are refused, a road given as a row is named by its row and levels the hills across it, and new villages drop and regenerate the stages that read them and not the hills or the road | Yes, and a Godot binary |
| History example | `examples/history/check.gd` | The example's toy history gives the same tables twice from one seed and makes rivers, roads and villages both standing and burned; given to the stages, a standing and a burned village each get a site named by its row with a town on it; a road levels the ground by a village; a second node given only the history saved as JSON holds the same ground; the history's time and the time to each town are printed | Yes, and a Godot binary |
| Sample world example | `examples/sample_world/check.gd` | The small city preset's city stands in grass with its modules and trees placed, the day goes on, and every one of the townsfolk walks along the navigation map inside the city | Yes, and a Godot binary |
| Godot noise | `wave_forge_godot/godot/verify_noise.gd` | A `FastNoiseLite` resource far from Godot's defaults (cellular, ridged, warped, offset), assigned to a pack's noise, gives the field exactly what its `get_noise_2d` gives at every column's centre, and a sample too | Yes, and a Godot binary |
| Godot assemblies | `wave_forge_godot/godot/verify_assemble.gd` | With cells twice as wide as tall, each piece of a village stands at the centre of its cells on the levelled ground, a street covers its cells once turned, and a house's door opens onto a street; the number of pieces and houses checked is printed | Yes, and a Godot binary |
| Godot scenes | `wave_forge_godot/godot/verify_scenes.gd` | Scenes bound to a village's pieces and to trees: a lone-mesh tree is drawn once per point as MultiMeshes; houses given as a `PackedScene` and streets given by a path loaded on the loader threads get exactly one node per piece, named by `instance_spawned`, at the piece's transform; with a promotion radius splitting the pieces, only those inside stay nodes and the rest are drawn as MultiMeshes of their mesh, and turning promotion off makes them nodes again; moving away, out of the pack's bound, frees every node and MultiMesh instance of the view left and places nothing where it arrives; placing's slowest frame is printed | Yes, and a Godot binary |
| Godot cave scenes | `wave_forge_godot/godot/verify_cave_scenes.gd` | Scenes bound to a cave level's rooms and the enemies spawned in them: every cavern gets one node, placed by the chunk its id names, standing at its room's transform, and every grunt, a lone mesh, is drawn once as MultiMeshes | Yes, and a Godot binary |
| Godot frozen directory | `wave_forge_godot/godot/verify_frozen.gd` | With `frozen_directory` set, walking away drops the origin's frozen trees into a file there, and walking back gives the origin the trees it had; a frozen city's origin leaves for the directory and walking back restores it as it was | Yes, and a Godot binary |
| Godot pooling | `wave_forge_godot/godot/verify_pooling.gd` | A village's pieces bound to a scene whose root script defines `_wave_forge_reset`, and then to one without it, freed and placed again five times by a promotion radius: with the reset every piece placed again reuses an earlier node, reset once per release and out of the tree while it waited, no more nodes are made than were placed at once, and freeing the node frees the pools; without it no node is reused; the time a node takes to place again is printed for both | Yes, and a Godot binary |
| Godot ground materials | `wave_forge_godot/godot/verify_ground.gd` | Every chunk's ground is drawn with a copy of the reference ground shader's material of its own, the shader the addon's file (`addons/wave_forge/shaders/ground.gdshader`, as the grass's is its own), whose id texture holds the category of every ground vertex, the chunks beyond its far edges included, and whose cell is the node's; a material stage that is no Rules or Area stage is refused; grass grows on every chunk of ground within its radius and none beyond, each chunk's grass material holding its cover per column and the ground's height per vertex; raising the first column of the chunk beside the origin builds the origin's ground again; `ground_height` is the column's height at every column centre and on each square's diagonal halfway across | Yes, and a Godot binary |
| Godot far ground | `wave_forge_godot/godot/verify_far.gd` | The far ground is drawn on every coarse chunk partly beyond the near ground and on none the near ground covers whole; every far vertex is in the colour of its coarse column's category, the near palette's for a category the near surface names; a far ground material stage at another scale is warned of and refused | Yes, and a Godot binary |
| Godot volume | `wave_forge_godot/godot/verify_volume.gd` | A Volume stage of ground over a cave: every chunk around the player gets its surface, a ray down from the sky lands on the ground's top, where `ground_height` stands with no ground stage, and from inside the cave a ray up meets its ceiling and one down its floor, each facing into the cave; the ground's top is grass and the cave rock, and the drawn surface carries each vertex's colour from `volume_palette`; ore embedded in the rock and bound to a scene is placed as nodes inside the rock; an Aquifer stage's pools, drawn as `fluid_stage`, fill the cave's floor below their levels as water and glowing lava, see-through in `fluid_palette`'s colours, and the ray to the floor passes through them; the navigation baked around the player walks on the ground's top and on the cave's floor, and a path crosses each chunk on its own `navigation_ready`; a ball dug where four chunks meet opens the cave to the sky, `ground_height` falling through it to the cave's floor, every chunk around it built again, the fluid beside it too, and its navigation baked again; a `volume_stage` that is a field, and digging a field, are refused | Yes, and a Godot binary |
| Godot bake | `wave_forge_godot/godot/verify_bake.gd` | Baking 3 by 3 chunks of ground with materials, trees drawn as a MultiMesh, a cave with pools of water and ore placed from a saved scene gives a scene that, saved as text, names no Wave Forge class, refers to no file but the ore's and opens in a second Godot project without the extension; loaded back, every chunk holds its ground standing where the node's does with a height-map body, its cave's surface and fluid with the node's vertices and a concave body, a tree for every point, standing where the stage placed it, and an ore for every embedded one; baking chunks not yet built is refused; a linked bake regenerated after a designer moved one ore, deleted another and added a node keeps the moved ore where it was put, turned as it was, drops the deleted one and keeps the designer's node | Yes, and a Godot binary |
| Godot kit import | `wave_forge_godot/godot/verify_import.gd` | A `MeshLibrary` of a block, an item without a mesh and a quarter-thick wall proposes a module set whose block sides share one symmetric connector, whose empty item is empty on every face, and whose wall's ends are one asymmetric connector's plain and flipped faces, its back the block's side and its front empty; the proposal loads as a rule set; `kit_connectors` lists the four connectors with the modules that have them, and the dock's kit import, named by the artist with the block's side walkable, saves a module set under those names that loads as a rule set, refuses one name for two connectors, and offers no walkable tick for a top; the same kit as a folder of scenes, the wall under a parent node, proposes the same modules, and the panel lists its four connectors from the folder; the scenes' static bodies are the items' shapes, where the scenes place them; a world generated from the folder's module set, its items' shapes its colliders, has a body under every walkable cell, a ray onto the top of every block with open air above meeting it | Yes, and a Godot binary |
| Godot parameters | `wave_forge_godot/godot/verify_params.gd` | The islands preset started with much land and no trees lists its three parameters with those values and places no tree; raising the tree density grows trees while the ground stays as it was; a value out of range and an undeclared name are refused; the inspector lists each parameter as a slider over its range that reads its default until set, reverts to it, and clears the trees again when moved back to 0 | Yes, and a Godot binary |
| Godot packs as data | `wave_forge_godot/godot/verify_pack_data.gd` | Every preset the plugin ships and every test pack of the project comes back from `pack_dictionary` and `pack_text` as the same data; an edited default and a stack put in another order save a pack a node starts from, the default as edited; a whole number typed as a float is taken where the pack holds an integer; data no pack holds, and a pack the library refuses, save no text | Yes, and a Godot binary |
| Godot stacks | `wave_forge_godot/godot/verify_stack.gd` | The islands preset made a `WaveForgeStack` saves the preset's own pack, saved as a resource and loaded back too; a node given the stack and no `pack_file` generates it and warns of no missing pack; the stack's height stage raised in place and its stages reordered, a node started again generates the edited pack; a stack of no valid pack does not start, and a stack beside a `pack_file` warns that the stack is generated; the stack lists its Rules stages' categories, a drag of one scene file drops it and any other drag nothing, and a Scatter stage added on the islands' grass, as a scene dropped onto it makes, is generated with every point on a grass column on the ground there and drawn from its scene (N3). Edit as a stack and the drag itself in the dock are not automated | Yes, and a Godot binary |
| Godot painting | `wave_forge_godot/godot/verify_paint.gd`, and the editor run in `verify.sh` | On the islands preset, all land: a remove stroke clears every tree near its path once its chunk comes back, a raise stroke then lifts the ground on its path by its strength at once, `edits_text` holds both, and a second node given that text before it starts has the raised ground from its start; a brush that names no brush is refused. The editor, run headless, loads the plugin without a script error, its dock the plugin's own `EditorDock` and Paint and the six brushes shortcuts in the editor settings, each taking its default key (`verify_plugin.gd`, an autoload that acts only in the editor). Painting with the mouse and undoing in the editor are not automated | Yes, and a Godot binary |
| Godot candidates | `wave_forge_godot/godot/verify_candidates.gd` | With `candidates_stage` set to the islands preset's trees, every chunk around the player gets its candidates drawn, the legend counts as many kept as there are trees, each verdict with its colour, and candidates rejected by either of the trees' conditions; `candidate_under` looking down at a tree, on a node without a `ground_stage`, gives a kept candidate, and `candidate_near` under it gives one with both conditions holding and no slope or water read, one off the grass reads 0 for its first condition, which fails, nothing lies far away, and the dock's candidate panel shows the verdict and readings | Yes, and a Godot binary |
| Godot world run | `wave_forge_godot/godot/verify_world.gd` | A node runs a finite world of 30 chunks into a directory through the editor dock's world run panel, reporting its progress by chunk and by stage, is cancelled after one chunk, the panel showing where it stopped, and run again, which resumes and finishes; a node playing that directory builds its ground from the stored fields, and its fields and trees equal those of a node that generates, with no stage run; both bake navigation from their ground, and a path across three chunks, on the ground, is the same in both | Yes, and a Godot binary |
| Godot continent | `wave_forge_godot/godot/verify_continent.gd` | `continent.tscn`, the maximal preset as its editor scene sets it up, started asking for towns alone: the scene's script gives it the history's 50 settlements, the first of the marsh folk; following it, its town arrives on its site within 300 s, solved with the marsh folk's module set, and every module its placements name is one of that set's | Yes, and a Godot binary |
| Godot typed maps | `wave_forge_godot/godot/verify_typed_maps.gd` | `target_radii`, `noises`, `sounds` and `proxy_colours` are typed Dictionaries of their key and value types; a scene written with untyped Dictionaries for all four loads with each map holding its entry, typed | Yes, and a Godot binary |
| Godot inspector | `wave_forge_godot/godot/verify_inspector.gd` | A `WaveForgeStages` node in a scene with no light, no environment, no camera and no pack warns of all four, and of nothing once the scene has them and the node a pack; an unknown target and a `candidates_stage` that is no Scatter stage warn, and `start` refuses the latter; colliders without Jolt and occluders with occlusion culling off warn, and not once culling is on; a `WaveForgeWorld` naming an interior bus the project lacks warns until the bus exists, and one set to start with no `rules_file` warns. Then `WaveForgeStages`' buttons: Start or regenerate starts it, Reroll seed takes another seed and keeps it running, and Bake the view refuses with nothing followed and then saves a scene of the nine chunks of a view of radius 1 | Yes, and a Godot binary |
| Godot presets | `wave_forge_godot/godot/verify_presets.gd` | Every preset the plugin ships, given to a fresh node in a lit scene, starts on its own and, once nothing is pending around the followed point, once every target's chunks within its radius have arrived, warns of nothing, stands ground there (the top of its volume for the cave level), has bodies and navigation, draws its sea when it has one and its props as MultiMesh instances, and has a palette colour per material of its ground or volume; `verify.sh` fails if the run prints an error or a warning (N1) | Yes, and a Godot binary |
| Godot walk | `wave_forge_godot/godot/verify_walk.gd` | A scene of a sun, an environment, a node given the default preset and the plugin's walker over land, with no code: nothing calls `follow`, the node follows the walker's camera and warns of nothing, the walker stands on the ground once its body is there and walks forward for three seconds, at least half its speed's distance, never below the ground; `verify.sh` fails if the run prints an error or a warning (N1) | Yes, and a Godot binary |
| Godot sound and surfaces | `wave_forge_godot/godot/verify_sound.gd` | Around the city's origin, every cell's `surface_at` is its module's surface, a chunk's interiors hold every building cell once and no other cell, and its emitters sit in its fountains' cells; within `audio_radius` the node has an `Area3D` per interior, reverbing on the given bus and playing on the other and a playing `AudioStreamPlayer3D` per emitter, whose `area_mask` holds the interiors' layer, and once the player has moved far away none remain where it was | Yes, and a Godot binary |
| Godot place names | `wave_forge_godot/godot/verify_names.gd` | A location table's sites carry the key `wf-place-stone-circle` and their region and index as `name_args`, and a translation built from the entries the plugin lists in translation templates, the pack's one key in the `wave_forge` context and nothing for a rule set, turns them into words through `tr` and `format`. Generating a template in the editor is not automated | Yes, and a Godot binary |
| Godot occlusion | `wave_forge_godot/godot/verify_occlusion.gd` | Around the city's origin, a chunk's occluders hold every solid building cell once and no other cell; within `occluder_radius` the node has one `OccluderInstance3D` per chunk with a solid cell, its `ArrayOccluder3D` holding eight corners per box; once the player has moved far away none remain where it was | Yes, and a Godot binary |
| Godot far proxies | `wave_forge_godot/godot/verify_proxies.gd` | With `proxy_distance` set, every generated chunk and no other has its proxy, and each with a coloured module an instance, before and after the player moves eight chunks away; turning proxies off frees them all | Yes, and a Godot binary |
| Far ground | `tests/far_ground.rs` | A far ground's corners are what a fine stage reading the coarse field gets; neighbouring coarse chunks meet exactly; with near grounds on a block of chunks, far and near ground cover every sampled point of the coarse chunk once; a wall stands at every near edge the far ground meets, from the edge or above to below it by the near ground's skirt | No |
| Godot far ground | `wave_forge_godot/godot/render_far.gd`, through `render_ground.sh` | The near ground and the far ground beyond it, from a coarse field of the same ground, over a magenta background from above at an angle, across their boundary from low down and from high above: no picture has a magenta pixel; prints the chunks each level generated per second of its stage's time | Yes, a display (Xvfb) |
| Godot ground levels | `wave_forge_godot/godot/render_lods.gd`, through `render_ground.sh` | 15 by 15 chunks of ground fill the picture over a magenta background, seen with levels of detail off and at thresholds of 1, 4 and 16 pixels: fewer primitives are drawn at every threshold, and no picture has a magenta pixel, a gap between neighbours at different levels (without skirts, 121 and 1 600 at 1 and 4 pixels) | Yes, a display (Xvfb) |
| Godot edits | `wave_forge_godot/godot/verify_edits.gd` | A tree felled by its id and ground raised under a position come back so; a second node given only the first one's edits log holds the same trees and ground; a third node given the first one's save keeps the felled tree and the raised ground, while cut grass of an ephemeral stage grows back; a point that is not there, a stage that is no field, and a log or a save that is not one are refused | Yes, and a Godot binary |
| Godot stages | `wave_forge_godot/godot/verify_stages.gd` | The `WaveForgeStages` node runs the valley test pack (`examples/valley.world.ron`): it finds a town over the sites alone, as a game would, then generates the ground, the towns and the trees around it; every town chunk holds a whole chunk of tiles on level ground at its site's height; the modules bound to a scene are drawn from it with no code, as many as the towns hold, a baked town chunk has each where `town_instance_sets` puts it, and given no shape they collide with the box they are drawn from; every tree stands on the ground and none in a town; every column's cover is the category its rule gives its height; every target stage reports its cost, printed per stage; no frame emits more than 256 signals; a sample and an atlas of the hills match their chunk, and locating the nearest town from inside it finds that town; the node starts again on its cached kernels and the time to the first chunk that holds a town is printed cold and warm; every chunk whose fields are held around it has ground, and every chunk within the collider radius a body; a path on the navigation baked from the ground and the town's shapes crosses the town nearly straight, on the ground and the streets, and the navigation reaches at least half the tops of the town's buildings; every town chunk within the occluder radius has occluders of its solid cells inside the town's layers; a capsule with gravity walks from open ground straight through a town, its feet never more than 0.3 below the ground's surface nor above it while standing; moving away drops the first view, its ground, its navigation and its occluders; the node's own time per frame stays under 2 ms at the 99th percentile | Yes, and a Godot binary |

## Running the tests

```bash
cargo test --workspace                 # the root workspace
cargo test --workspace --lib           # unit tests only, no GPU needed
cargo test -p wfc-devtools --test e2e_3d_city -- --nocapture   # one test, showing artifact paths
cargo test --manifest-path wave_forge_bevy/Cargo.toml          # no device needed
cargo test --manifest-path wave_forge_bevy/Cargo.toml --release -- --ignored --nocapture
cargo test --manifest-path wave_forge_godot/Cargo.toml
GODOT=/path/to/godot bash wave_forge_godot/verify.sh release
```

Set `CARGO_TARGET_DIR` per workspace first when building from a worktree, and in the dev container
start Godot with the `libd3d12core.so` preload ([environment.md](environment.md)).

**The workspace build does not test each crate's own feature set.** Cargo unifies features across a
workspace build, so a crate can compile there with a feature it does not enable itself and still
break anyone who depends on it alone. The CPU reference (`wfc-core/reference`) is enabled by several
dev-dependencies, and the wgpu backend is a feature an engine integration can build without. CI
builds each crate on its own:

```bash
cargo test -p wfc-core                             # default features (none)
cargo test -p wfc-core --all-features
cargo check -p wfc-gpu --no-default-features       # no wgpu
cargo check -p wave_forge --no-default-features    # the facade without a bundled solver
cargo test -p wfc-rules --no-default-features      # RON parsing disabled
```

**Recording the golden world again.** After a change that is meant to change worlds, run
`WAVE_FORGE_BLESS=1 cargo test -p wfc-devtools --test golden_world` on a real GPU and say in the
commit why the world changed.

## End-to-end tests

| Test | Rule set | Asserts |
|---|---|---|
| `e2e_2d::coastline_2d_obeys_its_rules_and_renders` | `fixtures::coast_2d`: water, sand, grass and forest bands | Full collapse, zero adjacency violations, rendered image matches the grid |
| `e2e_3d_city::small_city_is_structurally_sound_and_renders` | `city::city`: 81 module variants: roads, solid buildings with doors, balconies, arcades and upper passages, pitched and walkable flat roofs with railings, walkways on pillars, and stairs from the street, from roofs and along facades | Full collapse, zero violations, street-level modules only on the bottom layer, only air on top, every building column rises from the street to exactly one roof, every stair has headroom above it; reports the share of walkable cells in the largest network |

The city is a very crude version of [marian42's WFC city](https://marian42.de/article/wfc/). It is
not meant to look good. It is meant to be a **realistic workload**. The toy fixtures prove the solver
works at all, but a handful of tiles propagate almost instantly and never contradict, so they say
nothing about what real 3D generation costs. The city has more tiles than fit in one possibility
word, weights, and structure reaching across many cells, which is what the solver has to handle for
the project's goals.

Modules are described by the **connectors** on their six faces (`wfc-rules/src/modules.rs`), the way
marian42 does it: rotated variants, adjacency and weights are derived, so the set stays readable at a
size where hand-written adjacency tuples would not. One difference: our modules are centred on cells,
while marian42's sit on grid corners. A cell face can therefore separate two materials (a facade and
open air), and `ModuleSet::connect` declares which different connectors may meet. The module set is a
rule file, `examples/city.ron`, which the CLI and both engines load as well; its voxel models and
boundary constraints live in `wfc-devtools/src/city.rs`. Assertions such as "street level only on the
bottom layer" can be traced back to the connector that causes them (`bedrock` under street-level
modules, which nothing fits).

**Walkability** comes from how the modules are built, as in marian42's city, not from a global
constraint; [constraints.md](../architecture/constraints.md) explains the design. Local rules cannot
forbid a network that is cut off as a whole, so `city::disconnected_walkable_cells` flood-fills the
walk graph and reports the share of walkable cells in the largest network. The tests print it rather
than assert it, because it is a property of the module set.

**Artifacts** are written to `$CARGO_TARGET_DIR/tmp/e2e-artifacts/` (Cargo's per-target temporary
directory, `target/tmp/` by default), or to `WFC_ARTIFACT_DIR` if set:

- `coast_2d.png`
- `city_isometric.png`: every module drawn as its small voxel model
- `city_street_level.png`: the bottom layer, one colour per module variant

Other suites write straight into `$CARGO_TARGET_DIR/tmp/`:

- `town_<x>_<y>.png`: each town of the city towns test, by region
- `stitched.png` and `live.png`: whole worlds from the streaming suite
- `game_session.png`: the chunks held at the end of the game session, back at its start

### Benchmarks

All three are `#[ignore]`d and print their numbers; run them in release mode.

```bash
cargo test -p wfc-devtools --release --test cpu_reference -- --ignored --nocapture
cargo test -p wfc-gpu --release --test block_solver_bench -- --ignored --nocapture --test-threads=1
cargo test -p wfc-devtools --release --test streaming -- --ignored --nocapture --test-threads=1
```

`cpu_reference` times the single-threaded CPU solver (`wfc-core`, feature `reference`), the yardstick
every GPU number is printed against. `block_solver_bench` measures what only it can: one chunk's cost
against one CPU thread and against all of them, how that scales with the chunks in a dispatch, and
that every chunk a many-chunk dispatch reports as solved is valid. `streaming` measures whole worlds
through the library, so what a game would get is what is timed. A timing describes one build on one
machine and driver stack; [measurements.md](../research/measurements.md) records each one with its
protocol, and [performance.md](performance.md) says how to measure.

### The game session

```bash
cargo test -p wfc-devtools --release --test game_session -- --ignored --nocapture --test-threads=1
```

The streaming suite asks whether the work of a simulated tick fits the tick. The game session asks
what a player of a game like marian42's city would see. It plays the city once, in wall-clock time,
the way an engine drives the library: a 60 Hz frame loop hands the player's position to a `Worker`
when it enters another chunk, drains the worker's events, and never waits for generation. The player
follows a fixed 776 m route on an unbounded world: a walk at 1.4 m/s, runs at 4.2 m/s (marian42's
run multiplier of three), turns of 45 and 90 degrees, a diagonal, and a return to the start after the
start has been dropped. A chunk is in view when its ground comes within 30 m of the player, which is
marian42's generation range, and chunks are generated three out from the player's chunk. The session
takes about five minutes, most of it the walk, and each test reads it and checks one thing:

| Test | Holds when |
|---|---|
| `no_chunk_in_view_is_ever_missing_while_it_generates` | no chunk in view is without tiles on any frame |
| `no_chunk_in_view_is_a_hole` | no chunk in view is one that could not be placed |
| `no_generated_cell_breaks_a_rule_inside_a_chunk_or_across_a_seam` | every chunk, checked when it arrives, against the chunks beside it at that moment |
| `generation_costs_the_main_thread_under_a_millisecond_a_frame` | asking for chunks, dropping them and draining events costs at most 1 ms at the 99th percentile and 4 ms at worst |
| `memory_stays_bounded_while_the_player_travels` | the chunks held at once stay within the radius, the margin and one chunk of movement, over a route that generates at least three times as many |
| `a_chunk_walked_back_to_comes_back_the_same` | every chunk generated twice whose tiles its coordinate alone decides both times is identical, over at least 20 |
| `seams_cut_no_more_paths_than_chunk_interiors` | where a face someone can walk through meets another, the walk goes on across seams at least as often as inside chunks, less 0.05 |

A chunk's coordinate alone decides its tiles when no repair touched it, no neighbour failed, and it
was solved with no neighbour present (first parity) or against four such neighbours (second parity);
that is the determinism the library promises ([world.md](../architecture/world.md)), and anything
else is counted as not comparable. The share of walkable cells in the largest network at the end is
printed, not asserted.

`no_chunk_in_view_is_a_hole` holds the bar that no chunk is left unplaced, and so do both streaming
tests. When one fails, `hole_census.rs` says why: it streams five city worlds and solves every chunk
given up on again with more seeds, more budget and every halo, and reports where each failed solve
contradicted.

```bash
cargo test -p wfc-devtools --release --test hole_census -- --ignored --nocapture --test-threads=1
```

## Rendering tools

`wfc-devtools` is a developer-only crate and is never part of the shipped library. Its `wave-forge`
binary generates one chunk from a rule file and writes it as a text grid (`--output`, `output.txt` by
default). `wfc-render` draws that grid to a PNG, and `wfc-export-models` writes the city's voxel
models as glTF for the engines:

```bash
cargo run -p wfc-devtools --release --bin wave-forge -- --rule-file examples/simple-pattern.ron --width 12 --height 12 --depth 6 --output grid.txt
cargo run -p wfc-devtools --bin wfc-render -- grid.txt --out grid.png --empty-tile 0
cargo run -p wfc-devtools --bin wfc-render -- grid.txt --view layer --z 0 --out layer0.png
```

The four-view sheet (`--view four-view`, the default) shows the whole grid with `+z` up:

| | |
|---|---|
| **Top**: looking down, `+x` right, `+y` up | **Isometric**: seen from `+x`, `+y`, `+z` |
| **Front**: from `-y`, `+x` right | **Side**: from `+x`, `+y` right |

The orthographic views show exact positions, and the isometric view shows how they fit together.
Nearer surfaces are brighter, and tiles marked empty with `--empty-tile` are see-through. Each tile
index gets a fixed colour, so the same tile has the same colour in every picture.

The end-to-end tests draw their artifacts with the same code (`wfc_devtools::render`). It is a small
CPU rasteriser rather than an engine because the images are made inside tests and containers without
a display, have to be pixel-for-pixel reproducible, and must be cheap enough to write after every
run. For a picture through a real engine, `render_city.sh` renders a city in Godot (next section).

## The engine integrations

Both live in their own workspaces, so the library's `cargo test --workspace` does not compile an
engine. Run them explicitly:

```bash
cargo test --manifest-path wave_forge_bevy/Cargo.toml                       # no device needed
cargo test --manifest-path wave_forge_bevy/Cargo.toml --release -- --ignored --nocapture
GODOT=/path/to/godot bash wave_forge_godot/verify.sh release                # needs a Godot 4 binary
GODOT=/path/to/godot xvfb-run -a wave_forge_godot/render_city.sh           # a picture of the city
GODOT=/path/to/godot xvfb-run -a wave_forge_godot/render_ground.sh         # the ground, grass, wind and levels
GODOT=/path/to/godot xvfb-run -a wave_forge_godot/render_presets.sh        # a contact sheet per preset, each parameter at its minimum, default and maximum
GODOT=/path/to/godot xvfb-run -a wave_forge_godot/render_occlusion.sh      # what occluders cull and cost
GODOT=/path/to/godot xvfb-run -a wave_forge_godot/render_proxies.sh        # modules near, proxies far
```

`prepare.sh`, which both Godot scripts run first, builds the extension and puts what the Godot
project loads next to it ([environment.md](environment.md)). `verify.sh` then runs `verify.gd`,
`verify_stages.gd`, `verify_tables.gd`, `verify_noise.gd`, `verify_edits.gd`, `verify_frozen.gd`,
`verify_assemble.gd`, `verify_scenes.gd`, `verify_cave_scenes.gd`, `verify_pooling.gd`, `verify_ground.gd`, `verify_far.gd`, `verify_volume.gd`, `verify_bake.gd`, `verify_import.gd`, `verify_params.gd`, `verify_pack_data.gd`, `verify_stack.gd`, `verify_paint.gd`, `verify_candidates.gd`, `verify_world.gd`, `verify_continent.gd`, `verify_inspector.gd`, `verify_typed_maps.gd`, `verify_presets.gd`, `verify_walk.gd`, the editor with the plugin, `verify_sound.gd`,
`verify_names.gd`, `verify_occlusion.gd` and `verify_proxies.gd` headless, prepares `examples/history` and `examples/sample_world` and runs each one's `check.gd`. Each exits non-zero on any failure and prints what it generated and how
long frames took. Headless, Godot's renderer is a dummy, so the frame time they check is Godot's own
thread (the extension and the script), not drawing, and the node's own share of it comes from its
`stats()` ([debugging.md](debugging.md)). `verify.gd` walks with a four-tile rule set
(`godot/rules.ron`), because it tests the extension's contract; the city module set (`city.ron`) is
loaded for the tile catalogue, the models and a node that starts from its `rules_file`.

`render_city.sh` renders a generated city with the module models through the Compatibility renderer
and saves the picture: the check to look at after a change to the models, the tile catalogue or the
coordinate mapping. It draws either through the extension's `instance_sets` and `RenderingServer`
(`server`, the default), checking every instance of the first chunk against `tile_basis` and
`cell_position` as the renderer stores it, or through nodes from GDScript (`nodes`), and prints
Godot's cost per chunk. The layout check needs a real renderer: headless, Godot's dummy renderer
stores no multimesh data.

## Known gaps

- **Golden worlds cover tiles, not pictures.** `golden_world.rs` compares a generated city tile for
  tile across devices. The end-to-end tests that render still assert invariants rather than compare
  against a stored image, and `render_city.sh`'s picture is looked at, not compared. The tile
  comparison makes image comparison redundant for generation, but not for the renderers.
- **Kernel internals are tested through whole-region results.** A wrong sweep or a bad checkpoint
  shows up as an invalid or unsolved region, which is a coarse signal; the checkpoint-ring bug that
  `every_reported_success_is_a_valid_chunk` caught is the kind of thing a unit test would have caught
  sooner.
- **The opt-in suites run by hand only.** CI runs no benchmark, streaming suite, game session or
  hole census, and no Bevy test on a device, so a change that can affect them is checked by running
  them ([environment.md](environment.md)).
