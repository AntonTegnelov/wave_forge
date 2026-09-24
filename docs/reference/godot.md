# Godot extension

What `wave_forge_godot` gives a Godot 4.7 project, as built. Why it is shaped this way is in
[engine-integration.md](../architecture/engine-integration.md). Every item below also shows in
Godot's own help and tooltips, generated from the doc comments in `wave_forge_godot/src/`.

The extension adds two nodes. `WaveForgeWorld` streams one WFC world, the infinite city of the MVP.
`WaveForgeStages` generates a pack of stages ([packs.md](packs.md)). Both generate on a thread of
their own, with a GPU device of their own, so nothing on Godot's thread waits for the device and the
extension needs none of godot-rust's thread-safety features.

## WaveForgeWorld

### Properties

| Group | Property | Meaning |
|---|---|---|
| Rules | `rules_file` | a rule file (tile set or module set) to load on start |
| | `start_on_ready` | load `rules_file` and start when the node enters the tree |
| World | `seed` | every choice derives from it |
| | `chunk_cells` | cells per chunk, along the lattice's axes (x and y across the ground, z up) |
| | `cell_size` | one cell in Godot's world units, along Godot's axes |
| | `world_chunks` | the world's size in chunks along each lattice axis; 0 is unbounded |
| Streaming | `view_radius` | chunks kept generated around the followed position |
| | `evict_margin` | chunks further than `view_radius` plus this are dropped |
| Physics | `collider_radius` | chunks around the player that get colliders |
| Navigation | `navigation_radius` | chunks around the player that get navigation meshes |
| | `navigation_template` | the `NavigationMesh` settings chunks are baked with |
| Advanced | `halo` | the first parity's halo, in cells |
| | `warm_kernels` | compile the kernels a run needs when generation starts |

### Functions

- **Starting:** `load_rules(text)`, `start()`, `is_generating()`.
- **The prior:** `set_layer_tiles(layers)`, `ban_tiles_on_face(axis, tiles)`.
- **Streaming:** `follow(position)` asks for the chunks around a position in Godot's world space;
  `generated_chunks()`.
- **Tiles:** `tiles_at(chunk)`, `tile_count()`, `tile_name(tile)`, `tile_rotation(tile)`,
  `tile_basis(tile)`, `tiles_named(name)`, `tiles_tagged(tag)`.
- **Space:** `chunk_at(position)`, `position_of(chunk)`, `cell_position(chunk, cell)`,
  `chunk_size()`.
- **Drawing:** `instance_sets(chunk, names)` gives one dictionary per module with its `name`, its
  `transforms` as a MultiMesh buffer (twelve floats per instance, for
  `RenderingServer.multimesh_set_buffer` on a `TRANSFORM_3D` multimesh), and each instance's stable
  `ids`. An instance is known by its chunk and its id, the same in every run and session.
- **Colliders:** `set_collision_shape(module, shape)` gives every cell of a module a collider, in
  the chunks within `collider_radius`, at most three chunks' bodies per frame, nearest first; `collider_chunks()`; `collider_instance(rid, shape)` maps a
  ray or shape query's hit back to `{chunk, id}`.
- **Navigation:** `navigation_chunks()` lists the chunks whose mesh is in the map. The node bakes
  one region per chunk from the collider shapes and the library's navigation source, off Godot's
  thread.
- **Cost:** `stats()`, below.

### Signals

`chunk_updated(chunk)`, `chunk_evicted(chunk)`, `chunk_failed(chunk, status)`,
`navigation_ready(chunk)`, `generation_failed(reason)`.

### stats()

- **Generation:** `batches`, `solved`, `repaired`, `rewritten_by_repair`, `failed`, `solver_ms`,
  `repair_batches`, `repair_ms`.
- **Waiting:** `pending_colliders`, the chunks within `collider_radius` still waiting for a body.
- **Navigation:** `navigation_baked`, `navigation_polygons`.
- **The slowest frame since the start:** `slowest_frame_ms` in all, `slowest_frame_events` drained
  and `slowest_frame_signals_ms` emitting them (connected handlers included),
  `slowest_frame_colliders` chunks given bodies in `slowest_frame_colliders_ms`, and
  `slowest_frame_navigation_ms`.
- **Recent timings**, each as `_median`, `_p99` and `_max` in milliseconds: `process_ms` (the node's
  own time on Godot's thread per frame), `navigation_bake_ms` (from asking for a bake to its mesh
  being in place), `navigation_start_ms` and `navigation_finish_ms` (Godot's thread preparing a bake
  and putting its mesh in place).

The node times itself because Godot's `Performance.TIME_PROCESS` is the slowest frame of the last
second, published once a second, not the last frame's time.

## WaveForgeStages

### Properties

| Group | Property | Meaning |
|---|---|---|
| Pack | `pack_file` | the pack, a `*.world.ron` file |
| | `rules_files` | the rule sets Solve stages name, as name to rule file path |
| | `targets` | the stages to generate; what they read comes with them |
| | `start_on_ready` | start when the node enters the tree |
| World | `seed` | every choice derives from it |
| | `chunk_cells` | columns per chunk along the lattice's x and y, and a town chunk's height along z |
| | `cell_size` | one cell in Godot's world units |
| Streaming | `view_radius` | chunks kept generated around the followed position |
| Ground | `ground_stage` | the field stage the ground is built from, a height in cells per column; empty for none |
| | `ground_material` | the material the ground is drawn with |
| | `ground_material_stage` | a Rules or Area stage whose categories are the ground's materials ([Ground and colliders](#ground-and-colliders)); empty for none |
| | `ground_palette` | a colour per category of `ground_material_stage`, for the reference ground shader |
| Grass | `grass_stage` | a field stage whose value per column, 0 to 1, is how much of it grass covers ([Grass](#grass)); empty for none |
| | `grass_per_cell` | blades per column where the cover is 1 (default 8) |
| | `grass_radius` | chunks around the followed position that get grass (default 1) |
| | `grass_material` | a `ShaderMaterial` taking the reference grass shader's parameters; empty for that shader |
| Physics | `collider_radius` | chunks around the followed position that get a body; below zero, none |
| Scenes | `scenes` | a kind (a Scatter point's kind or an Assemble piece's name) to a `PackedScene` or a path to one ([Scenes](#scenes)) |
| | `placement_budget_ms` | how long a frame may spend placing scenes (default 2 ms) |
| | `promotion_radius` | chunks around the followed position within which a node scene is placed as nodes; beyond, its first mesh stands in for it (default -1, always nodes) |
| Advanced | `kernel_cache` | where compiled GPU kernels are kept across runs (default `user://wave_forge/kernels`); empty keeps none |

### Functions

- `noises`: a Dictionary of a pack's noise names to `FastNoiseLite` resources; each replaces the
  pack's noise of that name, so a Field reading `FastNoise(name)` holds exactly what the
  resource's `get_noise_2d` gives at each column's centre in cells.
- `remove_point(stage, chunk, id)` takes away a Scatter stage's point by the id `point_sets` gave
  it, and `raise(stage, position, by)` raises a field stage at the column under a position in
  Godot's world space ([packs.md](packs.md#edits)). `edits_log()` gives the player's edits as text
  for a save, and `set_edits_log(text)` restores them. A refused edit is reported as an error,
  returns false and changes nothing.
- `request_save()` asks the stages' thread for a save ([packs.md](packs.md#persistence-and-saves)),
  which arrives as the `saved` signal's text a game writes to disk; `load_save(text)` brings a world
  back from it. A text that is not a save, or holds an edit the pack refuses, is reported as an
  error, returns false and changes nothing.
- `target_radii`: a Dictionary of target stage names to a radius in chunks of their own; the
  other targets keep `view_radius`.
- `start()` loads the pack and the rule sets and starts the stages' thread, where a town solver
  builds its device.
- `follow(position)` generates around a position in Godot's world space, asking again only when it
  enters another chunk.
- `field_values(stage, chunk)`: a field's values, column by column, x fastest.
- `categories(stage, chunk)`: a Rules stage's categories, a byte per column, as indices into
  `category_names(stage)`.
- `sites(stage, chunk)`: the sites overlapping a chunk, each with what names it (its `region` for
  a Sites stage, its `row` for a TableSites stage, its `region` and `index` and its `kind` for a
  Locations stage), footprint `min` and `max` in chunks, and levelled `height`.
- `town(stage, chunk)`: a town's `region` or `row`, `height` and `tiles` in a chunk.
- `town_instance_sets(stage, chunk, names)`: a town chunk's placements in the layout of
  `WaveForgeWorld.instance_sets`, raised to the site's height.
- `point_sets(stage, chunk)`: a Scatter stage's points, one dictionary per kind, with `transforms`
  as a MultiMesh buffer standing on the field, turned, leant and scaled as each point is, and each
  point's `ids`.
- `stamps(stage, chunk)`: an Assemble stage's pieces overlapping a chunk, one dictionary each, with
  the `piece` name, what names its site (as `sites` gives it), its `id`, its `transform` in Godot's
  world space, where a scene of the piece authored at turn 0 with its footprint centred on its
  origin goes, and the cells it covers from `min` to `max` ([packs.md](packs.md#assemble)).
- `set_collision_shape(module, shape)` gives every cell of a town's module a collider in the
  chunks within `collider_radius`, for every Solve stage; `modules_tagged(rules, tag)` names the
  modules of a rule set that carry a tag, to assign shapes by tag.
- `sample(stage, position)` and `atlas(stage, min, size)`: a stage's value at a position on the
  ground plane, and a world map of its own columns, computed on Godot's thread without chunks, for
  Field, Rules, Blur, Delta and Area stages. An error, and NaN or an empty array, for another
  stage.
- `give_table(table, rows)` replaces a given table of facts ([packs.md](packs.md#tables-of-facts)):
  an Array of Dictionaries, typed or not, each with an `id`, a whole number from 0 (a float such as
  JSON reads back is taken if it is whole), and a number or, for a names column, a
  name per column. `focus_row(table, id)` focuses the stages on a row, and `table_rows(table)`
  returns a table's rows in the order of their ids, each with its `id` as a PackedInt64Array and
  its columns, names as names. Give tables and focus a row after `start` and before `follow`, since
  a stage that reads a row with none focused stops generation. A refused give or focus is reported
  as an error, returns false and changes nothing; the stages that read a changed table are
  generated again, with `stage_dropped` and `stage_ready` for their chunks.
- `curves(stage, chunk)`: a Region or TableCurves stage's curves through a chunk, each with what
  names it (its `region` and `index`, or its `row`), `points` on the ground plane in Godot's world
  space, and `values`.
- `ground_chunks()` and `collider_chunks()` list the chunks with ground and with a body.
- `stage_names()`, and `stats()`: `process_ms_median`, `_p99` and `_max`, and what the slowest frame
  since the start spent its time on (`slowest_frame_ms`, `slowest_frame_events` signals emitted in
  `slowest_frame_signals_ms`, `slowest_frame_grounds` built in `slowest_frame_grounds_ms`,
  `slowest_frame_bodies` built in `slowest_frame_bodies_ms`), and `stages`, each stage's cost on
  the stages' thread by name (`products`, `ms`, `slowest_ms`).

### Signals

`stage_ready(stage, chunk)`, `stage_dropped(stage, chunk)`, `generation_failed(reason)`,
`saved(text)`, `instance_spawned(node, chunk, id)`.

At most 256 `stage_ready` and `stage_dropped` signals are emitted per frame, in the order the
products arrived (nearest first), so after a wide request some come a few frames later; by then a
product can have been dropped again, and its `stage_dropped` follows. Ground is built for at most
8 chunks per frame, and bodies for at most 3, nearest the player first. `stats()` reports what
waits as `pending_signals`, `pending_grounds`, `pending_colliders` and `pending_placements`, and
what is placed as `placed_nodes` and `placed_instances`.

### Scenes

`scenes` binds a kind to a scene: a Scatter point's kind, or an Assemble piece's name, so a
dungeon's rooms bind the same way as trees. A value is a `PackedScene`, or a path the node loads on
Godot's loader threads when it starts; placing waits until every scene has loaded. A scene holding
another extension's Rust resource has to be given as a `PackedScene`, since such a resource aborts
the process when loaded on a loader thread
([engine-integration.md](../architecture/engine-integration.md#godot-tiers-of-disclosure)).

Each scene is drawn one of two ways, chosen when it is bound:

- **A lone mesh:** a scene whose root is a `MeshInstance3D` without children or a script is drawn as
  one `RenderingServer` MultiMesh per chunk and kind, never as nodes.
- **Nodes:** any other scene is instantiated as nodes under the `WaveForgeStages` node, at the
  point's or piece's transform, and `instance_spawned(node, chunk, id)` names each, with the id
  `point_sets` and `stamps` give it. A piece overlapping several chunks is placed once, by the chunk
  holding its footprint's centre.

With `promotion_radius` at zero or more, a scene placed as nodes is so only in chunks within that
many chunks of the followed position. Farther out it is drawn as a MultiMesh of its first mesh,
depth first, where that mesh sits in the scene, or not at all if it has none, which suits a spawner
that should act only near the player. A chunk that crosses the radius as the player moves is placed
again, under the same budget, and `instance_spawned` names its nodes again.

Chunks are placed nearest the followed position first, each whole, until `placement_budget_ms` is
spent, so a frame can go over by one chunk's placing. Everything a chunk placed is freed when the
chunk is dropped; a point removed by an edit is gone from its chunk, so its scene is never placed
again. Every chunk the stages hold is placed, those generated beyond the view for a stage that reads
them included.

### Ground and colliders

With `ground_stage` set, the node builds each chunk's ground once the fields of the chunk and the
eight around it have arrived ([packs.md](packs.md#ground)): a mesh through the `RenderingServer`,
drawn with `ground_material`. Within `collider_radius` of the followed chunk, each chunk gets one
static body holding its ground as a `HeightMapShape3D` and its towns' modules as the shapes
`set_collision_shape` assigned, every shape added before the body joins the space. A height map's
samples are one unit apart, so it is scaled by the cell's width, which needs cells as wide as they
are deep. Ground and bodies go when their chunk's field is dropped or the player moves away.

With `ground_material_stage` set, a chunk's ground also waits for that stage's categories of
itself and of the chunks beyond its far edges ([packs.md](packs.md#ground)), and gets a copy of
its material of its own. The copy takes three shader parameters: `wave_forge_materials`, a texture
of one texel per ground vertex holding its category in the red channel as id / 255;
`wave_forge_cell`, the cell's width along x and z; and `wave_forge_palette`, a 256 by 1 texture of
a colour per category from `ground_palette`, categories past its end taking colours of their own.
With `ground_material` empty, the copy is of the reference ground shader, embedded in the extension,
which blends the colours of the four vertices around every fragment, so materials meet in a smooth
band a cell wide on every renderer, Compatibility included. `ground_shader_code()` gives its code,
to start a game's own shader from; a game's `ground_material` has to be a `ShaderMaterial` taking
the same parameters. `ground_material_of(chunk)` gives a chunk's copy. Loading refuses a
`ground_material_stage` that is no Rules or Area stage, and a `ground_material` that is no
`ShaderMaterial` beside it.

### Grass

With `grass_stage` set, every chunk within `grass_radius` of the followed position whose ground is
built gets grass, a few chunks a frame, nearest first, and loses it when it leaves the radius. All
chunks share one MultiMesh of `grass_per_cell` blades per column, each with the identity transform,
so its buffer is uploaded once. Each chunk draws it as an instance of its own, with a copy of the
grass material holding `wave_forge_cover` (the chunk's cover per column, one byte each),
`wave_forge_heights` (the ground's height per vertex, one float each), `wave_forge_cell`,
`wave_forge_per_column` and `wave_forge_chunk`. The reference grass shader, embedded in the
extension (`grass_shader_code()` gives it), gives blade `INSTANCE_ID` its column, a hashed place in
it, a hashed turn and height, and shows it only where a hash is below the column's cover; it stands
on the ground's heights, blended between vertices, and sways in the global shader parameter
`wave_forge_wind` (a direction along x and z, a strength at the tip, a speed), which the node
registers blowing gently along +x unless the project's settings declare it. Grass casts no shadow,
takes no GI, and is bounded by its chunk rather than by its blades, which the shader moves. It
draws on every renderer, Compatibility included. `grass_chunks()` lists the chunks with grass and
`grass_material_of(chunk)` gives a chunk's material. Loading refuses a `grass_stage` the pack does
not have, or grass without `ground_stage`. What it costs is in
[measurements.md](../research/measurements.md) (E45).

### Wind

`wave_forge_wind` is a global shader parameter, a `vec4`: a direction along x and z, a strength
in world units, and a speed. The extension registers it as it loads, before any scene, blowing
gently along +x, unless the project's settings declare it (Project Settings, Shader Globals); a game
changes it with `RenderingServer.global_shader_parameter_set`. The grass shader reads it, and so does
the reference vegetation shader, which `vegetation_shader_code()` gives: on a plant's mesh bound in
`scenes`, it bends the plant downwind, more the higher up a vertex is (up to `bend_height`) and the
less stiff the plant, keeping each vertex's distance from the root, and flutters its outer parts
along their normals, after GPU Gems 3, chapter 16. Every MultiMesh the node places carries each
instance's custom data: its phase in the wind, as a fraction of a turn hashed from its id, and its
stiffness, one over a point's scale, so a larger plant bends less and no two move alike.

## Checking it

`wave_forge_godot/verify.sh` builds the extension and runs `godot/verify.gd` (the WFC world, with
colliders and navigation), `godot/verify_stages.gd` (the valley pack, its ground, and a walk
through a town), `godot/verify_tables.gd` (tables of facts given from GDScript) and
`godot/verify_noise.gd` (a `FastNoiseLite` resource read through a pack), `godot/verify_edits.gd`
(a felled tree and raised ground through an edits log and a save, and cut grass growing back) and
`godot/verify_assemble.gd` (a village's pieces placed by their transforms) and
`godot/verify_scenes.gd` (scenes bound to trees and pieces, as MultiMeshes and as nodes) and
`godot/verify_ground.gd` (the ground's materials per category, and grass) in a real headless
Godot. `render_ground.sh` renders the ground's materials and grass to pictures, to look at, times
grass, and checks that trees drawn with the vegetation shader move in the wind and stand still
without it.
How to run it in the dev container and in CI is in [environment.md](../guides/environment.md), and
what the checks assert is in [testing.md](../guides/testing.md).
