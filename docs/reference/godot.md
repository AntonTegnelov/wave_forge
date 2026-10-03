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
| Audio | `audio_radius` | chunks around the player that sound ([Sound and surfaces](#sound-and-surfaces)); below zero, none |
| | `sounds` | the stream each sound key plays, as key to `AudioStream` |
| | `interior_reverb_bus` | the audio bus interiors reverb the sounds inside them on; empty for none |
| | `interior_audio_bus` | the audio bus sounds inside interiors play on instead of their own; empty for none, and with both empty, no interiors |
| Occlusion | `occluder_radius` | chunks around the player that get occluders of their solid cells ([Occlusion](#occlusion)); below zero, none |
| Far | `proxy_distance` | from this distance on, each generated chunk is drawn as its far proxy ([Far proxies](#far-proxies)); below zero, never |
| | `proxy_colours` | each module's colour seen from afar, as module name to `Color`; a module without one is left out |
| Advanced | `halo` | the first parity's halo, in cells |
| | `warm_kernels` | compile the kernels a run needs when generation starts |
| | `frozen_directory` | freezes the world: where the chunks it evicts are kept as they are, a file each, and come back from rather than being generated, even after the rule set changes ([world.md](../architecture/world.md#regenerating-exactly)); `user://` paths are resolved, and the game keeps the directory with its saves. Empty generates an evicted chunk again, tile for tile |

### Functions

- **Starting:** `load_rules(text)`, `start()`, `is_generating()`.
- **A kit's rules:** `WaveForgeWorld.propose_module_set(library, cell_size)`, a static function,
  proposes a module set from a `MeshLibrary`: a module per item, whose faces get connectors from the
  shapes of the item's mesh on them, taken as a `GridMap` centres it in cells of `cell_size`
  ([constraints.md](../architecture/constraints.md#what-adjacency-can-express)). It returns the set
  as text for `load_rules`, for an artist to confirm, rename and mark walkable first.
  `kit_connectors(library, cell_size)` lists the connectors it finds, each with its `name`
  (`"side n"` or `"top n"`), whether it joins tops (`top`) and the `modules` that have it, and
  `name_module_set(library, cell_size, names, walkable)` writes the set with the connectors `names`
  maps under their new names and walkable sides on the connectors `walkable` lists; a naming that
  would lose or merge connectors is refused with an error and an empty result.
  `mesh_library_from_scenes(directory)` makes a `MeshLibrary` of a kit that comes as a folder of
  scenes: an item per `.tscn`, `.scn`, `.glb` or `.gltf` file, named as the file, whose mesh is
  every `MeshInstance3D` of the scene merged where the scene places them. The dock's kit import does
  all three ([Editor](#editor)).
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
- **Sound and surfaces:** `surface_at(position)`, `region_tags(chunk)`
  ([Sound and surfaces](#sound-and-surfaces)).
- **Occlusion:** `occluders(chunk)` ([Occlusion](#occlusion)).
- **Far proxies:** `proxy_instance(chunk)`, `proxy_chunks()` ([Far proxies](#far-proxies)).
- **Cost:** `stats()`, below.

### Signals

`chunk_updated(chunk)`, `chunk_evicted(chunk)`, `chunk_failed(chunk, status)`,
`navigation_ready(chunk)`, `generation_failed(reason)`.

### stats()

- **Generation:** `batches`, `solved`, `repaired`, `rewritten_by_repair`, `failed`, `solver_ms`,
  `repair_batches`, `repair_ms`, `replayed`, repairs run again to rewrite a chunk generated again,
  and `restored`, chunks a frozen world brought back from its directory
  ([world.md](../architecture/world.md#regenerating-exactly)).
- **Waiting:** `pending_colliders`, the chunks within `collider_radius` still waiting for a body.
- **Navigation:** `navigation_baked`, `navigation_polygons`.
- **The slowest frame since the start:** `slowest_frame_ms` in all, `slowest_frame_events` drained
  and `slowest_frame_signals_ms` emitting them (connected handlers included),
  `slowest_frame_colliders` chunks given bodies in `slowest_frame_colliders_ms`, and
  `slowest_frame_navigation_ms`.
- **Recent timings**, each as `_median`, `_p99` and `_max` in milliseconds: `process_ms` (the node's
  own time on Godot's thread per frame), `navigation_bake_ms` (from asking for a bake to its mesh
  being in place), `navigation_start_ms` and `navigation_finish_ms` (Godot's thread preparing a bake
  and putting its mesh in place), and `navigation_source_ms` (the part of a start that gathers the
  source triangles, the rest being their handover to Godot).

The node times itself because Godot's `Performance.TIME_PROCESS` is the slowest frame of the last
second, published once a second, not the last frame's time.

### Sound and surfaces

A module set's modules may say what walkers in their cell stand on (`surface`), that the cell is
inside (`indoor`) and where sounds play in it (`sounds`); the city example says all three
(`examples/city.ron`, and the format in `wfc-rules/src/formats/module_format.rs`).
`surface_at(position)` gives the surface of the module in the cell holding a position, or an empty
string, which is what a game's footsteps ask. `region_tags(chunk)` gives a generated chunk's
`interiors`, an `AABB` per box of indoor cells, the boxes together covering each indoor cell once,
and its `emitters`, a dictionary per sound with its `position` in Godot's world and its `key`.

Within `audio_radius` of the followed chunk, the node gives each chunk's interiors an `Area3D`, on
collision layer 1, which its emitters' `area_mask` holds, that reverbs the sounds inside it on
`interior_reverb_bus` and plays them on `interior_audio_bus` (a muffled indoor mix, say), each when
set, and each emitter whose key `sounds` maps an `AudioStreamPlayer3D`, playing,
all as children of the node; a key `sounds` does not map plays nothing, with a warning once per key.
At most three chunks get their sound per frame, nearest first, and again when their tiles change.
Since Godot 4.7 an `AudioStreamPlayer3D`'s default `area_mask` is empty, so a game's own players
need layer 1 in theirs for the interiors to reverb them. A chunk leaving the radius frees its areas and stops its players, which a pool keeps for the next
chunk, and the node stops them all when it leaves the tree. A game that wants its own mapping
leaves `audio_radius` below zero and reads `region_tags`.

### Occlusion

A module set's modules may say they fill their cell with opaque geometry (`solid`), which the city
example says of its closed buildings. `occluders(chunk)` gives a generated chunk's solid cells as
an `AABB` per box, the boxes together covering each solid cell once. Within `occluder_radius` of
the followed chunk, the node gives each chunk with a solid cell one `OccluderInstance3D` holding an
`ArrayOccluder3D` of its boxes, a child of the node, at most three chunks a frame, nearest first,
again when its tiles change, and frees it when the chunk leaves the radius. Godot's occlusion
culling uses them once the viewport's `use_occlusion_culling` is on.

Whether they pay depends on the view and on how the game draws
([measurements.md](../research/measurements.md) E50). From a street they hid half of what the
city draws and saved several milliseconds; from above the roofs they hid under 1%, because a
chunk's MultiMesh of a module is culled only when all of it is hidden, and culling cost more than it
saved. Rebuilding three chunks' occluders costs Godot a millisecond or more on the frames after the
player crosses into a chunk. `occluder_radius` is below zero until a game turns it on.

### Far proxies

With `proxy_distance` at zero or more, the node draws each generated chunk as one mesh of boxes
coloured by `proxy_colours`, the library's `proxy_mesh`: a box per cell of a coloured module, and coarser levels of boxes standing for two,
four and more cells a side as the surface's `lods`. It shows from `proxy_distance` on, as a
visibility range with no end, takes static global illumination and casts no shadow, and is built
at most three chunks a frame, nearest first, again when the chunk's tiles change, and freed when
the chunk is dropped or `proxy_distance` goes below zero.

A far chunk then costs one draw call where its modules cost one per module. The game draws the
chunk near the player as before and hands it over by giving each of its instances the chunk's
proxy as parent: `RenderingServer.instance_set_visibility_parent(instance,
world.proxy_instance(chunk))`, or a node's `visibility_parent`. The game's drawing of the chunk then
hides wherever its proxy shows, and the switch is exact per chunk. `proxy_instance(chunk)` is an
invalid RID until the chunk's proxy is built, so a game sets the parent once it is, and a chunk
with no coloured module has none. `proxy_chunks()` lists the chunks given their proxy.

## WaveForgeStages

### Properties

| Group | Property | Meaning |
|---|---|---|
| Pack | `pack_file` | the pack, a `*.world.ron` file |
| | `rules_files` | the rule sets Solve stages name, as name to rule file path |
| | `targets` | the stages to generate; what they read comes with them |
| | `params/<name>` | a slider per parameter of the pack over its range, listed from `pack_file`, which the scene saves, `start` gives the stages and moving it changes them while they run; reverting gives the pack's default ([packs.md](packs.md#parameters)). From code, `params` is the same values as name to number |
| | `start_on_ready` | start when the node enters the tree, when the game runs |
| | `preview_in_editor` | start in the editor too, as a preview the editor plugin follows with the editor's camera and brushes paint on ([Editor](#editor)) |
| | `edits_text` | the edits of the world as text, `edits_log`'s, which the scene saves and `start` applies: what brushes painted in the editor |
| World | `seed` | every choice derives from it |
| | `chunk_cells` | columns per chunk along the lattice's x and y, and a town chunk's height along z |
| | `cell_size` | one cell in Godot's world units |
| Streaming | `view_radius` | chunks kept generated around the followed position |
| Ground | `ground_stage` | the field stage the ground is built from, a height in cells per column; empty for none |
| | `ground_material` | the material the ground is drawn with |
| | `ground_material_stage` | a Rules, Area or Nearest stage whose categories are the ground's materials ([Ground and colliders](#ground-and-colliders)); empty for none |
| | `ground_palette` | a colour per category of `ground_material_stage`, for the reference ground shader |
| | `sea_material` | the material the pack's sea is drawn with: a plane at the pack's water level ([packs.md](packs.md#water)) under the followed chunk, as wide as the view; empty, or a pack without water, draws none. Lakes above the sea level are not drawn |
| | `far_ground_stage` | a coarse field stage the far ground beyond the ground is drawn from ([packs.md](packs.md#far-ground)), with `ground_material`; give it a radius of its own in `target_radii`, as far as the ground should reach. Empty for none |
| Volume | `volume_stage` | a Volume or Carve stage at scale 1 whose surface is drawn and collided with, for overhangs and caves ([packs.md](packs.md#volume)); empty for none |
| | `volume_material` | the material the volume's surface is drawn with; empty for Godot's default, or for a stage with materials one that takes its albedo from the vertices' colours |
| | `volume_budget_ms` | how long a frame may spend drawing volume surfaces, which are meshed on a thread of their own; one is drawn a frame whatever it costs (default 2 ms) |
| | `volume_palette` | a colour per material of `volume_stage`, by index, which each vertex of the surface carries; materials past its end take colours of their own from their index |
| | `fluid_stage` | a Volume, Carve or Aquifer stage at scale 1 whose surface is drawn as fluid and never collided with, the water and lava of an [Aquifer](packs.md#aquifer) stage say; meshed and drawn as the volume's, within the same `volume_budget_ms`; empty for none |
| | `fluid_material` | the material the fluid's surface is drawn with; empty for one that takes its albedo and opacity from the vertices' colours, seen from both sides |
| | `fluid_palette` | a colour per material of `fluid_stage`, by index, its alpha the fluid's opacity, as `volume_palette` is for the volume |
| | `fluid_glow` | how brightly each material of `fluid_stage` glows, by index, as a multiple of its colour, lava's say; materials past its end do not glow. The reference fluid shader reads it from each vertex's first UV (`fluid_shader_code()`) |
| Grass | `grass_stage` | a field stage whose value per column, 0 to 1, is how much of it grass covers ([Grass](#grass)); empty for none |
| | `grass_per_cell` | blades per column where the cover is 1 (default 8) |
| | `grass_radius` | chunks around the followed position that get grass (default 1) |
| | `grass_material` | a `ShaderMaterial` taking the reference grass shader's parameters; empty for that shader |
| Physics | `collider_radius` | chunks around the followed position that get a body; below zero, none |
| Navigation | `navigation_radius` | chunks around the followed position that get a navigation region ([Navigation](#navigation)); below zero, none (the default) |
| | `navigation_template` | the `NavigationMesh` settings chunks are baked with |
| Occlusion | `occluder_radius` | chunks around the followed position whose towns get occluders of their solid cells, for Godot's occlusion culling; below zero, none (the default) |
| Scenes | `scenes` | a kind (a Scatter, Embed, Deposit or Spawn point's kind, or an Assemble piece's or Cave room's name) to a `PackedScene` or a path to one ([Scenes](#scenes)) |
| | `placement_budget_ms` | how long a frame may spend placing scenes (default 2 ms) |
| | `promotion_radius` | chunks around the followed position within which a node scene is placed as nodes; beyond, its first mesh stands in for it (default -1, always nodes) |
| Advanced | `kernel_cache` | where compiled GPU kernels are kept across runs (default `user://wave_forge/kernels`); empty keeps none |
| | `frozen_directory` | where frozen stages' chunks the request no longer needs are kept, a file each, so they leave memory and come back unchanged ([packs.md](packs.md#persistence-and-saves)); `user://` paths are resolved, and the game keeps the directory with its saves. Empty keeps every frozen chunk in memory, and in the save |
| | `play_directory` | a directory `run_world` wrote the pack's whole world to, which the node plays instead of generating: its targets' chunks come from there, and no stage runs ([packs.md](packs.md#a-whole-world-ahead-of-time)); empty generates as usual. The stages read it on their own thread, where Godot's file API cannot be called, so it has to be a directory in the file system: in an exported game a `res://` directory is looked for beside the executable, and a game ships the run's directory there, outside the exported pack |
| Debug | `candidates_stage` | a Scatter stage whose candidates are drawn as small boxes over the ground, each coloured by what became of it, kept or the modifier that rejected it ([packs.md](packs.md#scatter)); empty draws none |

### Functions

- `noises`: a Dictionary of a pack's noise names to `FastNoiseLite` resources; each replaces the
  pack's noise of that name, so a Field reading `FastNoise(name)` holds exactly what the
  resource's `get_noise_2d` gives at each column's centre in cells.
- `update_params(values)` sets the pack's parameters named in a Dictionary of names to numbers
  while the stages run, generating again only what reads a changed one; `pack_params()` lists them,
  each with its `name`, `default`, `min`, `max` and `value` now ([packs.md](packs.md#parameters)).
- `remove_point(stage, chunk, id)` takes away a Scatter stage's point by the id `point_sets` gave
  it, `raise(stage, position, by)` raises a field stage at the column under a position in
  Godot's world space, and `dig(stage, position, radius)` and `fill(stage, position, radius)` dig a
  ball of `radius` cells out of a Volume or Carve stage around a position or fill one in
  ([packs.md](packs.md#edits)); every volume surface that reads a chunk an edit changed is built
  again, its body too. `edits_log()` gives the player's edits as text
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
  Locations stage), footprint `min` and `max` in chunks, and levelled `height`. A location's name
  is its `name_key` with its `name_args` ([packs.md](packs.md#locations)), which a game turns into
  words with `tr(site.name_key, "wave_forge").format(site.name_args)` and a translation of that key
  in the `wave_forge` context.
- `water()`: the pack's water ([packs.md](packs.md#water)), its `level` in cells of height, or an
  empty dictionary if the pack declares none.
- `occluder_chunks()` and `town_occluders(chunk)`: the chunks within `occluder_radius` that have
  an `OccluderInstance3D` of their towns' solid cells, as `WaveForgeWorld` gives its city's
  ([Occlusion](#occlusion)), and a chunk's boxes as `AABB`s, each town's raised to its site; a few
  chunks are built a frame, nearest first, and again when a town arrives anew.
- `town(stage, chunk)`: a town's `region` or `row`, `height`, `rules` (the rule set it was solved
  with, whose modules its `tiles` index) and `tiles` in a chunk. `town_instance_sets`, colliders
  and navigation read a town's tiles with that rule set.
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
- `ground_height(position)`: the height of the ground's surface at full detail above a position,
  from `ground_stage`'s fields ([packs.md](packs.md#ground)), what a game stands a player or an
  object on; NaN until the fields around it have arrived.
- `sample(stage, position)` and `atlas(stage, min, size)`: a stage's value at a position on the
  ground plane, and a world map of its own columns, computed on Godot's thread without chunks, for
  Field, Rules, Nearest, Blur, Delta and Area stages. An error, and NaN or an empty array, for another
  stage.
- `locate(stage, position, within)`: the site of a Sites stage nearest a position, as `sites`
  gives a site, found on Godot's thread without chunks ([packs.md](packs.md#sites)); empty if none
  lies within `within` regions.
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
- `ground_mesh_of(chunk)` is the `RenderingServer` mesh a chunk's ground is drawn with. A chunk's
  ground reads the fields and materials of the chunks around it, so when one of them is dropped and
  generated again, after a raise say, every ground that reads it is built again, its grass and
  body too.
- `ground_chunks()` and `collider_chunks()` list the chunks with ground and with a body,
  `far_ground_chunks()` the chunks of `far_ground_stage` whose far ground is drawn, and
  `volume_chunks()` the chunks of `volume_stage` whose surface is built. `volume_surface(chunk)`
  gives one: its `positions`, `normals`, `indices` in Godot's winding and each vertex's material in
  `materials`, relative to the chunk's corner; `volume_mesh_of(chunk)` is the `RenderingServer` mesh
  it is drawn with. A chunk's surface is drawn with `volume_material`, and within `collider_radius` its body holds it as a
  `ConcavePolygonShape3D`, whose front faces, the ones that collide, face from solid to empty:
  a ray from inside a cave meets its ceiling as it meets the ground from the sky.
  `fluid_chunks()`, `fluid_surface(chunk)` and `fluid_mesh_of(chunk)` give the same of
  `fluid_stage`, which has no body, and `fluid_shader_code()` the reference fluid shader's code.
- `bake(from, to)` returns a `PackedScene` of plain nodes holding what the node draws over the
  chunks from `from` to `to`, both included, for a game to save as `.tscn` and open without the
  extension. It has a node per chunk (`Chunk x y`) holding:
  - its ground (`Ground`, with its levels of detail and material) and a `GroundBody` with its
    `HeightMapShape3D`;
  - its volume surface (`Surface`) and a `SurfaceBody` with its `ConcavePolygonShape3D`;
  - its fluid (`Fluid`);
  - every bound scene its stages placed: a kind drawn as a MultiMesh as a `MultiMeshInstance3D`,
    any other as an instance of its scene, which the saved file refers to by path if the scene has
    one.

  Grass and the far ground are left out. It returns null, with an error, while a chunk's ground or
  surfaces are not built, or while a scene is still loading. Its nodes carry metadata a linked
  bake reads, all plain Godot values:
  - a chunk's node has `wave_forge_chunk` and the list of its points in `wave_forge_points`;
  - a point placed as a node has `wave_forge_point`, its stage, the chunk and id of its positional
    id, where it stood and its transform;
  - everything else the generator made has `wave_forge_generated`.
- `candidate_legend()` gives what the drawn candidates of `candidates_stage` came to, over every
  chunk drawn: each verdict's `name` (`"kept"`, or `"chance"`, `"height"`, `"slope"`,
  `"condition n"`, `"water"`, `"sites"`, `"blocked"` or `"spacing"`), the `colour` it is drawn in
  and its `count`; `candidate_chunks()` lists the chunks drawn. The node asks the stages' thread
  for each chunk's `scatter_report` as the chunk arrives. `candidate_near(position, radius)` gives
  the drawn candidate nearest a point in world space, no farther than `radius` along the ground:
  its `verdict` and `colour`, its `position`, and what the stage's modifiers read there, `height`,
  `slope` and `water_depth` when the stage has those modifiers, and `conditions`, each with the
  `value` it tests and whether it `holds` ([packs.md](packs.md#scatter)); empty when none lies that
  near. `candidate_under(from, along)` gives the same for the candidate under a ray, within a cell
  of where the ray first meets the ground the candidates stand on, the Scatter stage's height
  field, whatever `ground_stage` is.
- `run_world(directory)` generates the pack's whole finite world ahead of time on a thread of its
  own, the node's `targets` over every chunk of the bound, keeping each chunk's products under
  `directory` as it is done; `world_run_progress(done, total, stages)` reports each chunk, and
  every second while a block generates, with what each stage has generated so far in the form of
  `stats()["stages"]`, `world_run_finished(done, total)` the end, and `cancel_world_run()` stops
  it within about a second, which running again resumes. The process waits for a run under way,
  and for the node's generating thread, before it exits. It runs as the node generates, with the tables given, the edits made and the parameters set since it started, so the node has to have started; in the editor,
  with `preview_in_editor`, it is the bake of M1. A node whose `play_directory` is that directory
  then plays the world with nothing generated.
- `keep_bake_edits(baked)` turns what a designer changed in a bake, as instanced, into edits:
  - a point node moved or turned is moved there with `Edit::Move`;
  - a point node deleted is removed.

  The world is generated again with them. Pieces, the ground, surfaces and MultiMeshes are the
  generator's, and changes to them are not carried. `bake_keeping(from, to, old)` then bakes again,
  copying every node the designer added under a chunk of `old`, so a linked bake is regenerated
  with the designer's edits kept.
- `stage_names()`, and `stats()`: `process_ms_median`, `_p99` and `_max`, and what the slowest frame
  since the start spent its time on (`slowest_frame_ms`, `slowest_frame_events` signals emitted in
  `slowest_frame_signals_ms`, `slowest_frame_grounds` built in `slowest_frame_grounds_ms`,
  `slowest_frame_bodies` built in `slowest_frame_bodies_ms`, `slowest_frame_navigation_ms` keeping
  navigation regions), `navigation_baked`, the navigation bakes gone into their regions, and
  `stages`, each stage's cost on
  the stages' thread by name (`products`, `ms`, `slowest_ms`); `last_frame_ms`, the node's own time
  in its last frame; and `volume_surfaces` and `volume_surfaces_ms`, the surfaces drawn so far and
  the milliseconds drawing them took on Godot's thread, fluid included; `pending_volumes` counts
  those due, those being meshed and those meshed and waiting to be drawn.

### Signals

`stage_ready(stage, chunk)`, `stage_dropped(stage, chunk)`, `generation_failed(reason)`,
`saved(text)`, `instance_spawned(node, chunk, id)`, `navigation_ready(chunk)`.

At most 256 `stage_ready` and `stage_dropped` signals are emitted per frame, in the order the
products arrived (nearest first), so after a wide request some come a few frames later; by then a
product can have been dropped again, and its `stage_dropped` follows. Ground is built for at most
8 chunks per frame, volume surfaces for `volume_budget_ms`, far ground for at most 4 coarse chunks,
again when ground comes or goes on or beside one, and bodies for at most 3 and 2 ms, since a
volume's collider takes milliseconds ([measurements.md](../research/measurements.md) E53), nearest
the player first, each at least one a frame. `stats()` reports what waits as `pending_signals`,
`pending_grounds`, `pending_volumes`,
`pending_far_grounds`, `pending_colliders` and
`pending_placements`, and
what is placed as `placed_nodes` and `placed_instances`, and the nodes of pooled scenes waiting to
be placed again as `pooled_nodes` ([Scenes](#scenes)).

### Editor

Both nodes are tool classes and show configuration warnings in the editor's scene tree, refreshed
about twice a second, for settings that would leave the world dark, without bodies, occluders or
sound, or that name no fitting stage; `configuration_warnings()` gives the same list to a script.
`WaveForgeStages` warns when:
- the scene has no light, or no `WorldEnvironment` and no default environment (the editor lights its
  viewport with a preview sun and sky that a running game has not);
- `pack_file` is empty or not a pack;
- a target names no stage, or a stage setting (`ground_stage`, `grass_stage`, `candidates_stage`,
  `ground_material_stage`, `far_ground_stage`, `fluid_stage`, `volume_stage`) names no stage of
  the kind it needs, which `start` refuses with the same words.

`WaveForgeStages`' inspector has three buttons: Start or regenerate, the same as `start`; Reroll
seed, which takes a new `seed` at random and starts again if the node is running; and Bake the view,
which bakes the chunks within `view_radius` of the followed one, as `bake` does, into a scene saved
at `bake_path` (`res://wave_forge_bake.tscn` unless changed).

`WaveForgeWorld` warns when `start_on_ready` is set with no `rules_file`, or when an interior bus it
names is not in the project's bus layout. Both warn when `collider_radius` builds bodies while the
project's 3D physics is not Jolt, and when `occluder_radius` builds occluders while occlusion culling
is off. `WaveForgeWorld` never starts in the editor.

`WaveForgeStages` is a tool class, so it runs in the editor where `preview_in_editor` is on,
generating around the editor's camera. The editor plugin (`addons/wave_forge`, enabled in the
project's plugins) adds a Wave Forge dock:
- a Preset list of the packs shipped in `addons/wave_forge/presets`, which makes the chosen one
  the selected node's pack as one undo action, its parameters then showing as sliders;
- a Paint toggle;
- a brush (Raise, Lower, Smooth, Dig, Fill or Remove);
- the stage it paints, or the point stages Remove takes from;
- a radius in cells, and a strength;
- with Paint off, the candidate of the node's `candidates_stage` under the mouse, what became of it
  and what its stage's modifiers read there, as `candidate_under` gives it for the ray under the
  mouse; the panel is `addons/wave_forge/candidate_panel.gd`, which only renders
  that Dictionary;
- a Kit import: a `MeshLibrary` path, or a folder of scenes, and a cell size, which list the connectors the library's meshes
  propose, each with a field for a new name and, for a side, a walkable tick, and a path the named
  set is saved to as a rule file (`res://kit.ron` unless changed); the panel is
  `addons/wave_forge/kit_import_panel.gd`, which calls `mesh_library_from_scenes`,
  `kit_connectors` and `name_module_set`;
- a World run: a directory (`res://wave_forge_world` unless changed, so the result ships with the
  game), Run world and Cancel buttons, a progress bar of chunks, and what each stage has generated
  so far or how the run ended. Run world is the node's `run_world`, so the node has to have started,
  with `preview_in_editor` on; running again after a cancel resumes. The panel is
  `addons/wave_forge/world_run_panel.gd`, which only renders the node's `world_run_progress` and
  `world_run_finished`.

The project's `continent.tscn` is the maximal preset set up to bake this way
([M1](../product/user-stories.md#m1-bake-a-maximal-world-in-the-editor)). Its `WaveForgeStages` node
holds the continent's pack, its eight cultures as `rules_files` and the 81 targets a game draws, and
the scene's script (`continent.gd`) gives the node the history a game would simulate,
`continent/history.json`, as its `settlements` table whenever it has started without one.
`prepare.sh` copies the pack, the cultures and the history into `continent/`.

With Paint on and the node selected, a drag in the 3D viewport paints a stroke along the ground,
as one undo action that restores `edits_text`. The ground comes from `ground_height`, so painting
needs `ground_stage`. What a stroke does is the node's own `paint(brush, path)`:
- `brush` is a Dictionary with `brush` (`"raise"`, `"smooth"`, `"dig"`, `"fill"` or `"remove"`),
  `stage` or `stages`, `radius` in cells, and `strength` for raise and smooth;
- `path` holds points in Godot's world space.

It adds the edits `stages::brushes::stroke` gives ([packs.md](packs.md#edits)), so a game's own
tools paint the same way.

### Scenes

`scenes` binds a kind to a scene: a Scatter, Embed, Deposit or Spawn point's kind, or an Assemble
piece's or Cave room's name, so a dungeon's rooms bind the same way as trees. A value is a
`PackedScene`, or a path the node loads on
Godot's loader threads when it starts; placing waits until every scene has loaded. A scene holding
another extension's Rust resource has to be given as a `PackedScene`, since such a resource aborts
the process when loaded on a loader thread
([engine-integration.md](../architecture/engine-integration.md#godot-tiers-of-disclosure)).

Each scene is drawn one of two ways, chosen when it is bound:

- **A lone mesh:** a scene whose root is a `MeshInstance3D` without children or a script is drawn as
  one `RenderingServer` MultiMesh per chunk and kind, never as nodes. A point's MultiMesh takes no
  global illumination, as a prop's; an Assemble piece's or Cave room's is static, as a building's, so
  SDFGI and baked lighting take it in ([Global illumination](#global-illumination)).
- **Nodes:** any other scene is instantiated as nodes under the `WaveForgeStages` node, at the
  point's or piece's transform, and `instance_spawned(node, chunk, id)` names each, with the id
  `point_sets` and `stamps` give it. A piece overlapping several chunks is placed once, by the chunk
  holding its footprint's centre.

With `promotion_radius` at zero or more, a scene placed as nodes is so only in chunks within that
many chunks of the followed position. Farther out it is drawn as a MultiMesh of its first mesh,
depth first, where that mesh sits in the scene, or not at all if it has none, which suits a spawner
that should act only near the player. A chunk that crosses the radius as the player moves is placed
again, under the same budget, and `instance_spawned` names its nodes again.

**Pooling.** A scene whose root script defines `_wave_forge_reset()` is pooled. When its chunk is
dropped or crosses the promotion radius, each of its nodes is reset by that method, while still in
the tree, then taken out of the tree and kept for the next placement of the same kind, which moves
it and announces it with `instance_spawned` as if it were new. A scene without the method is always
instantiated fresh, so a node's script state (a chest opened, a timer running) never carries into
another place unless the scene clears it itself. A pool only holds nodes that were placed at the
same time before, so it never outgrows the most its kind has had placed; the pools are freed with
the node. A node the game frees itself is forgotten, never reused. Reusing a node is cheaper than
instantiating one, though placing either one is cheap next to the placing budget
([measurements.md](../research/measurements.md), E44 and E51).

Chunks are placed nearest the followed position first, each whole, until `placement_budget_ms` is
spent, so a frame can go over by one chunk's placing. Everything a chunk placed is freed when the
chunk is dropped; a point removed by an edit is gone from its chunk, so its scene is never placed
again. Every chunk the stages hold is placed, those generated beyond the view for a stage that reads
them included.

### Global illumination

What the node draws itself takes the global illumination of its content class, as
`GeometryInstance3D`'s `gi_mode` would set it
([engine-integration.md](../architecture/engine-integration.md#content-classes)): the ground and
Assemble pieces drawn as MultiMeshes are static, so SDFGI and baked lighting take them in; grass
and Scatter points drawn as MultiMeshes take none, being many, small and swaying. Scenes placed as
nodes keep their own `gi_mode`, and `WaveForgeWorld` leaves drawing, and so this, to the game: a
city's modules are buildings, static.

### Ground and colliders

With `ground_stage` set, the node builds each chunk's ground once the fields of the chunk and the
eight around it have arrived ([packs.md](packs.md#ground)): a mesh through the `RenderingServer`,
drawn with `ground_material`, whose surface holds full detail and the coarser levels as its `lods`,
skirts included. Godot draws a coarser level once its error, seen from the camera, shrinks below
the viewport's `mesh_lod_threshold` in pixels (1 by default), and ends at the first level that does
not, in order of key; so a level's key is its error in world units, or a finer level's when that
is larger, and the smallest positive float for a level that strays nowhere, since Godot skips a key
that is not positive. Of levels with the same key, the coarsest is drawn. Within `collider_radius` of the
followed chunk, each chunk gets one static body holding its ground as a `HeightMapShape3D` and its
towns' modules as the shapes `set_collision_shape` assigned, every shape added before the body
joins the space. A height map's
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
`ground_material_stage` that is no Rules, Area or Nearest stage, and a `ground_material` that is no
`ShaderMaterial` beside it.

### Navigation

Within `navigation_radius` of the followed chunk, each chunk gets a navigation region in the
viewport's navigation map, baked on the navigation server's threads from what its body would hold:
its ground at full detail, its volume's surface, and its towns' modules with the shapes
`set_collision_shape` assigned. The source (`wave_forge::surface_nav_source`) holds the chunk's
triangles and its neighbours' out to a border, the agent's radius in cells and three more, so
regions baked apart meet on the same vertices, and starts on a whole number of the map's cell
height for the same reason. A chunk is baked once it and every neighbour inside the world have
their ground and surface, so navigation reaches a chunk less far than the ground, and again when
any of them changes, keeping its last mesh until the new one is in. One bake is started a frame,
nearest first, and only on a frame that has spent under 2 ms of Godot's thread so far, since
starting one costs up to a millisecond. `navigation_template` gives the agent's size, climb and
slope; its cell size and height are replaced by the map's. `navigation_chunks()` lists the chunks
whose mesh is in the map, and `navigation_ready(chunk)` names each as it goes in. A played world
(`play_directory`) gets the same navigation as a generated one. The regions take no asynchronous
iterations: a region given new meshes while one was under way stopped the map synchronising in
Godot 4.7.2.

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
stiffness, a point's scale, so every plant's top sways by about the wind's strength in world units, a
larger plant leaning less, and no two move alike.

## Checking it

`wave_forge_godot/verify.sh` builds the extension and runs `godot/verify.gd` (the WFC world, with
colliders and navigation), `godot/verify_stages.gd` (the valley pack, its ground, and a walk
through a town), `godot/verify_tables.gd` (tables of facts given from GDScript) and
`godot/verify_noise.gd` (a `FastNoiseLite` resource read through a pack), `godot/verify_edits.gd`
(a felled tree and raised ground through an edits log and a save, and cut grass growing back) and
`godot/verify_assemble.gd` (a village's pieces placed by their transforms) and
`godot/verify_scenes.gd` (scenes bound to trees and pieces, as MultiMeshes and as nodes),
`godot/verify_pooling.gd` (pooled scenes reused and reset, others never) and
`godot/verify_ground.gd` (the ground's materials per category, and grass) and
`godot/verify_sound.gd` (the city's surfaces, interiors, emitters and the node's sound) and
`godot/verify_names.gd` (a location's name through a translation) and `godot/verify_occlusion.gd`
(the city's occluders, and the node's) and `godot/verify_proxies.gd` (a proxy for every generated
chunk) in a real headless Godot. `render_occlusion.sh` measures what occluders cull and cost from
above the city and from a street, and `render_proxies.sh` checks that a chunk's modules draw near
and its proxy far, never both. `render_ground.sh` renders the ground's materials and grass to pictures, to look at, times
grass, and checks that trees drawn with the vegetation shader move in the wind and stand still
without it.
How to run it in the dev container and in CI is in [environment.md](../guides/environment.md), and
what the checks assert is in [testing.md](../guides/testing.md).
