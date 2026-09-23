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
  the chunks within `collider_radius`; `collider_chunks()`; `collider_instance(rid, shape)` maps a
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

### Functions

- `start()` loads the pack and the rule sets and starts the stages' thread, where a town solver
  builds its device.
- `follow(position)` generates around a position in Godot's world space, asking again only when it
  enters another chunk.
- `field_values(stage, chunk)`: a field's values, column by column, x fastest.
- `sites(stage, chunk)`: the sites overlapping a chunk, each with its `region`, footprint `min` and
  `max` in chunks, and levelled `height`.
- `town(stage, chunk)`: a town's `region`, `height` and `tiles` in a chunk.
- `town_instance_sets(stage, chunk, names)`: a town chunk's placements in the layout of
  `WaveForgeWorld.instance_sets`, raised to the site's height.
- `point_sets(stage, chunk)`: a Scatter stage's points, one dictionary per kind, with `transforms`
  as a MultiMesh buffer standing on the field and each point's `ids`.
- `stage_names()`, `stats()` (`process_ms_median`, `_p99` and `_max`).

### Signals

`stage_ready(stage, chunk)`, `stage_dropped(stage, chunk)`, `generation_failed(reason)`.

Turning a field into a ground mesh and a collider is the game's for now
([#88](https://github.com/AntonTegnelov/wave_forge/issues/88)).

## Checking it

`wave_forge_godot/verify.sh` builds the extension and runs `godot/verify.gd` (the WFC world, with
colliders and navigation) and `godot/verify_stages.gd` (the valley pack) in a real headless Godot.
How to run it in the dev container and in CI is in [environment.md](../guides/environment.md), and
what the checks assert is in [testing.md](../guides/testing.md).
