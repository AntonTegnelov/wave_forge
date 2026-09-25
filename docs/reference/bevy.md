# Bevy plugin

What `wave_forge_bevy` gives a Bevy 0.20 app, as built. Why it is shaped this way is in
[engine-integration.md](../architecture/engine-integration.md). The crate tracks the Bevy release
that uses the library's wgpu version, because sharing Bevy's device needs both to agree.

## A streamed WFC world

`WaveForgePlugin` generates a world in chunks around every entity with a `GenerationFocus`, on the
GPU device Bevy renders with. Add it after the plugin that creates Bevy's renderer (`DefaultPlugins`,
or `RenderPlugin` alone), because the shared device exists only once that has run; the generator is
built in `Plugin::finish`.

| Constructor | Use |
|---|---|
| `WaveForgePlugin::new(ruleset, prior, settings)` | generate on Bevy's device |
| `WaveForgePlugin::from_rules(file, prior, settings)` | the same from a rule file, and insert `WaveForgeTiles` so systems can ask what each tile is |
| `WaveForgePlugin::on_own_device(ruleset, prior, settings)` | a device of its own, for a Bevy build on another wgpu version or to isolate generation from rendering |
| `.warm(radius)` | compile every kernel a focus of up to `radius` can dispatch, repairs included, while the plugin builds, instead of at the first dispatch |
| `WaveForgeSolverPlugin::new(ruleset, prior, settings, solver)` | generate on a solver the game built: another backend, or the CPU reference in a test |

**Settings** (`WaveForgeSettings`): `seed`; `extent`, the world in chunks, which carries the chunk
shape; `halo`; `repair`, a `RepairPolicy`; `cell_size` in Bevy's units; `evict_margin`, chunks
beyond a focus's radius plus this are dropped (keep it at one or more, because a request reaches one
chunk beyond its radius). It also converts between spaces: `chunk_at`, `translation_of`,
`cell_translation`, `chunk_size`.

**Components, resources and messages:**

- `GenerationFocus`, a component: generate around this entity, within its radius.
- `WaveForgeWorld<S>`, a resource: `chunk(coord)`, `is_idle`, `pending_chunks`.
- `WaveForgeTiles`, a resource: the rule file, and `rotation_of(tile)`.
- `ChunkUpdated`, `ChunkFailed`, `ChunkEvicted`: messages per chunk.
- `WaveForgeSystems`: the system set, to order a game's systems around the plugin's.

The plugin asks for new chunks, starts batches and collects results in two systems per frame, and
nothing blocks.

## Packs of stages

`WaveForgeStagesPlugin::new(targets, settings, build)` runs a pack of stages
([packs.md](packs.md)) on a thread of its own through `StageWorker`. `build` is a closure that makes
the `Runtime` on that thread, so a town solver builds its GPU device there: a device of its own, not
Bevy's. Whether towns should share Bevy's device is the same measurement as the solver's
([#39](https://github.com/AntonTegnelov/wave_forge/issues/39)). A Region stage's job is registered
there too, with `Runtime::with_region_job`. To sample a stage or read an atlas without chunks, a
game builds a `Runtime` of the same pack and seed and calls `sample` or `atlas` on it, on any
thread.

- `StagesSettings`: `chunk`, columns per chunk as the runtime was built with, and `cell_size`.
- `WaveForgeStages`, a resource: `field`, `categories`, `curves`, `sites`, `tiles`, `points` and `stamps` per stage and chunk, `timings()` per stage,
  `translation_of(point)` and `transform_of(point)` (turned, leant and scaled) in Bevy's world,
  `stamp_transform(stamp)`, where a scene of an assembled piece goes
  ([packs.md](packs.md#assemble)), and
  `failure()`. `set_facts(facts)` and `focus(table,
  id)` hand the stages new tables of facts and a focused row ([packs.md](packs.md#tables-of-facts)):
  a game keeps its own `Facts`, gives it rows, and hands a copy here; the runtime `build` makes is
  given its first facts and focus there. A location's `Site::name()` is a translation key with
  arguments ([packs.md](packs.md#locations)), for a game's localisation, a Fluent bundle say.
- `StageReady { stage, chunk }`, `StageDropped { stage, chunk }`, `StagesFailed(reason)`: messages.
- `WaveForgeStagesSystems`: the system set.
- `WaveForgeStages::set_edits(edits)` hands the stages the player's edits
  ([packs.md](packs.md#edits)). `request_save()` asks for a save, which arrives as a
  `StagesSaved(save)` message, and `load(save)` brings a world back from one
  ([packs.md](packs.md#persistence-and-saves)).
- `StagePlacements`, a resource: `StagePlacements::default().bind(kind, |entity| ...)` binds a
  kind, a Scatter point's kind or an Assemble piece's name, to what its entities hold: a
  `SceneRoot` of a glTF scene, a mesh and a material, anything. Every point or piece of a bound kind
  in a chunk that arrives gets an entity with its `Transform` (`transform_of` or `stamp_transform`)
  and a `Placed { stage, chunk, id }` component, announced by an `InstanceSpawned { entity, placed }`
  message, and despawned when its chunk is dropped. A piece overlapping several chunks gets one
  entity, from the chunk holding its footprint's centre. Insert it before the stages it binds arrive.
- `.with_ground_materials(stage)` gives the ground the categories of a Rules or Area stage as
  materials ([Ground materials](#ground-materials)); `WaveForgeStages::ground_materials(chunk)`
  gives them per ground vertex, and `settings()` how chunks and cells sit in Bevy's world.
- `.with_radius(stage, radius)` generates one target within a radius of its own around every
  `GenerationFocus`, the others keeping each focus's radius.
- `.with_ground(stage)` builds each chunk's ground from a field stage once the fields around it
  have arrived ([packs.md](packs.md#ground)): `WaveForgeStages::ground(chunk)` returns the
  `GroundMesh`, relative to `chunk_corner(chunk)`, `GroundReady(chunk)` and `GroundDropped(chunk)`
  announce it, `ground_mesh(&ground)` turns it into a Bevy `Mesh` at full detail, skirt included,
  or `ground_levels` into one per level of detail ([Ground levels of detail](#ground-levels-of-detail)),
  and its `heights` are the grid a physics crate's height-field collider takes (the plugin depends
  on no physics crate).

The plugin asks for the chunks around every `GenerationFocus`, drains the worker each frame and
sends a message per product.

## Checking it

`wave_forge_bevy/tests/` runs headless apps with the real `DefaultPlugins`: a city generated on
Bevy's device matches what the library generates on a device of its own, and a small pack's
products arrive as messages equal to what the runtime generates, placed where the lattice puts
them and dropped when the focus moves away, and its ground equals the library's for the same
fields. New facts drop the stages that read them and regenerate them with the new rows. See [testing.md](../guides/testing.md).

## Ground levels of detail

`ground_levels(&ground, detail)` gives a chunk's ground as one `GroundLevelMesh` per level of
detail ([packs.md](packs.md#ground)): its `step`, a `mesh` with the full positions and normals and
the level's triangles, skirt included, and the abrupt `VisibilityRange` it is drawn in, measured
from the centre of the mesh's bounds. Each goes on an entity of its own at the chunk's corner.
`GroundDetail { pixels, height, fov }` says how many pixels a level's error may span in a viewport
`height` pixels tall under a vertical field of view of `fov` radians: a coarser level is drawn from
where its error spans that many pixels, pushed out by half the diagonal of the chunk's bounds so no
point of the chunk is nearer, and a level's error counts as at least every finer level's. A level
no distance would draw is left out. Bevy chooses no mesh's level on its own, so a game that wants
the levels spawns these instead of `ground_mesh`.

## Ground materials

`WaveForgeMaterialsPlugin`, which a game that renders adds after Bevy's own plugins, registers the
reference materials and embeds their WESL shaders. `GroundMaterial` is an `ExtendedMaterial` over
`StandardMaterial`: everything but the base colour is the `StandardMaterial`'s, and the base colour
comes from the chunk's materials. `ground_material(stages, chunk, base, palette, images)` makes a
chunk's, once its ground is built with materials, and `ground_material_of(mesh, ids, corner, cell,
base, palette, images)` makes one for a ground built some other way. Its id image holds a texel per
ground vertex, the category / 255 in red; `palette_image(colours)` makes the 256 by 1 palette,
categories past the colours given taking colours of their own. The fragment shader blends the
colours of the four vertices around each fragment, so materials meet in a band a cell wide, then
applies Bevy's lighting, or none for an unlit base. A chunk's ground reads the materials of the
chunks beyond its far edges, so the material stage has to be generated one chunk beyond the ground,
with `.with_radius` say.

## Grass

`GrassMaterial` is an `ExtendedMaterial` over `StandardMaterial` with a vertex shader of its own.
Every chunk draws the same mesh, `grass_mesh(columns, per_column)`: `per_column` blades for each
column, each a triangle carrying nothing but its index, in its second UV channel.
`grass_material(stages, chunk, cover, per_column, base, images)` makes a chunk's, once its ground
is built and `cover`, a field stage's product for the chunk, has arrived; `grass_material_of(mesh,
cover, cell, per_column, base, images)` makes one for a ground built some other way. It holds the
ground's height per vertex (one float each) and the cover per column (one byte each) as images. The
grass shader gives blade *b* its column, a hashed place in it, a hashed turn and height, and shows
it only where a hash is below the column's cover; it stands on the ground's heights, blended between
vertices, and sways in the wind. A blade is drawn from both sides, casts no shadow and has no
prepass.

The shader moves the blades far from where the mesh has them, so a chunk's grass needs the bounds
`grass_bounds(stages, chunk)` gives, or `grass_bounds_of(mesh, cell)`: an `Aabb` over the chunk and
its heights, and `NoAutoAabb`. Both go on the grass's entity, since without `NoAutoAabb` Bevy
replaces the bounds with the mesh's own whenever the entity's mesh changes, and culls the chunk's
grass wherever the chunk's corner is out of view.

## Wind

The `Wind` resource is a `Vec4`: a direction along x and z, a strength in world units at a blade's
or a plant's tip, and a speed. It blows gently along +x unless a game sets it. A change reaches
every grass and vegetation material in the next frame, and a material added later gets the wind as
it is.

`VegetationMaterial` is an `ExtendedMaterial` over `StandardMaterial` whose vertex shader bends a
plant's mesh in the wind, after GPU Gems 3, chapter 16, while the `StandardMaterial` gives its
looks. `vegetation_material(base, bend_height, flutter)` makes one. It leans the plant downwind,
more the higher up a vertex is (fully at `bend_height`, in the mesh's own units) and the less stiff
the plant, keeping each vertex's distance from the root, and flutters its outer parts out from the
plant's axis by `flutter`. A plant's phase in the wind is hashed from where it stands, and its
stiffness is its scale, so every plant's top sways by about the wind's strength in world units, a
larger plant leaning less, and no two move alike. The same vertex shader runs in the prepass, so
depth, shadows and motion vectors follow the swaying plant; that is why the flutter goes out from
the axis rather than along the normals, which Bevy's prepass has only when a normal prepass runs.
A plant's mesh stands on its origin with +y up, and is neither skinned nor morphed.
