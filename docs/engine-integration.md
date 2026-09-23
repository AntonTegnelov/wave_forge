# Engine integration

How Wave Forge should fit into Godot and Bevy beyond handing out tile ids: what the library emits,
where it runs, how each engine consumes it, which engine features it feeds, and how users configure
it. This is the design the integration work in [roadmap.md](roadmap.md) builds towards. What exists
today is described in [architecture.md §5](architecture.md#5-dispatch-async-and-threading) and
[status.md](status.md).

It was researched in September 2026 against Godot 4.7.2 (4.8 in development), godot-rust 0.5.5 and
Bevy 0.20.0-rc.1, from the engines' documentation, class references and source at those tags,
merged pull requests, crates.io, and the procedural generation literature. Load-bearing claims were
checked a second time against primary sources. How claims are marked:

- no mark: verified against a primary source, which is linked or named;
- **(inferred)**: reasoned from verified facts, not tested;
- **(unverified)**: neither checked against a source nor measured.

Engine versions move quickly. Before building on a claim here, check it against the version you
are building for.

## 1. The boundary: products, not rendering

Wave Forge does not render. It never owns a draw call, a render pass or an engine object. What it
emits widens from "which tile goes where" to typed, engine-neutral **products**: instance sets,
meshes with levels of detail, colliders, navigation source geometry, region tags and spawn points,
laid out so that an engine can take each one in a single bulk call. The integrations map products
onto native engine systems and ship **reference assets** (a vegetation wind shader, a grass shader,
a ground material) and **authoring tools** (preview, brushes, bake). The domain logic of those
tools, such as applying a brush stroke to an edit layer, lives in the library, so that there is one
implementation and each surface stays thin.

Why: if the library emitted tiles only, every game would rebuild colliders, navigation, levels of
detail and scatter itself, and most would do it on the main thread. The value of a generator inside
an engine is in how well its output fits the engine's own systems.

What it costs:

- Reference shaders exist twice, in Godot's shading language and in WESL, which Bevy 0.20 uses for
  its shaders, and have to be kept in step.
- Level-of-detail generation needs a mesh simplifier. meshoptimizer is what Godot's own importer
  uses and is MIT licensed; the Rust binding to use is **(unverified)**.
- The public API grows and has to stay stable for two engines.
- Godot's Forward+ features (SDFGI, decals, fog volumes, texture streaming, compute) cannot be tested
  in the dev container, which only runs Godot's Compatibility renderer
  ([roadmap.md](roadmap.md#what-stands-between-the-integrations-and-the-criterion)). They stay
  unverified until someone runs them on a desktop.

## 2. Products

### 2.1 What the library emits

Every product is emitted per chunk, in **chunk-local coordinates** (an integer chunk coordinate plus
local `f32` values), with a stable identity. Chunk-local output is what makes moving the world
origin possible later. Godot documents single-precision comfort ranges of roughly 2048 to 4096
units for first-person games, 4096 to 8192 for third-person and 16384 to 32768 for top-down, and
has no built-in floating origin
([large world coordinates](https://docs.godotengine.org/en/stable/tutorials/physics/large_world_coordinates.html)).

| Product | Contents | Lives on |
|---|---|---|
| **TileGrid** (exists) | tile id per cell, which maps to a prototype and a rotation ([#33](https://github.com/AntonTegnelov/wave_forge/issues/33)) | CPU |
| **InstanceSet** | per chunk and prototype: transforms in Godot's flat MultiMesh layout (12 floats of transform, 4 of colour, 4 of custom data such as wind phase, stiffness and variant), a stable id per instance, a class (visual only, collidable, interactable), and the set's bounding box | CPU |
| **MeshProduct** | indexed triangle meshes with `u32` indices, normals, UVs and tangents; level-of-detail index lists; and a merged far **ProxyMesh** per chunk or group of chunks | CPU |
| **HeightTile, MaterialTile** (Phase 2) | 2^n+1 height samples with shared edges, minimum and maximum, coarser levels, `u8` material ids | CPU, and a GPU texture when a renderer consumes one |
| **CoverMap** (Phase 2) | ground-cover type, density and variation per cell | CPU; expanded into blades on the GPU by an engine shader every frame |
| **Colliders** | heightfield grids; a shape library per prototype (boxes or convex pieces) and the transforms that compose it per chunk; triangle meshes only as a last resort | CPU |
| **NavSource** | walkable and blocking triangles with a one-chunk halo, snapped to the navigation cell grid | CPU |
| **Occluders** | conservative boxes or solid faces per chunk | CPU |
| **RegionTags** | biome, indoor or outdoor, surface material per walkable face, interior volumes, audio emitter points, place names as translation keys with arguments | CPU |
| **Splines** (Phase 2) | roads and rivers: control points, width, material | CPU |
| **SpawnPoints** | kind id, transform, stable id (seed, chunk, layer, index), custom data | CPU |
| **ChunkHash** | a hash over the products gameplay depends on | CPU |

Configuration that goes with the products rather than being emitted per chunk:

- **Distance rings per product class.** Rendering reaches furthest, then navigation, then physics,
  then spawned objects. The library's scheduler decides which products are built at which distance,
  so no integration re-implements that decision.
- **Level-of-detail bands** (distance ranges with margins), and a shadow and global-illumination
  policy, per product class.

**Existence is decided with integers.** Whether an object exists, and which prototype it is, is
decided from integer hashes over quantised inputs, never by comparing a float against a threshold.
WGSL allows fused multiply-add and bounds `sin` and `cos` only to 2^-11
([WGSL spec](https://www.w3.org/TR/WGSL/)), so a float comparison could come out differently on two
GPU vendors and break the determinism the world relies on ([#35](https://github.com/AntonTegnelov/wave_forge/issues/35)).
Floats may still jitter positions that only affect visuals. InstanceSet and SpawnPoint hashes join
the golden-world test.

**Only ground cover stays on the GPU by default.** Everything else is read back, because gameplay
touches it. Shipped systems agree on this: Far Cry 5 keeps height data on the CPU for gameplay
queries ([GDC 2018](https://media.gdcvault.com/gdc2018/presentations/TerrainRenderingFarCry5.pdf)),
Horizon Zero Dawn copies its GPU placement results back to the CPU
([Guerrilla](https://www.guerrilla-games.com/read/gpu-based-procedural-placement-in-horizon-zero-dawn)),
and Unreal's GPU-only PCG instances get no collision, navigation or saving.

**Colour comes from textures, not vertex colours.** Godot 4.7's glTF importer reads a primitive's
material before it parses that primitive's `COLOR_0`, so the flag that uses vertex colours as albedo
is only ever set for a later primitive, never for a mesh of one (`modules/gltf/gltf_document.cpp`,
4.7-stable). Meshes Wave Forge writes (`wfc-export-models`, and MeshProducts later) colour through a
base colour texture and UVs, which every importer honours.

**Engine assets never enter the product model.** Rules emit ids; each integration binds ids to its
own assets (a `PackedScene`, a glTF scene, a material, a `FastNoiseLite`). That keeps one model for
both engines.

### 2.2 How each engine takes them

| Product | Godot 4.7 | Bevy 0.20 |
|---|---|---|
| TileGrid | `tiles_at` | `ChunkUpdated` message |
| InstanceSet, visual | one MultiMesh per chunk and prototype through `RenderingServer.multimesh_set_buffer` with a custom AABB (MultiMesh is culled all or nothing, so never world-wide) | entities sharing `Mesh3d` and material handles, which Bevy batches into multi-draw indirect calls |
| InstanceSet, collidable | shared shape RIDs on one static body per chunk | a compound collider through an avian or rapier adapter crate |
| MeshProduct | `ArrayMesh.add_surface_from_arrays(..., lods)`; the ProxyMesh behind `visibility_range_begin`/`end` with `visibility_parent` | `Mesh3d`; the ProxyMesh behind `VisibilityRange` |
| HeightTile | `HeightMapShape3D`, a mesh, or a texture | avian `Collider::heightfield`; a mesh |
| CoverMap | a reference `ShaderMaterial` expanding blades per `INSTANCE_ID` over a per-chunk MultiMesh | a reference WESL material |
| Colliders | `PhysicsServer3D`: create the body, add every shape, then `body_set_space` last | avian or rapier compound colliders |
| NavSource | `NavigationMeshSourceGeometryData3D`, baked with `NavigationServer3D.bake_from_source_geometry_data_async` | bevy_rerecast |
| Occluders | `OccluderInstance3D` with `ArrayOccluder3D` or `BoxOccluder3D` | not needed; Bevy's occlusion culling works from the depth buffer |
| RegionTags | `Area3D` reverb and bus zones, pooled `AudioStreamPlayer3D`, `tr()` with a context | audio zones through bevy_seedling or bevy_kira_audio, bevy_fluent |
| Splines | a road mesh or splat writes; decals on Forward+ and Mobile only | meshes or clustered decals |
| SpawnPoints | a user `PackedScene`, preloaded with `ResourceLoader.load_threaded_request` | an entity event to a user binding (a glTF scene or a scene function) |
| ChunkHash | multiplayer agreement and golden checks | the same |

## 3. Where the solver and product kernels run

### 3.1 Godot: the solver stays on its own device

| Option | Verdict |
|---|---|
| **Own wgpu device on a worker thread, results read back** (today) | **Keep.** It makes no Godot calls off the main thread, so it needs none of godot-rust's `experimental-threads`, which 0.5.5 requires for any engine call from another thread (it panics otherwise). It is the only option that works when Godot renders with the Compatibility renderer, and it works in the dev container. The readback is a few kilobytes per chunk. godot_voxel has the same shape. |
| Godot's main `RenderingDevice` | **Not for the solver.** In 4.7 graphics and compute share one queue (`main_queue`); there is a separate transfer queue for uploads but no compute queue. A batch dispatch that takes tens of milliseconds would serialise with frame rendering. It would also have to run through `RenderingServer.call_on_render_thread`, does not exist on Compatibility, and needs the WGSL kernel translated to SPIR-V by naga and checked against Godot's shader reflection. |
| A local `RenderingDevice` | **Not for performance.** It creates its own logical device, and its documentation says it "cannot draw to the screen nor share data with the global RenderingDevice" ([RenderingServer.xml](https://raw.githubusercontent.com/godotengine/godot/4.7-stable/doc/classes/RenderingServer.xml)). `sync()` blocks, and driving it from a worker needs `experimental-threads`. Its one real benefit would be leaving wgpu out of the extension binary, which is worth measuring as a size question only. |
| Wrapping Godot's Vulkan device in wgpu-hal | **No.** It would need external synchronisation of a queue Godot drives, is specific to one graphics API, and offers no way to import buffers. |

**Product kernels are a separate question.** Products that should stay on the GPU can only be
shared with Godot's renderer through its **main** `RenderingDevice`, on Forward+ and Mobile:

- `RenderingServer.multimesh_get_buffer_rd_rid`, `multimesh_get_command_buffer_rd_rid` and
  indirect MultiMeshes, since 4.4;
- `Texture2DRD`, since 4.2;
- mesh vertex, attribute and index buffers as `RenderingDevice` RIDs with
  `ARRAY_FLAG_USE_STORAGE_BUFFER`, **from 4.8 only**
  ([#118973](https://github.com/godotengine/godot/pull/118973), merged 2026-06-19; OpenGL returns an
  invalid RID).

Godot's spatial shaders cannot bind storage buffers in 4.7 (proposals #7516 and #6989 are open; the
draft pull request [#109951](https://github.com/godotengine/godot/pull/109951) would add them). Data
therefore reaches Godot materials only through textures, MultiMesh instance data or vertex
attributes, and every GPU product format has to fit one of those three.

Two traps on that path. With motion vectors enabled (TAA, FSR2), a MultiMesh's buffer is
reallocated at twice its size under a new RID, so a writer has to ask for the RID before every
write. And under the experimental separate render thread model, a `call_on_render_thread` callback
needs `Callable::from_sync_fn`, which again needs `experimental-threads`. Which hook suits one-shot
per-chunk writes best (`call_on_render_thread`, the `frame_pre_draw` signal, or a
`CompositorEffect`) was not settled.

None of this is built before the vertex-shader path for ground cover (§4) has been measured and
found too slow.

### 3.2 Bevy: shared device, with a risk to measure

The Bevy plugin builds the generator on Bevy's own `RenderDevice` and `RenderQueue` and polls it
from systems, which is idiomatic and avoids a second device. The risk is that wgpu has one queue per
device ([gfx-rs/wgpu#1066](https://github.com/gfx-rs/wgpu/issues/1066)), so a long dispatch on
Bevy's queue could delay frames the same way it would on Godot's main `RenderingDevice`. Sharing
the device pays off only for products that stay on the GPU; the risk applies to every solve.

**The device policy is decided by measurement.** Both integrations keep their current shape until
frame times have been measured while streaming: p50 and p99, own device against shared device, and
with batches limited to what fits a frame budget, on native Vulkan and Direct3D 12 with NVIDIA and
at least one other vendor. Results from the dev container's dozen driver do not count. From Pascal
on, NVIDIA GPUs preempt compute at instruction level, so a solve on a separate device should
interleave with rendering there; other vendors are **(unverified)**. The same measurement on a
desktop Godot answers the open `RenderingDevice` question in the roadmap. If one policy wins on
both engines, both adopt it, since diverging here is undesirable.

Other Bevy changes the research points to:

- Build the generator asynchronously when its configuration asset has loaded, instead of in
  `Plugin::finish`, which blocks start-up for the seconds a kernel specialisation takes to compile.
- Give the default solver a type alias, so users never spell the generic out.
- Keep any render-world kernel in a small, feature-gated module. The render graph was removed in
  0.19 and extraction changed in 0.20, so that API changes every release.

### 3.3 Platforms

- **Web is out of scope.** Godot's web export is WebGL 2 only, and wgpu's WebGPU backend is not
  available under Emscripten, so there is no GPU compute path, and Wave Forge has no CPU fallback
  ([vision.md](vision.md#non-goals)).
- **Mobile needs measurement.** Godot's own documentation calls mobile compute performance
  generally poor, and Vulkan guarantees only 16 KiB of workgroup memory, against the 32 KiB the city
  uses today.

## 4. Content classes

Each kind of content needs a different treatment. The detail of a house interior, the ground, tall
grass and a forest cannot share one approach.

| Class | Strategy | Engine features it feeds |
|---|---|---|
| **Ground** | A mesh per chunk with skirts and level-of-detail lists, built from the HeightTile; works on every renderer. The HeightTile layout can also feed clipmap or CDLOD terrain renderers such as Terrain3D (MIT) for users who prefer them. Wave Forge ships no clipmap renderer of its own. | mesh LOD, `GI_MODE_STATIC`, `HeightMapShape3D`, NavSource |
| **Grass and ground cover** | Never an instance list from the generator. The engine expands the CoverMap and HeightTile into blades every frame, the way Ghost of Tsushima and Far Cry 5 do: in Godot, a per-chunk MultiMesh of blades whose vertex shader places each one from its `INSTANCE_ID` and the map textures, which works on Compatibility **(inferred; speed unmeasured)**; in Bevy, a WESL material. Shadows and GI off, a visibility range to end it. | global wind parameter (`RenderingServer.global_shader_parameter_set`), visibility ranges |
| **Trees and large vegetation** | An InstanceSet per chunk and species. Meshes carry baked wind weights (the main and detail bending of GPU Gems 3, chapter 16); per-instance custom data carries phase and stiffness. A proxy or impostor beyond the middle band. Trunk colliders only inside the physics ring. In Bevy the wind displacement must also run in the prepass vertex shader, or depth prepass and occlusion culling disagree with what is drawn. | MultiMesh custom data, visibility ranges, shadows |
| **Modular buildings (WFC)** | Near: one shared mesh per prototype and rotation, one InstanceSet per chunk and prototype. Far: a merged ProxyMesh with its own level-of-detail lists, swapped by visibility range. The library emits conservative occluders from solid cells, and colliders from the per-prototype shape library. | instancing, `OccluderInstance3D`, Bevy multi-draw indirect and occlusion culling, SDFGI (`STATIC`) |
| **Detailed interiors** | Nested SpawnPoints and InstanceSets gated to a near ring, with full physics only there. VoxelGI as an opt-in per bounded structure. | rigid bodies, VoxelGI (opt-in) |
| **Interactables** | SpawnPoints promoted to user scenes inside a radius, pooled when their chunk is evicted, and persisted as changes keyed by stable id (the pattern of Unreal's Instanced Actors and godot_voxel's persistent items). | `ResourceLoader`, `MultiplayerSpawner` |
| **Water, roads, decals** | Splines feed the layers below them (flattening, splat, keeping scatter off roads), and the engine draws meshes or splat. Decals need Forward+ or Mobile, so Compatibility gets a mesh or splat path. Water is a sea level plus water tiles, drawn by an engine shader. | decals (optional), fog volumes (Forward+) |

### 4.1 The Godot features that were asked about, with Bevy's counterparts

| Feature | What exists | Verdict | Bevy |
|---|---|---|---|
| Texture streaming | Not in 4.7. Mip streaming arrived in 4.8 dev 5 ([#113429](https://github.com/godotengine/godot/pull/113429)): Forward+ and Mobile only, only imported "Texture2D Streamed" textures, no texture arrays, 3D textures or lightmaps. | **Feed optionally**, for user-imported ground textures on 4.8+. Generated textures cannot stream, so they stay small; terrain material arrays will not stream. | none found **(unverified)** |
| Automatic mesh LOD | Generated at import only. A mesh created at runtime gets levels of detail only through the `lods` dictionary of `add_surface_from_arrays`. The Compatibility renderer applies supplied levels (`rasterizer_scene_gles3.cpp` calls `mesh_surface_get_lod`), so this can be tested in the dev container. | **Feed** level-of-detail lists from the library | nothing automatic **(unverified)**; `VisibilityRange` swaps meshes |
| GPU occlusion culling | **Not a Godot feature**; the term is Unity's. Godot has CPU occlusion culling with `OccluderInstance3D` (Embree rasterised); MultiMeshes, particles and CSG are not baked into occluders ([docs](https://docs.godotengine.org/en/stable/tutorials/3d/occlusion_culling.html)). | **Feed** generated occluders; the cost of rebuilding the occluder structure as chunks stream is unmeasured | `OcclusionCulling` with a depth prepass, no longer experimental since 0.19; needs the GPU culling mode, which 0.20 enables by device feature and never on OpenGL |
| GPU Resident Drawer | **Not a Godot feature**; the term is Unity's. A GPU-driven renderer exists for Godot only as a design proposal. | **Does not exist.** The nearest equivalents are Forward+ automatic instancing of shared meshes, MultiMesh and RenderingServer instances. | automatic multi-draw indirect with the retained render world is the closest match |
| SDFGI | Forward+ only. Only `STATIC` geometry contributes, cascades update as they scroll, and there is no regenerate call ([docs](https://docs.godotengine.org/en/stable/tutorials/3d/global_illumination/using_sdfgi.html)). HDDAGI is an unmerged pull request (#119869). | **Feed**: ground and buildings `STATIC`, grass and props `DISABLED`; stream chunks in beyond the finest cascade **(inferred)** | none; Solari is experimental |
| VoxelGI | Forward+ only, a bake takes seconds, at most 8 per view | **Does not suit streaming**; an opt-in for bounded structures | irradiance volumes, not at runtime |
| LightmapGI | Baked in the editor only. `ArrayMesh.lightmap_unwrap` exists only in editor builds. | **Bake workflow only**, never in an exported game | lightmaps, not at runtime |
| Hardware path tracing | Stock 4.7 has Vulkan ray-tracing plumbing (#99119) that no built-in renderer uses. NVIDIA maintains an MIT-licensed fork with a Vulkan path tracer; its DLSS Ray Reconstruction denoiser is NVIDIA-only, a second denoiser is in progress, and NVIDIA says it intends to upstream the work. | **Do not target.** Stay compatible by keeping chunk geometry static and instanced (one acceleration structure per shared mesh) **(inferred)**. | Solari (experimental) |
| "Scene chunks" and world partitioning | **Not a Godot feature.** The phrase appears in Godot's thread-safety documentation about building scene subtrees outside the active tree. There is no world partition (proposals #7144 and #11728 are open); World Partition is Unreal's term. | **Does not exist.** Wave Forge's chunk lattice is the partition. Chunk subtrees are built off the tree and attached under a per-frame budget. | none; chunk entities with children |
| Background loading | `ResourceLoader.load_threaded_request`, `load_threaded_get_status`, `load_threaded_get` | **Use**, for user scenes that rules reference | `AssetServer`, asynchronous by design |
| Visibility ranges (HLOD) | on every `GeometryInstance3D`, with `visibility_parent` | **Use**; the most portable concept in this table | `VisibilityRange` |
| `DrawableTexture2D` | GPU blits into a texture through texture-blit shaders (`blit_rect`, `blit_rect_multi`, `BlitMaterial`) ([docs](https://docs.godotengine.org/en/stable/tutorials/rendering/drawable_textures.html)) | **Use** for painting masks and splat maps in brushes | a render-to-texture pass |

## 5. Systems other than rendering

| System | Godot | Bevy | The library provides |
|---|---|---|---|
| **Physics** | Jolt is the default for projects created with 4.6 or later, but the `DEFAULT` setting still means Godot Physics, so the integration must not assume Jolt. Add every shape before `body_set_space`: Jolt rebuilds a compound shape on each `add_shape` while the body is in a space. One static body per chunk, not per object. Map a ray hit back to its object through the shape index and the stable instance id. | avian or rapier adapters in separate crates; shapes built on `AsyncComputeTaskPool` | shape library, compound transforms, heightfields |
| **Navigation** | Bake from NavSource with `NavigationServer3D.bake_from_source_geometry_data_async`. Because the source is plain arrays, the whole bake runs off the main thread; parsing the scene tree is what makes navigation slow at runtime, and parsing visual meshes stalls the renderer ([performance](https://docs.godotengine.org/en/stable/tutorials/navigation/navigation_optimizing_performance.html)). One region per chunk, baked with `filter_baking_aabb` grown into the neighbours and `border_size` equal to that margin (xz only in 3D), edge connections off, and vertices snapped so chunks merge by vertex, which is cheap, instead of by edge connection, which checks distance and angle ([navigation meshes](https://docs.godotengine.org/en/stable/tutorials/navigation/navigation_using_navigationmeshes.html), the official [chunk demo](https://github.com/godotengine/godot-demo-projects/tree/master/3d/navigation_mesh_chunks)). A `navigation_ready(chunk)` signal fires once the region is in the map. | bevy_rerecast, or bevy_landmass | NavSource with its halo |
| **Audio** | Godot has no built-in audio occlusion. `Area3D` reverb and bus zones, pooled `AudioStreamPlayer3D` with a `max_distance`, an ambience bed. | bevy_seedling or bevy_kira_audio | surface tags, interior volumes, emitter points, biome zones |
| **Localisation** | `tr()` with a context such as `wave_forge.place`; a translation parser plugin for rule files; a separate translation domain for the plugin's own interface | bevy_fluent | keys and name tokens, never finished strings |
| **Loading** | Preload every scene the rules reference, instantiate each once to compile its pipelines, instantiate off the tree, attach under a time budget | `AssetServer` | ring scheduling |
| **User content** | an `instance_spawned(node, chunk, id)` signal, with metadata set before `add_child` | an entity event per spawn point | SpawnPoints with stable ids |
| **Saving edits** | an edits resource | an edits asset | the Edits log: Prior overrides, added and removed instance ids |
| **Multiplayer** | Peers regenerate from the seed. Node names come from stable ids, because RPCs need matching node paths. ChunkHash checks that peers agree. | the same, with the ecosystem's replication crates | stable ids, ChunkHash |

The navigation mesh is baked rather than written directly from tiles. Baking merges walkable area
into few convex polygons and shrinks it by the agent's radius, height, climb and slope; path queries
cost in proportion to polygons and edges and run far more often than a chunk bakes. Writing one
polygon per walkable module face would multiply polygons and ignore agent size. Whether a
library-merged polygon mesh could beat the bake is a measurement, not a plan.

Multiplayer regeneration relies on the same world coming out on every GPU vendor
([#35](https://github.com/AntonTegnelov/wave_forge/issues/35)). If that fails, a server has to send
tiles instead.

## 6. How users configure it

### 6.1 One model, three workflows

The library owns a **Recipe**, a graph of layers, each a pure function of the seed and a chunk, and
**Edits**, sparse overrides: height changes, painted masks, per-cell tile overrides, pinned or
removed placements. Together they produce the products for any region. RON is the file format in
both engines.

- **Runtime generation** streams around focus points.
- **Editor preview** is the same stream around the editor camera, with a smaller budget and a quick
  draft first. Preview nodes are never saved; they are rebuilt from the recipe.
- **Brushes** write Edits. Edits reach WFC as `Prior` overrides, so painted constraints survive
  regeneration.
- **Bake** writes a bounded region out as ordinary engine content. It either keeps the link (baked
  chunks become fixed, and neighbours solve against them through the `Prior`, so there is no visible
  border) or detaches (the plugin is not needed at runtime). Because generation is deterministic, a
  bake equals what runtime would have produced, which makes it a starting point for a handcrafted
  world.

### 6.2 Godot: tiers of disclosure

- **Tier 0, first run.** A gallery of presets at the top of the inspector, a seed with a reroll
  button (`@export_tool_button`, 4.4+), an instant preview, and configuration warnings that say what
  to do next. A preset is copied into the scene, never shared as a `.tres`, because Godot resources
  are shared by default and a first-timer would otherwise change the preset for every scene using it.
- **Tier 1, the inspector.** Groups for World, Terrain, Settlements and Vegetation & Props, and a
  folded Advanced group for `halo`, `evict_margin` and kernel warming. Fields that do not apply are
  hidden. Fields that force a kernel rebuild (chunk shape, halo, tile count) are marked as such,
  unlike fields that only regenerate.
- **Tier 2, layers.** A reorderable list of layers in the inspector; a graph editor over the same
  graph comes later, once real recipes branch.
- **Tier 3, experts.** Signals and per-chunk product hooks in GDScript, a custom-WGSL layer, and the
  Rust crate. GDScript is not offered as a generation layer: it would run on the main thread,
  outside the GPU path and outside the determinism guarantee.

The shape of nodes and resources: a `WaveForgeWorld` node (following the current camera by default)
holds a `WaveForgeRecipe` resource, which holds layers: a height layer (a `FastNoiseLite` and a
`Curve`), a settlement layer, a WFC layer (modules authored as a `MeshLibrary`, which 4.7 gives a
dedicated editor), and placement layers made of placement rules.

**Placing your own scenes by rule is the headline feature.** A placement rule holds a
`PackedScene`, what it attaches to (a layer, a tile, a face or a mask), filters (density, slope,
height, spacing) and a render mode: Auto, MultiMesh or Nodes. In Auto, a static single-mesh scene
becomes chunked MultiMeshes, and a scene with scripts or bodies is instantiated as real nodes,
optionally promoted from instance to node as the player approaches.

Brushes follow Terrain3D: a toolbar at the side of the 3D viewport, an `EditorDock` (4.6+), input
through `_forward_3d_gui_input` with a decal cursor, one `EditorUndoRedoManager` action per stroke
storing the Edits it changed, and the stroke logic in Rust. A scene brush mirrors 4.7's 2D scene
painting in 3D.

**Rust resource classes on Godot's loader threads.** A spike (Godot 4.7.2, godot-rust 0.5.5)
defined a recipe resource holding layer resources, each layer holding a `FastNoiseLite`, saved one
as `.tres` and loaded it back. On the main thread it loads correctly, sub-resources and noise
settings included. Through `ResourceLoader.load_threaded_request`, godot-rust panics inside Godot's
loader thread ("attempted to access binding from different thread than main thread") and the
**whole process aborts**, because the panic cannot unwind across the engine. That is not about
nesting (godot-rust issue #610 describes it that way): a flat Rust resource aborts the same way,
with or without sub-threads. With godot-rust's `experimental-threads` feature every case loads
correctly, and the node's own check runs as before (Godot's slowest frame 0.34 ms, no late
frames; the check printed that figure as a 99th percentile, but Godot publishes its process time
only as the slowest frame of each second, see §8 A). A game that loads scenes in the background, as the loading guidance recommends, would
crash on any scene holding a Rust resource unless that feature is on. So the recipe resources are
either GDScript resources the Rust side reads, which Godot's loader handles like any other script,
or Rust classes with `experimental-threads`, whose soundness godot-rust does not yet promise. The
choice is the owner's; GDScript resources are the conservative one.

The extension already has the basics of that: godot-rust's `register-docs` feature turns the doc
comments into Godot's own help and tooltips, the `.gdextension` gives the node an icon and asks for
Godot 4.7, the inspector groups the node's properties into Rules, World, Streaming and Advanced,
and a scene can name a rule file and let the node start on its own (`rules_file`,
`start_on_ready`) instead of calling `load_rules` and `start` from a script.

Distribution goes through the **Godot Asset Store**, which replaced the Asset Library on 2026-05-22
and is integrated in 4.7; the Asset Library is deprecated and will become read-only
([announcement](https://godotengine.org/article/introducing-the-godot-asset-store/)).

### 6.3 Bevy

Bevy has no editor: `bevy_editor_prototypes` is archived, and a first-party inspector is in
progress for 0.21. So in Bevy:

- a world is `commands.spawn(WaveForgeWorld(asset_server.load("city.world.ron")))`, with required
  components for its status (loading, compiling, ready, failed) and statistics;
- every configuration type derives `Reflect`, so any inspector can edit it;
- a small asset loader of our own reads `*.world.ron` and loads nested references through its load
  context, so hot reloading works (Bevy's own RON loader arrives in 0.21);
- optional `inspector` and `tools` features add egui panels, and brushes through picking and gizmos.

Brushes, bake and preview docks exist only in Godot, because Bevy has no editor to host them. That
is a deliberate, temporary exception: the Edits format and the brush logic live in the library, so
Bevy gets the same tools once its editor can host them.

### 6.4 Noise that means the same in both engines

Godot's `FastNoiseLite` is the default noise source in Godot. Its `Noise` base class has no
overridable sampling, so a custom noise subclass cannot occur.

The library defines a `NoiseConfig` that mirrors Godot's resource field for field: its enum order,
`offset` applied first, a separate domain-warp object sharing the seed, and Godot's defaults (seed 0,
SIMPLEX_SMOOTH, FBM, 5 octaves, warp amplitude 30, warp frequency 0.05, warp lacunarity 6), with
every field always serialised so defaults cannot drift between engines. Godot converts a
`FastNoiseLite` resource into this configuration on the main thread; Bevy edits it through
`Reflect`. The evaluator is our own port of FastNoiseLite 1.1.0, the version Godot bundles (whether
1.1.0 and the 1.1.1 crate differ is **(unverified)**), plus a WGSL port, since upstream has none. Golden
tests compare it with Godot's C++ output. Noise is quantised before any discrete decision, and the
library offers `sample_height()` so gameplay code does not resample noise itself.

Modules and props for Bevy are authored in Blender and exported as glTF, with Wave Forge metadata in
glTF extras, which Godot's importer reads as well.

## 7. Where the engines have to differ

Every concept above lives in the library, and integrations map it. These are the places where the
engines make that impossible or pointless:

1. **Editor tools exist only in Godot**, because Bevy has no editor (§6.3).
2. **Occluders are used only by Godot.** Bevy culls from the depth buffer.
3. **GPU-resident products travel differently.** Godot needs its main `RenderingDevice` (Forward+ or
   Mobile; MultiMesh buffers from 4.4, meshes from 4.8); Bevy uses its render world. The product
   formats are the same.
4. **Reference shaders are written twice**, in Godot's shading language and in WESL.
5. **Asset binding differs**: a `PackedScene` in Godot, a glTF or code-defined scene in Bevy.
6. **Godot has a node mode and a server mode**; Bevy has one entity path.
7. **Godot has renderer tiers.** Compatibility has no compute, decals, SDFGI or texture streaming,
   so every Godot feature needs a Compatibility path or a documented absence.
8. **Bevy's physics, navigation, audio and localisation come from third-party crates**, which today
   target Bevy 0.19; Godot has them built in. The integration tracks Bevy 0.20 for device sharing,
   so the adapter crates wait until those crates move.

## 8. Order of work

The stages fit [roadmap.md](roadmap.md): the MVP walk first, then Phase 2.

- **A, with the MVP walk** ([#38](https://github.com/AntonTegnelov/wave_forge/issues/38)). Only the products the walk needs: InstanceSet with stable ids, the
  per-prototype collider library, level-of-detail lists for the exported module meshes, and a
  collider ring smaller than the visual ring, applied in Godot through `RenderingServer` and
  `PhysicsServer3D` in one call per chunk. Measured here, on Compatibility and Jolt: the main-thread
  cost of creating a chunk's colliders (node path against server path, shapes added before or after
  `body_set_space`), the node path against the server path for visuals, and whether Compatibility
  applies supplied levels of detail.

  InstanceSets exist (`wave_forge::instance_sets`, `WaveForgeWorld.instance_sets`), and the visual
  paths are measured. Drawing a city chunk of 8×8×8 cells costs Godot's thread about 7 ms either way
  (13 chunks, Compatibility renderer on the RTX 3070, `render_city.sh`): computing the placements is
  0.09 ms in the library against a GDScript loop, and almost all the rest is
  `RenderingServer.multimesh_allocate_data` at about 280 µs per multimesh, against about 6 µs for
  creating one, setting its buffer or creating its instance. Reusing multimeshes does not help on
  this stack: a pool of 25 multimeshes of 512 instances, refilled per chunk with padded buffers and
  `multimesh_set_visible_instances`, cost 13.9 ms a chunk against 7.3 ms for fresh ones (20 chunks,
  same renderer). Every write to a buffer costs a few hundred microseconds here, more for one a
  draw has used, which points at the per-call cost of Compatibility on Mesa's OpenGL-on-Direct3D 12
  translation rather than at allocation as such. The lever that holds on any driver is fewer calls:
  one merged mesh per chunk instead of one multimesh per module, which Godot's own GPU
  optimization guidance recommends for static geometry, traded against the memory that instancing
  saves. Which wins, and whether Forward+ behaves differently, has to be measured on a desktop.

  Colliders are built by the node itself (`set_collision_shape`, `collider_radius`): one static
  body per chunk within the radius, a shape per instance, turned and centred as the models are
  but unscaled, with a ray's hit mapped back to the instance's id (`collider_instance`). How they
  are built was measured (headless Godot 4.7.2, 20 chunks of 200 boxes): through `PhysicsServer3D`
  with every shape added before the body joins the space, 0.12 ms a chunk on Jolt and 0.16 ms on
  Godot Physics; the body joining the space first, 3.1 ms on Jolt, which rebuilds the compound per
  shape, and 0.08 ms on Godot Physics; as `StaticBody3D` and `CollisionShape3D` nodes, 1.0 ms and
  0.68 ms. So the node adds every shape first. In the Godot check, with 512 boxes per chunk,
  Godot's process time stays at 1.4 ms at worst.

  Godot's `Performance.TIME_PROCESS` is the slowest frame of the last second, not the last frame's
  time: `main.cpp` keeps the maximum and publishes it once a second (4.7.2). A percentile taken over
  it per frame is the slowest frame. So the node times its own `process` every frame and reports the
  median, 99th percentile and maximum in `stats()`, and the check bounds both: Godot's slowest frame
  under 8 ms, and the node's own time under 2 ms at the 99th percentile.
- **B, hardening.** The device measurement of §3.2 on desktops ([#39](https://github.com/AntonTegnelov/wave_forge/issues/39)), InstanceSet and ChunkHash in the
  golden worlds, the Godot improvements of §6.2 ([#40](https://github.com/AntonTegnelov/wave_forge/issues/40)), and the godot-rust resource spike ([#41](https://github.com/AntonTegnelov/wave_forge/issues/41)).
- **C, systems.** NavSource with its halo and asynchronous baking, measuring bake time per chunk
  ([#42](https://github.com/AntonTegnelov/wave_forge/issues/42)); RegionTags with the audio and localisation helpers ([#43](https://github.com/AntonTegnelov/wave_forge/issues/43)); SpawnPoints with
  preloading, pooling and saved edits ([#44](https://github.com/AntonTegnelov/wave_forge/issues/44)); and the Bevy adapter crates once the ecosystem reaches Bevy 0.20.
- **D, Phase 2 layers.** `NoiseConfig` with the FastNoiseLite port and golden tests ([#45](https://github.com/AntonTegnelov/wave_forge/issues/45));
  HeightTile, MaterialTile and CoverMap with reference grass and wind shaders in both engines
  ([#46](https://github.com/AntonTegnelov/wave_forge/issues/46)); scatter from density with integer existence decisions; splines; merged far proxies and
  generated occluders ([#47](https://github.com/AntonTegnelov/wave_forge/issues/47)).
- **E, authoring** ([#48](https://github.com/AntonTegnelov/wave_forge/issues/48)). The recipe resources and layer list, the brush plugin, bake (linked and
  detached), and importing a `MeshLibrary` as WFC modules; a graph editor last.
- **F, GPU-resident products**, only where a measurement demands them: product kernels on Godot's
  main `RenderingDevice` (after checking naga's SPIR-V on lavapipe), 4.8 mesh buffers, Bevy
  render-world writes.

Open after this research: the cost of rebuilding occluders while streaming, the speed of
vertex-shader grass on Compatibility and Forward+, the best hook for writes on Godot's main
`RenderingDevice`, Bevy's occlusion culling on Direct3D 12 and Metal in 0.20, and one key scheme for
localisation catalogues shared by Godot's `.po`/`.csv` and Fluent's `.ftl`.
