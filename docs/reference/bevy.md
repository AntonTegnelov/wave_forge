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
- `WaveForgeStages`, a resource: `field`, `categories`, `curves`, `sites`, `tiles` and `points` per stage and chunk, `timings()` per stage,
  `translation_of(point)` in Bevy's world, and `failure()`.
- `StageReady { stage, chunk }`, `StageDropped { stage, chunk }`, `StagesFailed(reason)`: messages.
- `WaveForgeStagesSystems`: the system set.
- `.with_ground(stage)` builds each chunk's ground from a field stage once the fields around it
  have arrived ([packs.md](packs.md#ground)): `WaveForgeStages::ground(chunk)` returns the
  `GroundMesh`, relative to `chunk_corner(chunk)`, `GroundReady(chunk)` and `GroundDropped(chunk)`
  announce it, `ground_mesh(&ground)` turns it into a Bevy `Mesh`, and its `heights` are the grid a
  physics crate's height-field collider takes (the plugin depends on no physics crate).

The plugin asks for the chunks around every `GenerationFocus`, drains the worker each frame and
sends a message per product.

## Checking it

`wave_forge_bevy/tests/` runs headless apps with the real `DefaultPlugins`: a city generated on
Bevy's device matches what the library generates on a device of its own, and a small pack's
products arrive as messages equal to what the runtime generates, placed where the lattice puts
them and dropped when the focus moves away, and its ground equals the library's for the same
fields. See [testing.md](../guides/testing.md).
