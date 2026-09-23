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
| Kernel units | `wfc-gpu/src/`: `kernel`, `block_solver` | The generated kernel source has every constant substituted, a rule set wider than four words uses two vectors, the workgroup budget counts what the kernel declares, a kernel that does not fit says so in numbers | No |
| Library units | `src/`: `scheduler`, `space`, `products`, `stages/pack`, `stages/runtime` | Parities and repair classes, batches that never share a face, nearest chunks first; the Y-up mapping of the lattice; instance sets, navigation sources and instance ids; pack loading, ordering and reach; the stage runtime's reach, blur and dropping | No |
| Devtools units | `wfc-devtools/src/`: `city`, `fixtures`, `invariants`, `models`, `render` | The city module set's structure (roofs on buildings, stairs with headroom, walkways that never end at walls); the fixtures; the invariant checker; voxel models and glTF export; the renderers | No |
| GPU integration | `wfc-gpu/tests/block_solver.rs` | The kernel on a real device: propagation against the reference fixpoint, validity and reproducibility, the same result at 64 and 256 invocations, a portfolio that stops seeds above its winner, weights, rule sets two and five words wide, impossible borders, regions and invocation counts the device cannot take, polling, and malformed batches | Yes |
| Dropped dispatch | `wfc-gpu/tests/dropped_dispatch.rs` | A backend that copies buffers but never runs the kernel, over buffers holding a plausible stale result: waiting and polling both give `SolverError::NoReport`, never solved regions | No, a fake backend |
| Library contract | `tests/facade.rs` | The same requests give the same world, the order they are asked in does not matter, a chunk evicted with its neighbours comes back the same, no batch holds two chunks that share a face, second-parity chunks solve without a halo, repairs try several seeds and keep the lowest that solves, report every chunk they rewrote and give a chunk up when no seed solves, a worker generates the same world on its own thread | No, the CPU reference |
| Stages | `tests/stages.rs` | A pack's stages come out bit for bit the same over a 4×4-chunk area asked for all at once and one chunk at a time in either order; `Pack::reach` reports how far each stage is generated beyond the target; sites stay inside their region and two chunks apart; the ground inside a site is level at its height | No |
| Stage worker | `tests/stage_worker.rs` | A runtime on its own thread hands back exactly what a runtime run directly generates, a request elsewhere drops what it no longer needs, a build failure or an unknown stage is reported, and each stage counts the products it generated, as the worker reports them | No |
| Expressions | `tests/expressions.rs` | Coordinates, distance and angle are the column's; every operation computes its formula at every column; a named noise is the same in every stage and apart from other names; loading refuses parameters an expression cannot use, naming the stage; the ring world's height in one Field stage matches its formula computed from its noises | No |
| Rules | `tests/rules.rs` | The first rule that holds gives a column its category, else the fallback; `Is` reads categories back as a mask; loading refuses a category read as a field, a field read as categories, an unnamed category and more than 256 categories; the ring world's biomes (`examples/rings.world.ron`) are the ones its rules computed directly give, the same in any generation order | No |
| Blend | `tests/blend.rs` | A `Match` blends by its tent on a straight border and steps by at most its share; without a blend each column takes its own category's expression; the categories are generated as far out as it blends; loading refuses a match it cannot evaluate; the ring world's per-biome terrain equals each biome's shape inside it and steps by no more than the blend allows between any two neighbouring columns, across biome borders and chunk seams | No |
| Scatter | `tests/scatter.rs` | Points over a 12×12-chunk area come out the same in any order, no two are closer than `apart`, seams included, every one stands on the field and passes its height, slope and site-margin tests, and ids are unique and name their chunk and column | No |
| Towns | `tests/towns.rs` | A Solve stage puts a town only on its site, street level below and air above, the same in any request order, and names a missing town solver or rule set; on the CPU reference with a ground-and-air module set | No |
| City towns | `wfc-devtools/tests/towns.rs` | The city as a Solve stage over a 12×12-chunk area: four towns, each valid across its chunk seams, the same all at once and one chunk at a time in reverse, each rendered to PNG | Yes |
| Order independence | `wfc-devtools/tests/order_diff.rs` | A 4×4-chunk city generated all at once, chunk by chunk in raster order and in reverse is the same world, tile for tile, with the same chunks given up on, with repairs off and on (seeds 8 and 11); a mismatch names how many chunks differ and the first differing cell | Yes |
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
| Bevy stages | `wave_forge_bevy/tests/stages.rs` | A focus entity makes a pack's stages generate around it; what arrives equals what the library's runtime generates directly; a point's place in Bevy's world is the lattice's; moving away drops what was left behind, with messages; each chunk's ground equals the library's for the same fields, is announced once and is dropped with its field; a Rules stage's categories arrive as the runtime generates them; each stage reports its cost | No |
| Bevy on a device (opt-in) | `wave_forge_bevy/tests/shared_device.rs`, `real_render_plugin.rs` | A city generates on a device created the way `bevy_render` creates its own, and on the device Bevy's own render plugin hands over | Yes |
| Godot units | `wave_forge_godot/src/timings.rs` | The median, 99th percentile and maximum that `stats()` reports, and the window of recent samples | No |
| Godot extension | `wave_forge_godot/godot/verify.gd` | Inside a real Godot, headless, on Jolt: the city's tiles are named, turned and tagged the way a game places them, every module model loads as glTF inside its cell, and the inspector groups the node's properties. A focus runs a strip of chunks and back by the clock without waiting for generation: the chunks beside it are there on every frame, chunks behind are dropped, tiles obey the rules across seams, a chunk returned to is unchanged, the chunks near it get colliders a ray hits and navigation meshes an agent paths across two seams. Godot's slowest frame stays under 8 ms, and the node's own time per frame under 2 ms at the 99th percentile | Yes, and a Godot binary |
| Godot stages | `wave_forge_godot/godot/verify_stages.gd` | The `WaveForgeStages` node runs the valley test pack (`examples/valley.world.ron`): it finds a town over the sites alone, as a game would, then generates the ground, the towns and the trees around it; every town chunk holds a whole chunk of tiles on level ground at its site's height; every tree stands on the ground and none in a town; every column's cover is the category its rule gives its height; every target stage reports its cost, printed per stage; no frame emits more than 256 signals; every chunk whose fields are held around it has ground, and every chunk within the collider radius a body; a capsule with gravity walks from open ground straight through a town, its feet never more than 0.3 below the ground's surface nor above it while standing; moving away drops the first view and its ground; the node's own time per frame stays under 2 ms at the 99th percentile | Yes, and a Godot binary |

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
```

`prepare.sh`, which both Godot scripts run first, builds the extension and puts what the Godot
project loads next to it ([environment.md](environment.md)). `verify.sh` then runs `verify.gd` and
`verify_stages.gd` headless. Each exits non-zero on any failure and prints what it generated and how
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
