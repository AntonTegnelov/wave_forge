# Testing and inspection

How Wave Forge is tested, why it is tested this way, and how to look at what the generator produced. For debugging techniques see [debugging.md](debugging.md).

## Why testing needs extra structure here

Wave Forge is a parallel program whose main work happens on the GPU. Its bugs rarely crash; they produce **output that is almost right**: a few cells that break an adjacency rule, a constraint that silently doesn't propagate, a shader reading the wrong field. Such bugs are invisible in logs and easy to miss by eye. The test setup is built around three ideas:

1. **Check invariants mechanically.** Every generated grid is checked against the rules it came from, cell by cell, in all directions.
2. **Pin the contracts between host and shader.** Layouts that must match on both sides are asserted in unit tests, because a mismatch compiles fine and fails silently.
3. **Make results visible.** End-to-end tests render their output to PNG, so humans and LLM-assisted development can see what happened in seconds.

## Test layers

| Layer | Where | What it covers | Needs a GPU |
|---|---|---|---|
| Unit | `#[cfg(test)]` modules in each crate | Rule compilation and transformations, mask and domain layouts, chunk and region geometry, the prior, the chunk store, the scheduler, the kernel's generated source, the invariant checker, the renderers and the fixtures | No |
| GPU integration | `wfc-gpu/tests/block_solver.rs` | The kernel on a real device: propagation against the reference fixpoint, validity and reproducibility, the same result at 64 and 256 invocations, weights, rule sets two and five words wide, and the errors a refused batch gives | Yes |
| Library contract | `tests/facade.rs` | What the facade promises: the same requests give the same world, the order they are asked in does not matter, no batch holds two chunks that share a face, a repair reports every chunk it rewrote, a worker generates the same world on a thread | No, it runs on the CPU reference |
| Stages | `tests/stages.rs`, `src/stages/` | A pack's stages come out bit for bit the same over a 4×4-chunk area asked for all at once and one chunk at a time in either order; `Pack::reach` adds reaches along the longest path; loading names the stage of every error (unknown input, cycle, duplicate, version, bad parameter); a read beyond a stage's reach is an error naming it | No, the CPU |
| Order independence | `wfc-devtools/tests/order_diff.rs` | The facade's order contract on a real device with the city: a 4×4-chunk city generated all at once, chunk by chunk in raster order and in reverse order is the same world, tile for tile, with the same chunks given up on, both with repairs off and with repairs on (seeds 8 and 11); a mismatch names how many chunks differ and the first differing cell | Yes |
| Golden world | `wfc-devtools/tests/golden_world.rs` | A 4×4-chunk city compared tile for tile with one recorded on another device, repairs included; a mismatch names the first differing cell. Recorded on the RTX 3070 through dozen, run by CI on lavapipe, so it checks that a world is the same on two vendors' Vulkan | Yes |
| End to end | `wfc-devtools/tests/` | Whole runs on reference rule sets: a 2D coastline and a small 3D city, with invariants checked and images written | Yes |
| Streaming (opt-in) | `wfc-devtools/tests/streaming.rs` | A whole world asked for at once, and a city generated in front of a walking player against a 500 ms tick budget; `#[ignore]`d | Yes |
| Hole census (opt-in) | `wfc-devtools/tests/hole_census.rs` | Five city worlds streamed; every chunk given up on solved again with more seeds, budgets and halos, and where each failed solve contradicted; `#[ignore]`d | Yes |
| Game session (opt-in) | `wfc-devtools/tests/game_session.rs` | A player walks and runs through an unbounded city in wall-clock time, driving a `Worker` the way an engine does: nothing in view is ever missing or a hole, no rule breaks across seams, the main thread's cost per frame, bounded memory, chunks walked back to, paths across seams; `#[ignore]`d | Yes |
| Bevy plugin | `wave_forge_bevy/tests/` | `wiring.rs` on the CPU reference: a focus entity generates around itself, messages arrive, eviction is reported, chunks and cells sit where Bevy's Y-up space says, and a tile's rotation turns its model the way the lattice turns it. `shared_device.rs` and `real_render_plugin.rs` (`#[ignore]`d) generate a city on a device Bevy created | Only the two ignored ones |
| Godot extension | `wave_forge_godot/godot/verify.gd` | The city's tiles are named, turned and tagged the way a game needs to place them, and a focus runs a strip of chunks and back inside a real Godot, by the clock and without waiting for generation: the chunks beside it are there on every frame, chunks behind are dropped, tiles obey the rules across seams, a chunk returned to is unchanged, the chunks near it get colliders a ray hits and navigation meshes an agent paths across two seams on top of the ground, Godot's slowest frame stays under 8 ms, and the node's own time per frame under 2 ms at the 99th percentile | Yes, and a Godot binary |

"Needs a GPU" means a Vulkan, Metal or DirectX 12 device. Wave Forge has no CPU fallback ([vision.md](vision.md#non-goals)). In the dev container tests run on the host RTX 3070 through Mesa's dozen driver ([development.md](development.md#toolchain-and-environment)). Where no GPU is available, a software Vulkan device (Mesa's lavapipe, selected with `WGPU_ADAPTER_NAME=llvmpipe`) is good enough to check correctness, but never for performance. It has to be a recent Mesa: the lavapipe in Debian 12 (Mesa 22.3.6) ends a workgroup partway through a long solve without reporting an error, which the solver reports as `SolverError::NoReport` ([development.md](development.md#toolchain-and-environment)).

```bash
cargo test --workspace                 # everything
cargo test --workspace --lib           # unit tests only, no GPU needed
cargo test -p wfc-devtools --test e2e_3d_city -- --nocapture   # one E2E test, showing artifact paths
```

**The workspace build does not test each crate's own feature set.** Cargo unifies features across a workspace build, so a crate can compile there with a feature it does not enable itself and still break anyone who depends on it alone. The CPU reference (`wfc-core/reference`) is enabled by several dev-dependencies, and the wgpu backend is a feature a Godot extension will build without. There is no CI to catch either, so when a change touches optional dependencies or `#[cfg(feature = ...)]` code, also build the affected crates on their own:

```bash
cargo test -p wfc-core                             # default features (none)
cargo test -p wfc-core --all-features
cargo check -p wfc-gpu --no-default-features       # no wgpu: what a Godot backend builds against
cargo check -p wave_forge --no-default-features    # the facade without a bundled solver
cargo test -p wfc-rules --no-default-features      # RON parsing disabled
```

## End-to-end tests

| Test | Rule set | Asserts |
|---|---|---|
| `e2e_2d::coastline_2d_obeys_its_rules_and_renders` | `fixtures::coast_2d`: water–sand–grass–forest bands | Full collapse, zero adjacency violations, rendered image matches the grid |
| `e2e_3d_city::small_city_is_structurally_sound_and_renders` | `city::city`: 81 module variants: roads, solid buildings with doors, balconies, arcades and upper passages, pitched and walkable flat roofs with railings, walkways on pillars, and stairs from the street, from roofs and along facades | Full collapse, zero violations, street-level modules only on the bottom layer, only air on top, every building column rises from the street to exactly one roof, every stair has headroom above it; reports the share of walkable cells in the largest network |

The city is a very crude version of [marian42's WFC city](https://marian42.de/article/wfc/). It is not meant to look good. It is meant to be a **realistic workload**. The toy fixtures prove the solver works at all, but a handful of tiles propagate almost instantly and never contradict, so they say nothing about what real 3D generation costs. The city has more tiles than fit in one possibility word, weights, and structure reaching across many cells, which is what the solver has to handle for the project's goals.

Modules are described by the **connectors** on their six faces (`wfc-rules/src/modules.rs`), the way marian42 does it: rotated variants, adjacency and weights are derived, so the set stays readable at a size where hand-written adjacency tuples would not. One difference: our modules are centred on cells, while marian42's sit on grid corners. A cell face can therefore separate two materials (a facade and open air), and `ModuleSet::connect` declares which different connectors may meet. The module set is a rule file, `examples/city.ron`, which the CLI and both engines load as well; its voxel models and boundary constraints live in `wfc-devtools/src/city.rs`. Assertions such as "street level only on the bottom layer" can be traced back to the connector that causes them (`bedrock` under street-level modules, which nothing fits).

**Walkability** follows marian42, whose city is interesting because you can walk almost everywhere and the paths make sense. His set has no global connectivity constraint; connectivity comes from how the modules are built, and the same choices are used here:

- **Walkable faces and paths.** Street, flat-roof, stair and passage faces are walkable; walkway and stair faces additionally *enforce* a walkable face across, so a path can never end at a wall or in mid-air. This is baked into the adjacency rules, as in his `ModuleData`.
- **Stairs lead up beside their top step.** A stair fills one cell and the cell above it is headroom whose path continues at the next floor level, so stairs are never buried under a walkway. Stairs start from the street, from a roof, or on a facade as wall stairs whose flights chain storey by storey.
- **Walkable tops.** Flat roofs join neighbouring roofs and walkways; an edge with nothing to join gets a railing. Borders ban paths pointing out of the grid.
- **Ways through buildings.** Buildings are solid here (unlike his hollow interiors, the single biggest factor in his connectivity), so arcades at street level and passages on upper storeys carry the network through a block.

Local rules cannot forbid a network that is cut off as a whole, so `city::disconnected_walkable_cells` flood-fills the walk graph and reports the share of walkable cells in the largest network. Recent runs give 0.3 to 0.85; a re-simulation of marian42's own rules scores 0.6 to 0.8 with hollow buildings and 0.14 to 0.32 without them. A global connectivity constraint (DeBroglie-style path constraint) would guarantee one network, and is optional for the solver redesign in [#7](https://github.com/AntonTegnelov/wave_forge/issues/7).

**Artifacts** are written to `$CARGO_TARGET_DIR/tmp/e2e-artifacts/` (Cargo's per-target temporary directory, `target/tmp/` by default), or to `WFC_ARTIFACT_DIR` if set:

- `coast_2d.png`
- `city_isometric.png`: every module drawn as its small voxel model
- `city_street_level.png`: the bottom layer, one colour per module variant
- `stitched.png` and `live.png`: whole worlds from the streaming suite, in `$CARGO_TARGET_DIR/tmp/`
- `game_session.png`: the chunks held at the end of the game session, back at its start, in `$CARGO_TARGET_DIR/tmp/`

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
machine and driver stack; see [solver-fit.md](solver-fit.md) for what each number means.

### The game session

```bash
cargo test -p wfc-devtools --release --test game_session -- --ignored --nocapture --test-threads=1
```

The streaming suite asks whether the work of a simulated tick fits the tick. The game session asks
what a player of a game like marian42's city would see. It plays the city once, in wall-clock time,
the way an engine drives the library: a 60 Hz frame loop hands the player's position to a `Worker`
when it enters another chunk, drains the worker's events, and never waits for generation. The
player follows a fixed 776 m route on an unbounded world: a walk at 1.4 m/s, runs at 4.2 m/s
(marian42's run multiplier of three), turns of 45 and 90 degrees, a diagonal, and a return to the
start after the start has been dropped. A chunk is in view when its ground comes within 30 m of the
player, which is marian42's generation range, and chunks are generated three out from the player's
chunk. The session takes about five minutes, most of it the walk, and each test reads it and checks
one thing:

| Test | Holds when |
|---|---|
| `no_chunk_in_view_is_ever_missing_while_it_generates` | no chunk in view is without tiles on any frame |
| `no_chunk_in_view_is_a_hole` | no chunk in view is one that could not be placed |
| `no_generated_cell_breaks_a_rule_inside_a_chunk_or_across_a_seam` | every chunk, checked when it arrives, against the chunks beside it at that moment |
| `generation_costs_the_main_thread_under_a_millisecond_a_frame` | asking for chunks, dropping them and draining events costs at most 1 ms at the 99th percentile and 4 ms at worst |
| `memory_stays_bounded_while_the_player_travels` | the chunks held at once stay within the radius, the margin and one chunk of movement, over a route that generates at least three times as many |
| `a_chunk_walked_back_to_comes_back_the_same` | every chunk generated twice whose tiles its coordinate alone decides both times is identical, over at least 20 |
| `seams_cut_no_more_paths_than_chunk_interiors` | where a face someone can walk through meets another, the walk goes on across seams at least as often as inside chunks, less 0.05 |

A chunk's coordinate alone decides its tiles when no repair touched it, no neighbour failed, and
it was solved with no neighbour present (first parity) or against four such neighbours (second
parity); that is the determinism the library promises, and anything else is counted as not
comparable. The share of walkable cells in the largest network at the end is printed, not
asserted: it is a property of the module set ([constraints.md](constraints.md)).

`no_chunk_in_view_is_a_hole` holds the bar that no chunk is left unplaced, and so do both
streaming tests. When one fails, `hole_census.rs` says why: it streams five city worlds and solves
every chunk given up on again with more seeds, more budget and every halo, and reports where each
failed solve contradicted.

```bash
cargo test -p wfc-devtools --release --test hole_census -- --ignored --nocapture --test-threads=1
```

## Rendering tools

`wfc-devtools` is a developer-only crate and is never part of the shipped library. Its `wave-forge`
binary generates one chunk from a rule file and writes it as a text grid (`--output`, `output.txt`
by default). `wfc-render` draws that grid to a PNG:

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
CPU rasteriser rather than an engine because the images are made inside tests and containers
without a display, have to be pixel-for-pixel reproducible, and must be cheap enough to write after
every run. For a picture through a real engine, `render_city.sh` renders a city in Godot (next
section).

## The engine integrations

Both live in their own workspaces, so the library's `cargo test --workspace` does not compile an
engine. Run them explicitly:

```bash
cargo test --manifest-path wave_forge_bevy/Cargo.toml                       # no device needed
cargo test --manifest-path wave_forge_bevy/Cargo.toml --release -- --ignored --nocapture
GODOT=/path/to/godot wave_forge_godot/verify.sh release                     # needs a Godot 4 binary
GODOT=/path/to/godot xvfb-run -a wave_forge_godot/render_city.sh           # a picture of the city
```

`prepare.sh`, which both scripts run first, builds the extension, copies it into
`wave_forge_godot/godot` with `examples/city.ron`, exports the city's module models into
`godot/models` with `wfc-export-models`, and writes the extension list Godot would otherwise only
write from the editor. `verify.sh` then runs `verify.gd` headless, on Jolt. Besides the walk, it checks that
every module model loads as glTF, coloured by a texture, inside its cell, and that at the end of the
walk exactly the chunks within the collider radius have bodies and a ray down onto a cell hits that
cell's instance at the top of its column. `render_city.sh` renders a
generated city with those models through the Compatibility renderer and saves the picture: the
check to look at after a change to the models, the tile catalogue or the coordinate mapping. It
draws either through the extension's `instance_sets` and `RenderingServer` (`server`, the default),
checking every instance of the first chunk against `tile_basis` and `cell_position` as the renderer
stores it, or through nodes from GDScript (`nodes`), and prints Godot's cost per chunk. The layout
check needs a real renderer: headless, Godot's dummy renderer stores no multimesh data.

`verify.gd` runs headless. It exits non-zero on
any failure and prints what it generated and how long frames took. Headless, Godot's renderer is a
dummy, so the frame time it checks is Godot's own thread (the extension and the script), not drawing.
It uses a four-tile rule set because it tests the extension's contract; the city reaches Godot as a
rule file with [#30](https://github.com/AntonTegnelov/wave_forge/issues/30).

Two environment details matter in this dev container. Godot has to be started with
`LD_PRELOAD=/usr/lib/wsl/lib/libd3d12core.so`, or it crashes at shutdown for the same reason test
binaries do ([development.md](development.md#toolchain-and-environment)). And Godot's own
`RenderingDevice` cannot be created here at all, with or without a display, because Mesa's dozen does
not expose `VK_KHR_swapchain`; that only limits the backend discussed in
[roadmap.md](roadmap.md#still-open-generating-on-godots-own-renderingdevice), not the extension, which brings its own device.

## Known gaps

- **Golden worlds cover tiles, not pictures.** `golden_world.rs` compares a generated city tile for tile across devices; the end-to-end tests that render still assert invariants rather than comparing against a stored image, which the tile comparison makes redundant for generation but not for the renderers.
- **Kernel internals are tested through whole-region results** (A-16). A wrong sweep or a bad checkpoint shows up as an invalid or unsolved region, which is a coarse signal; the checkpoint-ring bug that `every_reported_success_is_a_valid_chunk` caught is the kind of thing a unit test would have caught sooner.
- **What a game draws is not tested here.** Meshes, colliders, draw cost, and the city inside Godot come with [#30](https://github.com/AntonTegnelov/wave_forge/issues/30), [#34](https://github.com/AntonTegnelov/wave_forge/issues/34) and the walk demo ([roadmap.md](roadmap.md#mvp-engine-integrations-and-a-walkable-city)).
