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
| End to end | `wfc-devtools/tests/` | Whole runs on reference rule sets: a 2D coastline and a small 3D city, with invariants checked and images written | Yes |
| Streaming (opt-in) | `wfc-devtools/tests/streaming.rs` | A whole world asked for at once, and a city generated in front of a walking player against a 500 ms tick budget; `#[ignore]`d | Yes |
| Bevy plugin | `wave_forge_bevy/tests/` | `wiring.rs` on the CPU reference: a focus entity generates around itself, messages arrive, eviction is reported, the lattice sits where Bevy's Y-up space says. `shared_device.rs` and `real_render_plugin.rs` (`#[ignore]`d) generate a city on a device Bevy created | Only the two ignored ones |
| Godot extension | `wave_forge_godot/godot/verify.gd` | A focus walks a strip of chunks and back inside a real Godot: chunks arrive, chunks behind are dropped, tiles obey the rules across seams, a chunk returned to is unchanged, and the main loop stays fast | Yes, and a Godot binary |

"Needs a GPU" means a Vulkan, Metal or DirectX 12 device. Wave Forge has no CPU fallback ([vision.md](vision.md#non-goals)). In the dev container tests run on the host RTX 3070 through Mesa's dozen driver ([development.md](development.md#toolchain-and-environment)). Where no GPU is available, a software Vulkan device (Mesa llvmpipe, or `WGPU_ADAPTER_NAME=llvmpipe`) is good enough to check correctness, but never for performance.

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

Modules are described by the **connectors** on their six faces (`wfc-rules/src/modules.rs`), the way marian42 does it: rotated variants, adjacency and weights are derived, so the set stays readable at a size where hand-written adjacency tuples would not. One difference: our modules are centred on cells, while marian42's sit on grid corners. A cell face can therefore separate two materials (a facade and open air), and `ModuleSet::connect` declares which different connectors may meet. The module set, its voxel models and its boundary constraints live in `wfc-devtools/src/city.rs`, so assertions such as "street level only on the bottom layer" can be traced back to the connector that causes them (`BEDROCK` under street-level modules, which nothing fits).

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

## The engine integrations

Both live in their own workspaces, so the library's `cargo test --workspace` does not compile an
engine. Run them explicitly:

```bash
cargo test --manifest-path wave_forge_bevy/Cargo.toml                       # no device needed
cargo test --manifest-path wave_forge_bevy/Cargo.toml --release -- --ignored --nocapture
GODOT=/path/to/godot wave_forge_godot/verify.sh                             # needs a Godot 4 binary
```

`verify.sh` builds the extension, copies it into `wave_forge_godot/godot`, writes the extension list
Godot would otherwise only write from the editor, and runs `verify.gd` headless. It exits non-zero on
any failure and prints what it generated.

Two environment details matter in this dev container. Godot has to be started with
`LD_PRELOAD=/usr/lib/wsl/lib/libd3d12core.so`, or it crashes at shutdown for the same reason test
binaries do ([development.md](development.md#toolchain-and-environment)). And Godot's own
`RenderingDevice` cannot be created here at all, with or without a display, because Mesa's dozen does
not expose `VK_KHR_swapchain`; that only limits the backend discussed in
[roadmap.md](roadmap.md#engine-integrations), not the extension, which brings its own device.

## Known gaps

- **Golden images are still missing.** Generation through the facade is reproducible, and `tests/facade.rs` compares whole worlds cell for cell, but the tests that render (the end-to-end pair) still assert invariants rather than comparing against a stored image.
- **Kernel internals are tested through whole-region results** (A-16). A wrong sweep or a bad checkpoint shows up as an invalid or unsolved region, which is a coarse signal; the checkpoint-ring bug that `every_reported_success_is_a_valid_chunk` caught is the kind of thing a unit test would have caught sooner.
- **Some city chunks cannot be placed** (3.2% in the streaming test). The suite asserts that the share stays small rather than zero, because it is a property of the module set ([constraints.md](constraints.md)).
