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
| Unit | `#[cfg(test)]` modules in each crate | Rule compilation and transformations, bit packing, host/shader struct layouts, output format, invariant checker, renderers, fixtures | No |
| GPU integration | `wfc-gpu/tests/` | Solver behaviour on a real device, for example that pre-constrained cells propagate before the first collapse | Yes |
| Library contract | `tests/facade.rs` | What the facade promises: the same requests give the same world, the order they are asked in does not matter, no batch holds two chunks that share a face, a repair reports every chunk it rewrote, a worker generates the same world on a thread | No, it runs on the CPU reference |
| End to end | `wfc-devtools/tests/` | Whole runs on reference rule sets: a 2D coastline and a small 3D city, with invariants checked and images written | Yes |
| Global constraint | `e2e_3d_city::the_connectivity_constraint_leaves_one_walkable_network` | The path constraint plus backtracking: every walkable cell of the city is connected | Yes |
| Stress (opt-in) | `wfc-devtools/tests/stress.rs` | Large cities and grids, timed; `#[ignore]`d so `cargo test` stays fast | Yes |
| Streaming (opt-in) | `wfc-devtools/tests/streaming.rs` | A whole world asked for at once, and a city generated in front of a walking player against a 500 ms tick budget; `#[ignore]`d | Yes |

"Needs a GPU" means a Vulkan, Metal or DirectX 12 device. Wave Forge has no CPU fallback ([vision.md](vision.md#non-goals)). In the dev container tests run on the host RTX 3070 through Mesa's dozen driver ([development.md](development.md#toolchain-and-environment)). Where no GPU is available, a software Vulkan device (Mesa llvmpipe, or `WGPU_ADAPTER_NAME=llvmpipe`) is good enough to check correctness, but never for performance.

```bash
cargo test --workspace                 # everything
cargo test --workspace --lib           # unit tests only, no GPU needed
cargo test -p wfc-devtools --test e2e_3d_city -- --nocapture   # one E2E test, showing artifact paths
```

**The workspace build does not test each crate's own feature set.** Cargo unifies features across a workspace build, and `wfc-gpu` enables `wfc-core/serde`, so `cargo test --workspace` compiles `wfc-core` with `serde` even though its default feature set is empty. An unguarded `use serde` in `wfc-core` therefore passes the workspace build and breaks anyone who depends on `wfc-core` alone. There is no CI to catch this, so when a change touches optional dependencies or `#[cfg(feature = ...)]` code, also build the affected crates on their own:

```bash
cargo test -p wfc-core                          # default features (none)
cargo test -p wfc-core --all-features
cargo test -p wfc-rules --no-default-features   # RON parsing disabled
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
- `stress_<name>.png`: from the stress suite

### Benchmarks

Both are `#[ignore]`d and print their numbers; run them in release mode.

```bash
cargo test -p wfc-devtools --release --test cpu_reference -- --ignored --nocapture
cargo test -p wfc-gpu --release --test block_solver_bench -- --ignored --nocapture --test-threads=1
cargo test -p wfc-devtools --release --test streaming -- --ignored --nocapture --test-threads=1
```

`cpu_reference` times the single-threaded CPU solver in `wfc_devtools::reference`, the yardstick every
GPU number is printed against. `block_solver_bench` holds the block-local chunk kernel: its
correctness tests check chunks against that reference and against the adjacency rules, and its
benchmarks report per-chunk cost, seams across a stitched world, and live streaming around a moving
focus. `streaming` measures the same two worlds through the library, so what a game would get is
timed rather than what a bench arranged by hand. A timing describes one build on one machine and
driver stack; see [solver-fit.md](solver-fit.md) for what each number means.

## Known gaps

- **Golden images are still missing.** Generation through the facade is reproducible, and `tests/facade.rs` compares whole worlds cell for cell, but the tests that render (the end-to-end pair) still assert invariants rather than comparing against a stored image.
- **Contradictions are retried** by rerunning from scratch (`tests/common/mod.rs`). That is a stopgap until the solver can backtrack or restart regions itself (A-9); the retry count is logged so frequent contradictions stay visible.
- **The city needs restarts.** Without backtracking, some runs contradict and start over. The count is part of what the stress suite reports.

## Stress and profiling suite

`wfc-devtools/tests/stress.rs` runs large grids that would make the normal test run slow: cities of 24×24×8, 48×48×10 and 96×96×12 cells, and a permissive two-tile 24³ grid that isolates per-collapse overhead from propagation cost. Every test is `#[ignore]`d. Run them in release mode, one at a time so they don't compete for the GPU:

```bash
cargo test -p wfc-devtools --release --test stress -- --ignored --nocapture --test-threads=1
```

Each run prints one line, for example `stress: city_medium 24x24x8 cells=4608 tiles=81 attempts=1 run_s=… total_s=… cells_per_s=…`. `run_s` is the successful run only, and `total_s` includes device setup and restarted attempts. Compare these lines before and after a performance change, and add `--trace-chrome` style tracing ([debugging.md](debugging.md)) to see where the time goes. The large runs are the baseline for the solver redesign in [#7](https://github.com/AntonTegnelov/wave_forge/issues/7).

## Rendering tools

`wfc-devtools` is a developer-only crate: it is never part of the shipped library.

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

Orthographic views show exact positions without perspective, and the isometric view shows how they fit together. Nearer surfaces are brighter, and tiles marked empty (`--empty-tile`) are see-through.

**Why a small CPU rasteriser and not Godot or Bevy:** the images have to be produced inside tests and containers without a display, must be pixel-for-pixel reproducible so they can be compared, and should cost nothing to generate after every run. An engine-based viewer can still be added for interactive inspection once the engine integrations exist.

## Writing tests

- **Name tests as statements of behaviour** (`pre_constrained_cells_propagate_before_first_collapse`) and explain in the doc comment *why* the behaviour matters, especially for regression tests.
- **Assert invariants, not incidental output.** Prefer "no adjacency violations" and structural checks over exact tile layouts until runs are deterministic.
- **Keep GPU tests independent.** Several tests create their own device in one process; nothing that holds GPU resources may be shared through globals (a process-wide pipeline cache once handed one device's pipelines to another).
- **When a host struct must match a shader struct, add a layout test** next to the Rust definition (see `wfc-gpu/src/buffers/mod.rs`).
- **Write an artifact whenever a failure would be hard to understand from the assertion alone.**
