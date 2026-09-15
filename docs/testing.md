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
| End to end | `wfc-devtools/tests/` | Whole runs on reference rule sets: a 2D coastline and a small 3D city, with invariants checked and images written | Yes |
| Stress (opt-in) | `wfc-devtools/tests/stress.rs` | Large cities and grids, timed; `#[ignore]`d so `cargo test` stays fast | Yes |

"Needs a GPU" means a Vulkan, Metal or DirectX 12 device. Wave Forge has no CPU fallback ([vision.md](vision.md#non-goals)). In the dev container tests run on the host RTX 3070 through Mesa's dozen driver ([development.md](development.md#toolchain-and-environment)). Where no GPU is available, a software Vulkan device (Mesa llvmpipe, or `WGPU_ADAPTER_NAME=llvmpipe`) is good enough to check correctness, but never for performance.

```bash
cargo test --workspace                 # everything
cargo test --workspace --lib           # unit tests only, no GPU needed
cargo test -p wfc-devtools --test e2e_3d_city -- --nocapture   # one E2E test, showing artifact paths
```

## End-to-end tests

| Test | Rule set | Asserts |
|---|---|---|
| `e2e_2d::coastline_2d_obeys_its_rules_and_renders` | `fixtures::coast_2d`: water–sand–grass–forest bands | Full collapse, zero adjacency violations, rendered image matches the grid |
| `e2e_3d_city::small_city_is_structurally_sound_and_renders` | `city::city`: 52 module variants: roads with corners, junctions and dead ends, buildings with doors, balconies and walkway doors, three roof types, walkways, stairs with landings | Full collapse, zero violations, street-level modules only on the bottom layer, only air on top, every building column rises from the street to exactly one roof, every stair has its landing; reports walkable cells unreachable from the street |

The city is a very crude version of [marian42's WFC city](https://marian42.de/article/wfc/). It is not meant to look good. It is meant to be a **realistic workload**. The toy fixtures prove the solver works at all, but a handful of tiles propagate almost instantly and never contradict, so they say nothing about what real 3D generation costs. The city has more tiles than fit in one possibility word, weights, and structure reaching across many cells, which is what the solver has to handle for the project's goals.

Modules are described by the **connectors** on their six faces (`wfc-rules/src/modules.rs`), the way marian42 does it: rotated variants, adjacency and weights are derived, so the set stays readable at a size where hand-written adjacency tuples would not. One difference: our modules are centred on cells, while marian42's sit on grid corners. A cell face can therefore separate two materials (a facade and open air), and `ModuleSet::connect` declares which different connectors may meet. The module set, its voxel models and its boundary constraints live in `wfc-devtools/src/city.rs`, so assertions such as "street level only on the bottom layer" can be traced back to the connector that causes them (`BEDROCK` under street-level modules, which nothing fits).

**Walkability** follows marian42, whose city is interesting because you can walk everywhere and paths make sense. Street-level faces are walkable. Walkway, door and landing faces are *paths*: they may only touch another walkable face. So a walkway can never end at a wall or in mid-air; it always runs between walkway doors, roof terraces and stair landings. The grid's side borders ban paths that point outward. These rules are local, like marian42's own, so they cannot rule out an island, for example a bridge between two buildings that have no street door. `city::unreachable_walkable_cells` finds such cells; the E2E and stress runs report the count rather than failing on it. A global connectivity constraint is planned for the solver redesign in [#7](https://github.com/AntonTegnelov/wave_forge/issues/7).

**Artifacts** are written to `$CARGO_TARGET_DIR/tmp/e2e-artifacts/` (Cargo's per-target temporary directory, `target/tmp/` by default), or to `WFC_ARTIFACT_DIR` if set:

- `coast_2d.png`
- `city_isometric.png`: every module drawn as its small voxel model
- `city_street_level.png`: the bottom layer, one colour per module variant
- `stress_<name>.png`: from the stress suite

### Known gaps

- **Results are not reproducible yet** (A-6 in [status.md](status.md#alignment-tasks)), so tests assert invariants rather than comparing against golden images. Once seeds work, add golden-image tests for fixed seeds.
- **Contradictions are retried** by rerunning from scratch (`tests/common/mod.rs`). That is a stopgap until the solver can backtrack or restart regions itself (A-9); the retry count is logged so frequent contradictions stay visible.
- **The city needs restarts.** Without backtracking, some runs contradict and start over. The count is part of what the stress suite reports.

## Stress and profiling suite

`wfc-devtools/tests/stress.rs` runs large grids that would make the normal test run slow: cities of 24×24×8, 48×48×10 and 96×96×12 cells, and a permissive two-tile 24³ grid that isolates per-collapse overhead from propagation cost. Every test is `#[ignore]`d. Run them in release mode, one at a time so they don't compete for the GPU:

```bash
cargo test -p wfc-devtools --release --test stress -- --ignored --nocapture --test-threads=1
```

Each run prints one line, for example `stress: city_medium 24x24x8 cells=4608 tiles=56 attempts=1 run_s=… total_s=… cells_per_s=…`. `run_s` is the successful run only, and `total_s` includes device setup and restarted attempts. Compare these lines before and after a performance change, and add `--trace-chrome` style tracing ([debugging.md](debugging.md)) to see where the time goes. The large runs are the baseline for the solver redesign in [#7](https://github.com/AntonTegnelov/wave_forge/issues/7).

## Rendering tools

`wfc-devtools` is a developer-only crate: it is never part of the shipped library.

```bash
cargo run --release -- --rule-file examples/simple-pattern.ron --width 12 --height 12 --depth 6 -o grid.txt
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
