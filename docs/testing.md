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

"Needs a GPU" means a Vulkan, Metal or DirectX 12 device. Wave Forge has no CPU fallback ([vision.md](vision.md#non-goals)). In containers without a GPU, a software Vulkan device (Mesa llvmpipe) is good enough to check correctness, but never for performance.

```bash
cargo test --workspace                 # everything
cargo test --workspace --lib           # unit tests only, no GPU needed
cargo test -p wfc-devtools --test e2e_3d_city -- --nocapture   # one E2E test, showing artifact paths
```

## End-to-end tests

| Test | Rule set | Asserts |
|---|---|---|
| `e2e_2d::coastline_2d_obeys_its_rules_and_renders` | `fixtures::coast_2d`: water–sand–grass–forest bands | Full collapse, zero adjacency violations, rendered image matches the grid |
| `e2e_3d_city::small_city_is_structurally_sound_and_renders` | `fixtures::city_3d`: ground, roads, crossings, walls, roofs, air | Full collapse, zero violations, bottom layer only ground, roads or buildings, only air on top, every building is walls up to exactly one roof |

The city is intentionally tiny and blocky; it exists to prove the generator can build structured 3D content in the style of [marian42's WFC city](https://marian42.de/article/wfc/), not to look good. Its rules live next to the tile constants in `wfc-devtools/src/fixtures.rs`, so assertions such as "ground only on the bottom layer" can be traced back to the rule that causes them.

**Artifacts** are written to `$CARGO_TARGET_DIR/tmp/e2e-artifacts/` (Cargo's per-target temporary directory, `target/tmp/` by default), or to `WFC_ARTIFACT_DIR` if set:

- `coast_2d.png`
- `city_3d_four_view.png`
- `city_3d_ground_layer.png`

### Known gaps

- **Results are not reproducible yet** (A-6 in [status.md](status.md#alignment-tasks)), so tests assert invariants rather than comparing against golden images. Once seeds work, add golden-image tests for fixed seeds.
- **Contradictions are retried** by rerunning from scratch (`tests/common/mod.rs`). That is a stopgap until the solver can backtrack or restart regions itself (A-9); the retry count is logged so frequent contradictions stay visible.
- **No benchmarks yet**; performance testing is planned with [#7](https://github.com/AntonTegnelov/wave_forge/issues/7).

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
