# Build profiles

Which Cargo profile to use for what, why each setting is there, and which options are deliberately not enabled yet. Runtime performance is the project's first priority ([vision.md](vision.md)), so compile time is traded away freely, but every setting that affects speed must be *measured*, and measurements are only meaningful in the right profile.

## Profiles

| Profile | Command | Use it for | Never use it for |
|---|---|---|---|
| `dev` | `cargo build`, `cargo run` | Everyday editing and debugging | Any timing: unoptimised code has a completely different performance profile |
| `test` | `cargo test` | Correctness tests | Performance assertions |
| `release` | `cargo build --release` | Anything shipped, and final benchmark numbers | Profiling with sampling or call-graph tools, because it has no symbols |
| `profiling` | `cargo build --profile profiling` | Profilers (callgrind, perf, Tracy, flame graphs) | Shipping: it carries full debug info |

Binaries land in `target/<profile>/`, for example `target/profiling/wave_forge`.

## Settings and why

### `release`

- **`lto = true` (fat link-time optimisation).** Lets LLVM inline and specialise across crate boundaries: our crates, `wgpu`, `bitvec` and friends. That matters most for the small, hot, generic functions that static dispatch is supposed to make cheap. Thin LTO is typically 10–20% faster than none; fat LTO can go further but costs the most compile time, which is the trade-off this project accepts.
- **`codegen-units = 1`.** Compiling each crate as one unit gives the optimiser the whole crate at once, for better inlining and dead-code removal, at the cost of parallel compilation.
- **`strip = true`.** Removes symbols from shipped binaries. Size matters for games and the Godot Asset Store, and nobody profiles a shipped build.

### `profiling`

- **Inherits `release`,** so the optimiser makes the same decisions as in a shipped build. Profiling an unoptimised or differently optimised build points at bottlenecks that do not exist in release.
- **`debug = true`, `strip = false`.** Profilers need symbols and line tables to attribute cost to functions, including inlined ones. Debug info does not change generated code.

## Options not enabled (yet), and why

| Option | Expected effect | Why it is not on |
|---|---|---|
| `panic = "abort"` | Slightly faster, smaller binaries (no unwinding tables) | Must be measured first. Also interacts with FFI: engines embedding the library (GDExtension) should never see a Rust unwind crossing the boundary, which argues *for* it, so decide together with the engine integration design. |
| `-C target-cpu=native` | Enables the newest SIMD instructions for the build machine | Produces binaries that crash on older CPUs, so it can never be used for anything shipped. Wide SIMD should instead use runtime feature detection (for example `std::arch` with `is_x86_feature_detected!`) so one binary serves every player. Acceptable only for local experiments, labelled as such. |
| Profile-guided optimisation (PGO) / BOLT | Often 10% or more | Needs representative workloads (the streaming suite) and a more complex build. Revisit once hot paths have stabilised. |
| Alternative allocator (mimalloc, jemalloc) | Can be large for allocation-heavy code | The hot path should not allocate per step at all; measure after allocation hot spots are removed rather than masking them. |
| `lto = "thin"` in `profiling` for faster builds | Shorter profiling build | Would make profiles describe a different binary than `release`. |

## Profiling in the dev container

- **GPU work** is measured by the counters a solve returns and wall clock around the dispatch ([debugging.md](debugging.md)); CPU profilers only see the CPU side waiting on the driver.
- **callgrind** (`valgrind --tool=callgrind`) works without extra permissions and can also simulate the L1/LL caches (`--cache-sim=yes`), which answers "is this loop cache-friendly" questions deterministically. It is slow (tens of times slower than native), so run it on small inputs or on CPU-only benchmarks.
- **perf** is installed but cannot open performance counters inside the container (`perf_event_paranoid` and the container's permissions). Enabling it requires changing the container's capabilities or the WSL kernel setting, which widens the sandbox. That is a deliberate decision, not a default.
- **Timings through dozen** (Vulkan on Direct3D 12) include translation overhead. Compare measurements within the same environment, and confirm conclusions about CPU/GPU crossover points on native hardware.

## Rules of thumb

1. Time `release`; profile `profiling`; never time `dev`.
2. Change one setting at a time and record the before/after numbers with the command that produced them.
3. Keep benchmark inputs fixed and large enough that per-run startup (device creation, shader compilation) does not dominate; report setup and steady-state separately.
