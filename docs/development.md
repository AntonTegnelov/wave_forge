# Development guide

How to build, test and contribute. For *what* we are building and *why*, read [vision.md](vision.md) and [architecture.md](architecture.md) first.

## Toolchain and environment

- **Rust 1.98.1**, edition 2024, pinned in [`rust-toolchain.toml`](../rust-toolchain.toml). The dev container's base image (`.devcontainer/Dockerfile`) pins the same version; bump both together so the container and every other checkout build identically.
- **Dev container:** see [`.devcontainer/README.md`](../.devcontainer/README.md). It provides the native libraries for windowing and Mesa's software Vulkan driver.
- **A GPU is required.** There is no CPU fallback by design ([vision.md](vision.md#non-goals)).
- **The dev container has the host GPU (an NVIDIA RTX 3070).** Inside WSL2 the GPU is only reachable through Direct3D 12, so the image ships Mesa's *dozen* driver, which implements Vulkan on top of D3D12; see the "GPU access" section of [`.devcontainer/README.md`](../.devcontainer/README.md) for how it is wired up without widening the sandbox. `vulkaninfo --summary` lists `Microsoft Direct3D12 (NVIDIA GeForce RTX 3070)` next to `llvmpipe` (a software Vulkan device).
- **wgpu only sees the RTX 3070 when the instance is built from the environment.** Mesa marks dozen non-conformant and wgpu hides such adapters unless `WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER=1` (set by the container) is applied, which happens only through `InstanceDescriptor::new_without_display_handle_from_env()`. All wave_forge instances are created that way; keep it so for new ones. `WGPU_ADAPTER_NAME=3070` or `WGPU_ADAPTER_NAME=llvmpipe` forces an adapter.
- **Timings through dozen are not native.** The translation layer adds dispatch and transfer overhead, so compare measurements within the container, and confirm conclusions about CPU/GPU crossover points on native hardware.
- **Known issue: GPU test binaries crash at thread exit on dozen.** When the wgpu instance is dropped, WSL's `libd3d12core.so` is unloaded while other threads still hold its thread-local destructors, so multi-threaded binaries such as test runners die with `SIGSEGV` after the tests themselves pass. The dev container image works around it by setting the Cargo runner variable `CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER` to preload `/usr/lib/wsl/lib/libd3d12core.so` for binaries Cargo runs. When running a test binary directly, set that variable yourself. The workaround lives in the container, not the repository, because the bug is in the WSL driver stack and a preload would be wrong on any other machine.
- **Keep GPU work bounded.** A runaway shader can hang the GPU and make Windows reset its graphics driver, so the kernel caps how many steps a region may take and tests keep their regions and batches small. `XDG_RUNTIME_DIR` warnings in test output come from the windowing libraries and are harmless.

## Building and testing

```bash
cargo build --workspace
cargo test --workspace
cargo run -p wfc-devtools --release --bin wave-forge -- --rule-file examples/simple-pattern.ron --width 8 --height 8 --depth 8
```

See [testing.md](testing.md) for the test layers, artifacts and rendering tools, [debugging.md](debugging.md) for debugging practices, and [build-profiles.md](build-profiles.md) before timing or profiling anything.

**Build output location matters.** The repository is bind-mounted from the host. The dev container redirects only the main checkout's `target/` to a named volume; git worktrees (for example under `.claude/worktrees/`) are not covered, and their build output would land on the host drive (several GB per worktree). When building from a worktree, point Cargo elsewhere:

```bash
export CARGO_TARGET_DIR="$HOME/.cache/cargo-target/wave_forge"
```

## Workflow

1. Branch from `develop` (in a worktree if you work on several things at once).
2. Commit atomically: one logical change per commit, and every commit builds.
3. Open a pull request against `develop`, never `main`, referencing the issue it resolves.
4. Merge when review (and CI, once it exists) is green, then close the issue and delete the branch.

## Documentation and comments

Runtime performance is our first priority, so parts of the code will be less obvious than the simplest possible implementation (static dispatch, GPU kernels, packed data layouts, SIMD). That trade-off is only acceptable if the reasoning is written down.

- **Explain why, not how.** The code shows what it does; comments and docs must explain why it is done this way, which alternatives were rejected, and what measurement justified it.
- **Big picture in `docs/`, details next to the code.** Architecture and cross-cutting decisions go in [architecture.md](architecture.md); anything local to one module belongs in comments in that module.
- **Keep [status.md](status.md) honest.** When a change fixes or introduces a misalignment with the architecture, update the alignment table and the matching callout in the same PR.
- **Performance claims need numbers.** When code is shaped by a benchmark or profile, say which one and what it showed.
