# Development guide

How to build, test and contribute. For *what* we are building and *why*, read [vision.md](vision.md) and [architecture.md](architecture.md) first.

## Toolchain and environment

- **Rust 1.98.1**, edition 2024, pinned in [`rust-toolchain.toml`](../rust-toolchain.toml). The dev container's base image (`.devcontainer/Dockerfile`) pins the same version; bump both together so the container and every other checkout build identically.
- **Dev container:** see [`.devcontainer/README.md`](../.devcontainer/README.md). It provides the native libraries for windowing and Mesa's software Vulkan driver.
- **A GPU is required.** There is no CPU fallback by design ([vision.md](vision.md#non-goals)). The dev container currently has no access to the host GPU, so wgpu uses Mesa llvmpipe, a software Vulkan device: good enough for correctness tests, useless for performance measurements. Getting the host GPU into the container is tracked in [roadmap.md](roadmap.md). `XDG_RUNTIME_DIR` warnings in test output come from the windowing libraries and are harmless.

## Building and testing

```bash
cargo build --workspace
cargo test --workspace
cargo run --release -- --rule-file examples/simple-pattern.ron --width 8 --height 8 --depth 8
```

See [testing.md](testing.md) for the test layers, artifacts and rendering tools, and [debugging.md](debugging.md) for tracing and debugging practices.

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
