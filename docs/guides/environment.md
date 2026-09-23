# Environment

What a machine needs to build and test Wave Forge, what the dev container's GPU and drivers do, and
what CI runs. This is the one place for GPU and driver notes; the dev container's setup and
operations are in [`.devcontainer/README.md`](../../.devcontainer/README.md).

## Toolchain

- **Rust 1.98.1**, edition 2024, pinned in [`rust-toolchain.toml`](../../rust-toolchain.toml) with
  Clippy, rustfmt and rust-analyzer. The dev container's base image (`.devcontainer/Dockerfile`,
  `rust:1.98.1-bookworm`) pins the same version; bump both together so the container and every
  other checkout build identically.
- **A GPU is required.** The solver has no CPU fallback by design ([vision.md](../product/vision.md)).
  The CPU reference solver (`wfc-core`, feature `reference`) exists for tests and benchmarks and is
  never compiled into a shipped build. A Vulkan, Metal or DirectX 12 device is enough; a software
  Vulkan device works for correctness if it is a recent Mesa (see [Software Vulkan](#software-vulkan-lavapipe)).

## Three workspaces

The library and the two engine integrations are separate Cargo workspaces with their own lock files,
so `cargo test --workspace` never compiles an engine.

| Workspace | Crates | Build and test |
|---|---|---|
| Root | `wave_forge` (the library), `wfc-core`, `wfc-rules`, `wfc-gpu`, `wfc-devtools` | `cargo test --workspace` |
| `wave_forge_bevy` | the Bevy plugin | `cargo test --manifest-path wave_forge_bevy/Cargo.toml` |
| `wave_forge_godot` | the GDExtension | `cargo test --manifest-path wave_forge_godot/Cargo.toml`, then [`verify.sh`](#godot) |

[testing.md](testing.md) lists what each workspace's tests cover and the per-crate feature checks.

### Build output goes outside the bind mount

The repository is bind-mounted from the host, and that mount is slow and can run out of space. The
dev container puts only the main checkout's `target/` on a named volume. Git worktrees (for example
under `.claude/worktrees/`) are not covered, and their build output would land on the host drive,
several GB per worktree. Point Cargo at `~/.cache`, which is a named volume, with one directory per
workspace so that builds of different workspaces do not wait on each other's build lock:

```bash
export CARGO_TARGET_DIR="$HOME/.cache/cargo-target/wave_forge"
cargo test --workspace

CARGO_TARGET_DIR="$HOME/.cache/cargo-target/wave_forge_bevy" \
  cargo test --manifest-path wave_forge_bevy/Cargo.toml

CARGO_TARGET_DIR="$HOME/.cache/cargo-target/wave_forge_godot" \
  cargo test --manifest-path wave_forge_godot/Cargo.toml
```

`wave_forge_godot/prepare.sh` asks Cargo where the extension's target directory is, so it honours
`CARGO_TARGET_DIR`. It also builds `wfc-export-models` from the root workspace into that same
directory.

- **Bevy is heavy to build.** Build it with `CARGO_BUILD_JOBS=4` and `nice`; one `rustc` job per
  core on the container's memory is what invites the out-of-memory killer.
- **A crashed container leaves zero-filled build artifacts.** `rustc` then reports "memory map must
  have a non-zero length", or a linked library fails with "invalid ELF header". `cargo clean` for
  the affected workspace is the fix; the files are not recoverable.

## The dev container's GPU

The dev container has the host's GPU, an NVIDIA RTX 3070. Inside WSL2 the GPU is only reachable
through Direct3D 12, and the NVIDIA Windows driver offers D3D12 and CUDA there but not Vulkan. The
image therefore builds Mesa's **dozen** driver, which implements Vulkan on top of D3D12, from source
(Mesa 26.2.2, `MESA_VERSION` in the Dockerfile) and installs only its ICD.

How the container reaches the card without widening its sandbox:

- `gpus: all` in `docker-compose.yml` uses Docker's standard GPU support (Docker Desktop on WSL2). It
  adds the WSL virtual GPU device `/dev/dxg` and the CUDA and dxcore libraries. No privileged mode,
  no extra capabilities, no Docker socket and no host drives are involved.
- Two read-only bind mounts from the Docker Desktop VM (not the Windows host) add what Vulkan on
  D3D12 needs beyond that: `/usr/lib/wsl/lib` (`libd3d12.so`, `libd3d12core.so`, `libdxcore.so`)
  and `/usr/lib/wsl/drivers`, the driver store where D3D12 loads NVIDIA's user-mode driver
  `libnvwgf2umx.so`.
- The entrypoint runs `ldconfig` so the WSL libraries are found.

### Adapters

`vulkaninfo --summary` lists two devices:

| Device name | Driver | What it is |
|---|---|---|
| `Microsoft Direct3D12 (NVIDIA GeForce RTX 3070)` | Dozen, Mesa 26.2.2 | The host GPU, through dozen |
| `llvmpipe (LLVM 15.0.6, 256 bits)` | llvmpipe, Mesa 22.3.6 | Debian 12's lavapipe, a software device; not usable for the GPU tests (below) |

**wgpu only sees the RTX 3070 when the instance is built from the environment.** Mesa marks dozen
non-conformant, and wgpu hides such adapters ("Adapter is not Vulkan compliant, hiding adapter" at
`RUST_LOG=wgpu_hal=warn`) unless `WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER=1` is applied. The
container sets that variable, but wgpu reads it only for an instance built with
`wgpu::InstanceDescriptor::new_without_display_handle_from_env()`. `wgpu::Instance::default()` and a
hand-built `InstanceDescriptor` ignore the environment and pick llvmpipe. Every wave_forge instance
is built from the environment; keep it so for new ones.

**`WGPU_ADAPTER_NAME` picks an adapter.** `WgpuBackend::from_env` takes the first adapter whose name
contains the value, ignoring case (`3070`, `llvmpipe`), and fails with the list of adapters when
none does. `WgpuBackend::describe` says which adapter and driver a run used.

### The preload for `libd3d12core.so`

When a wgpu instance on dozen is dropped, WSL's `libd3d12core.so` is unloaded while other threads
still have its thread-local destructors registered. Multi-threaded binaries such as test runners
then die with `SIGSEGV` after their work is done; a backtrace ends in `__nptl_deallocate_tsd`
calling an unmapped address. Preloading the library keeps it mapped.

- The image sets the Cargo runner `CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="env
  LD_PRELOAD=/usr/lib/wsl/lib/libd3d12core.so"`, so the preload applies to binaries that
  `cargo test` and `cargo run` start, and to nothing else. A global preload would add about 13 ms to
  every process start. A container built from an older image carries the same runner in
  `~/.cargo/config.toml` on the cargo-home volume.
- A test binary run directly, not through Cargo, needs `LD_PRELOAD` set by hand.
- Godot needs it too, or it dumps core at shutdown after a successful run.

The workaround lives in the container, not the repository, because the bug is in the WSL driver
stack and a preload would be wrong on any other machine.

### What dozen changes about measurements and tests

- **Timings through dozen are not native.** The translation layer adds dispatch and transfer
  overhead. Compare measurements within the container, and confirm conclusions about CPU and GPU
  crossover points on native hardware. [measurements.md](../research/measurements.md) records the
  stack of every number.
- **Most GPU test time is dozen compiling kernels.** Every test builds a device and compiles its
  kernels, and dozen translates each one to DXIL. `wfc-gpu`'s `block_solver` suite takes about 13 s
  on the RTX 3070 through dozen and 0.7 s on Mesa 25's lavapipe on four CPU cores. Solve timings
  are unaffected; this compilation cost is what `BlockSolver::warm` moves out of the way.
- **Keep GPU work bounded.** A shader that does not terminate hangs the GPU, and Windows resets its
  graphics driver after about two seconds (the desktop flickers and applications can crash). The
  kernel caps how many steps a region may take, and tests keep their regions and batches small.
- **Godot cannot create a `RenderingDevice` here.** Dozen does not expose `VK_KHR_swapchain`, which
  Godot requires of any device, so `RenderingServer.create_local_rendering_device()` returns null on
  the RTX 3070, with `--headless` and with a display from `xvfb-run` alike. Forcing the software
  device with `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json` gives a working
  `RenderingDevice`, slowly. The extension does not need one, since it brings its own wgpu device,
  so this only means a `RenderingDevice` backend can be checked for correctness here but never
  measured. Godot's Compatibility renderer (OpenGL) does run on the RTX 3070 under `xvfb-run`,
  which is what `render_city.sh` uses.
- `XDG_RUNTIME_DIR` warnings in test output come from the windowing libraries and are harmless.

### When the GPU is missing

- **`vulkaninfo` shows only llvmpipe, or dozen reports `ID3D12DeviceFactory::CreateDevice failed`:**
  the GPU mounts are missing. Inside the container, check
  `ls /dev/dxg /usr/lib/wsl/lib/libd3d12.so /usr/lib/wsl/drivers/nv_dispig*/libnvwgf2umx.so`. On the
  host, make sure Docker Desktop's WSL2 backend is in use and the NVIDIA driver is current, then
  run `docker compose -f .devcontainer/docker-compose.yml up -d`. After a Windows driver update, restart the container: the driver-store
  path changes, and `ldconfig` runs at start.
- **`vulkaninfo` fails with `vkEnumeratePhysicalDevices failed`:** the loader aborts when dozen loads
  but cannot create a D3D12 device; the fix is the same. To bypass it for a while, use the software
  device alone with `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`.
- **wgpu finds no adapter:** check that `vulkaninfo --summary` lists a device, and set
  `WGPU_BACKEND=vulkan` if another backend is probed first.
- **wgpu picks llvmpipe although dozen is listed:** the instance was not built from the environment
  (see [Adapters](#adapters)).

## Software Vulkan (lavapipe)

Mesa's lavapipe is a CPU implementation of Vulkan. It is good enough to check correctness, never
performance, and it is what CI runs the GPU tests on.

**The container's own lavapipe (Mesa 22.3.6, Debian 12) is not usable for the GPU tests.** It
silently ends a workgroup once a solve reaches its first backtrack or runs a few hundred steps: the
dispatch completes, no error is raised, and the kernel's writes never happen. It was found because
`the_result_does_not_depend_on_invocations_per_workgroup` returned an empty chunk. A trace of the
kernel's state machine showed lane 0 stopping at the same step every run, and the lavapipe of Mesa
25.0.7 runs the same kernel to the end and passes the whole suite. The solver clears each region's
statistics record before a dispatch and fails with `SolverError::NoReport` when a record stays
empty, so a dropped dispatch never passes as solved chunks; `wfc-gpu/tests/dropped_dispatch.rs`
pins that.

To run on a software device in the container, unpack Debian's `bookworm-backports` build of Mesa
25.0.7 into a scratch directory and point the Vulkan loader at it. Do not install the package: the
system Mesa provides dozen.

```bash
dir=/tmp/mesa25    # any scratch directory outside the checkout
mkdir -p "$dir" && cd "$dir"
curl -fsSLO "http://deb.debian.org/debian/pool/main/m/mesa/mesa-vulkan-drivers_25.0.7-2~bpo12+1_amd64.deb"
dpkg-deb -x mesa-vulkan-drivers_25.0.7-2~bpo12+1_amd64.deb root
sed "s|\"libvulkan_lvp.so\"|\"$dir/root/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so\"|" \
  root/usr/share/vulkan/icd.d/lvp_icd.json > lvp25.json
export VK_ICD_FILENAMES="$dir/lvp25.json" WGPU_ADAPTER_NAME=llvmpipe
```

CI installs Ubuntu 24.04's `mesa-vulkan-drivers`, which is Mesa 25.2.8, and prints the version in the
Library job so a failure can be read against it.

## CI

[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) runs on free GitHub-hosted runners
(`ubuntu-24.04`) for every pull request into `develop` or `main` and every push to `develop`. The
runners have no GPU, so `WGPU_ADAPTER_NAME=llvmpipe` selects lavapipe, and `WgpuBackend::from_env`
fails the job if it is not there.

A change to nothing but Markdown files, `docs/` or `LICENSE` does not trigger CI (`paths-ignore`), so
a docs-only pull request gets no checks.

| Job | What it runs |
|---|---|
| Library | `cargo fmt --all --check`; `cargo clippy --workspace --all-targets -- -D warnings`; `cargo test --workspace`, GPU tests included, on lavapipe; each crate on its own feature set ([testing.md](testing.md)) |
| Godot extension | Clippy with `-D warnings` and the unit tests of `wave_forge_godot`; then `verify.sh release` in Godot 4.7.2, downloaded from the Godot release and checked against a pinned SHA-512 |
| Bevy plugin | the system libraries Bevy links against on Linux; Clippy with `-D warnings` and the tests of `wave_forge_bevy` that need no device |

CI does not run the `#[ignore]`d suites (benchmarks, streaming, hole census, the game session, and
the two Bevy tests on a device) or `render_city.sh`. Run those by hand when a change can affect them.

## Godot

The Godot checks need a Godot 4 binary; the one CI uses is 4.7.2 for Linux x86_64 from the Godot
releases on GitHub. Unpack it outside the checkout. `~/.cache` is a named volume and survives a
container recreate; `/tmp` does not.

```bash
mkdir -p ~/.cache/godot && cd ~/.cache/godot
curl -fsSLO https://github.com/godotengine/godot/releases/download/4.7.2-stable/Godot_v4.7.2-stable_linux.x86_64.zip
unzip -q Godot_v4.7.2-stable_linux.x86_64.zip
```

Then, from the repository root:

```bash
LD_PRELOAD=/usr/lib/wsl/lib/libd3d12core.so \
GODOT="$HOME/.cache/godot/Godot_v4.7.2-stable_linux.x86_64" \
CARGO_TARGET_DIR="$HOME/.cache/cargo-target/wave_forge_godot" \
  bash wave_forge_godot/verify.sh release
```

- `prepare.sh` builds the extension in the profile given (`debug` by default), copies it into
  `wave_forge_godot/godot/bin`, copies `examples/city.ron` and `examples/valley.world.ron` next to it,
  exports the city's module models with `wfc-export-models`, and writes the extension list that
  Godot otherwise writes only from the editor.
- `verify.sh` runs `prepare.sh`, then `verify.gd`, `verify_stages.gd`, `verify_tables.gd` and
  `verify_noise.gd` headless, then the history example's `prepare.sh` and `check.gd`. Use `release`:
  the scripts' frame-time bars describe the extension a game would ship.
- `render_city.sh` renders a city through the Compatibility renderer and needs a display; run it
  under `xvfb-run -a`.
- `LD_PRELOAD` is for the dev container only ([the preload](#the-preload-for-libd3d12coreso)).

What each script checks is in [testing.md](testing.md).
