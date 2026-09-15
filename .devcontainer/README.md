# wave_forge dev container

An SSH-reachable Linux dev box for wave_forge, following the house pattern in
`E:\Programmering\Code\DEVCONTAINER_TEMPLATE.md` — **this one uses host
port 2239** (see the template's port table for the full allocation).

## What's inside

| Tool | Version / notes |
|---|---|
| Rust | 1.98.1 via rustup (`rust:1.98.1-bookworm` base, matches `rust-toolchain.toml`) + clippy, rustfmt, rust-analyzer |
| Native deps | X11/Wayland/xkbcommon headers (minifb), clang/libclang (bindgen), cmake, pkg-config |
| Vulkan | **the host RTX 3070** via Mesa dozen (Vulkan on D3D12, built from source in the image) plus lavapipe (CPU) as fallback — see "GPU access" |
| Node.js | 22 (NodeSource) — only for Claude Code |
| Claude Code | latest, installed globally via npm |
| gh | GitHub CLI (authenticate with the repo-scoped PAT: `gh auth login --with-token`) |
| sshd | hardened: pubkey-only, no root, `dev` user only, host port **2239** |

The repo is bind-mounted at `/workspaces/wave_forge`, so edits sync both ways.

**`target/` is a named volume inside the container** — it masks the Windows
`target/` from the bind mount (wrong-platform artifacts, and a bind-mounted
build dir is slow) and survives container recreates. Host and container each
keep their own build output.

**Named volumes** (survive recreates, including Zed's dev-container flow):
`target/`, `~/.cargo` (`CARGO_HOME`: registry/git caches, `cargo install`ed
binaries), `~/.cache` (out-of-repo build caches such as a worktree's
`CARGO_TARGET_DIR`), `~/.claude`, `~/.ssh` (incl. the PAT store), and the
sshd host keys. The toolchain itself (`/usr/local/rustup`) is on the image
layer and owned by `dev`, so `rustup update` works but is undone by a
rebuild — bump the toolchain in both the Dockerfile and
`rust-toolchain.toml`.

## GPU access

The container can run wgpu work on the host's NVIDIA GPU without changing
the sandbox boundary:

- `gpus: all` in `docker-compose.yml` uses Docker's standard GPU support
  (Docker Desktop on WSL2). It adds the WSL virtual GPU device `/dev/dxg`
  and the CUDA/dxcore user-space libraries. No privileged mode, no extra
  capabilities, no Docker socket, no host drives.
- Two read-only bind mounts from the Docker Desktop VM (not the Windows
  host): `/usr/lib/wsl/lib` (`libd3d12.so`, `libd3d12core.so`,
  `libdxcore.so`) and `/usr/lib/wsl/drivers` (the driver store, where D3D12
  loads NVIDIA's user-mode driver `libnvwgf2umx.so`). `gpus: all` alone
  mounts only the CUDA pieces, which is not enough for Vulkan-on-D3D12.
- The NVIDIA Windows driver offers D3D12 and CUDA inside WSL, not Vulkan, so
  the image builds Mesa's **dozen** driver (Vulkan -> D3D12) from source in a
  separate build stage and installs only its ICD. The entrypoint runs
  `ldconfig` so the WSL libraries are found.

Check: `vulkaninfo --summary` lists
`Microsoft Direct3D12 (NVIDIA GeForce RTX 3070)` (driver Dozen, discrete
GPU) and `llvmpipe` (CPU).

**wgpu needs two things to see it.** Mesa marks dozen non-conformant, and
wgpu hides non-conformant adapters ("Adapter is not Vulkan compliant, hiding
adapter" at `RUST_LOG=wgpu_hal=warn`). The container sets
`WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER=1`, but wgpu only reads it for
instances built from the environment:

```rust
let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
```

`wgpu::Instance::default()` and a hand-built `InstanceDescriptor { .. }`
ignore the environment and will keep selecting llvmpipe. With the env-built
instance, both the default and the `HighPerformance` adapter requests
return the RTX 3070 (verified 2026-09-15 with a small standalone crate:
enumerate, request, create device, submit an upload). Force a specific
adapter with `WGPU_ADAPTER_NAME=llvmpipe` or `WGPU_ADAPTER_NAME=3070`.

Caveats: dozen is a translation layer, so absolute timings carry extra
dispatch/transfer overhead; comparisons within the container are meaningful,
conclusions about GPU-vs-CPU crossover should be confirmed natively. A
runaway shader can hang the GPU and make Windows reset the graphics driver
(desktop flicker, possibly crashed apps), so keep GPU tests bounded with
timeouts and small default grid sizes. dozen is marked non-conformant by
Mesa; if a feature misbehaves, compare against lavapipe first.

## One-time setup

Make sure `%USERPROFILE%\.ssh\authorized_keys` on the host contains your
public key:

```
type %USERPROFILE%\.ssh\id_ed25519.pub >> %USERPROFILE%\.ssh\authorized_keys
```

The entrypoint installs this file into the container on every start, so key
changes only need `docker compose restart` — no rebuild.

## Build and start

```
docker compose -f .devcontainer/docker-compose.yml up -d --build
```

The container auto-restarts with Docker Desktop (`restart: unless-stopped`).

Stop: `docker compose -f .devcontainer/docker-compose.yml down`
(add `-v` to also wipe `target/`, the cargo caches, the Claude login, the PAT
store, and ssh host keys).

## Connect

Add to `~/.ssh/config` on the host:

```
Host wave-forge-dev
    HostName localhost
    Port 2239
    User dev
```

Then `ssh wave-forge-dev`, or point Claude Code / Cursor / JetBrains Gateway
at it. Zed: `zed ssh://dev@localhost:2239/workspaces/wave_forge`, or "Reopen
in Dev Container" (recreates once on first attach — state is on volumes, so
it survives). VS Code: "Dev Containers: Reopen in Container", which runs
`cargo fetch` via `postCreateCommand`.

## First-login project setup (SSH users)

```bash
cd /workspaces/wave_forge
cargo build --workspace
cargo clippy --workspace
cargo test --workspace      # GPU-path tests run on the RTX 3070 via dozen
```

The binary opens a minifb window when visualization is on; there is no
display in the container, so run headless modes (`--benchmark-mode`, etc.)
or build here and run on the host.

## Git identity / push

Commit identity and the credential helper are baked into the image's system
gitconfig — nothing to configure. Pushing uses a **fine-grained per-repo
PAT** over https, **never an SSH key**: GitHub SSH keys can't be scoped to
one repo, and this container must not reach beyond its own repo (see
DEVCONTAINER_TEMPLATE.md).

One-time, after minting the PAT (GitHub -> Settings -> Developer settings ->
Fine-grained tokens -> Repository access: only `AntonTegnelov/wave_forge` ->
Permissions: Contents = Read and write):

```bash
printf 'https://AntonTegnelov:%s@github.com\n' '<the PAT>' > ~/.ssh/git-credentials
chmod 600 ~/.ssh/git-credentials
```

(Or just `git push` once and answer the prompt — username `AntonTegnelov`,
password = the PAT; the credential helper writes the same file.) The store
lives on the `ssh-config` named volume, so the login survives container
recreates and rebuilds.

## Claude Code

`claude` is preinstalled. Log in once (`claude` -> follow the OAuth flow);
credentials live on the `claude-config` named volume and survive rebuilds.

## Troubleshooting

- **`Permission denied (publickey)`** — check `%USERPROFILE%\.ssh\authorized_keys`
  contains your pubkey, then `docker compose -f .devcontainer/docker-compose.yml restart`.
- **Host key changed after `down -v`** — the host-key volume was wiped; run
  `ssh-keygen -R "[localhost]:2239"` on the host and reconnect.
- **wgpu finds no adapter** — check `vulkaninfo --summary` lists `llvmpipe`;
  set `WGPU_BACKEND=vulkan` if another backend is being probed first.
- **`vulkaninfo` shows only llvmpipe, or dzn reports
  `ID3D12DeviceFactory::CreateDevice failed`** — the GPU mounts are missing.
  Inside the container check `ls /dev/dxg /usr/lib/wsl/lib/libd3d12.so
  /usr/lib/wsl/drivers/nv_dispig*/libnvwgf2umx.so`; on the host make sure
  Docker Desktop's WSL2 backend is in use and the NVIDIA driver is current,
  then `docker compose -f .devcontainer/docker-compose.yml up -d`. After a
  Windows driver update, restart the container (the driver-store path
  changes and `ldconfig` runs at start).
- **`vulkaninfo` errors with `vkEnumeratePhysicalDevices failed`** — the
  loader aborts when dzn loads but cannot create a D3D12 device; same fix as
  above. To bypass temporarily: `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`.
- **A build dir looks empty / stale** — you're seeing the `target` volume,
  not the Windows build; just `cargo build`.
