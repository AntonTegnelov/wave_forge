# wave_forge dev container

An SSH-reachable Linux dev box for wave_forge, following the house pattern in
`E:\Programmering\Code\DEVCONTAINER_TEMPLATE.md`. **This one uses host
port 2239** (see the template's port table for the full allocation).

## What's inside

| Tool | Version / notes |
|---|---|
| Rust | 1.98.1 via rustup (`rust:1.98.1-bookworm` base, matches `rust-toolchain.toml`) + clippy, rustfmt, rust-analyzer |
| Native deps | X11, Wayland and xkbcommon headers, clang and libclang (bindgen), cmake, pkg-config |
| Vulkan | **the host RTX 3070** via Mesa dozen (Vulkan on D3D12, built from source in the image), plus Debian's lavapipe (a CPU device); see "GPU access" |
| Node.js | 22 (NodeSource), only for Claude Code |
| Claude Code | latest, installed globally via npm |
| gh | GitHub CLI (authenticate with the repo-scoped PAT: `gh auth login --with-token`) |
| sshd | hardened: pubkey-only, no root, `dev` user only, host port **2239** |

The repo is bind-mounted at `/workspaces/wave_forge`, so edits sync both ways.

**`target/` is a named volume inside the container.** It masks the Windows
`target/` from the bind mount (wrong-platform artifacts, and a bind-mounted
build dir is slow) and survives container recreates. Host and container each
keep their own build output.

**Named volumes** (survive recreates, including Zed's dev-container flow):
`target/`, `~/.cargo` (`CARGO_HOME`: registry/git caches, `cargo install`ed
binaries), `~/.cache` (out-of-repo build caches such as a worktree's
`CARGO_TARGET_DIR`), `~/.claude`, `~/.ssh` (incl. the PAT store), and the
sshd host keys. The toolchain itself (`/usr/local/rustup`) is on the image
layer and owned by `dev`, so `rustup update` works but is undone by a
rebuild. Bump the toolchain in both the Dockerfile and
`rust-toolchain.toml`.

Only the main checkout's `target/` is on a volume. A git worktree (for
example under `.claude/worktrees/`) builds into its own `target/` on the host
drive unless `CARGO_TARGET_DIR` points under `~/.cache`; see
[docs/guides/environment.md](../docs/guides/environment.md).

## GPU access

The container runs wgpu work on the host's NVIDIA GPU through Mesa's dozen driver (Vulkan on
Direct3D 12), using Docker's standard `gpus: all` and two read-only mounts of WSL's GPU libraries,
without widening the sandbox. How that is wired, the adapters it gives, the `libd3d12core.so`
preload, the lavapipe caveats and what to do when the GPU is missing are in
[docs/guides/environment.md](../docs/guides/environment.md).

## One-time setup

Make sure `%USERPROFILE%\.ssh\authorized_keys` on the host contains your
public key:

```
type %USERPROFILE%\.ssh\id_ed25519.pub >> %USERPROFILE%\.ssh\authorized_keys
```

The entrypoint installs this file into the container on every start, so key
changes only need `docker compose restart`, not a rebuild.

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
in Dev Container" (it recreates once on first attach; state is on volumes, so
it survives). VS Code: "Dev Containers: Reopen in Container", which runs
`cargo fetch` via `postCreateCommand`.

## First-login project setup (SSH users)

```bash
cd /workspaces/wave_forge
cargo build --workspace
cargo clippy --workspace --all-targets
cargo test --workspace      # GPU tests run on the RTX 3070 via dozen
```

The engine integrations are separate workspaces, and the Godot checks need a
Godot binary; the commands are in
[docs/guides/environment.md](../docs/guides/environment.md) and
[docs/guides/testing.md](../docs/guides/testing.md).

## Git identity / push

Commit identity and the credential helper are baked into the image's system
gitconfig, so there is nothing to configure. Pushing uses a **fine-grained per-repo
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

(Or just `git push` once and answer the prompt with username `AntonTegnelov`
and the PAT as password; the credential helper writes the same file.) The store
lives on the `ssh-config` named volume, so the login survives container
recreates and rebuilds.

## Claude Code

`claude` is preinstalled. Log in once (`claude` -> follow the OAuth flow);
credentials live on the `claude-config` named volume and survive rebuilds.

## Troubleshooting

- **`Permission denied (publickey)`:** check that `%USERPROFILE%\.ssh\authorized_keys`
  contains your pubkey, then `docker compose -f .devcontainer/docker-compose.yml restart`.
- **Host key changed after `down -v`:** the host-key volume was wiped; run
  `ssh-keygen -R "[localhost]:2239"` on the host and reconnect.
- **The GPU is missing, or wgpu picks the wrong adapter:** see "When the GPU is
  missing" in [docs/guides/environment.md](../docs/guides/environment.md).
- **A build dir looks empty or stale:** you are seeing the `target` volume,
  not the Windows build; just `cargo build`.
