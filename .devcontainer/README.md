# wave_forge dev container

An SSH-reachable Linux dev box for wave_forge, following the house pattern in
`E:\Programmering\Code\DEVCONTAINER_TEMPLATE.md` — **this one uses host
port 2239** (see the template's port table for the full allocation).

## What's inside

| Tool | Version / notes |
|---|---|
| Rust | stable via rustup (`rust:1-bookworm` base) + clippy, rustfmt, rust-analyzer |
| Native deps | X11/Wayland/xkbcommon headers (minifb), clang/libclang (bindgen), cmake, pkg-config |
| Vulkan | Mesa lavapipe (CPU Vulkan) so wgpu can find a device headless — slow, no real GPU in the container |
| Node.js | 22 (NodeSource) — only for Claude Code |
| Claude Code | latest, installed globally via npm |
| sshd | hardened: pubkey-only, no root, `dev` user only, host port **2239** |

The repo is bind-mounted at `/workspaces/wave_forge`, so edits sync both ways.

**`target/` is a named volume inside the container** — it masks the Windows
`target/` from the bind mount (wrong-platform artifacts, and a bind-mounted
build dir is slow) and survives container recreates. Host and container each
keep their own build output.

**Named volumes** (survive recreates, including Zed's dev-container flow):
`target/`, `~/.cargo` (`CARGO_HOME`: registry/git caches, `cargo install`ed
binaries), `~/.claude`, `~/.ssh` (incl. the PAT store), and the sshd host
keys. The toolchain itself (`/usr/local/rustup`) is on the image layer and
owned by `dev`, so `rustup update` works but is undone by a rebuild — pin
toolchain changes in the Dockerfile.

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
cargo test --workspace      # GPU-path tests run on lavapipe (CPU) — slow
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
- **A build dir looks empty / stale** — you're seeing the `target` volume,
  not the Windows build; just `cargo build`.
