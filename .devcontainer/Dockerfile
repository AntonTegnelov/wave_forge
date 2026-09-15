# syntax=docker/dockerfile:1.6
# Official Rust image, pinned to the same toolchain as rust-toolchain.toml:
# rustup at /usr/local/rustup with proxies in /usr/local/cargo/bin (kept on
# PATH); CARGO_HOME is re-pointed to /home/dev/.cargo (a named volume) so
# registry caches survive recreates. Bump both files together.
FROM rust:1.98.1-bookworm

ARG NODE_MAJOR=22
ARG DEV_USER=dev
ARG DEV_UID=1000
ARG DEV_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# ---- Base packages ----------------------------------------------------------
# Native deps for the crate graph:
#   libx11-dev / libxkbcommon-dev / libwayland-dev / libxcursor-dev /
#   libxrandr-dev / libxi-dev  -> minifb (X11 + Wayland backends, C++ sources)
#   clang / libclang-dev       -> bindgen users
#   libvulkan1 / mesa-vulkan-drivers / vulkan-tools -> wgpu at runtime: Mesa's
#     lavapipe gives a CPU Vulkan device, so GPU-path tests can run headless
#     (slowly). No real GPU is reachable from the container.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg xz-utils unzip \
        git build-essential pkg-config cmake clang libclang-dev \
        libx11-dev libxkbcommon-dev libwayland-dev libxcursor-dev \
        libxrandr-dev libxi-dev libxinerama-dev \
        libvulkan1 mesa-vulkan-drivers vulkan-tools \
        ripgrep fd-find jq less vim \
        openssh-server sudo \
        locales tini \
 && sed -i '/en_US.UTF-8/s/^# //' /etc/locale.gen && locale-gen \
 && rm -rf /var/lib/apt/lists/*

# fd is installed as `fdfind` on Debian; provide the conventional `fd` alias.
RUN ln -s /usr/bin/fdfind /usr/local/bin/fd

# ---- Rust components --------------------------------------------------------
# Baked into the image (rustup home is on the image layer, not a volume).
RUN rustup component add clippy rustfmt rust-analyzer

# ---- Node.js (NodeSource) + Claude Code -------------------------------------
RUN curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/* \
 && npm install -g @anthropic-ai/claude-code \
 && npm cache clean --force

# ---- Dev user ---------------------------------------------------------------
RUN groupadd --gid "${DEV_GID}" "${DEV_USER}" \
 && useradd  --uid "${DEV_UID}" --gid "${DEV_GID}" --create-home --shell /bin/bash "${DEV_USER}" \
 && echo "${DEV_USER} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/${DEV_USER}" \
 && chmod 0440 "/etc/sudoers.d/${DEV_USER}" \
 && install -d -m 0700 -o "${DEV_USER}" -g "${DEV_USER}" "/home/${DEV_USER}/.ssh" \
 && install -d -m 0755 -o "${DEV_USER}" -g "${DEV_USER}" \
        "/home/${DEV_USER}/.cache" "/home/${DEV_USER}/.local" \
        "/home/${DEV_USER}/.config" "/home/${DEV_USER}/.claude" \
        "/home/${DEV_USER}/.cargo" \
 # Let dev manage toolchains/components (rustup update, component add) without sudo.
 && chown -R "${DEV_UID}:${DEV_GID}" /usr/local/rustup /usr/local/cargo

# ---- Env for sshd/PAM login shells ------------------------------------------
# Docker ENV does not propagate into ssh sessions; PAM reads /etc/environment.
# PATH must keep the system dirs or ssh logins lose cargo/node/git/etc.
# ~/.cargo/bin first (cargo-installed tools), then the rustup proxies.
RUN printf '%s\n' \
        'PATH="/home/dev/.cargo/bin:/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"' \
        'CARGO_HOME="/home/dev/.cargo"' \
        'RUSTUP_HOME="/usr/local/rustup"' \
        'LANG="en_US.UTF-8"' \
        >> /etc/environment

# Non-login shells (VS Code terminals, docker exec) get the same via Docker ENV.
ENV CARGO_HOME="/home/${DEV_USER}/.cargo" \
    PATH="/home/${DEV_USER}/.cargo/bin:${PATH}"

# ---- SSH server -------------------------------------------------------------
# Hardened sshd config (pubkey-only, no root, dev user only).
COPY --chmod=0644 .devcontainer/sshd_config /etc/ssh/sshd_config.d/00-dev.conf
RUN sed -i 's/\r$//' /etc/ssh/sshd_config.d/00-dev.conf

# Privilege separation directory required by sshd.
RUN install -d -m 0755 /var/run/sshd

# Entrypoint: install authorized_keys, generate host keys, exec sshd.
COPY --chmod=0755 .devcontainer/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh

# The repo is bind-mounted with host ownership; without this git refuses to
# operate ("dubious ownership") for both SSH and VS Code sessions.
RUN git config --system --add safe.directory /workspaces/wave_forge

# ---- Git auth: fine-grained per-repo PAT over https -------------------------
# The credential store sits on the ssh-config named volume so the token
# survives container recreates. NEVER an SSH key to GitHub in here — SSH keys
# cannot be scoped per-repo, and this container must not reach beyond its own
# repo (see DEVCONTAINER_TEMPLATE.md).
RUN git config --system credential.helper 'store --file /home/dev/.ssh/git-credentials' \
 && git config --system user.name "Anton Tegnelöv" \
 && git config --system user.email "anton.tegnelov@gmail.com" \
 # The Windows host checked the worktree out with core.autocrlf=true (CRLF
 # on disk, LF in the index). The container shares that worktree via the
 # bind mount, so it must apply the same conversion or every text file
 # shows as modified. .gitattributes still forces LF for .devcontainer/**.
 && git config --system core.autocrlf true

# Used by the entrypoint to self-heal ownership of workspace-nested volumes.
ENV WORKSPACE_DIR=/workspaces/wave_forge

WORKDIR /workspaces/wave_forge
EXPOSE 22

# tini is provided by docker-compose's `init: true`, which runs it as PID 1.
# Don't wrap with another tini here or you get nested-tini warnings and
# broken signal handling.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
