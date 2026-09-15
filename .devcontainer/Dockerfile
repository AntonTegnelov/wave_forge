# syntax=docker/dockerfile:1.6

# ---- Stage 1: Mesa "dozen" (Vulkan on Direct3D 12) -------------------------
# Inside WSL2 the NVIDIA Windows driver exposes D3D12 and CUDA, not Vulkan.
# Mesa's dozen driver translates Vulkan to D3D12, which is how wgpu reaches
# the host GPU from the container. Neither Debian nor Ubuntu packages it, so
# it is built from source here (only the dzn ICD; no GL, no gallium, no LLVM)
# and copied into the final image. Runtime needs /usr/lib/wsl/{lib,drivers}
# mounted by docker-compose and `gpus: all`.
# Same pinned base as the final stage: reproducible, and the layers are shared.
FROM rust:1.98.1-bookworm AS dzn-build

ARG MESA_VERSION=26.2.2
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates curl xz-utils git \
        build-essential pkg-config bison flex \
        python3 python3-pip python3-venv \
        zlib1g-dev libzstd-dev libexpat1-dev libdrm-dev \
 && rm -rf /var/lib/apt/lists/*

# Debian bookworm's meson (1.0) is too old for current Mesa; use a venv.
RUN python3 -m venv /opt/mesa-venv \
 && /opt/mesa-venv/bin/pip install --no-cache-dir meson ninja mako pyyaml packaging

RUN curl -fsSL "https://archive.mesa3d.org/mesa-${MESA_VERSION}.tar.xz" -o /tmp/mesa.tar.xz \
 && mkdir -p /src/mesa \
 && tar -xJf /tmp/mesa.tar.xz -C /src/mesa --strip-components=1 \
 && rm /tmp/mesa.tar.xz

WORKDIR /src/mesa
# DirectX-Headers is fetched by meson's wrap (Debian's 1.606 is too old).
RUN PATH="/opt/mesa-venv/bin:${PATH}" meson setup build \
        --prefix=/opt/dzn \
        --buildtype=release \
        -Dplatforms= \
        -Dglx=disabled -Degl=disabled -Dgbm=disabled \
        -Dgles1=disabled -Dgles2=disabled -Dopengl=false \
        -Dgallium-drivers= \
        -Dvulkan-drivers=microsoft-experimental \
        -Dvulkan-layers= \
        -Dllvm=disabled \
        -Dtools= \
        -Dvideo-codecs= \
        -Dlibunwind=disabled -Dvalgrind=disabled \
 && PATH="/opt/mesa-venv/bin:${PATH}" ninja -C build install

# ---- Stage 2: the dev container ----------------------------------------------
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
#     lavapipe gives a CPU Vulkan device as a fallback; the host GPU is
#     reached through the dozen driver installed below.
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

# ---- GitHub CLI ---------------------------------------------------------------
# Used for PRs/issues with the repo-scoped PAT (`gh auth login --with-token`).
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
 && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends gh \
 && rm -rf /var/lib/apt/lists/*

# ---- Vulkan on the host GPU: Mesa dozen ICD ---------------------------------
# The driver library stays under /opt/dzn; only its ICD manifest goes into the
# Vulkan loader's search path (with the library path rewritten to match).
# /usr/lib/wsl/lib holds libd3d12.so etc. at runtime (compose bind mount);
# the entrypoint runs ldconfig so the dynamic linker finds them. lavapipe's
# ICD stays installed, so `vulkaninfo --summary` lists both the RTX and
# llvmpipe; wgpu's default (high performance) adapter request picks the GPU.
COPY --from=dzn-build /opt/dzn /opt/dzn
RUN mkdir -p /usr/share/vulkan/icd.d \
 && src="$(find /opt/dzn -name 'dzn_icd.*.json' | head -1)" \
 && lib="$(find /opt/dzn -name 'libvulkan_dzn.so' | head -1)" \
 && sed "s#\"library_path\": *\"[^\"]*\"#\"library_path\": \"${lib}\"#" "$src" \
        > /usr/share/vulkan/icd.d/dzn_icd.x86_64.json \
 && echo /usr/lib/wsl/lib > /etc/ld.so.conf.d/wsl.conf

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
# WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER: Mesa marks dozen non-conformant
# (conformanceVersion 0.0.0.0) and wgpu hides such adapters unless this flag
# is set. It only takes effect for instances built with
# `wgpu::InstanceDescriptor::new_without_display_handle_from_env()` in wgpu 30
# (Instance::default() does not read the environment).
RUN printf '%s\n' \
        'PATH="/home/dev/.cargo/bin:/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"' \
        'CARGO_HOME="/home/dev/.cargo"' \
        'RUSTUP_HOME="/usr/local/rustup"' \
        'LANG="en_US.UTF-8"' \
        'WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER="1"' \
        >> /etc/environment

# Non-login shells (VS Code terminals, docker exec) get the same via Docker ENV.
ENV CARGO_HOME="/home/${DEV_USER}/.cargo" \
    PATH="/home/${DEV_USER}/.cargo/bin:${PATH}" \
    WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER=1

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
