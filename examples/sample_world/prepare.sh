#!/usr/bin/env bash
# Builds the Wave Forge extension and installs it in this project as a game would: the extension
# library in bin/ and the Wave Forge addon, with its presets and city kit, in addons/.
#
# Usage: prepare.sh [debug|release]. Needs Rust (https://rustup.rs); the first build takes a few
# minutes.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$here/../../wave_forge_godot/install.sh" "$here" "${1:-release}"
