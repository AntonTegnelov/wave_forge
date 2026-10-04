#!/usr/bin/env bash
# Builds the Wave Forge extension and installs it in this project as a game would: the extension
# library in bin/ and the Wave Forge addon, with its presets and city kit, in addons/.
#
# Usage: prepare.sh [debug|release]. Needs Rust (https://rustup.rs); the first build takes a few
# minutes.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$here/../.."
profile="${1:-release}"

# The extension's own project puts the library and the addon's presets and city kit in place.
"$repo/wave_forge_godot/prepare.sh" "$profile"
mkdir -p "$here/bin" "$here/addons"
cp "$repo/wave_forge_godot/godot/bin/libwave_forge_godot.so" "$here/bin/"
rm -rf "$here/addons/wave_forge"
cp -r "$repo/wave_forge_godot/godot/addons/wave_forge" "$here/addons/"
# Godot writes this list only when the editor opens the project; a run without it needs it too.
mkdir -p "$here/.godot"
echo "res://wave_forge.gdextension" > "$here/.godot/extension_list.cfg"
