#!/usr/bin/env bash
# Builds the Wave Forge extension and installs it in a Godot project as a game would: the extension
# library in its bin/ and the Wave Forge addon, with its presets and city kit, in its addons/. The
# example projects' prepare.sh run it.
#
# Usage: install.sh <project folder> [debug|release]. Needs Rust (https://rustup.rs); the first build
# takes a few minutes.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$1"
profile="${2:-release}"

# The extension's own project puts the library and the addon's presets and city kit in place.
"$here/prepare.sh" "$profile"
mkdir -p "$project/bin" "$project/addons"
cp "$here/godot/bin/libwave_forge_godot.so" "$project/bin/"
rm -rf "$project/addons/wave_forge"
cp -r "$here/godot/addons/wave_forge" "$project/addons/"
# Godot writes this list only when the editor opens the project; a run without it needs it too.
mkdir -p "$project/.godot"
echo "res://wave_forge.gdextension" > "$project/.godot/extension_list.cfg"
