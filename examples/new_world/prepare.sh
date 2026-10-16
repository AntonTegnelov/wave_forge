#!/usr/bin/env bash
# Builds the Wave Forge extension and installs it in this project as a game would, the extension
# library in bin/ and the Wave Forge addon in addons/, and copies in the maximal preset's continent
# (its scene, its pack, its eight cultures and its towns' models), the memory reading measure.gd takes, and the walker
# the checks use, which stands in for a game's own player.
#
# Usage: prepare.sh [debug|release]. Needs Rust (https://rustup.rs); the first build takes a few
# minutes.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
godot="$here/../../wave_forge_godot"
"$godot/install.sh" "$here" "${1:-release}"
rm -rf "$here/continent"
cp -r "$godot/godot/continent" "$here/continent"
cp "$godot/godot/continent.tscn" "$godot/godot/continent.gd" "$godot/godot/peak_memory.gd" \
	"$godot/godot/walker.tscn" "$godot/godot/walker.gd" "$here/"
# The scenes the continent's towns are drawn with, once tools/make_continent_kits.gd has bound some.
rm -rf "$here/continent_kits"
if [ -d "$godot/godot/continent_kits" ]; then
	cp -r "$godot/godot/continent_kits" "$here/continent_kits"
fi
