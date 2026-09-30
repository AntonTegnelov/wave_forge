#!/usr/bin/env bash
# Builds the extension and puts everything the Godot project in ./godot loads next to it: the
# extension library, the city rule set, the valley pack, and the city's module models.
#
# Usage: prepare.sh [debug|release]. Build in release when timing: that is what a game ships.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
profile="${1:-debug}"

case "$profile" in
	debug) cargo build --manifest-path "$here/Cargo.toml" ;;
	release) cargo build --release --manifest-path "$here/Cargo.toml" ;;
	*) echo "usage: prepare.sh [debug|release]" >&2; exit 2 ;;
esac

target="$(cargo metadata --no-deps --format-version 1 --manifest-path "$here/Cargo.toml" | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')"
mkdir -p "$here/godot/bin"
cp "$target/$profile/libwave_forge_godot.so" "$here/godot/bin/"
# The city the library's own tests load, so the project reads the same file, and its models.
cp "$here/../examples/city.ron" "$here/godot/city.ron"
cp "$here/../examples/valley.world.ron" "$here/godot/valley.world.ron"
cp "$here/../examples/presets/islands.world.ron" "$here/godot/islands.world.ron"
# The maximal preset's continent, its eight cultures and the history a game gives it.
mkdir -p "$here/godot/continent/cultures"
cp "$here/../examples/continent/continent.world.ron" "$here/../examples/continent/history.json" "$here/godot/continent/"
cp "$here/../examples/continent/cultures/"*.ron "$here/godot/continent/cultures/"
# The presets the editor plugin lists, shipped with it.
mkdir -p "$here/godot/addons/wave_forge/presets"
cp "$here/../examples/presets/"*.world.ron "$here/godot/addons/wave_forge/presets/"
cargo run --quiet --manifest-path "$here/../Cargo.toml" -p wfc-devtools --bin wfc-export-models -- \
	--out "$here/godot/models"

# Godot only scans for .gdextension files when the editor opens a project, so a run without the
# editor needs the list it would have written.
mkdir -p "$here/godot/.godot"
echo "res://wave_forge.gdextension" > "$here/godot/.godot/extension_list.cfg"
