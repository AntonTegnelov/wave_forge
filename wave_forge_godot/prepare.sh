#!/usr/bin/env bash
# Builds the extension and puts everything the Godot project in ./godot loads next to it: the
# extension library, the city rule set, and the city's module models.
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
cargo run --quiet --manifest-path "$here/../Cargo.toml" -p wfc-devtools --bin wfc-export-models -- \
	--out "$here/godot/models"

# Godot only scans for .gdextension files when the editor opens a project, so a run without the
# editor needs the list it would have written.
mkdir -p "$here/godot/.godot"
echo "res://wave_forge.gdextension" > "$here/godot/.godot/extension_list.cfg"
