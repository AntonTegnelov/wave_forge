#!/usr/bin/env bash
# Builds the Wave Forge extension and puts what this project loads next to it: the extension
# library, the city rule set its towns are built from, and the city's module models.
#
# Usage: prepare.sh [debug|release]. Needs Rust (https://rustup.rs); the first build takes a few
# minutes.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$here/../.."
profile="${1:-release}"

case "$profile" in
	debug) cargo build --manifest-path "$repo/wave_forge_godot/Cargo.toml" ;;
	release) cargo build --release --manifest-path "$repo/wave_forge_godot/Cargo.toml" ;;
	*) echo "usage: prepare.sh [debug|release]" >&2; exit 2 ;;
esac

target="$(cargo metadata --no-deps --format-version 1 --manifest-path "$repo/wave_forge_godot/Cargo.toml" | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')"
mkdir -p "$here/bin"
cp "$target/$profile/libwave_forge_godot.so" "$here/bin/"
cp "$repo/examples/city.ron" "$here/city.ron"
cargo run --quiet --manifest-path "$repo/Cargo.toml" -p wfc-devtools --bin wfc-export-models -- \
	--out "$here/models"
# Godot writes this list only when the editor opens the project; a run without it needs it too.
mkdir -p "$here/.godot"
echo "res://wave_forge.gdextension" > "$here/.godot/extension_list.cfg"
