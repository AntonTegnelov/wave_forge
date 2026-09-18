#!/usr/bin/env bash
# Builds the extension, copies it into the verification project and drives it with Godot headless.
#
# Needs a Godot 4 binary: set GODOT, or have `godot` on PATH. The check itself is `godot/verify.gd`,
# which generates a 3x3-chunk world through the node and asserts what a game would rely on.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
godot="${GODOT:-godot}"
profile="${1:-debug}"

case "$profile" in
	debug) cargo build --manifest-path "$here/Cargo.toml" ;;
	release) cargo build --release --manifest-path "$here/Cargo.toml" ;;
	*) echo "usage: verify.sh [debug|release]" >&2; exit 2 ;;
esac

target="$(cargo metadata --no-deps --format-version 1 --manifest-path "$here/Cargo.toml" | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')"
mkdir -p "$here/godot/bin"
cp "$target/$profile/libwave_forge_godot.so" "$here/godot/bin/"

# Godot only scans for .gdextension files when the editor opens a project, so a headless run needs
# the list it would have written.
mkdir -p "$here/godot/.godot"
echo "res://wave_forge.gdextension" > "$here/godot/.godot/extension_list.cfg"

exec "$godot" --headless --path "$here/godot" --script verify.gd
