#!/usr/bin/env bash
# Builds the extension, prepares the verification project and drives it with Godot headless.
#
# Needs a Godot 4 binary: set GODOT, or have `godot` on PATH. The check itself is `godot/verify.gd`,
# which runs a focus through a streamed world in real time and asserts what a game would rely on,
# then `godot/verify_stages.gd`, which generates the valley pack's stages and checks what they hold,
# then `godot/verify_tables.gd`, which gives a pack tables of facts from GDScript.
# Build in release: its frame-time bars describe the extension a game would ship.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$here/prepare.sh" "${1:-debug}"
"${GODOT:-godot}" --headless --path "$here/godot" --script verify.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_stages.gd
exec "${GODOT:-godot}" --headless --path "$here/godot" --script verify_tables.gd
