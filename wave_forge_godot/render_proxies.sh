#!/usr/bin/env bash
# Draws the city near and far, handing each chunk from its modules to its far proxy, saves a
# picture from each and checks only one of the two draws (developer tool).
#
# Needs a Godot 4 binary (GODOT, or `godot` on PATH) and a display; without one, run it under
# `xvfb-run`. It uses the Compatibility renderer, which runs wherever OpenGL 3.3 does.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$here/prepare.sh" "${1:-release}"
exec "${GODOT:-godot}" --rendering-driver opengl3 --path "$here/godot" --script render_proxies.gd
