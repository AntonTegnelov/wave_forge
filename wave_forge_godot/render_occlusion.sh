#!/usr/bin/env bash
# Measures occluders in the city, seen from above the roofs and from a street: objects drawn and
# frame times with occlusion culling off, on, and on while occluders are rebuilt every frame
# (developer tool).
#
# Needs a Godot 4 binary (GODOT, or `godot` on PATH) and a display; without one, run it under
# `xvfb-run`. It uses the Compatibility renderer, which runs wherever OpenGL 3.3 does.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$here/prepare.sh" "${1:-release}"
"${GODOT:-godot}" --rendering-driver opengl3 --path "$here/godot" --script render_occlusion.gd -- above
exec "${GODOT:-godot}" --rendering-driver opengl3 --path "$here/godot" --script render_occlusion.gd -- street
