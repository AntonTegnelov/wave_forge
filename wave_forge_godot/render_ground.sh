#!/usr/bin/env bash
# Renders a pack's ground with a material per category and grass and saves pictures, to look at,
# timing grass; then draws trees with the vegetation shader and checks they move in the wind; then
# draws a wide view at the ground's levels of detail and checks no gap opens between chunks; then
# draws the far ground beyond the near ground and checks no view sees between them (developer tool).
#
# Needs a Godot 4 binary (GODOT, or `godot` on PATH) and a display; without one, run it under
# `xvfb-run`. It uses the Compatibility renderer, which runs wherever OpenGL 3.3 does. The picture
# lands in Godot's user directory, and the script prints where.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$here/prepare.sh" "${1:-release}"
"${GODOT:-godot}" --rendering-driver opengl3 --path "$here/godot" --script render_ground.gd
"${GODOT:-godot}" --rendering-driver opengl3 --path "$here/godot" --script render_wind.gd
"${GODOT:-godot}" --rendering-driver opengl3 --path "$here/godot" --script render_lods.gd
exec "${GODOT:-godot}" --rendering-driver opengl3 --path "$here/godot" --script render_far.gd
