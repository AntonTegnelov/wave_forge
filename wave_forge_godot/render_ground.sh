#!/usr/bin/env bash
# Renders a pack's ground with a material per category and grass and saves pictures, to look at,
# timing grass; then draws trees with the vegetation shader and checks they move in the wind; then
# draws a wide view at the ground's levels of detail and checks no gap opens between chunks; then
# draws the far ground beyond the near ground and checks no view sees between them, at a coarse
# scale of 8 and of 4; then draws a valley's lakes and river and checks the water shows with no gap
# against its banks; then draws the continent's far ground toward a mountain range 1 to 2 km away
# (developer tool).
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
"${GODOT:-godot}" --rendering-driver opengl3 --path "$here/godot" --script render_far.gd
"${GODOT:-godot}" --rendering-driver opengl3 --path "$here/godot" --script render_far.gd -- far4
"${GODOT:-godot}" --rendering-driver opengl3 --path "$here/godot" --script render_water.gd
exec "${GODOT:-godot}" --rendering-driver opengl3 --resolution 1600x900 --path "$here/godot" --script render_continent_far.gd
