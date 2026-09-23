#!/usr/bin/env bash
# Renders a generated city with the module models and saves a picture, to look at (developer tool).
#
# Needs a Godot 4 binary (GODOT, or `godot` on PATH) and a display; without one, run it under
# `xvfb-run`. It uses the Compatibility renderer, which runs wherever OpenGL 3.3 does. The picture
# lands in Godot's user directory, and the script prints where. After the build profile, `server`
# (the default) or `nodes` chooses how the city is drawn; see render_city.gd.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$here/prepare.sh" "${1:-release}"
exec "${GODOT:-godot}" --rendering-driver opengl3 --path "$here/godot" --script render_city.gd -- \
	"${2:-server}"
