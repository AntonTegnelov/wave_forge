#!/usr/bin/env bash
# Renders a contact sheet of every preset the plugin ships, each parameter at its minimum, default
# and maximum, for judging the presets' looks (developer tool).
#
# Needs a Godot 4 binary (GODOT, or `godot` on PATH) and a display; without one, run it under
# `xvfb-run`. It uses the Compatibility renderer, which runs wherever OpenGL 3.3 does. The sheets
# land in Godot's user directory, and the script prints where.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$here/prepare.sh" "${1:-release}"
exec "${GODOT:-godot}" --rendering-driver opengl3 --path "$here/godot" --script render_presets.gd
