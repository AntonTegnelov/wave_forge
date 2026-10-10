#!/usr/bin/env bash
# Renders every preset the plugin ships at its defaults from the reference views (eye level,
# oblique and top-down) under the reference look, once on the Compatibility renderer and once on
# Forward+, for judging how the terrain looks (developer tool).
#
# Needs a Godot 4 binary (GODOT, or `godot` on PATH), a display (without one, run it under
# `xvfb-run`) and a Vulkan driver for Forward+. The sheets land in Godot's user directory, and the
# script prints where.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$here/prepare.sh" "${1:-release}"
"${GODOT:-godot}" --rendering-method gl_compatibility --rendering-driver opengl3 --path "$here/godot" --script render_presets.gd -- views
exec "${GODOT:-godot}" --rendering-method forward_plus --rendering-driver vulkan --path "$here/godot" --script render_presets.gd -- views
