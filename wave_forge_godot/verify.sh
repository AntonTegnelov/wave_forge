#!/usr/bin/env bash
# Builds the extension, prepares the verification project and drives it with Godot headless.
#
# Needs a Godot 4 binary: set GODOT, or have `godot` on PATH. The check itself is `godot/verify.gd`,
# which runs a focus through a streamed world in real time and asserts what a game would rely on,
# then `godot/verify_stages.gd`, which generates the valley pack's stages and checks what they hold,
# then `godot/verify_tables.gd`, which gives a pack tables of facts from GDScript, then
# `godot/verify_noise.gd`, which reads a FastNoiseLite resource through a pack, then
# `godot/verify_edits.gd`, which fells a tree and raises the ground, then `godot/verify_frozen.gd`,
# which keeps frozen chunks in a directory while they are out of memory, then `godot/verify_assemble.gd`,
# which places a village's pieces, then `godot/verify_scenes.gd`, which binds scenes to them, then
# `godot/verify_cave_scenes.gd`, which binds scenes to a cave's rooms and spawns, then
# `godot/verify_pooling.gd`, which reuses the nodes of scenes that reset themselves, then
# `godot/verify_ground.gd`, which gives the ground a material per category, then
# `godot/verify_water.gd`, which draws a valley's lakes and river as water, then
# `godot/verify_ambience.gd`, which plays the terrain's sounds along a river and at a shore, then
# `godot/verify_far.gd`, which draws the far ground where the near ground is missing, then
# `godot/verify_volume.gd`, which draws a volume's surface and collides with its cave, then
# `godot/verify_bake.gd`, which bakes an area into a scene of plain nodes, then
# `godot/verify_import.gd`, which proposes a module set from a MeshLibrary, then
# `godot/verify_params.gd`, which tunes a preset's parameters, then `godot/verify_pack_data.gd`,
# which edits packs as plain data and saves them, then `godot/verify_stack.gd`, which generates a
# pack from a stack of stages edited in place, then `godot/verify_paint.gd`, which
# paints strokes as the editor's brushes do, then `godot/verify_candidates.gd`, which draws a
# Scatter stage's candidates by what became of each, then `godot/verify_world.gd`, which runs a
# finite world whole, stopping and resuming, and plays it back, then `godot/verify_continent.gd`,
# which follows the maximal preset's first settlement to its town, then `godot/verify_inspector.gd`,
# which checks the nodes' configuration warnings and inspector buttons, then
# `godot/verify_typed_maps.gd`, which loads a scene saved with untyped maps into the typed ones, then
# `godot/verify_presets.gd`, which takes every preset the plugin ships as a new node and checks it
# stands a lit, walkable world with nothing printed, then `godot/verify_walk.gd`, which walks
# the default preset with the plugin's walker and no code, then the editor itself, headless,
# which has to load the editor plugin (`godot/addons/wave_forge`) without a script error, its dock
# an `EditorDock` and its shortcuts in the editor settings (`godot/verify_plugin.gd`), then
# `godot/verify_sound.gd`, which checks the city's region tags and sound, then
# `godot/verify_names.gd`, which names a location through a translation, then
# `godot/verify_occlusion.gd`, which checks the city's occluders, then `godot/verify_proxies.gd`,
# which checks its far proxies, then the history example's own check
# (examples/history/check.gd), and last the new world's (examples/new_world/check.gd).
# Build in release: its frame-time bars describe the extension a game would ship.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$here/prepare.sh" "${1:-debug}"
"${GODOT:-godot}" --headless --path "$here/godot" --script verify.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_stages.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_tables.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_noise.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_edits.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_frozen.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_assemble.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_scenes.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_cave_scenes.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_pooling.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_ground.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_water.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_ambience.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_far.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_volume.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_bake.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_import.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_params.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_pack_data.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_stack.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_paint.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_candidates.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_world.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_continent.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_inspector.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_typed_maps.gd
# N1: a preset's first play, and walking it, print no error and no warning; the device's own
# notices aside.
for script in verify_presets.gd verify_walk.gd; do
	log="$(mktemp)"
	"${GODOT:-godot}" --headless --path "$here/godot" --script "$script" 2>&1 | tee "$log"
	if grep -E "^(ERROR|WARNING|SCRIPT ERROR)" "$log" | grep -v -E "dzn is not a conformant|XDG_RUNTIME_DIR"; then
		echo "verify: $script printed an error or a warning" >&2
		exit 1
	fi
done
editor_log="$(mktemp)"
"${GODOT:-godot}" --headless --editor --path "$here/godot" --quit-after 300 >"$editor_log" 2>&1
if grep -E "SCRIPT ERROR|Failed to load script|verify_plugin" "$editor_log" | grep -v "^verify_plugin: the dock"; then
	echo "verify: the editor plugin does not load, or its dock or shortcuts are wrong" >&2
	exit 1
fi
if ! grep -q "^verify_plugin: the dock" "$editor_log"; then
	echo "verify: the editor never checked the plugin's dock and shortcuts" >&2
	exit 1
fi
echo "verify: the editor loads the plugin without a script error, its dock an EditorDock with shortcuts"
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_sound.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_names.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_occlusion.gd
"${GODOT:-godot}" --headless --path "$here/godot" --script verify_proxies.gd
"$here/../examples/history/prepare.sh" "${1:-debug}"
"${GODOT:-godot}" --headless --path "$here/../examples/history" --script check.gd
"$here/../examples/new_world/prepare.sh" "${1:-debug}"
exec "${GODOT:-godot}" --headless --path "$here/../examples/new_world" --script check.gd
