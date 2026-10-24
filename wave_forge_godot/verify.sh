#!/usr/bin/env bash
# Builds the extension, prepares the verification project and drives it with Godot headless.
#
# Needs a Godot 4 binary: set GODOT, or have `godot` on PATH. The check itself is `godot/verify.gd`,
# which runs a focus through a streamed world in real time and asserts what a game would rely on,
# then `godot/verify_follow.gd`, which lets a world follow a camera with no code and keep its
# kernels, then `godot/verify_stages.gd`, which generates the valley pack's stages and checks what they hold,
# then `godot/verify_signals.gd`, which spreads many products' signals over frames by time,
# then `godot/verify_tables.gd`, which gives a pack tables of facts from GDScript, then
# `godot/verify_decals.gd`, which lays decals along a road, then
# `godot/verify_continent_kits.gd`, which checks and binds authored town models, then
# `godot/verify_painted.gd`, which gives a pack a painted map, then
# `godot/verify_noise.gd`, which reads a FastNoiseLite resource through a pack, then
# `godot/verify_edits.gd`, which fells a tree and raises the ground, then `godot/verify_frozen.gd`,
# which keeps frozen chunks in a directory while they are out of memory, then `godot/verify_assemble.gd`,
# which places a village's pieces, then `godot/verify_scenes.gd`, which binds scenes to them, then
# `godot/verify_variants.gd`, which binds a kind's variants to scenes of their own, then
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
# which follows the maximal preset's first settlement to its town, then
# `godot/verify_run_exit.gd`, which runs a world into a directory and quits, Godot exiting cleanly,
# then `godot/verify_inspector.gd`,
# which checks the nodes' configuration warnings and inspector buttons, then
# `godot/verify_typed_maps.gd`, which loads a scene saved with untyped maps into the typed ones, then
# `godot/verify_presets.gd`, which takes every preset the plugin ships as a new node and checks it
# stands a lit, walkable world with nothing printed, then `godot/verify_walk.gd`, which walks
# the default preset with the plugin's walker and no code, then `godot/verify_demo.gd`, which runs
# the addon's demo scene, lit by its sun and sky, with nothing printed, then
# `godot/verify_scene_paths.gd`, which loads many scenes bound by path with nothing printed, then
# the editor itself, headless,
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
# Each run of Godot gets at most this many seconds. A script error in a SceneTree script's
# `_initialize`, or a wait that never ends, leaves headless Godot running; the limit fails it by
# name instead of holding the job until CI gives up on it.
limit_s="${VERIFY_TIMEOUT_S:-600}"
# Runs Godot with the given arguments under the limit, naming `$1`, the run, if it hits it.
godot_run() {
	local name="$1" status=0
	shift
	timeout --kill-after=10 "$limit_s" "${GODOT:-godot}" "$@" || status=$?
	if ((status == 124 || status == 137)); then
		echo "verify: $name did not finish within $limit_s s" >&2
	fi
	return "$status"
}
"$here/prepare.sh" "${1:-debug}"
godot_run verify.gd --headless --path "$here/godot" --script verify.gd
godot_run verify_follow.gd --headless --path "$here/godot" --script verify_follow.gd
godot_run verify_stages.gd --headless --path "$here/godot" --script verify_stages.gd
godot_run verify_signals.gd --headless --path "$here/godot" --script verify_signals.gd
godot_run verify_tables.gd --headless --path "$here/godot" --script verify_tables.gd
godot_run verify_decals.gd --headless --path "$here/godot" --script verify_decals.gd
godot_run verify_continent_kits.gd --headless --path "$here/godot" --script verify_continent_kits.gd
godot_run verify_painted.gd --headless --path "$here/godot" --script verify_painted.gd
godot_run verify_noise.gd --headless --path "$here/godot" --script verify_noise.gd
godot_run verify_edits.gd --headless --path "$here/godot" --script verify_edits.gd
godot_run verify_frozen.gd --headless --path "$here/godot" --script verify_frozen.gd
godot_run verify_assemble.gd --headless --path "$here/godot" --script verify_assemble.gd
godot_run verify_scenes.gd --headless --path "$here/godot" --script verify_scenes.gd
godot_run verify_variants.gd --headless --path "$here/godot" --script verify_variants.gd
godot_run verify_cave_scenes.gd --headless --path "$here/godot" --script verify_cave_scenes.gd
godot_run verify_pooling.gd --headless --path "$here/godot" --script verify_pooling.gd
godot_run verify_ground.gd --headless --path "$here/godot" --script verify_ground.gd
godot_run verify_water.gd --headless --path "$here/godot" --script verify_water.gd
godot_run verify_ambience.gd --headless --path "$here/godot" --script verify_ambience.gd
godot_run verify_far.gd --headless --path "$here/godot" --script verify_far.gd
godot_run verify_volume.gd --headless --path "$here/godot" --script verify_volume.gd
godot_run verify_bake.gd --headless --path "$here/godot" --script verify_bake.gd
godot_run verify_import.gd --headless --path "$here/godot" --script verify_import.gd
godot_run verify_params.gd --headless --path "$here/godot" --script verify_params.gd
godot_run verify_pack_data.gd --headless --path "$here/godot" --script verify_pack_data.gd
godot_run verify_stack.gd --headless --path "$here/godot" --script verify_stack.gd
godot_run verify_paint.gd --headless --path "$here/godot" --script verify_paint.gd
godot_run verify_candidates.gd --headless --path "$here/godot" --script verify_candidates.gd
godot_run verify_world.gd --headless --path "$here/godot" --script verify_world.gd
godot_run verify_continent.gd --headless --path "$here/godot" --script verify_continent.gd
godot_run verify_run_exit.gd --headless --path "$here/godot" --script verify_run_exit.gd
godot_run verify_inspector.gd --headless --path "$here/godot" --script verify_inspector.gd
godot_run verify_typed_maps.gd --headless --path "$here/godot" --script verify_typed_maps.gd
# N1: a preset's first play, and walking it, print no error and no warning; the device's own
# notices aside. So do many scenes bound by path loading at once (#465).
for script in verify_presets.gd verify_walk.gd verify_demo.gd verify_scene_paths.gd; do
	log="$(mktemp)"
	godot_run "$script" --headless --path "$here/godot" --script "$script" 2>&1 | tee "$log"
	if grep -E "^(ERROR|WARNING|SCRIPT ERROR)" "$log" | grep -v -E "dzn is not a conformant|XDG_RUNTIME_DIR"; then
		echo "verify: $script printed an error or a warning" >&2
		exit 1
	fi
done
editor_log="$(mktemp)"
godot_run "the editor" --headless --editor --path "$here/godot" --quit-after 300 >"$editor_log" 2>&1
if grep -E "SCRIPT ERROR|Failed to load script|verify_plugin" "$editor_log" | grep -v "^verify_plugin: the dock"; then
	echo "verify: the editor plugin does not load, or its dock or shortcuts are wrong" >&2
	exit 1
fi
if ! grep -q "^verify_plugin: the dock" "$editor_log"; then
	echo "verify: the editor never checked the plugin's dock and shortcuts" >&2
	exit 1
fi
echo "verify: the editor loads the plugin without a script error, its dock an EditorDock with shortcuts"
godot_run verify_sound.gd --headless --path "$here/godot" --script verify_sound.gd
godot_run verify_names.gd --headless --path "$here/godot" --script verify_names.gd
godot_run verify_occlusion.gd --headless --path "$here/godot" --script verify_occlusion.gd
godot_run verify_proxies.gd --headless --path "$here/godot" --script verify_proxies.gd
"$here/../examples/history/prepare.sh" "${1:-debug}"
godot_run "the history example's check" --headless --path "$here/../examples/history" --script check.gd
"$here/../examples/new_world/prepare.sh" "${1:-debug}"
godot_run "the new world's check" --headless --path "$here/../examples/new_world" --script check.gd
