@tool
## The presets the plugin ships (docs/reference/godot.md, "Editor"): each a scene of a configured
## WaveForgeStages node beside its pack in `addons/wave_forge/presets`. A preset is copied onto a
## node's own settings, never shared, so changing the node changes no other scene.

## Where the presets lie.
const DIRECTORY := "res://addons/wave_forge/presets"
## The preset a node added with nothing set takes in the editor (N1).
const DEFAULT := "res://addons/wave_forge/presets/islands.tscn"

## The presets, by path, in the order of their names.
static func paths() -> PackedStringArray:
	var found := PackedStringArray()
	for file in DirAccess.get_files_at(DIRECTORY):
		if file.ends_with(".tscn"):
			found.append(DIRECTORY.path_join(file))
	found.sort()
	return found

## The settings the preset at `path` gives a node, as property name to value: every property
## WaveForgeStages itself declares, and its parameters back to the pack's defaults.
static func settings(path: String) -> Dictionary:
	var preset: Node = (load(path) as PackedScene).instantiate()
	var values := {}
	for property: Dictionary in ClassDB.class_get_property_list("WaveForgeStages", true):
		if property["usage"] & PROPERTY_USAGE_STORAGE:
			values[property["name"]] = preset.get(property["name"])
	values["params"] = {}
	preset.free()
	return values

## Whether `node` has nothing of its own set yet: no pack, no targets and no ground.
static func is_fresh(node: Node) -> bool:
	return String(node.pack_file).is_empty() and node.stack == null and node.targets.is_empty() and String(node.ground_stage).is_empty()
