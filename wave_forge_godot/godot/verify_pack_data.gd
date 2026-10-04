## A pack as plain data GDScript edits: `pack_dictionary` and `pack_text`.
##
## Run by `../verify.sh` after `verify_params.gd`. Every preset the plugin ships and every test pack
## of the project comes back from its data as the same data. An edited default and a stack of
## stages put in another order save a pack a node starts from, the default as edited. A whole
## number typed as a float is taken where the pack holds an integer. Data no pack holds is refused
## with an empty text, and so is a pack the library refuses.
extends SceneTree

const EDITED := "user://verify_pack_data.world.ron"

func _initialize() -> void:
	var problem: String = _check()
	if problem != "ok":
		printerr("verify_pack_data: " + problem)
		quit(1)
		return
	quit(0)

## What is wrong, or "ok". A script error gives null.
func _check() -> String:
	var paths := PackedStringArray()
	for directory in ["res://addons/wave_forge/presets", "res://"]:
		for file in DirAccess.get_files_at(directory):
			if file.ends_with(".world.ron"):
				paths.append(directory.path_join(file))
	if paths.size() < 15:
		return "only %d packs found" % paths.size()
	for path in paths:
		var data: Dictionary = WaveForgeStages.pack_dictionary(FileAccess.get_file_as_string(path))
		if data.is_empty():
			return "%s gave no data" % path
		var again: Dictionary = WaveForgeStages.pack_dictionary(WaveForgeStages.pack_text(data))
		if again != data:
			return "%s came back from its data as other data" % path
	print("verify_pack_data: %d packs come back from their data as the same data" % paths.size())

	var islands: Dictionary = WaveForgeStages.pack_dictionary(FileAccess.get_file_as_string("res://addons/wave_forge/presets/islands.world.ron"))
	islands["params"]["land"]["default"] = 0.8
	var stages: Array = islands["stages"]
	stages.push_front(stages.pop_back())
	islands["version"] = 1.0
	var text: String = WaveForgeStages.pack_text(islands)
	if text.is_empty():
		return "the edited islands saved no pack"
	var file := FileAccess.open(EDITED, FileAccess.WRITE)
	file.store_string(text)
	file.close()
	var node: Node = ClassDB.instantiate("WaveForgeStages")
	node.pack_file = EDITED
	node.targets = PackedStringArray(["height"])
	root.add_child(node)
	if not node.start():
		return "a node did not start from the edited pack"
	var land: Array = node.pack_params().filter(func(param: Dictionary) -> bool: return param["name"] == "land")
	node.queue_free()
	if land.size() != 1 or not is_equal_approx(land[0]["default"], 0.8):
		return "the edited pack's land parameter is %s" % [land]
	var order: Array = WaveForgeStages.pack_dictionary(text)["stages"].map(func(stage: Dictionary) -> String: return stage["name"])
	if order[0] != stages[0]["name"]:
		return "the stages came back in the order %s" % [order]
	print("verify_pack_data: an edited default and a reordered stack save a pack a node starts from, the default as edited")

	var refused := [
		{"version": 1, "stages": [{"name": "height", "kind": {"Nonsense": {}}}]},
		{"version": 1, "stages": [{"name": "height", "kind": {"Field": {"Constant": Vector3.ONE}}}]},
		{"version": 1, "stages": [{"name": "height", "kind": {"Field": {"Input": "nothing"}}}]},
	]
	for data in refused:
		if not WaveForgeStages.pack_text(data).is_empty():
			return "%s was taken as a pack" % [data]
	print("verify_pack_data: data no pack holds, and a pack the library refuses, save no text")
	return "ok"
