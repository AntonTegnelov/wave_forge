## Binds the continent's authored town pieces (#310): every model under
## `examples/continent/kits/<batch>/<piece>.glb` that meets its module's faces becomes
## `continent_kits/<batch>/<piece>.tscn`, and `continent.tscn` binds each module to its scene in
## `scenes`. Prints why each refused model was refused and which modules still have no model, and
## exits non-zero if any model was refused. Run `../prepare.sh` first, then from
## `wave_forge_godot/godot`: `godot --headless --path . --script tools/make_continent_kits.gd`.
extends SceneTree

const ContinentKits := preload("res://tools/continent_kits.gd")
const OUT := "res://continent_kits"
const SCENE := "res://continent.tscn"

func _initialize() -> void:
	var examples := ProjectSettings.globalize_path("res://").path_join("../../examples/continent").simplify_path()
	var sets: Dictionary[String, String] = {}
	for file in DirAccess.get_files_at(examples.path_join("cultures")):
		if file.get_extension() == "ron":
			sets[file.get_basename()] = FileAccess.get_file_as_string(examples.path_join("cultures").path_join(file))
	var outcome := ContinentKits.build(examples.path_join("kits"), sets, OUT)
	for why in outcome.refused:
		printerr("make_continent_kits: " + why)
	var scene := FileAccess.get_file_as_string(SCENE)
	var file := FileAccess.open(SCENE, FileAccess.WRITE)
	file.store_string(ContinentKits.bind(scene, outcome.bound))
	file.close()
	var missing := ContinentKits.missing(sets, outcome.bound)
	print("make_continent_kits: bound %d modules in %s; %d refused; %d have no model yet%s" % [
		outcome.bound.size(), SCENE, outcome.refused.size(), missing.size(),
		"" if missing.is_empty() else ": " + ", ".join(missing)])
	quit(0 if outcome.refused.is_empty() else 1)
