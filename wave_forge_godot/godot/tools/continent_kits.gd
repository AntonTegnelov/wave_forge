## The continent's authored town pieces as scenes its towns are drawn with (#310,
## docs/reference/godot.md, "Scenes"): each model a glTF binary named for its module, in a folder per
## batch, `shared` for the pieces every culture has and a culture's file name for its own. A model is
## checked against its module's faces in every module set it belongs to
## (`WaveForgeWorld.check_kit_model`) and holds exactly one mesh, unmoved; one that passes is saved
## as a mesh and a scene of a lone `MeshInstance3D`, which the node draws as a MultiMesh and collides
## with. `make_continent_kits.gd` runs it over the repository's kits.
extends RefCounted

## Modules that draw nothing, air and a stair's headroom: a model for one is not looked at.
const NOTHING_TO_DRAW := ["air", "stair_head"]

## What came of a run: the scene path bound to each module, and why each refused model was refused,
## as "batch/piece: why".
class Outcome:
	var bound: Dictionary[String, String] = {}
	var refused: PackedStringArray = []

## Imports every model under `kits`, a folder of batch folders, checking each against the module
## sets `sets` (a culture's file name to its text), and saves those that pass under `out`, a
## `res://` folder, as `<batch>/<piece>.res` and `.tscn`.
static func build(kits: String, sets: Dictionary[String, String], out: String) -> Outcome:
	var outcome := Outcome.new()
	for batch in DirAccess.get_directories_at(kits):
		var batch_sets: Array[String] = []
		if batch == "shared":
			batch_sets.assign(sets.keys())
		elif sets.has(batch):
			batch_sets.append(batch)
		else:
			outcome.refused.append("%s: no culture's module set is named %s" % [batch, batch])
			continue
		DirAccess.make_dir_recursive_absolute(out.path_join(batch))
		for file in DirAccess.get_files_at(kits.path_join(batch)):
			if file.get_extension() != "glb":
				continue
			var module := file.get_basename()
			if module in NOTHING_TO_DRAW:
				continue
			var problems := PackedStringArray()
			var mesh := _mesh(kits.path_join(batch).path_join(file), problems)
			if mesh != null:
				for culture in batch_sets:
					for problem in WaveForgeWorld.check_kit_model(sets[culture], module, mesh.get_faces()):
						problems.append("%s (in %s)" % [problem, culture])
			if not problems.is_empty():
				for problem in problems:
					outcome.refused.append("%s/%s: %s" % [batch, module, problem])
				continue
			var path := out.path_join(batch).path_join(module)
			var saved := ResourceSaver.save(mesh, path + ".res")
			if saved != OK:
				outcome.refused.append("%s/%s: %s.res could not be written: %s" % [batch, module, path, error_string(saved)])
				continue
			var instance := MeshInstance3D.new()
			instance.name = module
			instance.mesh = load(path + ".res")
			var scene := PackedScene.new()
			scene.pack(instance)
			instance.free()
			saved = ResourceSaver.save(scene, path + ".tscn")
			if saved != OK:
				outcome.refused.append("%s/%s: %s.tscn could not be written: %s" % [batch, module, path, error_string(saved)])
				continue
			outcome.bound[module] = path + ".tscn"
	return outcome

## The modules of `sets` with something to draw that `bound` has no scene for, sorted.
static func missing(sets: Dictionary[String, String], bound: Dictionary[String, String]) -> PackedStringArray:
	var names := {}
	var module := RegEx.create_from_string("\\(name: \"([a-z0-9_]+)\"")
	for text: String in sets.values():
		for found in module.search_all(text):
			names[found.get_string(1)] = true
	var result := PackedStringArray()
	for name: String in names:
		if not name in NOTHING_TO_DRAW and not bound.has(name):
			result.append(name)
	result.sort()
	return result

## `scene_text`, a scene file's text, binding each module of `bound` to its scene: an external
## resource per scene, `kit_<module>`, after the scene's own, and its `scenes` property as module
## name to that resource, after `rules_files`, both sorted and replacing what an earlier run wrote.
## Scenes bound as resources load with the scene on Godot's main thread, as the small city preset
## binds its kit; bound as paths they would load on loader threads.
static func bind(scene_text: String, bound: Dictionary[String, String]) -> String:
	var text := RegEx.create_from_string("(?m)^\\[ext_resource type=\"PackedScene\" path=\"[^\"]*\" id=\"kit_[a-z0-9_]+\"\\]\\n").sub(scene_text, "", true)
	text = RegEx.create_from_string("(?ms)^scenes = \\{\\n.*?^\\}\\n").sub(text, "")
	# The resource count Godot reads as a hint changes with the kit, and Godot no longer writes it.
	text = RegEx.create_from_string(" load_steps=[0-9]+").sub(text, "")
	if bound.is_empty():
		return text
	var modules := bound.keys()
	modules.sort()
	var resources := PackedStringArray()
	var entries := PackedStringArray()
	for module: String in modules:
		resources.append("[ext_resource type=\"PackedScene\" path=\"%s\" id=\"kit_%s\"]\n" % [bound[module], module])
		entries.append("\"%s\": ExtResource(\"kit_%s\")" % [module, module])
	var last := text.rfind("[ext_resource ")
	var resources_at := text.find("\n", last) + 1 if last >= 0 else text.find("\n") + 2
	text = text.substr(0, resources_at) + "".join(resources) + text.substr(resources_at)
	var rules := text.find("rules_files = {")
	var after := text.find("\n}\n", rules) + 3
	return text.substr(0, after) + "scenes = {\n%s\n}\n" % ",\n".join(entries) + text.substr(after)

## The one mesh of the model at `path`, or null with why in `problems`.
static func _mesh(path: String, problems: PackedStringArray) -> Mesh:
	var document := GLTFDocument.new()
	var state := GLTFState.new()
	if document.append_from_file(path, state) != OK:
		problems.append("it could not be read as glTF")
		return null
	var scene := document.generate_scene(state)
	var found := scene.find_children("*", "MeshInstance3D", true, false)
	var mesh: Mesh = null
	if found.size() != 1:
		problems.append("it holds %d mesh objects, not one" % found.size())
	elif not _unmoved(found[0], scene):
		problems.append("its mesh object is moved, turned or scaled; the cell's centre is its origin")
	else:
		mesh = (found[0] as MeshInstance3D).mesh
	scene.free()
	return mesh

## Whether `node` and every parent up to `root` stand at the origin unturned and unscaled.
static func _unmoved(node: Node, root: Node) -> bool:
	var at := node
	while at != null:
		if at is Node3D and not (at as Node3D).transform.is_equal_approx(Transform3D.IDENTITY):
			return false
		if at == root:
			return true
		at = at.get_parent()
	return true
