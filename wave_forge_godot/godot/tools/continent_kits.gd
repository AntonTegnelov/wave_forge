## The continent's authored town pieces as scenes its towns are drawn with (#310,
## docs/reference/godot.md, "Scenes"): each model a glTF binary named for its module, in a folder per
## batch, `shared` for the pieces every culture has and a culture's file name for its own. A model is
## checked against its module's faces in every module set it belongs to
## (`WaveForgeWorld.check_kit_model`) and holds exactly one mesh, unmoved; one that passes is saved
## as a mesh and a scene of a lone `MeshInstance3D`, which the node draws as a MultiMesh and collides
## with. `make_continent_kits.gd` runs it over the repository's kits.
extends RefCounted

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
			ResourceSaver.save(mesh, path + ".res")
			var instance := MeshInstance3D.new()
			instance.name = module
			instance.mesh = load(path + ".res")
			var scene := PackedScene.new()
			scene.pack(instance)
			ResourceSaver.save(scene, path + ".tscn")
			instance.free()
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
		# Air and a stair's headroom draw nothing.
		if name != "air" and name != "stair_head" and not bound.has(name):
			result.append(name)
	result.sort()
	return result

## `scene_text`, a scene file's text, with its `scenes` property replaced by `bound`, placed after
## the `rules_files` property, as module name to scene path, sorted.
static func bind(scene_text: String, bound: Dictionary[String, String]) -> String:
	var text := RegEx.create_from_string("(?ms)^scenes = \\{\\n.*?^\\}\\n").sub(scene_text, "")
	if bound.is_empty():
		return text
	var modules := bound.keys()
	modules.sort()
	var lines := PackedStringArray()
	for module: String in modules:
		lines.append("\"%s\": \"%s\"" % [module, bound[module]])
	var block := "scenes = {\n%s\n}\n" % ",\n".join(lines)
	var rules := text.find("rules_files = {")
	var after := text.find("\n}\n", rules) + 3
	return text.substr(0, after) + block + text.substr(after)

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
