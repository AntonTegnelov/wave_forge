## The continent's kit tool (`tools/continent_kits.gd`, #310) on models made here and written as
## glTF binaries: a shared ground tile on its slab is bound, as a scene of a lone mesh; one whose
## ground lies too low and one holding two meshes are refused, saying why; a batch named for no
## culture is refused; the modules still without a model leave the bound one out; and binding a
## scene's text twice leaves one `scenes` property, after `rules_files`.
##
## Run headless by `../verify.sh`, after `prepare.sh` has copied the continent's cultures.
extends SceneTree

const ContinentKits := preload("res://tools/continent_kits.gd")
const KITS := "user://continent_kits_check/kits"
const OUT := "user://continent_kits_check/out"

func _initialize() -> void:
	var sets: Dictionary[String, String] = {}
	for file in DirAccess.get_files_at("res://continent/cultures"):
		if file.get_extension() == "ron":
			sets[file.get_basename()] = FileAccess.get_file_as_string("res://continent/cultures".path_join(file))
	if sets.size() != 8:
		_fail("%d cultures in res://continent/cultures, not 8; run prepare.sh" % sets.size())
		return
	# Nothing from an earlier run.
	_remove(ProjectSettings.globalize_path("user://continent_kits_check"))
	_write("shared", "grass", [_slab(-0.25)])
	_write("shared", "plaza", [_slab(-0.35)])
	_write("shared", "plaza_lamp", [_slab(-0.25), _slab(-0.25)])
	_write("nowhere", "grass", [_slab(-0.25)])

	var outcome := ContinentKits.build(ProjectSettings.globalize_path(KITS), sets, OUT)

	if outcome.bound.keys() != ["grass"]:
		_fail("bound %s; only the good grass tile should be" % [outcome.bound])
		return
	var scene: Node = load(outcome.bound["grass"]).instantiate()
	var lone := scene is MeshInstance3D and scene.get_child_count() == 0 and (scene as MeshInstance3D).mesh != null
	scene.free()
	if not lone:
		_fail("the grass tile's scene is no lone MeshInstance3D with a mesh")
		return
	var refused := "\n".join(outcome.refused)
	for expected in ["shared/plaza: its +X side is walkable", "shared/plaza_lamp: it holds 2 mesh objects", "nowhere: no culture's module set is named nowhere"]:
		if not expected in refused:
			_fail("nothing refused as \"%s\" in:\n%s" % [expected, refused])
			return
	var missing := ContinentKits.missing(sets, outcome.bound)
	if "grass" in missing or not "plaza" in missing or "air" in missing:
		_fail("the modules without a model are %s" % [missing])
		return
	print("verify_continent_kits: a good tile is bound as a lone mesh, and a tile walking too low, one of two meshes and an unknown batch are refused, saying why")

	var text := "[node name=\"Continent\" type=\"WaveForgeStages\"]\nrules_files = {\n\"a\": \"a.ron\"\n}\nseed = 11\n"
	var once := ContinentKits.bind(text, outcome.bound)
	var twice := ContinentKits.bind(once, outcome.bound)
	var expected_text := "[node name=\"Continent\" type=\"WaveForgeStages\"]\nrules_files = {\n\"a\": \"a.ron\"\n}\nscenes = {\n\"grass\": \"%s\"\n}\nseed = 11\n" % outcome.bound["grass"]
	if twice != expected_text:
		_fail("binding twice gave:\n%s" % twice)
		return
	print("verify_continent_kits: binding writes one scenes property after rules_files, however often it runs")
	quit(0)

## A ground slab across the whole cell, its top at `top`, as a mesh in the model's own frame.
func _slab(top: float) -> ArrayMesh:
	var box := BoxMesh.new()
	box.size = Vector3(1.0, top + 0.5, 1.0)
	var arrays := box.get_mesh_arrays()
	var vertices: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
	for i in vertices.size():
		vertices[i].y += (top - 0.5) / 2.0
	arrays[Mesh.ARRAY_VERTEX] = vertices
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	return mesh

## Writes `meshes`, each as its own unmoved mesh object, as `<batch>/<piece>.glb` under KITS.
func _write(batch: String, piece: String, meshes: Array) -> void:
	var folder := ProjectSettings.globalize_path(KITS).path_join(batch)
	DirAccess.make_dir_recursive_absolute(folder)
	var root := Node3D.new()
	for mesh: Mesh in meshes:
		var instance := MeshInstance3D.new()
		instance.mesh = mesh
		root.add_child(instance)
	var document := GLTFDocument.new()
	var state := GLTFState.new()
	document.append_from_scene(root, state)
	document.write_to_filesystem(state, folder.path_join(piece + ".glb"))
	root.free()

func _remove(path: String) -> void:
	if not DirAccess.dir_exists_absolute(path):
		return
	for folder in DirAccess.get_directories_at(path):
		_remove(path.path_join(folder))
	for file in DirAccess.get_files_at(path):
		DirAccess.remove_absolute(path.path_join(file))
	DirAccess.remove_absolute(path)

func _fail(message: String) -> void:
	printerr("verify_continent_kits: " + message)
	quit(1)
