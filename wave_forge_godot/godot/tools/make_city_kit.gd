## Writes the plugin's built-in city kit, the scenes the small city preset binds to the modules of
## the city module set (`examples/city.ron`): for each module with something to draw,
## `addons/wave_forge/city/<module>.tres`, its model's mesh, and `<module>.tscn`, a lone
## `MeshInstance3D` of it, which the node draws as a MultiMesh and collides with. The models are
## the ones `../prepare.sh` exports to `models/`, so run that first, then from `wave_forge_godot/godot`:
## `godot --headless --path . --script tools/make_city_kit.gd`.
extends SceneTree

const MODELS := "res://models"
const KIT := "res://addons/wave_forge/city"

func _initialize() -> void:
	DirAccess.make_dir_recursive_absolute(KIT)
	var written := 0
	for file in DirAccess.get_files_at(MODELS):
		if file.get_extension() != "glb":
			continue
		var module := file.get_basename()
		var mesh := _mesh(MODELS.path_join(file))
		if mesh == null:
			continue
		var path := KIT.path_join(module)
		ResourceSaver.save(mesh, path + ".tres")
		var instance := MeshInstance3D.new()
		instance.name = module
		instance.mesh = load(path + ".tres")
		var scene := PackedScene.new()
		scene.pack(instance)
		ResourceSaver.save(scene, path + ".tscn")
		instance.free()
		written += 1
	print("make_city_kit: wrote %d modules to %s" % [written, KIT])
	quit()

## The mesh of the model at `path`, or null for a model with nothing to draw.
func _mesh(path: String) -> Mesh:
	var document := GLTFDocument.new()
	var state := GLTFState.new()
	if document.append_from_file(path, state) != OK:
		push_error("make_city_kit: %s could not be read" % path)
		return null
	var scene := document.generate_scene(state)
	var found := scene.find_children("*", "MeshInstance3D", true, false)
	var mesh: Mesh = null if found.is_empty() else (found[0] as MeshInstance3D).mesh
	scene.free()
	return mesh
