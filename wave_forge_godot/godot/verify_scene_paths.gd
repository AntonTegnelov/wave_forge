## Many scenes bound by path load without an error in a headless Godot (#465). A headless Godot's
## dummy renderer cannot create meshes and materials on several threads at once, and loading
## dozens of mesh scenes on Godot's loader threads printed "Attempting to initialize the wrong RID"
## and could crash it. In each of several rounds this binds 64 new scenes of a lone mesh, each with a
## material of its own, by path, and the islands' trees and bushes to a scene of nodes, also by path;
## placing waits until every bound scene has loaded, so the first tree placed ends the round.
##
## Run headless by `../verify.sh`, which fails it on any error it prints.
extends SceneTree

const CELLS := 8
const SCENES := 64
## New files every round, so no round loads what Godot's resource cache already holds.
const ROUNDS := 16
const FOLDER := "user://verify_scene_paths"
const TIMEOUT_S := 60.0

var world: Node
var round := 0
var started_usec := 0
var placed := false

func _initialize() -> void:
	_start_round()

func _start_round() -> void:
	var folder := FOLDER.path_join("round_%d" % round)
	DirAccess.make_dir_recursive_absolute(folder)
	var tree := _tree(folder)
	var scenes := {"broadleaf": tree, "bush": tree}
	for i in SCENES:
		scenes["kind_%d" % i] = _scene(folder, i)
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://islands.world.ron"
	world.targets = PackedStringArray(["height", "trees"])
	world.seed = 6
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.view_radius = 1
	world.collider_radius = -1
	world.params = {"land": 1.0, "trees": 1.0}
	world.scenes = scenes
	root.add_child(world)
	world.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	placed = false
	world.instance_spawned.connect(func(_node: Node3D, _chunk: Vector3i, _id: int) -> void: placed = true)
	if not world.start():
		_fail("the stages did not start")
		return
	world.follow(Vector3(8, 0, 8))
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("no tree was placed from its scene in round %d" % round)
		return true
	if not placed:
		return false
	world.free()
	round += 1
	if round < ROUNDS:
		_start_round()
		return false
	print("verify_scene_paths: %d rounds of %d scenes bound by path loaded, and the trees were placed from theirs" % [ROUNDS, SCENES + 1])
	quit(0)
	return true

## A tree scene of nodes, a mesh under a plain root, which the node places as nodes, saved in
## `folder`, as its path.
func _tree(folder: String) -> String:
	var root_node := Node3D.new()
	var trunk := MeshInstance3D.new()
	trunk.mesh = BoxMesh.new()
	root_node.add_child(trunk)
	trunk.owner = root_node
	var scene := PackedScene.new()
	scene.pack(root_node)
	root_node.free()
	var path := folder.path_join("tree.tscn")
	ResourceSaver.save(scene, path)
	return path

## A scene of a lone mesh with a material of its own, saved in `folder`, as its path.
func _scene(folder: String, index: int) -> String:
	var box := BoxMesh.new()
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, box.get_mesh_arrays())
	var material := StandardMaterial3D.new()
	material.albedo_color = Color.from_hsv(float(index) / SCENES, 0.6, 0.8)
	mesh.surface_set_material(0, material)
	var mesh_path := folder.path_join("mesh_%d.res" % index)
	ResourceSaver.save(mesh, mesh_path)
	var instance := MeshInstance3D.new()
	instance.mesh = load(mesh_path)
	var scene := PackedScene.new()
	scene.pack(instance)
	instance.free()
	var path := folder.path_join("scene_%d.tscn" % index)
	ResourceSaver.save(scene, path)
	return path

func _fail(message: String) -> void:
	printerr("verify_scene_paths: " + message)
	quit(1)
