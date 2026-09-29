## Bakes an area into a scene of plain nodes and loads it back.
##
## Run by `../verify.sh` after `verify_volume.gd`. The pack has ground with materials, trees drawn
## as a MultiMesh, a cave volume with pools of water, and ore placed as instances of a saved scene.
## Baking 3 by 3 chunks gives a scene that, saved as text, names no Wave Forge class and refers to
## no file but the ore's scene. Loaded back, every chunk holds its ground standing where the node's
## does with a height-map body, its cave's surface and fluid with the node's vertices and a concave
## body, a tree for every point and an ore for every embedded one. Baking a chunk not yet built is
## refused.
extends SceneTree

const CELLS := 8
const CELL := Vector3(2, 1, 2)
const TIMEOUT_S := 30.0
const ORE_PATH := "user://wave_forge_bake_ore.tscn"
const BAKED_PATH := "user://wave_forge_baked.tscn"
const FROM := Vector3i(-1, -1, 0)
const TO := Vector3i(1, 1, 0)

var world: Node
var started_usec := 0

func _initialize() -> void:
	var tree := MeshInstance3D.new()
	tree.mesh = BoxMesh.new()
	var trees := PackedScene.new()
	trees.pack(tree)
	tree.free()
	var top := Node3D.new()
	var child := MeshInstance3D.new()
	child.mesh = SphereMesh.new()
	top.add_child(child)
	child.owner = top
	var ore := PackedScene.new()
	ore.pack(top)
	top.free()
	if ResourceSaver.save(ore, ORE_PATH) != OK:
		_fail("the ore scene could not be saved")
		return
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://bake.world.ron"
	world.targets = PackedStringArray(["height", "surface", "trees", "cave", "water", "ore"])
	world.seed = 7
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = CELL
	world.view_radius = 3
	world.collider_radius = -1
	world.ground_stage = "height"
	world.ground_material_stage = "surface"
	world.ground_palette = PackedColorArray([Color(0.8, 0.7, 0.4), Color(0.5, 0.5, 0.5), Color(0.2, 0.6, 0.2)])
	world.volume_stage = "cave"
	world.fluid_stage = "water"
	world.scenes = {"tree": trees, "ore": ORE_PATH}
	root.add_child(world)
	world.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	if not world.start():
		_fail("the stages did not start")
		return
	world.follow(Vector3(1, 0, 1))
	started_usec = Time.get_ticks_usec()

func _area() -> Array[Vector3i]:
	var chunks: Array[Vector3i] = []
	for y in range(FROM.y, TO.y + 1):
		for x in range(FROM.x, TO.x + 1):
			chunks.append(Vector3i(x, y, 0))
	return chunks

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("the area had not arrived")
		return true
	var stats: Dictionary = world.stats()
	for chunk in _area():
		if not world.ground_chunks().has(chunk) or not world.volume_chunks().has(chunk) or not world.fluid_chunks().has(chunk):
			return false
	if stats["pending_placements"] > 0 or stats["pending_volumes"] > 0:
		return false
	return _check()

func _check() -> bool:
	if world.bake(Vector3i(40, 40, 0), Vector3i(41, 41, 0)) != null:
		_fail("baking chunks not built was taken")
		return true
	var scene: PackedScene = world.bake(FROM, TO)
	if scene == null:
		_fail("the bake failed")
		return true
	if ResourceSaver.save(scene, BAKED_PATH) != OK:
		_fail("the baked scene could not be saved")
		return true
	var text := FileAccess.get_file_as_string(BAKED_PATH)
	if text.contains("WaveForge") or text.contains("gdextension"):
		_fail("the baked scene names the extension")
		return true
	var files := 0
	for line in text.split("\n"):
		if line.begins_with("[ext_resource"):
			files += 1
			if not line.contains(ORE_PATH):
				_fail("the baked scene refers to a file other than the ore's: %s" % line)
				return true
	if files != 1:
		_fail("the baked scene refers to %d files, not the ore's alone" % files)
		return true
	var loaded: PackedScene = ResourceLoader.load(BAKED_PATH, "", ResourceLoader.CACHE_MODE_IGNORE)
	var baked: Node3D = loaded.instantiate()
	var trees := 0
	var ores := 0
	for chunk in _area():
		var holder: Node3D = baked.get_node("Chunk %d %d" % [chunk.x, chunk.y])
		if not _check_chunk(holder, chunk):
			return true
		for set: Dictionary in world.point_sets("trees", chunk):
			trees += set["ids"].size()
		for set: Dictionary in world.point_sets("ore", chunk):
			ores += set["ids"].size()
	var baked_trees := 0
	var baked_ores := 0
	for node in baked.find_children("*", "", true, false):
		if node is MultiMeshInstance3D and node.name == "tree":
			baked_trees += node.multimesh.instance_count
		elif node.scene_file_path == ORE_PATH:
			baked_ores += 1
	baked.free()
	if trees < 20 or ores < 5 or baked_trees != trees or baked_ores != ores:
		_fail("%d of %d trees and %d of %d ores baked" % [baked_trees, trees, baked_ores, ores])
		return true
	print("verify_bake: 9 chunks baked into a scene naming no Wave Forge class, loaded back with their ground, bodies, cave, fluid, %d trees and %d ores" % [trees, ores])
	quit(0)
	return true

## A baked chunk holds its ground standing where the node's does, with a material and a height-map
## body; its cave's surface and its fluid with the node's vertices; and a concave body whose faces
## are the cave's triangles.
func _check_chunk(holder: Node3D, chunk: Vector3i) -> bool:
	var ground: MeshInstance3D = holder.get_node("Ground")
	var corner := Vector3(chunk.x * CELLS * CELL.x, 0, chunk.y * CELLS * CELL.z)
	var on_grid := 0
	for vertex: Vector3 in ground.mesh.surface_get_arrays(0)[Mesh.ARRAY_VERTEX]:
		var at := corner + vertex
		# A skirt vertex hangs below the ground's edge.
		if at.y < world.ground_height(at) - 0.01:
			continue
		if absf(at.y - world.ground_height(at)) > 0.001:
			_fail("a ground vertex of %s stands at %.3f, the node's ground at %.3f" % [chunk, at.y, world.ground_height(at)])
			return false
		on_grid += 1
	if on_grid < (CELLS + 1) * (CELLS + 1) or ground.mesh.surface_get_material(0) == null:
		_fail("the ground of %s has %d vertices on its grid, or no material" % [chunk, on_grid])
		return false
	var ground_body: StaticBody3D = holder.get_node("GroundBody")
	if not (ground_body.get_child(0).shape is HeightMapShape3D):
		_fail("the ground of %s has no height map" % chunk)
		return false
	for part in [["Surface", world.volume_surface(chunk)], ["Fluid", world.fluid_surface(chunk)]]:
		var baked: MeshInstance3D = holder.get_node(part[0])
		var vertices: PackedVector3Array = baked.mesh.surface_get_arrays(0)[Mesh.ARRAY_VERTEX]
		if vertices != part[1]["positions"] or vertices.is_empty():
			_fail("the %s of %s is baked with %d vertices, the node's has %d" % [part[0], chunk, vertices.size(), part[1]["positions"].size()])
			return false
	var surface_body: StaticBody3D = holder.get_node("SurfaceBody")
	var faces: PackedVector3Array = surface_body.get_child(0).shape.get_faces()
	if faces.size() != world.volume_surface(chunk)["indices"].size():
		_fail("the cave of %s is baked with %d faces for %d corners" % [chunk, faces.size(), world.volume_surface(chunk)["indices"].size()])
		return false
	return true

func _fail(message: String) -> void:
	printerr("verify_bake: " + message)
	quit(1)
