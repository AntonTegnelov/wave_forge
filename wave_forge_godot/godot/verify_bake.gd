## Bakes an area into a scene of plain nodes and loads it back.
##
## Run by `../verify.sh` after `verify_volume.gd`. The pack has ground with materials, trees drawn
## as a MultiMesh, a cave volume with pools of water, and ore placed as instances of a saved scene.
## Baking 3 by 3 chunks gives a scene that, saved as text, names no Wave Forge class, refers to no
## file but the ore's scene, and opens in a second Godot project that has no extension at all. Loaded back, every chunk holds its ground standing where the node's
## does with a height-map body, its cave's surface and fluid with the node's vertices and a concave
## body, a tree for every point and an ore for every embedded one. Baking a chunk not yet built is
## refused. Then, linked: a designer moves one ore of the baked scene, deletes another and adds a
## node of their own; the node keeps those edits, the world is generated again with the ore moved
## and the other gone, and a bake keeping the old one holds the moved ore where the designer put it,
## not the deleted one, and the designer's node.
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
var phase := "arrive"
## The first bake, as instanced, which the designer edits.
var baked: Node3D
var ores_before := 0
## The moved ore's key and where it was moved to, and the deleted ore's key.
var moved := []
var moved_to := Transform3D()
var deleted := []
## The products dropped since the designer's edits were kept and not generated again yet: the
## linked bake waits for every one of them.
var regenerating := {}

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
	world.stage_dropped.connect(func(stage: String, chunk: Vector3i) -> void:
		if phase == "linked":
			regenerating[[stage, chunk]] = true)
	world.stage_ready.connect(func(stage: String, chunk: Vector3i) -> void: regenerating.erase([stage, chunk]))
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
		_fail("the %s phase timed out" % phase)
		return true
	if phase == "linked":
		return _check_linked()
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
	if not _opens_without_the_extension():
		return true
	var loaded: PackedScene = ResourceLoader.load(BAKED_PATH, "", ResourceLoader.CACHE_MODE_IGNORE)
	baked = loaded.instantiate()
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
	if trees < 20 or ores < 5 or baked_trees != trees or baked_ores != ores:
		_fail("%d of %d trees and %d of %d ores baked" % [baked_trees, trees, baked_ores, ores])
		return true
	if not _trees_stand_where_placed():
		return true
	print("verify_bake: 9 chunks baked into a scene naming no Wave Forge class, opened in a project without the extension, loaded back with their ground, bodies, cave, fluid, %d trees and %d ores" % [trees, ores])
	return _edit_as_a_designer(ores)

## Every baked tree stands where the stage placed it.
func _trees_stand_where_placed() -> bool:
	var placed := []
	for chunk in _area():
		for set: Dictionary in world.point_sets("trees", chunk):
			var transforms: PackedFloat32Array = set["transforms"]
			for i in range(0, transforms.size(), 12):
				placed.append(Vector3(transforms[i + 3], transforms[i + 7], transforms[i + 11]))
	var drawn := []
	for node in baked.find_children("*", "MultiMeshInstance3D", true, false):
		if node.name == "tree":
			var at: Transform3D = node.get_parent().transform * node.transform
			# Read from the buffer the scene saved: the headless renderer keeps no instance data.
			var buffer: PackedFloat32Array = node.multimesh.buffer
			var stride: int = buffer.size() / node.multimesh.instance_count
			for i in node.multimesh.instance_count:
				var o: int = i * stride
				drawn.append(at * Vector3(buffer[o + 3], buffer[o + 7], buffer[o + 11]))
	placed.sort()
	drawn.sort()
	for i in placed.size():
		if not placed[i].is_equal_approx(drawn[i]):
			_fail("a baked tree stands at %s, where the stage placed one at %s" % [drawn[i], placed[i]])
			return false
	return true

## Opens the saved bake in a project of its own, with no extension, in a second Godot process: the
## project has this one's name, so `user://` and the ore's scene there are the same.
func _opens_without_the_extension() -> bool:
	var dir := OS.get_user_data_dir().path_join("plugin_free")
	DirAccess.make_dir_recursive_absolute(dir)
	var project := FileAccess.open(dir.path_join("project.godot"), FileAccess.WRITE)
	project.store_string("config_version=5\n\n[application]\nconfig/name=\"wave_forge_verify\"\n")
	project.close()
	DirAccess.copy_absolute(ProjectSettings.globalize_path(BAKED_PATH), dir.path_join("baked.tscn"))
	var opener := FileAccess.open(dir.path_join("open.gd"), FileAccess.WRITE)
	opener.store_string("\n".join([
		"extends SceneTree",
		"func _initialize() -> void:",
		"\tvar scene: PackedScene = load(\"res://baked.tscn\")",
		"\tvar baked: Node = scene.instantiate() if scene != null else null",
		"\tvar chunks := baked.get_child_count() if baked != null else 0",
		"\tif baked != null: baked.free()",
		"\tprint(\"opened %d chunks\" % chunks)",
		"\tquit(0 if chunks == 9 else 1)",
		"",
	]))
	opener.close()
	var output := []
	var code := OS.execute(OS.get_executable_path(), ["--headless", "--path", dir, "--script", "res://open.gd"], output, true)
	if code != 0 or DirAccess.dir_exists_absolute(dir.path_join("bin")):
		_fail("the bake does not open in a project without the extension (exit %d):\n%s" % [code, "\n".join(output)])
		return false
	return true

## Moves one ore of the baked scene a cell along x, deletes another, adds a node of the designer's
## own, and hands the edits to the node.
func _edit_as_a_designer(ores: int) -> bool:
	ores_before = ores
	var holder: Node3D = baked.get_node("Chunk 0 0")
	var ore_nodes := []
	for child in holder.get_children():
		if child.scene_file_path == ORE_PATH:
			ore_nodes.append(child)
	if ore_nodes.size() < 2:
		_fail("chunk 0 0 holds %d ores" % ore_nodes.size())
		return true
	var mark: Dictionary = ore_nodes[0].get_meta("wave_forge_point")
	moved = [mark["stage"], mark["chunk"], mark["id"]]
	ore_nodes[0].position += Vector3(CELL.x, 0, 0)
	moved_to = ore_nodes[0].transform
	mark = ore_nodes[1].get_meta("wave_forge_point")
	deleted = [mark["stage"], mark["chunk"], mark["id"]]
	holder.remove_child(ore_nodes[1])
	ore_nodes[1].free()
	var house := Node3D.new()
	house.name = "Designer house"
	holder.add_child(house)
	if not world.keep_bake_edits(baked):
		_fail("the designer's edits were refused")
		return true
	phase = "linked"
	started_usec = Time.get_ticks_usec()
	return false

## Waits for the world generated again, every product the edits dropped back and placed, then
## bakes it keeping the old bake.
func _check_linked() -> bool:
	var stats: Dictionary = world.stats()
	if stats["pending_placements"] > 0 or stats["pending_signals"] > 0 or not regenerating.is_empty():
		return false
	for set: Dictionary in world.point_sets("ore", deleted[1]):
		if set["ids"].has(deleted[2]):
			return false
	var scene: PackedScene = world.bake_keeping(FROM, TO, baked)
	if scene == null:
		_fail("the linked bake failed")
		return true
	var kept: Node3D = scene.instantiate()
	var ores := 0
	var found_moved := false
	for node in kept.find_children("*", "", true, false):
		if node.scene_file_path != ORE_PATH:
			continue
		ores += 1
		var mark: Dictionary = node.get_meta("wave_forge_point")
		var key := [mark["stage"], mark["chunk"], mark["id"]]
		if key == deleted:
			_fail("the deleted ore is baked again")
			return true
		if key == moved:
			# Where the designer put it, turned as it was.
			found_moved = node.transform.is_equal_approx(moved_to)
	var house_kept: bool = kept.get_node("Chunk 0 0").has_node("Designer house")
	kept.free()
	baked.free()
	if not found_moved or not house_kept or ores != ores_before - 1:
		_fail("the linked bake has the moved ore where it was put: %s, the designer's node: %s, and %d ores for %d" % [found_moved, house_kept, ores, ores_before - 1])
		return true
	print("verify_bake: a linked bake regenerated with the designer's edits keeps the moved ore where it was put, drops the deleted one and keeps the designer's node")
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
