## Proposes a module set from a MeshLibrary and loads it as a rule set.
##
## Run by `../verify.sh` after `verify_bake.gd`. The library holds a block filling its cell, an item
## without a mesh, and a wall a quarter of a cell thick along the cell's -z side, all in cells of
## 2 by 1 by 2 as a GridMap centres them. The block's four sides share one symmetric connector, the
## empty item's faces are empty, the wall's ends are the plain and flipped faces of one asymmetric
## connector, its back is the block's side and its front is empty; and the proposal loads as a rule
## set. The dock's kit import (N6) lists the four connectors, with the modules that have them;
## named by the artist, with the block's side walkable, it saves a module set under those names
## that loads as a rule set, refuses one name for two connectors, and offers no walkable tick for
## a top. The same kit as a folder of scenes, the wall placed under a parent node, proposes the
## same modules and the panel lists the same four connectors from the folder; its scenes' static
## bodies come with it as the items' shapes, where the scenes place them. Last, a world generated from
## the folder's module set with those shapes as its colliders (`set_collision_shapes`) has a body
## under every walkable cell: a ray dropped onto the top of every block with open air above meets it.
extends SceneTree

const CELL := Vector3(2, 1, 2)
const CELLS := Vector3i(4, 4, 3)
const TIMEOUT_S := 60.0
## Frames to wait once the bodies are built, as the physics space takes them in.
const SETTLE_FRAMES := 10

## The kit as a folder of scenes, its proposal, and the world generated from them.
var scene_library: MeshLibrary
var scene_proposal := ""
var city: Node
var started_usec := 0
var settled_frames := 0

func _initialize() -> void:
	var library := MeshLibrary.new()
	var block := BoxMesh.new()
	block.size = CELL
	library.create_item(0)
	library.set_item_name(0, "block")
	library.set_item_mesh(0, block)
	library.create_item(1)
	library.set_item_name(1, "air")
	var wall := BoxMesh.new()
	wall.size = Vector3(CELL.x, CELL.y, CELL.z / 4.0)
	library.create_item(2)
	library.set_item_name(2, "wall")
	library.set_item_mesh(2, wall)
	library.set_item_mesh_transform(2, Transform3D(Basis(), Vector3(0, 0, -CELL.z * 3.0 / 8.0)))

	var proposed: String = ClassDB.class_call_static("WaveForgeWorld", "propose_module_set", library, CELL)

	var expected := [
		'(name: "block", sides: ["side 0", "side 0", "side 0", "side 0"], up: "top 0", down: "top 0")',
		'(name: "air", sides: ["empty side", "empty side", "empty side", "empty side"], up: "empty top", down: "empty top")',
		'(name: "wall", sides: ["side 1 plain", "side 1 flipped", "empty side", "side 0"]',
	]
	for line in expected:
		if not proposed.contains(line):
			_fail("the proposal has no %s:\n%s" % [line, proposed])
			return
	var world: Node = ClassDB.instantiate("WaveForgeWorld")
	if not world.load_rules(proposed):
		_fail("the proposal does not load as a rule set:\n%s" % proposed)
		world.free()
		return
	world.free()
	print("verify_import: a MeshLibrary's block, empty item and wall get the connectors their faces' shapes give, and load as a rule set")
	var problem := _named(library)
	if not problem.is_empty():
		_fail(problem)
		return
	print("verify_import: the kit import lists the four connectors, saves the artist's names and walkable side as a rule set that loads, and refuses a name for two connectors")
	problem = _from_scenes()
	if not problem.is_empty():
		_fail(problem)
		return
	print("verify_import: the same kit as a folder of scenes proposes the same modules, and the panel lists its four connectors, its scenes' bodies the items' shapes")
	city = ClassDB.instantiate("WaveForgeWorld")
	if not city.load_rules(scene_proposal):
		_fail("the folder's proposal does not load as a rule set")
		return
	city.seed = 3
	city.chunk_cells = CELLS
	city.cell_size = CELL
	city.world_chunks = Vector3i(2, 2, 1)
	city.view_radius = 1
	city.collider_radius = 1
	city.set_collision_shapes(scene_library)
	root.add_child(city)
	if not city.start():
		_fail("the world of the folder's kit did not start")
		return
	city.follow(Vector3(CELLS.x * CELL.x, 0, CELLS.y * CELL.z))
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	if city == null:
		return false
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("the world of the folder's kit had %d bodies after %.0f s" % [city.collider_chunks().size(), TIMEOUT_S])
		return true
	# A chunk of nothing but air has no body, so the wait is for nothing to be pending.
	if city.collider_chunks().is_empty() or city.stats()["pending_colliders"] > 0:
		return false
	settled_frames += 1
	if settled_frames < SETTLE_FRAMES:
		return false
	return _check_walkable()

## A ray dropped onto the top of every block with open air above, in every chunk with a body, meets
## the block's collider there.
func _check_walkable() -> bool:
	var space := root.get_world_3d().direct_space_state
	var walkable := 0
	for chunk: Vector3i in city.collider_chunks():
		var tiles: PackedInt32Array = city.tiles_at(chunk)
		for cell in tiles.size():
			var level := cell / (CELLS.x * CELLS.y)
			var above := cell + CELLS.x * CELLS.y
			if city.tile_name(tiles[cell]) != "block":
				continue
			if level < CELLS.z - 1 and city.tile_name(tiles[above]) != "air":
				continue
			var top: Vector3 = city.cell_position(chunk, cell) + Vector3.UP * (CELL.y / 2.0)
			var query := PhysicsRayQueryParameters3D.create(top + Vector3.UP * 0.25, top + Vector3.DOWN * 0.25)
			var hit := space.intersect_ray(query)
			if hit.is_empty() or absf(hit["position"].y - top.y) > 0.01:
				_fail("a ray onto the top of a block at %s met %s" % [top, hit.get("position", "nothing")])
				return true
			walkable += 1
	if walkable == 0:
		_fail("no block with open air above in the world of the folder's kit")
		return true
	print("verify_import: a world of the folder's kit, its items' shapes its colliders, has a body under each of its %d walkable cells" % walkable)
	quit(0)
	return true

## What is wrong with proposing from the kit as a folder of scenes, or nothing.
func _from_scenes() -> String:
	var folder := "user://kit_scenes"
	DirAccess.make_dir_recursive_absolute(folder)
	var block := MeshInstance3D.new()
	var block_mesh := BoxMesh.new()
	block_mesh.size = CELL
	block.mesh = block_mesh
	var block_box := BoxShape3D.new()
	block_box.size = CELL
	_add_body(block, block_box, Vector3.ZERO)
	var wall := Node3D.new()
	var wall_mesh := MeshInstance3D.new()
	var thin := BoxMesh.new()
	thin.size = Vector3(CELL.x, CELL.y, CELL.z / 4.0)
	wall_mesh.mesh = thin
	wall_mesh.position = Vector3(0, 0, -CELL.z * 3.0 / 8.0)
	wall.add_child(wall_mesh)
	wall_mesh.owner = wall
	var thin_box := BoxShape3D.new()
	thin_box.size = thin.size
	_add_body(wall, thin_box, wall_mesh.position)
	for scene in [["air", Node3D.new()], ["block", block], ["wall", wall]]:
		var packed := PackedScene.new()
		packed.pack(scene[1])
		ResourceSaver.save(packed, "%s/%s.tscn" % [folder, scene[0]])
		scene[1].free()
	var library: MeshLibrary = ClassDB.class_call_static("WaveForgeWorld", "mesh_library_from_scenes", folder)
	if library == null:
		return "no MeshLibrary from the folder"
	for item in library.get_item_list():
		var shapes: Array = library.get_item_shapes(item)
		var expected: Array = {
			"air": [],
			"block": [block_box, Transform3D.IDENTITY],
			"wall": [thin_box, Transform3D(Basis(), Vector3(0, 0, -CELL.z * 3.0 / 8.0))],
		}[library.get_item_name(item)]
		if shapes.size() != expected.size() or (shapes.size() == 2 and (shapes[0].size != expected[0].size or not shapes[1].is_equal_approx(expected[1]))):
			return "the %s item's shapes are %s" % [library.get_item_name(item), shapes]
	var proposed: String = ClassDB.class_call_static("WaveForgeWorld", "propose_module_set", library, CELL)
	scene_library = library
	scene_proposal = proposed
	for line in [
		'(name: "air", sides: ["empty side", "empty side", "empty side", "empty side"], up: "empty top", down: "empty top")',
		'(name: "block", sides: ["side 0", "side 0", "side 0", "side 0"], up: "top 0", down: "top 0")',
		'(name: "wall", sides: ["side 1 plain", "side 1 flipped", "empty side", "side 0"]',
	]:
		if not proposed.contains(line):
			return "the folder's proposal has no %s:\n%s" % [line, proposed]
	var panel: VBoxContainer = (load("res://addons/wave_forge/kit_import_panel.gd") as GDScript).new()
	var listed: int = panel.propose_path(folder, CELL)
	panel.free()
	if listed != 4:
		return "the panel listed %d connectors from the folder" % listed
	return ""

## Gives `node` a static body holding `shape` at `at`, owned by `node` so the body is packed with it.
func _add_body(node: Node3D, shape: Shape3D, at: Vector3) -> void:
	var body := StaticBody3D.new()
	var collision := CollisionShape3D.new()
	collision.shape = shape
	collision.position = at
	body.add_child(collision)
	node.add_child(body)
	body.owner = node
	collision.owner = node

## What is wrong with naming the proposal through the dock's kit import, or nothing.
func _named(library: MeshLibrary) -> String:
	var found: Array = ClassDB.class_call_static("WaveForgeWorld", "kit_connectors", library, CELL)
	var listed := {}
	for connector: Dictionary in found:
		listed[connector["name"]] = Array(connector["modules"])
	if listed.keys() != ["side 0", "side 1", "top 0", "top 1"] or listed["side 0"] != ["block", "wall"]:
		return "the connectors listed: %s" % listed
	var panel: VBoxContainer = (load("res://addons/wave_forge/kit_import_panel.gd") as GDScript).new()
	var problem := _through(panel, library)
	panel.free()
	return problem

func _through(panel: VBoxContainer, library: MeshLibrary) -> String:
	if panel.propose(library, CELL) != 4:
		return "the panel listed %d connectors" % panel.rows.size()
	if not panel.rows["top 0"][1].disabled or panel.rows["side 0"][1].disabled:
		return "a walkable tick on a top, or none on a side"
	panel.rows["side 0"][0].text = "brick"
	panel.rows["side 1"][0].text = "wall end"
	panel.rows["side 0"][1].button_pressed = true
	if not panel.save_to("user://kit.ron"):
		return "the named set was not saved: %s" % panel.status.text
	var saved := FileAccess.get_file_as_string("user://kit.ron")
	for line in ['"brick": Side(connector: "brick", symmetry: Symmetric, walkable: true)', '"wall end plain"', 'sides: ["wall end plain", "wall end flipped", "empty side", "brick"]']:
		if not saved.contains(line):
			return "the saved set has no %s:\n%s" % [line, saved]
	var world: Node = ClassDB.instantiate("WaveForgeWorld")
	var loads: bool = world.load_rules(saved)
	world.free()
	if not loads:
		return "the saved set does not load as a rule set:\n%s" % saved
	panel.rows["side 1"][0].text = "brick"
	if panel.save_to("user://clash.ron"):
		return "one name for two connectors was saved"
	return ""

func _fail(message: String) -> void:
	printerr("verify_import: " + message)
	quit(1)
