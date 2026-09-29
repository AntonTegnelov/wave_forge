## Proposes a module set from a MeshLibrary and loads it as a rule set.
##
## Run by `../verify.sh` after `verify_bake.gd`. The library holds a block filling its cell, an item
## without a mesh, and a wall a quarter of a cell thick along the cell's -z side, all in cells of
## 2 by 1 by 2 as a GridMap centres them. The block's four sides share one symmetric connector, the
## empty item's faces are empty, the wall's ends are the plain and flipped faces of one asymmetric
## connector, its back is the block's side and its front is empty; and the proposal loads as a rule
## set.
extends SceneTree

const CELL := Vector3(2, 1, 2)

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
	quit(0)

func _fail(message: String) -> void:
	printerr("verify_import: " + message)
	quit(1)
