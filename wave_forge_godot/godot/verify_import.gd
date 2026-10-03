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
## a top.
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
	var problem := _named(library)
	if not problem.is_empty():
		_fail(problem)
		return
	print("verify_import: the kit import lists the four connectors, saves the artist's names and walkable side as a rule set that loads, and refuses a name for two connectors")
	quit(0)

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
