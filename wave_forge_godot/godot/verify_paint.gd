## Paints strokes on a stages node's world, as the editor plugin's brushes do.
##
## Run by `../verify.sh` after `verify_params.gd`. On the islands preset, all land and with no
## roughness, so its rock spines stay low and trees grow along the stroke: a remove stroke
## takes every tree near its path once the chunks come back, a raise stroke then lifts the ground
## on its path by its strength at once, and `edits_text` holds both; a second node given that text
## before it starts has the raised ground from its start. A brush that names no brush is refused.
## The editor's brush cursor rings as much ground as its brush reaches, and hides off the ground.
extends SceneTree

const CELLS := 8
const CELL := Vector3(2, 1, 2)
const TIMEOUT_S := 30.0
## A stroke along x across the middle of chunk 0 0, through the centres of a row of columns, and
## a column centre on it where the ground is measured.
const STROKE := [Vector3(2, 0, 9), Vector3(14, 0, 9)]
const PROBE := Vector3(9, 0, 9)

var world: Node
var ready := {}
var started_usec := 0
var phase := "arrive"
var ground_before := 0.0
## Whether the brush cursor has been checked, on the first frame, once the tree is running.
var cursor_checked := false

func _initialize() -> void:
	world = _world()
	world.stage_ready.connect(func(stage: String, chunk: Vector3i) -> void: ready[[stage, chunk]] = true)
	if not world.start():
		_fail("the stages did not start")
		return
	world.follow(Vector3(8, 0, 8))
	started_usec = Time.get_ticks_usec()

## The cursor shown for a brush of 3 cells over PROBE is centred there, 3 cells either way, deep
## enough to lie on steep ground, and hidden when the mouse is off the ground.
func _check_cursor() -> bool:
	var cursor: Decal = preload("res://addons/wave_forge/brush_cursor.gd").new()
	root.add_child(cursor)
	cursor.show_at(PROBE, 3.0, CELL)
	var reach := 3.0 * CELL.x
	var placed := cursor.visible and cursor.global_position == PROBE
	var sized := cursor.size.x == 2.0 * reach and cursor.size.z == 2.0 * reach and cursor.size.y >= 2.0 * reach
	cursor.show_at(null, 3.0, CELL)
	var hidden := not cursor.visible
	cursor.free()
	if not (placed and sized and hidden):
		_fail("the brush cursor was placed %s, sized %s, hidden %s" % [placed, sized, hidden])
		return false
	print("verify_paint: the brush cursor rings the ground a brush reaches, and hides off the ground")
	return true

func _world() -> Node:
	var node: Node = ClassDB.instantiate("WaveForgeStages")
	node.pack_file = "res://islands.world.ron"
	node.targets = PackedStringArray(["height", "trees"])
	node.seed = 6
	node.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	node.cell_size = CELL
	node.view_radius = 1
	node.collider_radius = -1
	node.params = {"land": 1.0, "trees": 1.0, "roughness": 0.0}
	root.add_child(node)
	node.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	return node

## The trees within `radius` world units of the stroke's line.
func _trees_near(radius: float) -> int:
	var count := 0
	for y in range(-1, 2):
		for x in range(-1, 2):
			for set: Dictionary in world.point_sets("trees", Vector3i(x, y, 0)):
				var transforms: PackedFloat32Array = set["transforms"]
				for i in range(0, transforms.size(), 12):
					var at := Vector3(transforms[i + 3], 0, transforms[i + 11])
					var along := clampf(at.x, STROKE[0].x, STROKE[1].x)
					if Vector2(at.x - along, at.z - STROKE[0].z).length() < radius:
						count += 1
	return count

func _process(_delta: float) -> bool:
	if phase == "arrive" and not cursor_checked:
		cursor_checked = true
		if not _check_cursor():
			return true
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("the %s phase timed out" % phase)
		return true
	for y in range(-1, 2):
		for x in range(-1, 2):
			if not ready.has(["trees", Vector3i(x, y, 0)]):
				return false
	match phase:
		"arrive":
			# The islands' trees stand 3 cells apart, so a stroke 6 cells long passes two or three.
			if _trees_near(3.0 * CELL.x) < 2:
				_fail("only %d trees near the stroke to remove" % _trees_near(3.0 * CELL.x))
				return true
			if world.paint({"brush": "nothing"}, PackedVector3Array(STROKE)):
				_fail("a brush that names no brush was taken")
				return true
			# Removed first: a raise regenerates the trees, and a stroke paints on what the node
			# holds when it is painted.
			if not world.paint({"brush": "remove", "stages": PackedStringArray(["trees"]), "radius": 3.0}, PackedVector3Array(STROKE)):
				_fail("the remove stroke was refused")
				return true
			# The stroke lies in chunk 0 0: wait for it to arrive again, since while it is being
			# generated it holds no trees at all.
			ready.erase(["trees", Vector3i.ZERO])
			phase = "removed"
			started_usec = Time.get_ticks_usec()
			return false
		"removed":
			if _trees_near(3.0 * CELL.x) > 0:
				return false
			ground_before = world.sample("height", PROBE)
			if not world.paint({"brush": "raise", "stage": "height", "radius": 3.0, "strength": 3.0}, PackedVector3Array(STROKE)):
				_fail("the raise stroke was refused")
				return true
			var lifted: float = world.sample("height", PROBE) - ground_before
			if not absf(lifted - 3.0) <= 0.001:
				_fail("the raise lifted the ground on its path by %.3f cells, not 3" % lifted)
				return true
			if world.edits_text != world.edits_log() or world.edits_text.is_empty():
				_fail("edits_text does not hold the strokes")
				return true
			var second := _world()
			second.edits_text = world.edits_text
			if not second.start():
				_fail("a node given edits_text did not start")
				return true
			lifted = second.sample("height", PROBE) - ground_before
			if not absf(lifted - 3.0) <= 0.001:
				_fail("a node given edits_text starts with the ground lifted by %.3f" % lifted)
				return true
			print("verify_paint: a remove stroke clears the trees near it, a raise stroke lifts its path at once, and a node given edits_text starts with both")
			quit(0)
			return true
	return false

func _fail(message: String) -> void:
	printerr("verify_paint: " + message)
	quit(1)
