## Decals along a road (docs/reference/godot.md, "Road decals"): the tables check's road, given as
## a row, gets an instance of its decal scene per piece in the chunks within `decal_radius`. The
## pieces lie along the road, as long as it is wide, end to end from its start, on the ground and
## keeping the scene's projection depth, each decal a hair longer so its neighbours overlap; leaving
## the radius frees them, and `curve_decals` naming a stage that makes no curves is refused.
##
## Run headless by `../verify.sh`. The renderer draws nothing here; how the decals look is checked
## by hand, by rendering `render_decals.gd`.
extends SceneTree

const CELLS := 8
const CELL := Vector3(2, 1, 2)
const TIMEOUT_S := 30.0
## The road, in cells: along x at row 12, from column 1, with a radius of 1.
const ROAD_START := 1.0
const ROAD_ROW := 12.0
const RADIUS := 1.0
## The decal scene's projection depth, which every instance keeps.
const DEPTH := 6.0
## How much longer than its piece of the road each decal is drawn (`decals.rs`, OVERLAP).
const OVERLAP := 1.02

var world: Node
var started_usec := 0
var moved_away := false

func _initialize() -> void:
	if not _check_refusal():
		return
	world = _world({"roads": _decal_scene()})
	root.add_child(world)
	world.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	if not world.start():
		_fail("the stages did not start")
		return
	if not world.give_table("roads", [{"id": 1, "x0": ROAD_START, "y0": ROAD_ROW, "x1": 60, "y1": ROAD_ROW, "width": RADIUS}]):
		_fail("the road was refused")
		return
	world.follow(Vector3.ZERO)
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("timed out with %d decals" % _decals().size())
		return true
	var decals := _decals()
	if moved_away:
		if decals.is_empty():
			print("verify_decals: leaving the decal radius frees the road's decals")
			quit(0)
			return true
		return false
	# The road crosses chunks 0 and 1 along x within the radius; wait for both.
	var seam := CELLS * CELL.x
	if decals.is_empty() or decals.front().position.x >= seam or decals.back().position.x < seam:
		return false
	if not _check_pieces(decals):
		return true
	moved_away = true
	world.follow(Vector3(400, 0, 400))
	return false

## The decals under the node, by their position along x.
func _decals() -> Array:
	var decals := world.find_children("*", "Decal", false, false)
	decals.sort_custom(func(a: Decal, b: Decal) -> bool: return a.position.x < b.position.x)
	return decals

func _check_pieces(decals: Array) -> bool:
	var width := 2.0 * RADIUS * CELL.x
	var first: Decal = decals[0]
	if not is_equal_approx(first.position.x - first.size.x / OVERLAP / 2.0, ROAD_START * CELL.x):
		_fail("the first piece starts at x %.3f; the road starts at %.3f" % [first.position.x - first.size.x / OVERLAP / 2.0, ROAD_START * CELL.x])
		return false
	for i in decals.size():
		var decal: Decal = decals[i]
		if not decal.basis.x.is_equal_approx(Vector3.RIGHT) or not decal.basis.y.is_equal_approx(Vector3.UP):
			_fail("decal %d is turned %s; the road runs along x" % [i, decal.basis])
			return false
		if not is_equal_approx(decal.position.z, ROAD_ROW * CELL.z):
			_fail("decal %d stands at z %.3f, off the road at %.3f" % [i, decal.position.z, ROAD_ROW * CELL.z])
			return false
		if not is_equal_approx(decal.size.z, width) or not is_equal_approx(decal.size.y, DEPTH) or decal.size.x > width * OVERLAP + 0.001:
			_fail("decal %d is %s; it should be %.1f wide, at most as long but for its overlap, and %.1f deep" % [i, decal.size, width, DEPTH])
			return false
		var ground: float = world.ground_height(decal.position)
		if absf(decal.position.y - ground) > DEPTH / 4.0:
			_fail("decal %d stands at y %.3f, the ground at %.3f" % [i, decal.position.y, ground])
			return false
		if i > 0:
			var before: Decal = decals[i - 1]
			# The pieces under the decals meet exactly; the decals overlap by their margin.
			var gap := (decal.position.x - decal.size.x / OVERLAP / 2.0) - (before.position.x + before.size.x / OVERLAP / 2.0)
			if absf(gap) > 0.001:
				_fail("the pieces of decals %d and %d leave a gap of %.4f" % [i - 1, i, gap])
				return false
	print("verify_decals: %d decals lie end to end along the road from its start, as long as it is wide, on the ground" % decals.size())
	return true

## A stage that makes no curves is refused before anything generates.
func _check_refusal() -> bool:
	var wrong := _world({"hills": _decal_scene()})
	root.add_child(wrong)
	var started: bool = wrong.start()
	wrong.free()
	if started:
		_fail("curve_decals naming the hills field was taken")
		return false
	print("verify_decals: curve_decals naming a stage that makes no curves is refused")
	return true

func _world(decals: Dictionary) -> Node:
	var node: Node = ClassDB.instantiate("WaveForgeStages")
	node.pack_file = "res://facts.world.ron"
	node.targets = PackedStringArray(["roads", "paved"])
	node.ground_stage = "paved"
	node.seed = 5
	node.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	node.cell_size = CELL
	node.view_radius = 2
	node.collider_radius = -1
	node.decal_radius = 1
	var typed: Dictionary[StringName, PackedScene] = {}
	for stage: String in decals:
		typed[StringName(stage)] = decals[stage]
	node.curve_decals = typed
	return node

func _decal_scene() -> PackedScene:
	var decal := Decal.new()
	decal.size = Vector3(1, DEPTH, 1)
	decal.texture_albedo = GradientTexture2D.new()
	var scene := PackedScene.new()
	scene.pack(decal)
	decal.free()
	return scene

func _fail(message: String) -> void:
	printerr("verify_decals: " + message)
	quit(1)
