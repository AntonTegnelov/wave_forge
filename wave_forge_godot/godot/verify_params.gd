## Tunes a preset's parameters before and while its stages run.
##
## Run by `../verify.sh` after `verify_import.gd`. The islands preset starts with much land and no
## trees: `pack_params` lists its three parameters with those values, and no tree stands anywhere.
## Raising the tree density then grows trees while the ground stays as it was, and a value outside
## a parameter's range or a name the pack does not declare is refused.
extends SceneTree

const CELLS := 8
const RADIUS := 2
const TIMEOUT_S := 30.0

var world: Node
var ready := {}
var started_usec := 0
var phase := "arrive"
var height_before := PackedFloat32Array()

func _initialize() -> void:
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://islands.world.ron"
	world.targets = PackedStringArray(["height", "trees"])
	world.seed = 5
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.view_radius = RADIUS
	world.collider_radius = -1
	world.params = {"land": 0.9, "trees": 0.0}
	root.add_child(world)
	world.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	world.stage_ready.connect(func(stage: String, chunk: Vector3i) -> void: ready[[stage, chunk]] = true)
	if not world.start():
		_fail("the stages did not start")
		return
	var listed := {}
	for param: Dictionary in world.pack_params():
		listed[param["name"]] = param["value"]
	if listed.size() != 3 or not is_equal_approx(listed["land"], 0.9) or not is_equal_approx(listed["roughness"], 0.4) or listed["trees"] != 0.0:
		_fail("pack_params lists %s" % listed)
		return
	world.follow(Vector3(1, 0, 1))
	started_usec = Time.get_ticks_usec()

func _area() -> Array[Vector3i]:
	var chunks: Array[Vector3i] = []
	for y in range(-RADIUS, RADIUS + 1):
		for x in range(-RADIUS, RADIUS + 1):
			chunks.append(Vector3i(x, y, 0))
	return chunks

func _trees() -> int:
	var count := 0
	for chunk in _area():
		for set: Dictionary in world.point_sets("trees", chunk):
			count += set["ids"].size()
	return count

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("the %s phase timed out" % phase)
		return true
	for chunk in _area():
		if not ready.has(["trees", chunk]) or not ready.has(["height", chunk]):
			return false
	match phase:
		"arrive":
			if _trees() != 0:
				_fail("%d trees at a tree density of 0" % _trees())
				return true
			height_before = world.field_values("height", Vector3i.ZERO)
			if world.update_params({"land": 2.0}) or world.update_params({"mountains": 1.0}):
				_fail("a value out of range or an undeclared name was taken")
				return true
			if not world.update_params({"trees": 1.0}):
				_fail("a tree density of 1 was refused")
				return true
			phase = "grow"
			started_usec = Time.get_ticks_usec()
			return false
		"grow":
			if _trees() == 0:
				return false
			if world.field_values("height", Vector3i.ZERO) != height_before:
				_fail("the ground changed with the tree density")
				return true
			print("verify_params: the islands start with no trees, grow %d when the density rises, and keep their ground; out-of-range values and unknown names are refused" % _trees())
			quit(0)
			return true
	return false

func _fail(message: String) -> void:
	printerr("verify_params: " + message)
	quit(1)
