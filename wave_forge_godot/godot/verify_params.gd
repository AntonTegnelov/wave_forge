## Tunes a preset's parameters before and while its stages run.
##
## Run by `../verify.sh` after `verify_import.gd`. The islands preset starts with much land and no
## trees: `pack_params` lists its three parameters with those values, and no tree stands anywhere.
## Raising the tree density then grows trees while the ground stays as it was, and a value outside
## a parameter's range or a name the pack does not declare is refused. The inspector lists each
## parameter as a slider over its range, `params/<name>`, reading its default until set and
## reverting to it; moving the tree slider back to 0 clears the trees again while they run.
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
	if not _check_sliders():
		return
	world.follow(Vector3(1, 0, 1))
	started_usec = Time.get_ticks_usec()

## The parameters as the inspector shows them.
func _check_sliders() -> bool:
	var sliders := {}
	for property: Dictionary in world.get_property_list():
		if property["name"].begins_with("params/"):
			sliders[property["name"]] = property
	if sliders.size() != 3:
		_fail("the inspector lists %s" % sliders.keys())
		return false
	var land: Dictionary = sliders["params/land"]
	if land["hint"] != PROPERTY_HINT_RANGE or land["hint_string"] != "0,1,0.01" or land["type"] != TYPE_FLOAT:
		_fail("land is listed as %s" % land)
		return false
	if not is_equal_approx(world.get("params/land"), 0.9) or not is_equal_approx(world.get("params/roughness"), 0.4):
		_fail("the sliders read %s and %s" % [world.get("params/land"), world.get("params/roughness")])
		return false
	if not is_equal_approx(world.property_get_revert("params/trees"), 0.5):
		_fail("the tree slider reverts to %s" % world.property_get_revert("params/trees"))
		return false
	return true

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
			world.set("params/trees", 0.0)
			# Wait for every chunk to come back: one being generated holds no trees at all.
			for chunk in _area():
				ready.erase(["trees", chunk])
			phase = "slide"
			started_usec = Time.get_ticks_usec()
			return false
		"slide":
			if _trees() > 0:
				return false
			print("verify_params: the islands start with no trees, grow them when the density rises, and keep their ground; out-of-range values and unknown names are refused; the inspector's sliders read, revert, and clear the trees again as they move")
			quit(0)
			return true
	return false

func _fail(message: String) -> void:
	printerr("verify_params: " + message)
	quit(1)
