## Generates the valley pack's stages in a real Godot and checks what a game reads from them.
##
## Run by `../verify.sh` after `verify.gd`. The pack (examples/valley.world.ron) is the test pack of
## the first slice: rolling ground, towns on levelled sites built by the city module set, and trees
## between them. The node runs the stages on its own thread. This first asks for the sites alone over
## a wide area and finds a town the way a game would, then asks for everything around it and waits
## until every chunk has arrived, then checks that towns stand on level ground at their height, that trees
## stand on the ground and never on a town, that moving away drops what is no longer needed, and
## that Godot's thread stayed free.
extends SceneTree

const CELLS := 8
const CELL_SIZE := 2.0
const RADIUS := 3
## How far to look for a town first, in chunks: sixteen of the pack's regions of six chunks.
const SEARCH_RADIUS := 12
const TARGETS := ["level", "city", "trees"]
const LOAD_TIMEOUT_S := 180.0
## The node's own time on Godot's thread, at the 99th percentile and at worst.
const NODE_P99_MS := 2.0
const NODE_MAX_MS := 8.0

var world: Node
## The chunk the checks look around: a town's, once one is found.
var centre := Vector3i.ZERO
var searching := true
var ready := {}
var dropped := {}
var started_usec := 0
var moved := false

func _initialize() -> void:
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://valley.world.ron"
	world.rules_files = {"city": "res://city.ron"}
	world.targets = PackedStringArray(["towns"])
	world.seed = 5
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = Vector3.ONE * CELL_SIZE
	world.view_radius = SEARCH_RADIUS
	root.add_child(world)
	world.stage_ready.connect(func(stage: String, chunk: Vector3i) -> void: ready[[stage, chunk]] = true)
	world.stage_dropped.connect(func(stage: String, chunk: Vector3i) -> void: dropped[[stage, chunk]] = true)
	world.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	if not world.start():
		_fail("the stages did not start")
		return
	world.follow(_position_of(Vector3i.ZERO))
	started_usec = Time.get_ticks_usec()

func _position_of(chunk: Vector3i) -> Vector3:
	return Vector3((chunk.x + 0.5) * CELLS * CELL_SIZE, 0, (chunk.y + 0.5) * CELLS * CELL_SIZE)

## Once the sites of the search area have arrived, the first town's chunk becomes the centre.
func _find_town(waited: float) -> bool:
	var found := false
	for y in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
		for x in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
			var chunk := Vector3i(x, y, 0)
			if not ready.has(["towns", chunk]):
				if waited > LOAD_TIMEOUT_S:
					_fail("the sites of %s had not arrived after %.0f s" % [chunk, waited])
				return false
			if not found and not world.sites("towns", chunk).is_empty():
				centre = chunk
				found = true
	if not found:
		_fail("no town within %d chunks of the origin" % SEARCH_RADIUS)
		return false
	var site: Dictionary = world.sites("towns", centre)[0]
	print("verify_stages: found the town of region %s at height %.2f, chunks %s to %s" % [site["region"], site["height"], site["min"], site["max"]])
	world.targets = PackedStringArray(TARGETS)
	world.view_radius = RADIUS
	world.follow(_position_of(centre))
	searching = false
	started_usec = Time.get_ticks_usec()
	return false

func _view() -> Array[Vector3i]:
	var chunks: Array[Vector3i] = []
	for y in range(-RADIUS, RADIUS + 1):
		for x in range(-RADIUS, RADIUS + 1):
			chunks.append(centre + Vector3i(x, y, 0))
	return chunks

func _process(_delta: float) -> bool:
	var waited := (Time.get_ticks_usec() - started_usec) / 1e6
	if searching:
		_find_town(waited)
		return false
	if moved:
		return _check_dropped(waited)
	for chunk in _view():
		for stage: String in TARGETS:
			if not ready.has([stage, chunk]):
				if waited > LOAD_TIMEOUT_S:
					_fail("%s of %s had not arrived after %.0f s" % [stage, chunk, waited])
					return true
				return false
	print("verify_stages: %d chunks of %s arrived in %.1f s" % [_view().size(), TARGETS, waited])
	if not _check_towns() or not _check_trees():
		return true
	var stats: Dictionary = world.stats()
	print("verify_stages: the node's own process per frame p50 %.3f ms, p99 %.3f ms, max %.3f ms" % [
		stats["process_ms_median"], stats["process_ms_p99"], stats["process_ms_max"]])
	if stats["process_ms_p99"] > NODE_P99_MS or stats["process_ms_max"] > NODE_MAX_MS:
		_fail("the node's process took %.2f ms at the 99th percentile, %.2f ms at worst" % [stats["process_ms_p99"], stats["process_ms_max"]])
		return true
	world.follow(Vector3(40 * CELLS * CELL_SIZE, 0, 40 * CELLS * CELL_SIZE))
	moved = true
	started_usec = Time.get_ticks_usec()
	return false

## Every town chunk holds a whole chunk of tiles, and the ground under it is level at its height.
func _check_towns() -> bool:
	var towns := 0
	for chunk in _view():
		var town: Dictionary = world.town("city", chunk)
		if town.is_empty():
			continue
		towns += 1
		var tiles: PackedInt32Array = town["tiles"]
		if tiles.size() != CELLS * CELLS * CELLS:
			_fail("the town in %s has %d tiles" % [chunk, tiles.size()])
			return false
		var level: PackedFloat32Array = world.field_values("level", chunk)
		for value in level:
			if not is_equal_approx(value, town["height"]):
				_fail("the ground under the town in %s is at %.3f, the town at %.3f" % [chunk, value, town["height"]])
				return false
		# The town's models stand on it: no instance's centre is below the ground, and the lowest
		# layer's are half a cell above it.
		var lowest := INF
		for set: Dictionary in world.town_instance_sets("city", chunk, PackedStringArray()):
			var transforms: PackedFloat32Array = set["transforms"]
			for i in range(0, transforms.size(), 12):
				lowest = minf(lowest, transforms[i + 7])
		var ground: float = town["height"] * CELL_SIZE
		if not is_equal_approx(lowest, ground + CELL_SIZE / 2.0):
			_fail("the town's lowest models in %s are centred at %.3f, the ground is at %.3f" % [chunk, lowest, ground])
			return false
	if towns == 0:
		_fail("no town in the %d chunks around the focus" % _view().size())
		return false
	print("verify_stages: %d town chunks, each on level ground at its site's height" % towns)
	return true

## Every tree stands on the ground of its column, and none stands in a town's chunk.
func _check_trees() -> bool:
	var trees := 0
	for chunk in _view():
		var level: PackedFloat32Array = world.field_values("level", chunk)
		var in_town: bool = not world.town("city", chunk).is_empty()
		for set: Dictionary in world.point_sets("trees", chunk):
			var transforms: PackedFloat32Array = set["transforms"]
			for i in range(0, transforms.size(), 12):
				trees += 1
				if in_town:
					_fail("a tree stands in the town in %s" % chunk)
					return false
				var x := int(floor(transforms[i + 3] / CELL_SIZE)) - chunk.x * CELLS
				var y := int(floor(transforms[i + 11] / CELL_SIZE)) - chunk.y * CELLS
				var ground := level[y * CELLS + x] * CELL_SIZE
				if absf(transforms[i + 7] - ground) > 0.001:
					_fail("a tree in %s stands at %.3f, the ground at %.3f" % [chunk, transforms[i + 7], ground])
					return false
	if trees == 0:
		_fail("no tree around the focus")
		return false
	print("verify_stages: %d trees, each on the ground and outside the towns" % trees)
	return true

## Moving far away drops every chunk of the first view.
func _check_dropped(waited: float) -> bool:
	for chunk in _view():
		if not dropped.has(["city", chunk]) or not dropped.has(["level", chunk]):
			if waited > 30.0:
				_fail("%s was not dropped after moving away" % chunk)
				return true
			return false
	print("verify_stages: moving away dropped the first view")
	quit(0)
	return true

func _fail(message: String) -> void:
	printerr("verify_stages: " + message)
	quit(1)
