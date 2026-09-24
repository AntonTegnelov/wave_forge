## Generates the valley pack's stages in a real Godot and checks what a game reads from them.
##
## Run by `../verify.sh` after `verify.gd`. The pack (examples/valley.world.ron) is the test pack of
## the first slice: rolling ground, towns on levelled sites built by the city module set, and trees
## between them. The node runs the stages on its own thread. This first asks for the sites alone over
## a wide area and finds a town the way a game would, then asks for everything around it and waits
## until every chunk has arrived, then checks that towns stand on level ground at their height, that trees
## stand on the ground and never on a town, and that Godot's thread stayed free. Then a character with
## gravity walks from the open ground straight through the town and out the other side, and must never
## sink below the ground's surface. Then moving away drops what is no longer needed. Last, the node
## starts again on the kernels the first start compiled and cached, and the time to the first town
## is printed for both starts, cold and warm.
##
## The town's modules get simple shapes here: a thin floor whose top is the bottom of every street-level cell, and
## a full box for everything above street level, so buildings are hollow at street level, as in
## marian42's city, and the walk can cross a town in a straight line.
extends SceneTree

const CELLS := 8
const CELL_SIZE := 2.0
const RADIUS := 3
## How far to look for a town first, in chunks: sixteen of the pack's regions of six chunks.
const SEARCH_RADIUS := 12
const TARGETS := ["level", "city", "trees", "cover", "hills"]
const LOAD_TIMEOUT_S := 180.0
## The node's own time on Godot's thread, at the 99th percentile and at worst.
const NODE_P99_MS := 2.0
const NODE_MAX_MS := 8.0
const COLLIDER_RADIUS := 2
## The walker: a capsule, how fast it walks, how long it may take, and how far its feet may be
## below the ground's surface before that counts as falling through, or above it while standing
## before that counts as a collider that does not match the mesh.
const WALKER_RADIUS := 0.4
const WALKER_HEIGHT := 1.5
const WALK_SPEED := 4.0
const WALK_TIMEOUT_S := 60.0
const SINK_TOLERANCE := 0.3
const GRAVITY := 20.0

var world: Node
## The chunk the checks look around: a town's, once one is found.
var centre := Vector3i.ZERO
var searching := true
var ready := {}
var dropped := {}
var started_usec := 0
var moved := false
var walker: CharacterBody3D
var walking := false
var walk_to_x := 0.0
var lowest_clearance := INF
var highest_standing := -INF
var walk_started_usec := 0
## When the first town chunk arrived after it was asked for, cold and then warm; 0 until then.
var towns_asked_usec := 0
var first_town_usec := 0
var cold_first_town_s := -1.0
var warm := false
const KERNEL_CACHE := "user://verify_kernels"

func _initialize() -> void:
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://valley.world.ron"
	world.rules_files = {"city": "res://city.ron"}
	world.targets = PackedStringArray(["towns"])
	world.seed = 5
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = Vector3.ONE * CELL_SIZE
	world.view_radius = SEARCH_RADIUS
	if DirAccess.dir_exists_absolute(KERNEL_CACHE):
		for file in DirAccess.get_files_at(KERNEL_CACHE):
			DirAccess.remove_absolute(KERNEL_CACHE.path_join(file))
	world.kernel_cache = KERNEL_CACHE
	world.ground_stage = "level"
	world.collider_radius = COLLIDER_RADIUS
	root.add_child(world)
	world.stage_ready.connect(_on_stage_ready)
	world.stage_dropped.connect(func(stage: String, chunk: Vector3i) -> void: dropped[[stage, chunk]] = true)
	world.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	if not world.start():
		_fail("the stages did not start")
		return
	_give_town_shapes()
	world.follow(_position_of(Vector3i.ZERO))
	started_usec = Time.get_ticks_usec()

## A floor slab at the bottom of every street-level cell, a full box for every other module but air.
func _give_town_shapes() -> void:
	var half := CELL_SIZE / 2.0
	var box := BoxShape3D.new()
	box.size = Vector3.ONE * CELL_SIZE
	for tag in ["building", "roof", "walkway", "pillar", "stair"]:
		for module in world.modules_tagged("city", tag):
			world.set_collision_shape(module, box)
	var slab := ConvexPolygonShape3D.new()
	var points := PackedVector3Array()
	for y in [-half - 0.2, -half]:
		for x in [-half, half]:
			for z in [-half, half]:
				points.append(Vector3(x, y, z))
	slab.points = points
	var street: PackedStringArray = world.modules_tagged("city", "street_level")
	if street.is_empty():
		_fail("the city has no street-level modules")
	for module in street:
		world.set_collision_shape(module, slab)

func _on_stage_ready(stage: String, chunk: Vector3i) -> void:
	ready[[stage, chunk]] = true
	# A Solve stage's chunks outside every site arrive at once; only one in a town has waited for it.
	if stage == "city" and first_town_usec == 0 and towns_asked_usec != 0 and not world.town("city", chunk).is_empty():
		first_town_usec = Time.get_ticks_usec()

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
	towns_asked_usec = started_usec
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
	if warm:
		return _check_warm(waited)
	if moved:
		return _check_dropped(waited)
	if walking:
		return false
	for chunk in _view():
		for stage: String in TARGETS:
			if not ready.has([stage, chunk]):
				if waited > LOAD_TIMEOUT_S:
					_fail("%s of %s had not arrived after %.0f s" % [stage, chunk, waited])
					return true
				return false
	var queued: Dictionary = world.stats()
	if queued["pending_signals"] > 0 or queued["pending_grounds"] > 0 or queued["pending_colliders"] > 0:
		return false
	print("verify_stages: %d chunks of %s arrived in %.1f s" % [_view().size(), TARGETS, waited])
	if not _check_towns() or not _check_trees() or not _check_cover():
		return true
	var stats: Dictionary = world.stats()
	print("verify_stages: the node's own process per frame p50 %.3f ms, p99 %.3f ms, max %.3f ms; its slowest frame: %d events in %.3f ms, %d grounds in %.3f ms, %d bodies in %.3f ms" % [
		stats["process_ms_median"], stats["process_ms_p99"], stats["process_ms_max"],
		stats["slowest_frame_events"], stats["slowest_frame_signals_ms"],
		stats["slowest_frame_grounds"], stats["slowest_frame_grounds_ms"],
		stats["slowest_frame_bodies"], stats["slowest_frame_bodies_ms"]])
	var costs: Dictionary = stats["stages"]
	for stage: String in costs:
		var cost: Dictionary = costs[stage]
		print("verify_stages: stage %s: %d products in %.1f ms, %.3f ms each, %.1f ms at most" % [stage, cost["products"], cost["ms"], cost["ms"] / maxi(cost["products"], 1), cost["slowest_ms"]])
	for stage: String in TARGETS:
		if not costs.has(stage) or costs[stage]["products"] == 0:
			_fail("no cost recorded for the stage %s" % stage)
			return true
	if stats["slowest_frame_events"] > 256:
		_fail("a frame emitted %d signals; at most 256 are allowed" % stats["slowest_frame_events"])
		return true
	if stats["process_ms_p99"] > NODE_P99_MS or stats["process_ms_max"] > NODE_MAX_MS:
		_fail("the node's process took %.2f ms at the 99th percentile, %.2f ms at worst" % [stats["process_ms_p99"], stats["process_ms_max"]])
		return true
	if not _check_ground() or not _check_sampling():
		return true
	_start_walk()
	return false

## A sample and an atlas of the ground give what its chunks hold, without generating any.
func _check_sampling() -> bool:
	var level: PackedFloat32Array = world.field_values("hills", centre)
	var atlas: PackedFloat32Array = world.atlas("hills", Vector2i(centre.x * CELLS, centre.y * CELLS), Vector2i(CELLS, CELLS))
	if atlas != level:
		_fail("the atlas of the hills in %s differs from its chunk" % centre)
		return false
	var at := Vector3((centre.x * CELLS + 3.5) * CELL_SIZE, 0, (centre.y * CELLS + 5.5) * CELL_SIZE)
	if world.sample("hills", at) != level[5 * CELLS + 3]:
		_fail("a sample of the hills differs from its chunk")
		return false
	print("verify_stages: a sample and an atlas of the hills match their chunk")
	return true

## A chunk has ground exactly when its height field and the eight around it are held, which the
## trees' reach makes every chunk of the view; every chunk within the collider radius has a body.
func _check_ground() -> bool:
	var grounds: Array = world.ground_chunks()
	var bodies: Array = world.collider_chunks()
	for chunk in _view():
		var complete := true
		for dy in range(-1, 2):
			for dx in range(-1, 2):
				if world.field_values("level", chunk + Vector3i(dx, dy, 0)).is_empty():
					complete = false
		if grounds.has(chunk) != complete:
			_fail("%s has ground: %s, and its fields and its neighbours' are held: %s" % [chunk, grounds.has(chunk), complete])
			return false
		if not complete:
			_fail("the height field around %s is not all held, though the trees read it" % chunk)
			return false
		var near := maxi(absi(chunk.x - centre.x), absi(chunk.y - centre.y)) <= COLLIDER_RADIUS
		if bodies.has(chunk) != near:
			_fail("%s has a body: %s, within the collider radius: %s" % [chunk, bodies.has(chunk), near])
			return false
	print("verify_stages: %d chunks have ground, %d have bodies" % [grounds.size(), bodies.size()])
	return true

## Puts the walker on the ground a little west of the town, level with the town's middle.
func _start_walk() -> void:
	var site: Dictionary = world.sites("towns", centre)[0]
	var chunk_size := CELLS * CELL_SIZE
	var low: Vector2i = site["min"]
	var high: Vector2i = site["max"]
	var z := (low.y + high.y) * 0.5 * chunk_size
	var from := Vector3(low.x * chunk_size - 12.0, 0.0, z)
	walk_to_x = high.x * chunk_size + 12.0
	from.y = _surface(from) + WALKER_HEIGHT / 2.0 + 0.5
	walker = CharacterBody3D.new()
	var capsule := CapsuleShape3D.new()
	capsule.radius = WALKER_RADIUS
	capsule.height = WALKER_HEIGHT
	var collision := CollisionShape3D.new()
	collision.shape = capsule
	walker.add_child(collision)
	root.add_child(walker)
	walker.global_position = from
	walking = true
	walk_started_usec = Time.get_ticks_usec()
	print("verify_stages: walking from x %.1f to %.1f at z %.1f, through the town of chunks %s to %s" % [from.x, walk_to_x, z, site["min"], site["max"]])

func _physics_process(delta: float) -> bool:
	if not walking:
		return false
	world.follow(walker.global_position)
	walker.velocity.x = WALK_SPEED
	walker.velocity.z = 0.0
	walker.velocity.y = 0.0 if walker.is_on_floor() else walker.velocity.y - GRAVITY * delta
	walker.move_and_slide()
	var at := walker.global_position
	var surface := _surface(at)
	if is_nan(surface):
		return true
	var feet := at.y - WALKER_HEIGHT / 2.0
	lowest_clearance = minf(lowest_clearance, feet - surface)
	if feet < surface - SINK_TOLERANCE:
		_fail("the walker sank through the ground at %s: feet at %.2f, the surface at %.2f" % [at, feet, surface])
		return true
	if walker.is_on_floor():
		highest_standing = maxf(highest_standing, feet - surface)
		if feet > surface + SINK_TOLERANCE:
			_fail("the walker stands at %s with its feet %.2f above the ground's surface" % [at, feet - surface])
			return true
	if at.x >= walk_to_x:
		print("verify_stages: walked through the town in %.1f s; the feet were at most %.2f below the ground's surface, and at most %.2f above it while standing" % [(Time.get_ticks_usec() - walk_started_usec) / 1e6, maxf(0.0, -lowest_clearance), highest_standing])
		walking = false
		walker.queue_free()
		world.follow(Vector3(40 * CELLS * CELL_SIZE, 0, 40 * CELLS * CELL_SIZE))
		moved = true
		started_usec = Time.get_ticks_usec()
		return false
	if (Time.get_ticks_usec() - walk_started_usec) / 1e6 > WALK_TIMEOUT_S:
		_fail("the walker was stuck at %s after %.0f s" % [at, WALK_TIMEOUT_S])
		return true
	return false

## The ground's surface under a point, as the ground mesh has it: the heights of the four column
## centres around it, on the mesh's two triangles per square. NAN, after failing, where the height
## field has not arrived.
func _surface(at: Vector3) -> float:
	var u := at.x / CELL_SIZE - 0.5
	var v := at.z / CELL_SIZE - 0.5
	var i := floori(u)
	var j := floori(v)
	var fu := u - i
	var fv := v - j
	var h00 := _column_height(i, j)
	var h10 := _column_height(i + 1, j)
	var h01 := _column_height(i, j + 1)
	var h11 := _column_height(i + 1, j + 1)
	if is_nan(h00 + h10 + h01 + h11):
		return NAN
	if fu + fv <= 1.0:
		return h00 + fu * (h10 - h00) + fv * (h01 - h00)
	return h11 + (1.0 - fu) * (h01 - h11) + (1.0 - fv) * (h10 - h11)

func _column_height(x: int, y: int) -> float:
	var chunk := Vector3i(floori(float(x) / CELLS), floori(float(y) / CELLS), 0)
	var level: PackedFloat32Array = world.field_values("level", chunk)
	if level.is_empty():
		_fail("the ground's height field of %s has not arrived under the walker" % chunk)
		return NAN
	return level[(y - chunk.y * CELLS) * CELLS + (x - chunk.x * CELLS)] * CELL_SIZE

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

## Every column's cover is highland above 30 cells of height and lowland elsewhere, as the pack's
## rule says.
func _check_cover() -> bool:
	var names: PackedStringArray = world.category_names("cover")
	if names != PackedStringArray(["highland", "lowland"]):
		_fail("the cover's categories are %s" % [names])
		return false
	var counts := {"highland": 0, "lowland": 0}
	for chunk in _view():
		var level: PackedFloat32Array = world.field_values("level", chunk)
		var cover: PackedByteArray = world.categories("cover", chunk)
		if cover.size() != level.size():
			_fail("the cover of %s has %d columns, its height %d" % [chunk, cover.size(), level.size()])
			return false
		for i in level.size():
			var expected := "highland" if level[i] > 30.0 else "lowland"
			if names[cover[i]] != expected:
				_fail("column %d of %s is %s at height %.2f" % [i, chunk, names[cover[i]], level[i]])
				return false
			counts[expected] += 1
	print("verify_stages: the cover follows its rule: %s" % [counts])
	return true

## Moving far away drops every chunk of the first view.
func _check_dropped(waited: float) -> bool:
	for chunk in _view():
		if not dropped.has(["city", chunk]) or not dropped.has(["level", chunk]):
			if waited > 30.0:
				_fail("%s was not dropped after moving away" % chunk)
				return true
			return false
	var grounds: Array = world.ground_chunks()
	for chunk in _view():
		if grounds.has(chunk):
			if waited > 30.0:
				_fail("the ground of %s was not freed after moving away" % chunk)
				return true
			return false
	print("verify_stages: moving away dropped the first view and its ground")
	cold_first_town_s = (first_town_usec - towns_asked_usec) / 1e6
	# Start again, on the kernels the first start compiled into the cache.
	warm = true
	ready.clear()
	first_town_usec = 0
	world.targets = PackedStringArray(TARGETS)
	world.view_radius = RADIUS
	if not world.start():
		_fail("the stages did not start again")
		return true
	world.follow(_position_of(centre))
	started_usec = Time.get_ticks_usec()
	towns_asked_usec = started_usec
	return false

## The first town after starting again on cached kernels, against the first start's.
func _check_warm(waited: float) -> bool:
	if first_town_usec == 0:
		if waited > LOAD_TIMEOUT_S:
			_fail("no town arrived %.0f s after starting again" % waited)
			return true
		return false
	var warm_s := (first_town_usec - towns_asked_usec) / 1e6
	print("verify_stages: the first town arrived %.1f s after it was asked for on a cold start and %.1f s on a start with cached kernels" % [cold_first_town_s, warm_s])
	quit(0)
	return true

func _fail(message: String) -> void:
	printerr("verify_stages: " + message)
	quit(1)
