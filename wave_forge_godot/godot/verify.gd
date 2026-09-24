## Drives the extension the way a game would and checks what it produced.
##
## Run through `../verify.sh`, which builds the extension, copies it in and starts Godot headless.
## A focus runs across a strip of chunks and back in real time, which is the whole of what a game
## does: chunks appear ahead of it, chunks behind it are dropped, and the ones it returns to come back
## the same. It never waits for generation, so a generator that falls behind shows up as a chunk
## missing beside the player, and every frame is timed, so a stall on Godot's own thread shows up
## too. The script exits non-zero with a message on any failure, so a shell can tell whether it
## passed.
extends SceneTree

## The world is this many chunks along x, three along y, one tall.
const CHUNKS_X := 8
const CHUNKS_Y := 3
## Cells along each axis of one chunk.
const CELLS := 8
## World units per cell, so one chunk is 16 units across.
const CELL_SIZE := 2.0
## How many chunks to generate around the focus, and how far beyond that to keep them. A column two
## chunks out is asked for when the focus enters a chunk and is needed when it enters the next one,
## 16 units later.
const VIEW_RADIUS := 2
const EVICT_MARGIN := 1
## Chunks this close to the focus's chunk have colliders.
const COLLIDER_RADIUS := 1
## Every chunk this close to the focus's chunk has to have tiles on every frame of the walk.
const READY_RADIUS := 1
## Units per second: a running player, three times a walking pace of 1.4.
const PACE := 4.2
## The frame rate a game would cap its main loop at.
const FPS := 60
## How long Godot's own thread may spend processing any frame, including the extension, the signal
## handlers and the navigation server's sync: half a frame, leaving the rest to rendering.
const PROCESS_MAX_MS := 8.0
## How long the node's own `process` may take on its busiest frames, the 99th percentile.
const NODE_P99_MS := 2.0
## Building the device, compiling kernels and generating the first view is loading, not play.
const LOAD_TIMEOUT_S := 180.0
## Tile indices, in the order `rules.ron` declares them.
const WATER := 0
const SAND := 1
const GRASS := 2
const FOREST := 3
## The focus starts in the middle of this chunk, runs to the middle of the other and back.
const WALK_FROM := 1
const WALK_TO := CHUNKS_X - 2

var world: Node
var updated := {}
var failed := {}
var evicted := {}
## The chunks that have tiles now, as the signals tell it.
var present := {}
## The tiles of the chunk the walk starts in, to compare against when it comes back.
var first_tiles: PackedInt32Array
var started_usec := 0
var walk_started_usec := -1
var last_frame_usec := 0
## The slowest frame Godot's thread processed during the walk, and per frame the time since the
## previous one. Godot publishes its process time once a second, as the slowest frame of that
## second, so the slowest is all it tells; the node's own `stats` time every frame.
var slowest_process_ms := 0.0
var period_ms := PackedFloat64Array()
## Frames on which a chunk within the ready radius had no tiles, and the first of them.
var late_frames := 0
var first_late := ""
var done := false
## The chunks `navigation_ready` named, and when the walk's checks finished and the wait for the
## last bakes began.
var navigable := {}
var navigation_wait_usec := -1
## A node set to start on its own from a rule file, checked on the first frame.
var starting_on_ready: Node = null

func _initialize() -> void:
	world = ClassDB.instantiate("WaveForgeWorld")
	if world == null:
		_fail("the extension did not register WaveForgeWorld")
		return
	world.seed = 7
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = Vector3(CELL_SIZE, CELL_SIZE, CELL_SIZE)
	world.view_radius = VIEW_RADIUS
	world.halo = 1
	world.evict_margin = EVICT_MARGIN
	world.world_chunks = Vector3i(CHUNKS_X, CHUNKS_Y, 1)
	# What a game does with a prior: no water on the ground layer, and no forest against the edges
	# of a bounded world. The rule set only lets a material meet itself vertically, so a ground
	# layer without water is a world without water, which is easy to check. The layers above it are
	# left open with an empty array, which is how a scene says "anything goes here".
	var layers: Array[PackedInt32Array] = [PackedInt32Array([SAND, GRASS, FOREST]), PackedInt32Array()]
	world.set_layer_tiles(layers)
	# Every material's cell is solid ground: a box the size of the cell, in the chunks near the focus.
	var box := BoxShape3D.new()
	box.size = Vector3.ONE * CELL_SIZE
	for material in 4:
		world.set_collision_shape(["water", "sand", "grass", "forest"][material], box)
	world.collider_radius = COLLIDER_RADIUS
	# The same boxes are what agents walk on, in the chunks near the focus.
	var agent := NavigationMesh.new()
	agent.agent_radius = 0.5
	agent.agent_height = 1.5
	agent.agent_max_climb = 0.25
	world.navigation_template = agent
	world.navigation_radius = COLLIDER_RADIUS
	world.navigation_ready.connect(func(chunk: Vector3i) -> void: navigable[chunk] = true)
	for axis in 4:
		world.ban_tiles_on_face(axis, PackedInt32Array([FOREST]))
	world.chunk_updated.connect(_on_chunk_updated)
	world.chunk_failed.connect(_on_chunk_failed)
	world.chunk_evicted.connect(_on_chunk_evicted)
	world.generation_failed.connect(_on_generation_failed)
	root.add_child(world)

	var rules := FileAccess.get_file_as_string("res://rules.ron")
	if rules.is_empty():
		_fail("res://rules.ron is missing")
		return
	if not world.load_rules(rules):
		_fail("the rule set was refused")
		return
	# A tile set's tiles keep the names the file gives them, and have no rotations.
	if world.tile_name(GRASS) != "grass" or world.tiles_named("forest") != PackedInt32Array([FOREST]) or world.tile_rotation(SAND) != 0:
		_fail("the tile set's names are not the file's: %s" % world.tile_name(GRASS))
		return
	if not _check_module_catalogue():
		return
	if not _check_editor_setup():
		return
	if not world.start():
		_fail("generation did not start")
		return
	Engine.max_fps = FPS
	started_usec = Time.get_ticks_usec()
	world.follow(_position_at(_start_x()))
	print("verify: starting in chunk ", world.chunk_at(_position_at(_start_x())))

## What a scene sets in the editor: grouped properties, and a node that starts on its own from a
## rule file.
func _check_editor_setup() -> bool:
	var node: Node = ClassDB.instantiate("WaveForgeWorld")
	var groups := []
	for property in node.get_property_list():
		if property["usage"] & PROPERTY_USAGE_GROUP:
			groups.append(property["name"])
	# The node's own groups come first; Node's inherited ones follow.
	if groups.slice(0, 7) != ["Rules", "World", "Streaming", "Physics", "Navigation", "Audio", "Advanced"]:
		node.free()
		_fail("the inspector groups are %s" % [groups])
		return false
	node.rules_file = "res://city.ron"
	node.start_on_ready = true
	# A node added now is readied on the first frame, which is where _process looks at it.
	root.add_child(node)
	starting_on_ready = node
	print("verify: the inspector groups its properties")
	return true

## Whether the node set to start on ready did, once it has been readied.
func _check_started_on_ready() -> bool:
	var started: bool = starting_on_ready.is_generating()
	starting_on_ready.queue_free()
	starting_on_ready = null
	if not started:
		_fail("a node set to start on ready did not start from its rules_file")
		return false
	print("verify: a node set to start on ready started from its rules_file")
	return true

## What a game needs to place a module set's models: names, rotations, tags, and where cells are.
## Loading the rules is enough, so this node is never started.
func _check_module_catalogue() -> bool:
	var city: Node = ClassDB.instantiate("WaveForgeWorld")
	var text := FileAccess.get_file_as_string("res://city.ron")
	var ok := _check_city(city, text)
	city.free()
	return ok

func _check_city(city: Node, text: String) -> bool:
	if text.is_empty() or not city.load_rules(text):
		_fail("res://city.ron could not be loaded")
		return false
	if city.tile_count() != 81:
		_fail("the city has %d tiles, expected 81" % city.tile_count())
		return false
	var roads: PackedInt32Array = city.tiles_named("road_straight")
	if roads.size() != 2 or city.tile_rotation(roads[1]) != 1 or city.tile_name(roads[1]) != "road_straight":
		_fail("a straight road's turns are wrong: %s" % roads)
		return false
	# A quarter turn takes the lattice's +x to its +y, which is Godot's +z.
	var turned: Vector3 = city.tile_basis(roads[1]) * Vector3.RIGHT
	if not turned.is_equal_approx(Vector3.BACK):
		_fail("a quarter turn takes +x to %s, expected +z" % turned)
		return false
	var street_level: PackedInt32Array = city.tiles_tagged("street_level")
	if not street_level.has(city.tiles_named("grass")[0]) or street_level.has(city.tiles_named("air")[0]):
		_fail("street level is %s" % street_level)
		return false
	# Cells run along x first, then the lattice's y (Godot's z), then up.
	var origin := Vector3i(0, 0, 0)
	var expected := {0: Vector3(0.5, 0.5, 0.5), 1: Vector3(1.5, 0.5, 0.5), 8: Vector3(0.5, 0.5, 1.5), 64: Vector3(0.5, 1.5, 0.5)}
	for cell: int in expected:
		var at: Vector3 = city.cell_position(origin, cell)
		if not at.is_equal_approx(expected[cell]):
			_fail("cell %d is at %s, expected %s" % [cell, at, expected[cell]])
			return false
	print("verify: the city's 81 tiles are named, turned and tagged as a game needs to place them")
	return _check_models(city)

## Every module of the city has a model a game can load at runtime, named as its tiles are: a mesh
## coloured by a texture, or nothing at all for air.
func _check_models(city: Node) -> bool:
	var names := {}
	for tile in city.tile_count():
		names[city.tile_name(tile)] = true
	var triangles := 0
	for module: String in names:
		var doc := GLTFDocument.new()
		var state := GLTFState.new()
		var path := "res://models/%s.glb" % module
		if doc.append_from_file(path, state) != OK:
			_fail("%s did not load as glTF" % path)
			return false
		var scene := doc.generate_scene(state)
		var meshes := scene.find_children("*", "MeshInstance3D", true, false)
		var drawn := not meshes.is_empty()
		if drawn != (module != "air" and module != "stair_head"):
			scene.free()
			_fail("%s has %d meshes" % [module, meshes.size()])
			return false
		for instance: MeshInstance3D in meshes:
			var material: StandardMaterial3D = instance.mesh.surface_get_material(0)
			if material == null or material.albedo_texture == null:
				scene.free()
				_fail("%s is not coloured by a texture" % module)
				return false
			var aabb := instance.mesh.get_aabb()
			if not Rect2(-0.5, -0.5, 1.0, 1.0).encloses(Rect2(aabb.position.x, aabb.position.z, aabb.size.x, aabb.size.z)):
				scene.free()
				_fail("%s reaches outside its cell: %s" % [module, aabb])
				return false
			triangles += instance.mesh.surface_get_array_index_len(0) / 3
		scene.free()
	print("verify: %d module models load as glTF, %d triangles in all" % [names.size(), triangles])
	return true

## Loads until the chunks around the start are there, then runs the route by the clock, whatever
## generation is doing.
func _process(_delta: float) -> bool:
	if done:
		return navigation_wait_usec < 0 or _await_navigation()
	if starting_on_ready != null and not _check_started_on_ready():
		return true
	if not world.is_generating():
		_fail("generation stopped")
		return true
	var now := Time.get_ticks_usec()
	if walk_started_usec < 0:
		var loading := (now - started_usec) / 1e6
		if loading > LOAD_TIMEOUT_S:
			_fail("the first view was not there after %.0f s, %d generated" % [loading, updated.size()])
			return true
		if _ready_around(WALK_FROM):
			print("verify: loaded in %.2f s" % loading)
			first_tiles = world.tiles_at(Vector3i(WALK_FROM, 1, 0))
			walk_started_usec = now
			last_frame_usec = now
		return false

	# The slowest frame of Godot's thread in the last second: the extension draining its worker and
	# emitting signals, this script's handlers, and the navigation server's sync.
	slowest_process_ms = maxf(slowest_process_ms, Performance.get_monitor(Performance.TIME_PROCESS) * 1000.0)
	period_ms.append((now - last_frame_usec) / 1000.0)
	last_frame_usec = now
	var x := _x_at((now - walk_started_usec) / 1e6)
	if is_nan(x):
		# Colliders are built a few chunks a frame; the check waits for the last of them.
		if world.stats()["pending_colliders"] > 0:
			return false
		# The loop goes on for the navigation check; a failed check has already quit.
		_check()
		return false
	world.follow(_position_at(x))
	var chunk_x := int(floor(x / _chunk_units()))
	if not _ready_around(chunk_x):
		late_frames += 1
		if first_late.is_empty():
			first_late = "around chunk %d at %.2f s" % [chunk_x, (now - walk_started_usec) / 1e6]
	return false

## Where along x the focus is `t` seconds into the run, or NAN once it is back.
func _x_at(t: float) -> float:
	var one_way := (WALK_TO - WALK_FROM) * _chunk_units()
	var run := t * PACE
	if run > 2.0 * one_way:
		return NAN
	return _start_x() + (run if run <= one_way else 2.0 * one_way - run)

## Whether every chunk of the world within the ready radius of column `chunk_x` has tiles.
func _ready_around(chunk_x: int) -> bool:
	for x in range(chunk_x - READY_RADIUS, chunk_x + READY_RADIUS + 1):
		if x < 0 or x >= CHUNKS_X:
			continue
		for y in CHUNKS_Y:
			if not present.has(Vector3i(x, y, 0)):
				return false
	return true

func _chunk_units() -> float:
	return CELLS * CELL_SIZE

func _start_x() -> float:
	return (WALK_FROM + 0.5) * _chunk_units()

func _position_at(x: float) -> Vector3:
	# Godot's xz plane is the lattice's xy: the strip runs along x, and the middle row is y = 1.
	return Vector3(x, 0.0, 1.5 * _chunk_units())

## The value below which a share `q` of `values` lie.
func _quantile(values: PackedFloat64Array, q: float) -> float:
	var sorted := values.duplicate()
	sorted.sort()
	return sorted[int(round((sorted.size() - 1) * q))]

func _has_chunk(chunk: Vector3i) -> bool:
	return not world.tiles_at(chunk).is_empty()

func _on_chunk_updated(chunk: Vector3i) -> void:
	updated[chunk] = updated.get(chunk, 0) + 1
	present[chunk] = true

func _on_chunk_failed(chunk: Vector3i, status: String) -> void:
	failed[chunk] = status

func _on_chunk_evicted(chunk: Vector3i) -> void:
	evicted[chunk] = true
	present.erase(chunk)

func _on_generation_failed(reason: String) -> void:
	_fail("generation failed: " + reason)

## Once the chunks near the focus have their navigation meshes, a path runs from one side of them to
## the other across two seams, as straight as the flat ground allows.
func _await_navigation() -> bool:
	var focus := Vector3i(WALK_FROM, 1, 0)
	var waited := (Time.get_ticks_usec() - navigation_wait_usec) / 1e6
	var baked := 0
	for x in range(focus.x - COLLIDER_RADIUS, focus.x + COLLIDER_RADIUS + 1):
		for y in range(focus.y - COLLIDER_RADIUS, focus.y + COLLIDER_RADIUS + 1):
			var chunk := Vector3i(x, y, 0)
			if x >= 0 and y >= 0 and y < CHUNKS_Y and navigable.has(chunk) and world.navigation_chunks().has(chunk):
				baked += 1
	if baked < 9:
		if waited > 60.0:
			_fail("%d of 9 chunks near the focus have navigation after %.0f s" % [baked, waited])
		return waited > 60.0
	# The map takes the new regions in on its next synchronisation.
	if waited < 0.5:
		return false
	var top := func(chunk: Vector3i) -> Vector3:
		var cell := (7 * CELLS + 4) * CELLS + 4
		return world.cell_position(chunk, cell) + Vector3.UP * (CELL_SIZE / 2.0)
	var from: Vector3 = top.call(Vector3i(focus.x - 1, 1, 0))
	var to: Vector3 = top.call(Vector3i(focus.x + 1, 1, 0))
	var map := root.get_world_3d().navigation_map
	var path := NavigationServer3D.map_get_path(map, from, to, true)
	if path.is_empty() or path[path.size() - 1].distance_to(to) > 0.5:
		_describe_navigation(map, [from, to])
		_fail("no path from %s to %s across the seams: %s" % [from, to, path])
		return true
	# Agents walk on the ground's top, not on a floor Recast found inside it.
	for point: Vector3 in path:
		if absf(point.y - from.y) > 0.5:
			_describe_navigation(map, [from, to])
			_fail("the path runs at height %.2f, the ground's top is %.2f: %s" % [point.y, from.y, path])
			return true
	var length := 0.0
	for i in range(1, path.size()):
		length += path[i - 1].distance_to(path[i])
	if length > from.distance_to(to) * 1.05:
		_fail("the path across flat ground is %.1f long for %.1f straight" % [length, from.distance_to(to)])
		return true
	var stats: Dictionary = world.stats()
	print("verify: navigation on the 9 chunks near the focus, a path of %.1f across two seams for %.1f straight; %d bakes, median %.0f ms, max %.0f ms, %d polygons" % [
		length, from.distance_to(to), stats["navigation_baked"], stats["navigation_bake_ms_median"], stats["navigation_bake_ms_max"], stats["navigation_polygons"]])
	quit(0)
	return true

## What the navigation map holds, to explain a path that was not found: each region's bounds, and
## the map's nearest point to each of `points` with the region it lies in.
func _describe_navigation(map: RID, points: Array) -> void:
	for region: RID in NavigationServer3D.map_get_regions(map):
		printerr("verify: region %s bounds %s" % [region, NavigationServer3D.region_get_bounds(region)])
	for point: Vector3 in points:
		printerr("verify: nearest the map gets to %s is %s, in region %s" % [point, NavigationServer3D.map_get_closest_point(map, point), NavigationServer3D.map_get_closest_point_owner(map, point)])

## The chunks near the focus have colliders and no others do, and a ray down onto a cell hits that
## cell's instance.
func _check_colliders() -> bool:
	var focus := Vector3i(WALK_FROM, 1, 0)
	var expected := {}
	for x in range(focus.x - COLLIDER_RADIUS, focus.x + COLLIDER_RADIUS + 1):
		for y in range(focus.y - COLLIDER_RADIUS, focus.y + COLLIDER_RADIUS + 1):
			if x >= 0 and x < CHUNKS_X and y >= 0 and y < CHUNKS_Y:
				expected[Vector3i(x, y, 0)] = true
	var bodies := {}
	for chunk: Vector3i in world.collider_chunks():
		bodies[chunk] = true
	if bodies.keys().size() != expected.keys().size() or not bodies.keys().all(func(c: Vector3i) -> bool: return expected.has(c)):
		_fail("colliders on %s, expected the chunks within %d of %s" % [bodies.keys(), COLLIDER_RADIUS, focus])
		return false
	# Straight down onto the cell at (2, 5) of the focus's chunk: it hits the top of that column,
	# cell z = 7, whose index is (7 * 8 + 5) * 8 + 2.
	var cell_index := (7 * CELLS + 5) * CELLS + 2
	var above: Vector3 = world.cell_position(focus, cell_index) + Vector3.UP * 10.0
	var query := PhysicsRayQueryParameters3D.create(above, above + Vector3.DOWN * 40.0)
	var hit := root.get_world_3d().direct_space_state.intersect_ray(query)
	if hit.is_empty():
		_fail("a ray down onto chunk %s hit nothing" % focus)
		return false
	var instance: Dictionary = world.collider_instance(hit["rid"], hit["shape"])
	if instance.is_empty() or instance["chunk"] != focus or instance["id"] & 0xFFFFFFFF != cell_index or hit["collider"] != world:
		_fail("the ray hit instance %s of %s, expected cell %d of chunk %s" % [instance, hit["collider"], cell_index, focus])
		return false
	var top: float = world.cell_position(focus, cell_index).y + CELL_SIZE / 2.0
	if absf(hit["position"].y - top) > 0.001:
		_fail("the ray hit at height %.3f, the top of the column is %.3f" % [hit["position"].y, top])
		return false
	print("verify: colliders on the %d chunks near the focus, and a ray down hits the cell below it" % bodies.size())
	return true

## Everything the extension promised a game, checked against what it can see from GDScript.
func _check() -> void:
	done = true
	var seconds := (Time.get_ticks_usec() - walk_started_usec) / 1e6
	var slow_frames := 0
	for period in period_ms:
		if period > 1000.0 / FPS + 2.0:
			slow_frames += 1
	print("verify: ran to chunk %d and back at %.1f units/s in %.2f s over %d frames" % [WALK_TO, PACE, seconds, period_ms.size()])
	var stats: Dictionary = world.stats()
	print("verify: Godot's slowest frame %.3f ms; the node's own process per frame p50 %.3f ms, p99 %.3f ms, max %.3f ms; frame period p50 %.2f ms, p99 %.2f ms, max %.2f ms, %d frames over %.1f ms" % [
		slowest_process_ms, stats["process_ms_median"], stats["process_ms_p99"], stats["process_ms_max"],
		_quantile(period_ms, 0.5), _quantile(period_ms, 0.99), _quantile(period_ms, 1.0),
		slow_frames, 1000.0 / FPS + 2.0])
	print("verify: %d chunks generated, %d dropped, %d late frames, stats %s" % [updated.size(), evicted.size(), late_frames, stats])

	# The point of the worker thread: building the device, compiling kernels and dispatching all
	# happen off Godot's own thread, so the main loop keeps its frame time while a world appears.
	if slowest_process_ms > PROCESS_MAX_MS:
		_fail("Godot's thread spent %.1f ms on its slowest frame; the node's slowest took %.1f ms: %d events and their signals %.1f ms, %d chunks' colliders %.1f ms, navigation %.1f ms" % [
			slowest_process_ms, stats["slowest_frame_ms"], stats["slowest_frame_events"], stats["slowest_frame_signals_ms"],
			stats["slowest_frame_colliders"], stats["slowest_frame_colliders_ms"], stats["slowest_frame_navigation_ms"]])
		return
	if stats["process_ms_p99"] > NODE_P99_MS:
		_fail("the node's own process took %.2f ms at the 99th percentile" % stats["process_ms_p99"])
		return
	# And generation keeps ahead of a running player, which it cannot fake by making them wait.
	if late_frames > 0:
		_fail("%d frames had a chunk beside the focus without tiles, first %s" % [late_frames, first_late])
		return
	if not failed.is_empty():
		_fail("chunks could not be placed: %s" % failed)
		return
	if evicted.is_empty():
		_fail("nothing was dropped although the focus walked away from it")
		return
	for chunk: Vector3i in evicted:
		if _has_chunk(chunk) and not updated.has(chunk):
			_fail("chunk %s was reported as dropped but its tiles are still there" % chunk)
			return

	var generated: Array = world.generated_chunks()
	if generated.size() < (2 * READY_RADIUS + 1) * CHUNKS_Y:
		_fail("only %d chunks around the focus at the end: %s" % [generated.size(), generated])
		return

	var names := ["water", "sand", "grass", "forest"]
	var world_tiles := {}
	for chunk: Vector3i in generated:
		var tiles: PackedInt32Array = world.tiles_at(chunk)
		if tiles.size() != CELLS * CELLS * CELLS:
			_fail("chunk %s has %d tiles, expected %d" % [chunk, tiles.size(), CELLS * CELLS * CELLS])
			return
		for z in CELLS:
			for y in CELLS:
				for x in CELLS:
					var tile := tiles[(z * CELLS + y) * CELLS + x]
					if tile < 0 or tile >= names.size():
						_fail("chunk %s cell (%d, %d, %d) holds tile %d" % [chunk, x, y, z, tile])
						return
					# The lattice's y is the second axis of a chunk coordinate, as in Rust.
					world_tiles[Vector3i(chunk.x * CELLS + x, chunk.y * CELLS + y, z)] = tile

	# The rule set only lets a material meet itself vertically, so every column is one material.
	for cell: Vector3i in world_tiles:
		if cell.z != 0:
			continue
		var material: int = world_tiles[cell]
		for z in range(1, CELLS):
			var above: int = world_tiles[Vector3i(cell.x, cell.y, z)]
			if above != material:
				_fail("column (%d, %d) holds %s at z=0 and %s at z=%d" % [cell.x, cell.y, names[material], names[above], z])
				return

	# And horizontally, a material only meets itself or the next band along, which is the rule the
	# generated world has to obey across chunk seams as well as inside a chunk.
	for cell: Vector3i in world_tiles:
		var material: int = world_tiles[cell]
		for step: Vector3i in [Vector3i(1, 0, 0), Vector3i(0, 1, 0)]:
			var beside: Vector3i = cell + step
			if not world_tiles.has(beside):
				continue
			var other: int = world_tiles[beside]
			if absi(other - material) > 1:
				_fail("%s at %s sits beside %s" % [names[material], cell, names[other]])
				return

	# The prior the scene set: the ground layer had no water, so nothing does, and the world's
	# edges had no forest.
	for cell: Vector3i in world_tiles:
		var material: int = world_tiles[cell]
		if material == WATER:
			_fail("water at %s although the ground layer forbade it" % cell)
			return
		var on_edge: bool = cell.x == 0 or cell.x == CHUNKS_X * CELLS - 1 or cell.y == 0 or cell.y == CHUNKS_Y * CELLS - 1
		if on_edge and material == FOREST:
			_fail("forest at %s although the world's faces forbade it" % cell)
			return

	# A chunk the focus left and came back to is generated again from its coordinate alone, so it
	# has to hold what it held the first time.
	var again: PackedInt32Array = world.tiles_at(Vector3i(WALK_FROM, 1, 0))
	if first_tiles.is_empty():
		_fail("the starting chunk was never read")
		return
	if again != first_tiles:
		_fail("chunk (%d, 1, 0) came back different after being dropped" % WALK_FROM)
		return

	print("verify: %d cells, every column one material, every neighbour legal, the prior obeyed, and the chunk walked back to is unchanged" % world_tiles.size())
	if not _check_colliders():
		return
	# The chunks near the focus bake their navigation meshes after it arrives; the check waits.
	navigation_wait_usec = Time.get_ticks_usec()

func _fail(message: String) -> void:
	printerr("verify: " + message)
	quit(1)
