## Generates the city and checks its occluders and the ones the node keeps near the player.
##
## Run by `../verify.sh` after `verify_names.gd`. The city's module set says which modules fill
## their cell (`solid`, its closed buildings). A chunk's occluders hold every solid cell once and no
## other; within `occluder_radius` the node gives each chunk with a solid cell one
## `OccluderInstance3D` whose `ArrayOccluder3D` holds those boxes, eight corners each; once the player
## has moved far away, none remain where it was.
extends SceneTree

const CELLS := 8
const CELL_SIZE := 2.0
const RADIUS := 1
const TIMEOUT_S := 120.0
const SOLID := ["building_base", "building_door", "building_floor"]

var world: Node
var started_usec := 0
var phase := "arrive"
var waited := 0

func _initialize() -> void:
	world = ClassDB.instantiate("WaveForgeWorld")
	world.seed = 11
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = Vector3(CELL_SIZE, CELL_SIZE, CELL_SIZE)
	world.view_radius = RADIUS + 1
	world.collider_radius = -1
	world.occluder_radius = RADIUS
	if not world.load_rules(FileAccess.get_file_as_string("res://city.ron")):
		_fail("res://city.ron could not be loaded")
		return
	var layers: Array[PackedInt32Array] = [world.tiles_tagged("street_level")]
	for storey in CELLS - 2:
		layers.append(PackedInt32Array())
	layers.append(world.tiles_named("air"))
	world.set_layer_tiles(layers)
	world.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	root.add_child(world)
	if not world.start():
		_fail("generation did not start")
		return
	world.follow(Vector3.ZERO)
	started_usec = Time.get_ticks_usec()

## The chunks within the occluder radius of the origin's.
func _near() -> Array[Vector3i]:
	var out: Array[Vector3i] = []
	for y in range(-RADIUS, RADIUS + 1):
		for x in range(-RADIUS, RADIUS + 1):
			out.append(Vector3i(x, y, 0))
	return out

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("timed out in %s" % phase)
		return true
	match phase:
		"arrive":
			for chunk in _near():
				if world.tiles_at(chunk).is_empty():
					return false
			# A few frames for the node to give the chunks their occluders, a few a frame.
			waited += 1
			if waited < 20:
				return false
			if not (_check_boxes() and _check_nodes()):
				return true
			world.follow(Vector3(1000, 0, 1000))
			phase = "leave"
			waited = 0
		"leave":
			waited += 1
			if waited < 10:
				return false
			return _check_left()
	return false

## Each chunk's occluders against its tiles.
func _check_boxes() -> bool:
	var boxes := 0
	for chunk in _near():
		var tiles: PackedInt32Array = world.tiles_at(chunk)
		var occluders: Array = world.occluders(chunk)
		boxes += occluders.size()
		for cell in tiles.size():
			var name: String = world.tile_name(tiles[cell])
			var centre: Vector3 = world.cell_position(chunk, cell)
			var holding := 0
			for box: AABB in occluders:
				if box.has_point(centre):
					holding += 1
			if holding != (1 if SOLID.has(name) else 0):
				_fail("%s cell %d, %s, is in %d occluders" % [chunk, cell, name, holding])
				return false
	if boxes == 0:
		_fail("the city near the origin has no solid cell, so nothing was checked")
		return false
	print("verify_occlusion: %d boxes hold every solid cell once and no other" % boxes)
	return true

## The node's occluders for the chunks within the radius.
func _check_nodes() -> bool:
	var expected := 0
	var corners := 0
	for chunk in _near():
		var count: int = world.occluders(chunk).size()
		if count > 0:
			expected += 1
			corners += count * 8
	var instances := world.get_children().filter(func(node: Node) -> bool: return node is OccluderInstance3D)
	var held := 0
	for instance: OccluderInstance3D in instances:
		held += (instance.occluder as ArrayOccluder3D).vertices.size()
	if instances.size() != expected or held != corners:
		_fail("%d occluders with %d corners for %d chunks with %d corners" % [instances.size(), held, expected, corners])
		return false
	print("verify_occlusion: %d chunks have an occluder of their boxes, %d corners in all" % [instances.size(), held])
	return true

## Once the player has left, no occluder is left where it was.
func _check_left() -> bool:
	var reach := (RADIUS + 1) * CELLS * CELL_SIZE
	var behind := world.get_children().filter(func(node: Node) -> bool:
		if not node is OccluderInstance3D or node.is_queued_for_deletion():
			return false
		var at: Vector3 = (node.occluder as ArrayOccluder3D).vertices[0]
		return absf(at.x) < reach and absf(at.z) < reach)
	if not behind.is_empty():
		_fail("after leaving, %d occluders remain where the player was" % behind.size())
		return true
	print("verify_occlusion: leaving frees the occluders left behind")
	quit(0)
	return true

func _fail(message: String) -> void:
	printerr("verify_occlusion: " + message)
	quit(1)
