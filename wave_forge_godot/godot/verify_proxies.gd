## Generates the city and checks the far proxies the node keeps for its chunks.
##
## Run by `../verify.sh` after `verify_occlusion.gd`. With `proxy_distance` set, every generated
## chunk gets a proxy, and one with a coloured module an instance a game can take as its drawing's
## `visibility_parent`; a chunk dropped loses its proxy, and turning proxies off frees them all.
extends SceneTree

const CELLS := 8
const CELL_SIZE := 2.0
const TIMEOUT_S := 120.0

var world: Node
var started_usec := 0
var phase := "arrive"
var waited := 0
## How many generated chunks there were last frame, and for how many frames that has held.
var last_count := -1
var steady := 0

func _initialize() -> void:
	world = ClassDB.instantiate("WaveForgeWorld")
	world.seed = 11
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = Vector3(CELL_SIZE, CELL_SIZE, CELL_SIZE)
	world.view_radius = 2
	world.evict_margin = 1
	world.collider_radius = -1
	world.proxy_distance = 60.0
	world.proxy_colours = {"building_base": Color(0.8, 0.7, 0.6), "building_floor": Color(0.8, 0.7, 0.6), "grass": Color(0.3, 0.6, 0.3)}
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

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("timed out in %s" % phase)
		return true
	waited += 1
	match phase:
		"arrive":
			if not _settled():
				return false
			if not _check_every_chunk():
				return true
			world.follow(Vector3(CELLS * CELL_SIZE * 8, 0, 0))
			phase = "move"
			waited = 0
		"move":
			if not _settled():
				return false
			if not _check_every_chunk():
				return true
			world.proxy_distance = -1.0
			phase = "off"
			waited = 0
		"off":
			if waited < 3:
				return false
			if not world.proxy_chunks().is_empty():
				_fail("with proxies off, %d chunks keep theirs" % world.proxy_chunks().size())
				return true
			print("verify_proxies: turning proxies off frees them all")
			quit(0)
			return true
	return false

## Whether the generated chunks have held still for a second, long enough for proxies built a
## few a frame to catch up.
func _settled() -> bool:
	var count: int = world.generated_chunks().size()
	steady = steady + 1 if count == last_count and count > 0 else 0
	last_count = count
	if steady < 60:
		return false
	steady = 0
	last_count = -1
	return true

## Every generated chunk, and no other, has its proxy, and each with a coloured module an instance.
func _check_every_chunk() -> bool:
	var generated := {}
	for chunk: Vector3i in world.generated_chunks():
		generated[chunk] = true
	var proxied := {}
	for chunk: Vector3i in world.proxy_chunks():
		proxied[chunk] = true
	if generated.keys().size() != proxied.keys().size() or not generated.keys().all(func(chunk: Vector3i) -> bool: return proxied.has(chunk)):
		_fail("%d chunks generated, %d with proxies" % [generated.size(), proxied.size()])
		return false
	if generated.is_empty():
		_fail("no chunk was generated, so nothing was checked")
		return false
	var drawn := 0
	for chunk: Vector3i in generated:
		var tiles: PackedInt32Array = world.tiles_at(chunk)
		var coloured := false
		for tile in tiles:
			coloured = coloured or world.proxy_colours.has(world.tile_name(tile))
		if world.proxy_instance(chunk).is_valid() != coloured:
			_fail("%s has a coloured module: %s, a proxy instance: %s" % [chunk, coloured, world.proxy_instance(chunk).is_valid()])
			return false
		drawn += 1 if coloured else 0
	if drawn == 0:
		_fail("no chunk had a coloured module, so no proxy was drawn")
		return false
	print("verify_proxies: %d generated chunks have their proxy, %d of them drawn" % [generated.size(), drawn])
	return true

func _fail(message: String) -> void:
	printerr("verify_proxies: " + message)
	quit(1)
