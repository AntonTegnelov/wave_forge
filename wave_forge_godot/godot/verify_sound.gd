## Generates the city and checks its region tags and the sound the node gives them.
##
## Run by `../verify.sh` after `verify_ground.gd`. The city's module set says what walkers stand on
## (`surface`), which modules are inside (`indoor`, its buildings) and where fountains play
## (`sounds`). Every cell's surface is its module's; a chunk's interiors hold every building cell
## once and no other; its emitters sit in its fountains' cells. Within `audio_radius` the node gives
## each interior an `Area3D` reverbing on `interior_reverb_bus` and playing the sounds inside it on
## `interior_audio_bus`, and each emitter a playing
## `AudioStreamPlayer3D`, and once the player has moved away, it frees the areas and stops the
## players.
extends SceneTree

const CELLS := 8
const CELL_SIZE := 2.0
const RADIUS := 2
const TIMEOUT_S := 120.0
const BUS := &"Rooms"
const INDOORS := &"Indoors"
const SURFACES := {
	"grass": "grass", "plaza": "stone", "plaza_lamp": "stone", "pillar_base": "stone",
	"plaza_fountain": "stone", "road_straight": "asphalt", "road_corner": "asphalt",
	"road_t": "asphalt", "road_cross": "asphalt", "road_end": "asphalt", "building_base": "wood",
	"building_door": "wood", "building_arcade": "wood", "building_floor": "wood",
	"building_balcony": "wood", "building_passage": "wood", "roof_flat": "tiles",
	"roof_flat_edge": "tiles", "roof_flat_corner": "tiles", "roof_flat_strip": "tiles",
	"roof_flat_end": "tiles", "walkway_straight": "wood", "walkway_corner": "wood",
	"walkway_on_pillar": "wood", "stair_head": "wood", "stair": "stone", "stair_roof": "stone",
	"stair_wall_street": "stone", "stair_wall": "stone",
}

var world: Node
var started_usec := 0
var phase := "arrive"
var waited := 0

func _initialize() -> void:
	for bus in [BUS, INDOORS]:
		AudioServer.add_bus()
		AudioServer.set_bus_name(AudioServer.bus_count - 1, bus)
	world = ClassDB.instantiate("WaveForgeWorld")
	world.seed = 11
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = Vector3(CELL_SIZE, CELL_SIZE, CELL_SIZE)
	world.view_radius = RADIUS
	world.collider_radius = -1
	world.audio_radius = RADIUS
	# A second of silence, looped, as an ambience plays until it is stopped.
	var fountain := AudioStreamWAV.new()
	fountain.format = AudioStreamWAV.FORMAT_16_BITS
	fountain.mix_rate = 44100
	var silence := PackedByteArray()
	silence.resize(44100 * 2)
	fountain.data = silence
	fountain.loop_mode = AudioStreamWAV.LOOP_FORWARD
	fountain.loop_end = 44100
	world.sounds = {"fountain": fountain}
	world.interior_reverb_bus = BUS
	world.interior_audio_bus = INDOORS
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

## The chunks within the audio radius of the origin's.
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
			# A few frames for the node to give the chunks their sound, a few a frame.
			waited += 1
			if waited < 20:
				return false
			if not (_check_tags() and _check_nodes()):
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

## Each cell's surface, and each chunk's interiors and emitters, against its tiles.
func _check_tags() -> bool:
	var fountains := 0
	var interiors := 0
	for chunk in _near():
		var tiles: PackedInt32Array = world.tiles_at(chunk)
		var tags: Dictionary = world.region_tags(chunk)
		var boxes: Array = tags["interiors"]
		interiors += boxes.size()
		var expected_emitters := 0
		for cell in tiles.size():
			var name: String = world.tile_name(tiles[cell])
			var centre: Vector3 = world.cell_position(chunk, cell)
			var surface: String = world.surface_at(centre)
			if surface != SURFACES.get(name, ""):
				_fail("%s cell %d, %s, stands on %s" % [chunk, cell, name, surface])
				return false
			var holding := 0
			for box: AABB in boxes:
				if box.has_point(centre):
					holding += 1
			if holding != (1 if name.begins_with("building_") else 0):
				_fail("%s cell %d, %s, is in %d interiors" % [chunk, cell, name, holding])
				return false
			if name == "plaza_fountain":
				expected_emitters += 1
				var cell_box := AABB(centre - Vector3.ONE * CELL_SIZE / 2, Vector3.ONE * CELL_SIZE)
				var found := false
				for emitter: Dictionary in tags["emitters"]:
					found = found or (emitter["key"] == "fountain" and cell_box.has_point(emitter["position"]))
				if not found:
					_fail("the fountain in %s cell %d plays nowhere in its cell" % [chunk, cell])
					return false
		if tags["emitters"].size() != expected_emitters:
			_fail("%s has %d emitters for %d fountains" % [chunk, tags["emitters"].size(), expected_emitters])
			return false
		fountains += expected_emitters
	if fountains == 0 or interiors == 0:
		_fail("the city near the origin has %d fountains and %d interiors, so nothing was checked" % [fountains, interiors])
		return false
	print("verify_sound: every cell stands on its module's surface; %d interiors hold every building cell once; %d fountains play in their cells" % [interiors, fountains])
	return true

## The node's areas and players for the chunks within the radius.
func _check_nodes() -> bool:
	var interiors := 0
	var emitters := 0
	for chunk in _near():
		var tags: Dictionary = world.region_tags(chunk)
		interiors += tags["interiors"].size()
		emitters += tags["emitters"].size()
	var areas := world.get_children().filter(func(node: Node) -> bool: return node is Area3D)
	var playing := world.get_children().filter(func(node: Node) -> bool: return node is AudioStreamPlayer3D and node.playing)
	if areas.size() != interiors or playing.size() != emitters:
		_fail("%d areas for %d interiors, %d players playing for %d emitters" % [areas.size(), interiors, playing.size(), emitters])
		return false
	for area: Area3D in areas:
		if not area.reverb_bus_enabled or area.reverb_bus_name != BUS:
			_fail("an interior reverbs on %s" % area.reverb_bus_name)
			return false
		if not area.audio_bus_override or area.audio_bus_name != INDOORS:
			_fail("the sounds in an interior play on %s" % area.audio_bus_name)
			return false
	print("verify_sound: %d interiors reverb on %s and play on %s, %d players play the fountains" % [areas.size(), BUS, INDOORS, playing.size()])
	return true

## Once the player has left, no area is left and no player plays where it was; the chunks it went to
## have sound of their own.
func _check_left() -> bool:
	var reach := (RADIUS + 1) * CELLS * CELL_SIZE
	var behind := func(node: Node) -> bool:
		var at: Vector3 = node.position
		return absf(at.x) < reach and absf(at.z) < reach
	var areas := world.get_children().filter(func(node: Node) -> bool: return node is Area3D and not node.is_queued_for_deletion() and behind.call(node))
	var playing := world.get_children().filter(func(node: Node) -> bool: return node is AudioStreamPlayer3D and node.playing and behind.call(node))
	if not areas.is_empty() or not playing.is_empty():
		_fail("after leaving, %d areas and %d playing players remain where the player was" % [areas.size(), playing.size()])
		return true
	print("verify_sound: leaving frees the interiors and stops the players left behind")
	quit(0)
	return true

func _fail(message: String) -> void:
	printerr("verify_sound: " + message)
	quit(1)
