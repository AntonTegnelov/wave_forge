## Plays the terrain's sounds and checks each plays where and as loud as its emitter says.
##
## Run by `../verify.sh` after `verify_water.gd`. The water checks' valley with the default
## ambience added (running water along its rivers, lapping at its lake's shore) and its river
## given as a table row: within `audio_radius` of the followed chunk, every emitter of the chunks
## with ground has a player of the stream `sounds` maps its key to, playing at the emitter's
## position and volume, and the river's emitters stand along the river.
extends SceneTree

const CELLS := 8
const CELL := Vector3(2, 1, 2)
const TIMEOUT_S := 30.0
const RADIUS := 2
const AMBIENCE := """ambience: [
		(key: "water_river", kind: River(curves: "rivers", height: "terrain")),
		(key: "water_lake", kind: Lake(lakes: "lakes", height: "terrain")),
	],
	stages: ["""

var world: Node
var started_usec := 0
var settled := 0

func _initialize() -> void:
	var text := FileAccess.get_file_as_string("res://water.world.ron").replace("stages: [", AMBIENCE)
	var pack := FileAccess.open("user://ambience.world.ron", FileAccess.WRITE)
	pack.store_string(text)
	pack.close()
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "user://ambience.world.ron"
	world.targets = PackedStringArray(["ground", "rivers", "terrain", "lakes"])
	world.seed = 9
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = CELL
	world.view_radius = 3
	world.collider_radius = -1
	world.ground_stage = "ground"
	world.audio_radius = RADIUS
	var sounds: Dictionary[StringName, AudioStream] = {
		&"water_river": AudioStreamGenerator.new(),
		&"water_lake": AudioStreamGenerator.new(),
	}
	world.sounds = sounds
	root.add_child(world)
	if not world.start():
		_fail("the stages did not start")
		return
	var river := {"id": 1, "x0": 2.0, "y0": 32.0, "x1": 62.0, "y1": 32.0, "width": 1.5}
	if not world.give_table("rivers", [river]):
		_fail("the river was not taken")
		return
	# Over the river, near the lake's hollow.
	world.follow(Vector3(36 * CELL.x, 0, 30 * CELL.z))
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("the sounds had not settled")
		return true
	if world.stats()["pending_grounds"] > 0:
		return false
	var focus := Vector3i(4, 3, 0)
	var expected := []
	for chunk: Vector3i in world.ground_chunks():
		if maxi(absi(chunk.x - focus.x), absi(chunk.y - focus.y)) <= RADIUS:
			expected.append_array(world.ambience(chunk))
	var players := []
	for child in world.get_children():
		if child is AudioStreamPlayer3D and child.playing:
			players.append(child)
	# A few frames with every emitter's player playing, so none are still to come.
	if players.size() != expected.size() or expected.is_empty():
		settled = 0
		return false
	settled += 1
	if settled < 10:
		return false
	var rivers := 0
	for emitter: Dictionary in expected:
		var found := false
		for player: AudioStreamPlayer3D in players:
			if player.position.is_equal_approx(emitter["position"]):
				found = true
				var db := 20.0 * log(maxf(emitter["volume"], 1e-4)) / log(10.0)
				if absf(player.volume_db - db) > 1e-3:
					_fail("a player at %s plays at %f dB, its emitter at %f" % [player.position, player.volume_db, db])
					return true
		if not found:
			_fail("no player at %s for %s" % [emitter["position"], emitter["key"]])
			return true
		if emitter["key"] == "water_river":
			rivers += 1
			if absf(emitter["position"].z - 32 * CELL.z) > 1e-3:
				_fail("a river sound at %s, off the river" % emitter["position"])
				return true
	if rivers == 0:
		_fail("no river sounds")
		return true
	print("verify_ambience: %d sounds play within %d chunks, %d along the river, each at its emitter's position and volume" % [expected.size(), RADIUS, rivers])
	quit(0)
	return true

func _fail(message: String) -> void:
	printerr("verify_ambience: FAILED: %s" % message)
	quit(1)
