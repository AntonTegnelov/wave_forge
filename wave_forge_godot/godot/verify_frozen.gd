## Frozen chunks leave memory for `frozen_directory` and come back from it unchanged.
##
## Run by `../verify.sh` after `verify_edits.gd`. First a WaveForgeStages node on that check's pack,
## whose trees are frozen, keeps its frozen chunks in a fresh directory: walking away drops the trees
## at the origin, which are then a file there, and walking back gives the trees the origin had.
## Then a frozen WaveForgeWorld of the city does the same with the origin's tiles.
extends SceneTree

const CELLS := 8
const TIMEOUT_S := 120.0
const ORIGIN := Vector3i.ZERO
const DIRECTORY := "user://wave_forge_verify_frozen"

var world: Node
var ready := {}
var dropped := {}
var started_usec := 0
var phase := "arrive"
var first := ""
var city: Node
var updated := {}
var evicted := {}
var first_tiles := PackedInt32Array()
var last_update_usec := 0

func _initialize() -> void:
	_remove(ProjectSettings.globalize_path(DIRECTORY))
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://edits.world.ron"
	world.targets = PackedStringArray(["ground", "trees"])
	world.seed = 5
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.view_radius = 1
	world.collider_radius = -1
	world.frozen_directory = DIRECTORY
	root.add_child(world)
	world.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	world.stage_ready.connect(func(stage: String, chunk: Vector3i) -> void: ready[[stage, chunk]] = true)
	world.stage_dropped.connect(func(stage: String, chunk: Vector3i) -> void:
		dropped[[stage, chunk]] = true
		ready.erase([stage, chunk]))
	if not world.start():
		_fail("the stages did not start")
		return
	world.follow(Vector3(4, 0, 4))
	started_usec = Time.get_ticks_usec()

## Deletes a directory and everything in it, if it is there.
func _remove(path: String) -> void:
	var directory := DirAccess.open(path)
	if directory == null:
		return
	for file in directory.get_files():
		DirAccess.remove_absolute(path.path_join(file))
	for inner in directory.get_directories():
		_remove(path.path_join(inner))
	DirAccess.remove_absolute(path)

func _trees() -> String:
	return str(world.point_sets("trees", ORIGIN))

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("the %s phase timed out" % phase)
		return true
	match phase:
		"arrive":
			if ready.has(["trees", ORIGIN]):
				first = _trees()
				if first == "[]":
					_fail("no trees at the origin to compare")
					return true
				world.follow(Vector3(2000, 0, 4))
				phase = "away"
		"away":
			if dropped.has(["trees", ORIGIN]):
				var file := ProjectSettings.globalize_path(DIRECTORY).path_join("trees").path_join("0_0_0")
				if not FileAccess.file_exists(file):
					_fail("the origin's trees were dropped but are not in %s" % file)
					return true
				print("verify_frozen: the origin's trees left memory for %s" % file)
				world.follow(Vector3(4, 0, 4))
				phase = "back"
		"back":
			if ready.has(["trees", ORIGIN]):
				if _trees() != first:
					_fail("the origin's trees came back different")
					return true
				print("verify_frozen: walking back gives the origin the trees it had")
				_remove(ProjectSettings.globalize_path(DIRECTORY))
				world.queue_free()
				_city()
				phase = "city_arrive"
				started_usec = Time.get_ticks_usec()
		"city_arrive":
			# Repairs can rewrite a chunk after it arrives, so the view counts as settled once
			# nothing has changed for a second.
			if updated.has(ORIGIN) and Time.get_ticks_usec() - last_update_usec > 1000000:
				first_tiles = city.tiles_at(ORIGIN)
				city.follow(Vector3(2000, 0, 8))
				phase = "city_away"
		"city_away":
			if evicted.has(ORIGIN):
				var file := ProjectSettings.globalize_path(DIRECTORY).path_join("tiles").path_join("0_0_0")
				if not FileAccess.file_exists(file):
					_fail("the origin's tiles were evicted but are not in %s" % file)
					return true
				print("verify_frozen: the city's origin left memory for %s" % file)
				updated.erase(ORIGIN)
				city.follow(Vector3(8, 0, 8))
				phase = "city_back"
		"city_back":
			if updated.has(ORIGIN):
				if city.tiles_at(ORIGIN) != first_tiles:
					_fail("the city's origin came back different")
					return true
				var restored: int = city.stats()["restored"]
				if restored == 0:
					_fail("the city's origin was generated again, not restored")
					return true
				print("verify_frozen: walking back restores the city's origin as it was, %d chunks from the directory" % restored)
				_remove(ProjectSettings.globalize_path(DIRECTORY))
				quit(0)
				return true
	return false

## A frozen city around the origin, keeping what it evicts in the directory.
func _city() -> void:
	city = ClassDB.instantiate("WaveForgeWorld")
	city.seed = 11
	city.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	city.cell_size = Vector3(2, 2, 2)
	city.view_radius = 1
	city.evict_margin = 1
	city.collider_radius = -1
	city.frozen_directory = DIRECTORY
	if not city.load_rules(FileAccess.get_file_as_string("res://city.ron")):
		_fail("res://city.ron could not be loaded")
		return
	var layers: Array[PackedInt32Array] = [city.tiles_tagged("street_level")]
	for storey in CELLS - 2:
		layers.append(PackedInt32Array())
	layers.append(city.tiles_named("air"))
	city.set_layer_tiles(layers)
	root.add_child(city)
	city.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	city.chunk_updated.connect(func(chunk: Vector3i) -> void:
		updated[chunk] = true
		last_update_usec = Time.get_ticks_usec())
	city.chunk_evicted.connect(func(chunk: Vector3i) -> void: evicted[chunk] = true)
	if not city.start():
		_fail("the city did not start")
		return
	city.follow(Vector3(8, 0, 8))

func _fail(message: String) -> void:
	printerr("verify_frozen: " + message)
	quit(1)
