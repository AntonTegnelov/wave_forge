## A frozen stage's chunks leave memory for `frozen_directory` and come back from it unchanged.
##
## Run by `../verify.sh` after `verify_edits.gd`, on its pack, whose trees are frozen. The node keeps
## frozen chunks in a fresh directory. Walking away drops the trees at the origin, which are then a
## file there, and walking back gives the trees the origin had before.
extends SceneTree

const CELLS := 8
const TIMEOUT_S := 30.0
const ORIGIN := Vector3i.ZERO
const DIRECTORY := "user://wave_forge_verify_frozen"

var world: Node
var ready := {}
var dropped := {}
var started_usec := 0
var phase := "arrive"
var first := ""

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
				quit(0)
				return true
	return false

func _fail(message: String) -> void:
	printerr("verify_frozen: " + message)
	quit(1)
