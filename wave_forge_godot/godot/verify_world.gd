## Runs a finite world whole ahead of time, stopping and resuming, and plays it back.
##
## Run by `../verify.sh` after `verify_candidates.gd`. A node runs the pack's 30 chunks into a
## directory, reporting its progress; it is cancelled as soon as it starts, so it stops after its
## first chunk, and run again, which resumes and finishes. A second node given that directory as `play_directory` then plays the world
## around the player: its fields and trees equal those of a third node that generates as usual, its
## ground is built from them, and no stage of it ran.
extends SceneTree

const CELLS := 8
const TIMEOUT_S := 60.0
const DIRECTORY := "user://wave_forge_world_run"

var runner: Node
var player: Node
var generator: Node
var started_usec := 0
var phase := "run"
var progressed := 0
var finished := []

func _initialize() -> void:
	DirAccess.remove_absolute(DIRECTORY)
	_clear(ProjectSettings.globalize_path(DIRECTORY))
	runner = _node("")
	if not runner.start():
		_fail("the runner did not start")
		return
	runner.world_run_progress.connect(func(_done: int, _total: int) -> void: progressed += 1)
	runner.world_run_finished.connect(func(done: int, total: int) -> void: finished.append([done, total]))
	if not runner.run_world(DIRECTORY):
		_fail("the world run did not start")
		return
	# The run checks for a cancel after each chunk, so cancelling at once stops it after its first
	# whatever the machine's speed.
	runner.cancel_world_run()
	started_usec = Time.get_ticks_usec()

## Deletes what an earlier run left in `path`.
func _clear(path: String) -> void:
	if not DirAccess.dir_exists_absolute(path):
		return
	for directory in DirAccess.get_directories_at(path):
		_clear(path.path_join(directory))
	for file in DirAccess.get_files_at(path):
		DirAccess.remove_absolute(path.path_join(file))
	DirAccess.remove_absolute(path)

func _node(played: String) -> Node:
	var node: Node = ClassDB.instantiate("WaveForgeStages")
	node.pack_file = "res://world.world.ron"
	node.targets = PackedStringArray(["height", "surface", "trees"])
	node.seed = 23
	node.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	node.view_radius = 1
	node.collider_radius = -1
	node.ground_stage = "height"
	node.play_directory = played
	root.add_child(node)
	node.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	return node

## The chunks around the player, all inside the world.
func _area() -> Array[Vector3i]:
	var chunks: Array[Vector3i] = []
	for y in range(1, 4):
		for x in range(1, 4):
			chunks.append(Vector3i(x, y, 0))
	return chunks

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("the %s phase timed out" % phase)
		return true
	match phase:
		"run":
			if finished.is_empty():
				return false
			if finished[0][0] >= finished[0][1] or finished[0][1] != 30:
				_fail("the cancelled run ended at %s" % [finished[0]])
				return true
			phase = "resume"
			if not runner.run_world(DIRECTORY):
				_fail("the run did not start again")
				return true
			return false
		"resume":
			if finished.size() < 2:
				return false
			if finished[1] != [30, 30] or progressed < 30:
				_fail("the resumed run ended at %s after %d reports" % [finished[1], progressed])
				return true
			player = _node(DIRECTORY)
			generator = _node("")
			if not player.start() or not generator.start():
				_fail("the player or the generator did not start")
				return true
			player.follow(Vector3(2 * CELLS + 1, 0, 2 * CELLS + 1))
			generator.follow(Vector3(2 * CELLS + 1, 0, 2 * CELLS + 1))
			phase = "play"
			return false
		"play":
			for chunk in _area():
				if not player.ground_chunks().has(chunk) or not generator.ground_chunks().has(chunk):
					return false
				if player.point_sets("trees", chunk).is_empty() and generator.point_sets("trees", chunk).size() > 0:
					return false
			for chunk in _area():
				if player.field_values("height", chunk) != generator.field_values("height", chunk):
					_fail("the played height of %s is not the generated one" % chunk)
					return true
				if player.point_sets("trees", chunk) != generator.point_sets("trees", chunk):
					_fail("the played trees of %s are not the generated ones" % chunk)
					return true
			if not player.stats()["stages"].is_empty():
				_fail("a stage ran in the played world: %s" % player.stats()["stages"])
				return true
			print("verify_world: a run of 30 chunks cancelled after one and resumed finishes; the played world's fields and trees are the generated ones, its ground built from them, and no stage of it ran")
			quit(0)
			return true
	return false

func _fail(message: String) -> void:
	printerr("verify_world: " + message)
	quit(1)
