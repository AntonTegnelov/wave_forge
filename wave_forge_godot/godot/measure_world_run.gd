## The maximal preset's continent baked ahead of time into a directory, as the editor dock's World
## run bakes it, timed for M1 on a desktop (docs/guides/desktop-measurements.md). Run headless:
##     godot --headless --path . --script measure_world_run.gd -- --directory <dir> --out results.txt
## `continent.tscn`'s node, given the continent's history, runs its whole world of every target into
## `--directory`, resuming from what the directory holds. It prints one line of `key=value` fields
## (the seconds the run took, the chunks done and in the world, and the most memory held), appended to
## `--out` as well when it is given, then quits; `measure_playback.gd` plays the directory.
extends SceneTree

const Continent := preload("res://continent.gd")
const PeakMemory := preload("res://peak_memory.gd")

var directory := ""
var out := ""
var stages: Node
var started_usec := 0

func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	var i := 0
	while i < args.size():
		if i + 1 >= args.size():
			_fail("%s needs a value" % args[i])
			return
		match args[i]:
			"--directory": directory = args[i + 1]
			"--out": out = args[i + 1]
			_:
				_fail("unknown option %s" % args[i])
				return
		i += 2
	if directory.is_empty():
		_fail("--directory names where the world goes")
		return
	var scene: Node = (load("res://continent.tscn") as PackedScene).instantiate()
	stages = scene.get_node("Continent")
	# The run generates every chunk of the world whatever the view, so the node draws nothing
	# around the player meanwhile.
	stages.view_radius = 0
	stages.collider_radius = -1
	stages.follow_camera = false
	root.add_child(scene)
	if not stages.start():
		_fail("the continent did not start")
		return
	if not Continent.give_history(stages):
		_fail("the continent did not take its history")
		return
	stages.world_run_finished.connect(_finished)
	if not stages.run_world(directory):
		_fail("the world run did not start")
		return
	started_usec = Time.get_ticks_usec()

func _finished(done: int, total: int) -> void:
	var seconds := (Time.get_ticks_usec() - started_usec) / 1e6
	var peak_mb := PeakMemory.peak_resident_mb()
	if peak_mb < 0.0:
		_fail("the system did not say how much memory the process held")
		return
	var line := "measure_world_run seconds=%.1f done=%d total=%d peak_mb=%.0f" % [seconds, done, total, peak_mb]
	print(line)
	if not out.is_empty():
		var file := FileAccess.open(out, FileAccess.READ_WRITE) if FileAccess.file_exists(out) else FileAccess.open(out, FileAccess.WRITE)
		if file == null:
			_fail("%s cannot be written" % out)
			return
		file.seek_end()
		file.store_line(line)
	if done != total:
		_fail("the world run stopped at %d of %d chunks" % [done, total])
		return
	quit(0)

func _fail(message: String) -> void:
	printerr("measure_world_run: " + message)
	quit(1)
