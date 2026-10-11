## Runs a world into a directory and quits, for `../verify.sh` to check Godot exits cleanly.
##
## Run by `../verify.sh` after `verify_continent.gd`. `continent.tscn`'s node runs its far height
## alone, the whole continent at a scale of 4, into an emptied directory under the user directory,
## then quits once the run has finished. Its threads have used a GPU device; the process must still
## exit with status 0 rather than fault as they end, which `verify.sh` checks since it stops at the
## first command that fails.
extends SceneTree

const Continent := preload("res://continent.gd")
const DIRECTORY := "user://verify_run_exit"
const TIMEOUT_S := 300.0

var started_usec := 0

func _initialize() -> void:
	_empty(DIRECTORY)
	var scene: Node = (load("res://continent.tscn") as PackedScene).instantiate()
	var stages: Node = scene.get_node("Continent")
	stages.targets = PackedStringArray(["far_height"])
	stages.far_ground_stage = ""
	stages.far_ground_material_stage = ""
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
	if not stages.run_world(ProjectSettings.globalize_path(DIRECTORY)):
		_fail("the world run did not start")
		return
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	if started_usec > 0 and (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("the world run had not finished after %.0f s" % TIMEOUT_S)
		return true
	return false

func _finished(done: int, total: int) -> void:
	if done != total:
		_fail("the world run stopped at %d of %d chunks" % [done, total])
		return
	print("verify_run_exit: ran %d chunks of the continent's far height; quitting" % total)
	quit(0)

## Removes everything under `path` and makes it again, empty.
func _empty(path: String) -> void:
	var absolute := ProjectSettings.globalize_path(path)
	if DirAccess.dir_exists_absolute(absolute):
		_remove(absolute)
	DirAccess.make_dir_recursive_absolute(absolute)

func _remove(absolute: String) -> void:
	for directory in DirAccess.get_directories_at(absolute):
		_remove(absolute.path_join(directory))
	for file in DirAccess.get_files_at(absolute):
		DirAccess.remove_absolute(absolute.path_join(file))
	DirAccess.remove_absolute(absolute)

func _fail(message: String) -> void:
	printerr("verify_run_exit: FAILED: %s" % message)
	quit(1)
