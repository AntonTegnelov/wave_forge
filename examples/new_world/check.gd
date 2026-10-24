## Checks the new world headless: `godot --headless --path . --script check.gd`. A world in a folder
## of its own: the game gives the continent its history and saves it, and the same seed gives the
## same history again; the run reports its progress by stage, and the button stops it; the game
## started again takes the saved history and resumes the run, which the button stops again. A whole
## run of the continent takes minutes, so finishing it and playing what it wrote is for the desktop
## measurements (docs/guides/desktop-measurements.md).
extends SceneTree

const FOLDER := "user://new_world_check"
const HISTORY := preload("res://history.gd")
const TIMEOUT_S := 120.0

var main: Node
var step := "first"
var started_usec := 0
## The node whose progress the check listens to, and whether it has reported what its stages have
## generated.
var listened: Node
var reported := false
var saved := ""
## The chunks the run had done, as last reported.
var latest_done := 0
## The products the stages had generated when each run first reported a chunk done, the first
## run's and the one started again's: a run reports its chunks once their block is done, so the
## first run's holds the whole of its first block, and a run that resumes generates only what the
## first did not keep of it.
var first_products := -1
var resumed_products := -1
## How far the first run had got when it stopped.
var stopped_done := 0

func _initialize() -> void:
	_remove(ProjectSettings.globalize_path(FOLDER))
	_start()

func _start() -> void:
	main = (load("res://main.tscn") as PackedScene).instantiate()
	main.folder = FOLDER
	root.add_child(main)
	reported = false
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		return _fail("the %s run was at %s after %d s" % [step, main.phase, TIMEOUT_S])
	if main.stages != null and main.stages != listened:
		listened = main.stages
		listened.world_run_progress.connect(func(done: int, _total: int, stages: Dictionary) -> void:
			latest_done = done
			if done > 0 and step == "first" and first_products < 0:
				first_products = _products(stages)
			elif done > 0 and step == "again" and resumed_products < 0:
				resumed_products = _products(stages)
			reported = reported or not stages.is_empty())
	match step:
		"first":
			# Stopped once a chunk is done, so the run started again has something to resume from.
			if main.phase != "generating" or not reported or latest_done == 0 or first_products <= 0:
				return false
			var history: Array = main.history
			if history.is_empty() or main.stages.table_rows("settlements").size() != history.size():
				return _fail("the continent holds %d settlements of a history of %d" % [main.stages.table_rows("settlements").size(), history.size()])
			if JSON.stringify(HISTORY.new().run(main.stages, main.stages.seed)) != JSON.stringify(history):
				return _fail("the same seed gave another history")
			saved = FileAccess.get_file_as_string(FOLDER.path_join("history.json"))
			if saved.is_empty():
				return _fail("the history was not saved")
			if main.costs.text.is_empty():
				return _fail("the screen shows no stage's cost")
			main.button.pressed.emit()
			step = "stopping"
		"stopping":
			if main.phase != "stopped":
				return false
			if main.button.text != "Resume":
				return _fail("the stopped run's button says %s" % main.button.text)
			stopped_done = latest_done
			main.free()
			step = "again"
			_start()
		"again":
			if main.phase != "generating" or not reported or resumed_products < 0:
				return false
			if JSON.stringify(main.history) != JSON.stringify(JSON.parse_string(saved)):
				return _fail("the game started again took another history than the saved one")
			if resumed_products >= first_products:
				return _fail("the run started again had generated %d products by its first chunk, as many as the first run's %d: it did not resume" % [resumed_products, first_products])
			main.button.pressed.emit()
			step = "stopping again"
		"stopping again":
			if main.phase != "stopped":
				return false
			print("check: the continent took a history of %d settlements, saved and the same again for its seed; the run reported its stages and stopped at %d chunks, and started again with the saved history it resumed, having generated %d products by its first chunk where the first run generated %d, and stopped" % [main.history.size(), stopped_done, resumed_products, first_products])
			quit(0)
			return true
	return false

## How many products the stages of a progress report have generated.
func _products(stages: Dictionary) -> int:
	var products := 0
	for cost: Dictionary in stages.values():
		products += cost["products"]
	return products

func _remove(path: String) -> void:
	var folder := DirAccess.open(path)
	if folder == null:
		return
	for name in folder.get_directories():
		_remove(path.path_join(name))
	for name in folder.get_files():
		folder.remove(name)
	DirAccess.remove_absolute(path)

func _fail(message: String) -> bool:
	printerr("check: " + message)
	quit(1)
	return true
