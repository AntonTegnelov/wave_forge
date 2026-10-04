## A new world generated whole before play (story M2). The first time the game starts, it starts
## the maximal preset's continent (`continent.tscn`), simulates the game's history over its map
## (`history.gd`) and gives it to the stages, then runs the whole world into the game's folder,
## showing how far it has got and what each stage has cost, with a button to stop it. Stopped, or
## closed part way, the run resumes from what it wrote, by the button or the next time the game
## starts, with the history it saved. Once every chunk is there, the game plays the world from the
## folder, generating nothing, the walker standing at the first settlement.
extends Node3D

const CONTINENT := preload("res://continent.tscn")
const HISTORY := preload("res://history.gd")
const WALKER := preload("res://addons/wave_forge/walker.tscn")

## Where the game keeps the world: its history, its chunks, and a mark once they are all there.
@export var folder := "user://new_world"

## What the game is doing: `history`, `generating`, `stopped`, `failed` or `playing`.
var phase := ""
## The node generating or playing the world.
var stages: Node
## The settlements of the world's history.
var history: Array = []

@onready var status: Label = $Screen/Panel/Margin/Box/Status
@onready var progress: ProgressBar = $Screen/Panel/Margin/Box/Progress
@onready var costs: Label = $Screen/Panel/Margin/Box/Costs
@onready var button: Button = $Screen/Panel/Margin/Box/Button

func _ready() -> void:
	DirAccess.make_dir_recursive_absolute(folder)
	button.pressed.connect(_on_button)
	if FileAccess.file_exists(_path("done")):
		_play()
	else:
		_generate()

func _path(name: String) -> String:
	return folder.path_join(name)

## Starts the continent, gives it the saved history or a new one, and runs the world.
func _generate() -> void:
	phase = "history"
	var scene: Node = CONTINENT.instantiate()
	stages = scene.get_node("Continent")
	# The run generates every chunk whatever the view, so nothing is drawn around a player meanwhile.
	stages.view_radius = 0
	stages.collider_radius = -1
	stages.follow_camera = false
	add_child(scene)
	stages.world_run_progress.connect(_on_progress)
	stages.world_run_finished.connect(_on_finished)
	if not stages.start():
		_fail("the continent did not start")
		return
	history = _load_history()
	if history.is_empty():
		history = HISTORY.new().run(stages, stages.seed)
		if history.is_empty():
			_fail("the history founded no settlement, so there is no world to play")
			return
		FileAccess.open(_path("history.json"), FileAccess.WRITE).store_string(JSON.stringify(history))
	if not stages.give_table("settlements", history):
		_fail("the continent did not take the history")
		return
	_run()

func _load_history() -> Array:
	if not FileAccess.file_exists(_path("history.json")):
		return []
	# The file is the game's own, but a crash part way through writing it leaves something else.
	var parsed: Variant = JSON.parse_string(FileAccess.get_file_as_string(_path("history.json")))
	return parsed if parsed is Array else []

func _run() -> void:
	if not stages.run_world(_path("world")):
		_fail("the world run did not start")
		return
	phase = "generating"
	status.text = "Generating a world of %d settlements" % history.size()
	button.text = "Stop"

func _on_progress(done: int, total: int, stage_costs: Dictionary) -> void:
	progress.max_value = total
	progress.value = done
	status.text = "Generating a world of %d settlements: %d of %d chunks" % [history.size(), done, total]
	var names := stage_costs.keys()
	names.sort_custom(func(a: String, b: String) -> bool: return stage_costs[a]["ms"] > stage_costs[b]["ms"])
	var lines := PackedStringArray()
	for name: String in names.slice(0, 5):
		lines.append("%s: %.1f s" % [name, stage_costs[name]["ms"] / 1000.0])
	costs.text = "\n".join(lines)

func _on_finished(done: int, total: int) -> void:
	if total == 0:
		_fail("the world run failed; the log says why")
		return
	if done < total:
		phase = "stopped"
		status.text = "Stopped at %d of %d chunks. Resume now, or the next time the game starts." % [done, total]
		button.text = "Resume"
		return
	FileAccess.open(_path("done"), FileAccess.WRITE).store_string("")
	stages.get_parent().queue_free()
	_play()

func _on_button() -> void:
	match phase:
		"generating":
			stages.cancel_world_run()
		"stopped":
			_run()

## Plays the world the folder holds, the walker at the first settlement of its history.
func _play() -> void:
	phase = "playing"
	$Screen.hide()
	history = _load_history()
	if history.is_empty():
		_fail("the folder holds a world but no history; delete it for a new world")
		return
	var scene: Node = CONTINENT.instantiate()
	stages = scene.get_node("Continent")
	stages.play_directory = _path("world")
	add_child(scene)
	# A world the folder no longer holds whole fails as it plays: the screen says so.
	stages.generation_failed.connect(func(reason: String) -> void: _fail("the world could not be played: " + reason))
	if not stages.start():
		_fail("the world did not start")
		return
	var walker: Node3D = WALKER.instantiate()
	var first: Dictionary = history[0]
	walker.position = Vector3(first["x"] * stages.cell_size.x, 0.0, first["y"] * stages.cell_size.z)
	add_child(walker)

func _fail(message: String) -> void:
	phase = "failed"
	$Screen.show()
	status.text = message[0].to_upper() + message.substr(1)
	button.hide()
	push_error("new_world: " + message)
