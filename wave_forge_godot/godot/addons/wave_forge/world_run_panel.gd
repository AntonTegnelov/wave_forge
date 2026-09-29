@tool
extends VBoxContainer
## The dock's world run (docs/reference/godot.md, "Editor"): runs the selected WaveForgeStages
## node's whole finite world ahead of time into a directory with the node's `run_world`, shows how
## far it has got by chunk and by stage, and cancels it; running again resumes from what the
## directory holds. The node does the work; the panel only renders its signals.

## Where a world is run to unless the directory is changed: inside the project, so the result ships
## with the game.
const DEFAULT_DIRECTORY := "res://wave_forge_world"

var directory_edit: LineEdit
var run_button: Button
var cancel_button: Button
var progress: ProgressBar
## What the run is doing: chunks done, and what each stage has generated, or how it ended.
var status: Label
## The node whose world is run, while one is selected.
var stages: Node

func _init() -> void:
	name = "World run"
	directory_edit = LineEdit.new()
	directory_edit.text = DEFAULT_DIRECTORY
	directory_edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	var row := HBoxContainer.new()
	var label := Label.new()
	label.text = "Directory"
	row.add_child(label)
	row.add_child(directory_edit)
	add_child(row)
	var buttons := HBoxContainer.new()
	run_button = Button.new()
	run_button.text = "Run world"
	run_button.pressed.connect(run)
	buttons.add_child(run_button)
	cancel_button = Button.new()
	cancel_button.text = "Cancel"
	cancel_button.disabled = true
	cancel_button.pressed.connect(cancel)
	buttons.add_child(cancel_button)
	add_child(buttons)
	progress = ProgressBar.new()
	add_child(progress)
	status = Label.new()
	status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	add_child(status)

## Shows the run of `node`, or of nothing when it is null.
func bind(node: Node) -> void:
	if stages != null and is_instance_valid(stages):
		stages.world_run_progress.disconnect(_on_progress)
		stages.world_run_finished.disconnect(_on_finished)
	stages = node
	if stages != null:
		stages.world_run_progress.connect(_on_progress)
		stages.world_run_finished.connect(_on_finished)

## Starts the node's world run into the directory; returns whether it started. It needs a node
## that has started, which in the editor is one whose `preview_in_editor` is on.
func run() -> bool:
	if stages == null or not is_instance_valid(stages):
		status.text = "Select a WaveForgeStages node."
		return false
	if not stages.run_world(directory_edit.text):
		status.text = "The run did not start: see the output. The node has to have started, with Preview In Editor on."
		return false
	run_button.disabled = true
	cancel_button.disabled = false
	status.text = "Starting."
	return true

## Stops the run after the chunk it is on; running again resumes it.
func cancel() -> void:
	if stages != null and is_instance_valid(stages):
		stages.cancel_world_run()

func _on_progress(done: int, total: int, costs: Dictionary) -> void:
	progress.max_value = total
	progress.value = done
	var generated := PackedStringArray()
	for stage: String in costs:
		generated.append("%s %d" % [stage, costs[stage]["products"]])
	status.text = "%d of %d chunks. Generated: %s." % [done, total, ", ".join(generated)]

func _on_finished(done: int, total: int) -> void:
	run_button.disabled = false
	cancel_button.disabled = true
	progress.max_value = total
	progress.value = done
	if total == 0:
		status.text = "The run failed: see the output."
	elif done == total:
		status.text = "Finished: %d chunks in %s." % [total, directory_edit.text]
	else:
		status.text = "Stopped at %d of %d chunks; run again to resume." % [done, total]
