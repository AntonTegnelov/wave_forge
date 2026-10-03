@tool
extends VBoxContainer
## The dock's kit import (docs/reference/godot.md, "Editor"; N6): proposes a module set from a
## MeshLibrary's meshes, lists the connectors the proposal finds for the artist to rename and mark
## walkable, and saves the named set as a rule file. WaveForgeWorld does the work
## (`kit_connectors`, `name_module_set`); the panel only gathers the artist's names.

## Where the rule file is saved unless the path is changed: inside the project, so it ships.
const DEFAULT_RULES := "res://kit.ron"

var library_edit: LineEdit
var cell_spins: Array[SpinBox] = []
## The connectors listed: a row of the proposal's name and modules, a new name and a walkable tick.
var connectors_grid: GridContainer
var path_edit: LineEdit
var status: Label
## The library and cell proposed from, and each connector's name field and walkable tick, by the
## proposal's name.
var library: MeshLibrary
var cell := Vector3.ONE
var rows := {}

func _init() -> void:
	name = "Kit import"
	library_edit = LineEdit.new()
	library_edit.placeholder_text = "res://kit.meshlib"
	library_edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	add_child(_row("MeshLibrary", library_edit))
	var cell_row := HBoxContainer.new()
	for axis in ["x", "y", "z"]:
		var spin := SpinBox.new()
		spin.min_value = 0.01
		spin.max_value = 1000.0
		spin.step = 0.01
		spin.value = 1.0
		spin.prefix = axis
		cell_spins.append(spin)
		cell_row.add_child(spin)
	add_child(_row("Cell", cell_row))
	var propose_button := Button.new()
	propose_button.text = "Propose connectors"
	propose_button.pressed.connect(_on_propose)
	add_child(propose_button)
	connectors_grid = GridContainer.new()
	connectors_grid.columns = 3
	add_child(connectors_grid)
	path_edit = LineEdit.new()
	path_edit.text = DEFAULT_RULES
	path_edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	add_child(_row("Save as", path_edit))
	var save_button := Button.new()
	save_button.text = "Save module set"
	save_button.pressed.connect(func() -> void: save_to(path_edit.text))
	add_child(save_button)
	status = Label.new()
	status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	add_child(status)

## Lists the connectors the proposal of `kit`'s meshes, in cells of `cell_size`, finds, each with
## a field for a new name and, for a side, a walkable tick; returns how many.
func propose(kit: MeshLibrary, cell_size: Vector3) -> int:
	library = kit
	cell = cell_size
	rows.clear()
	for child in connectors_grid.get_children():
		connectors_grid.remove_child(child)
		child.queue_free()
	var found: Array = ClassDB.class_call_static("WaveForgeWorld", "kit_connectors", kit, cell_size)
	for connector: Dictionary in found:
		var label := Label.new()
		label.text = connector["name"]
		label.tooltip_text = "On " + ", ".join(connector["modules"])
		connectors_grid.add_child(label)
		var rename := LineEdit.new()
		rename.placeholder_text = connector["name"]
		rename.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		connectors_grid.add_child(rename)
		var walkable := CheckBox.new()
		walkable.text = "walkable"
		walkable.disabled = connector["top"]
		connectors_grid.add_child(walkable)
		rows[connector["name"]] = [rename, walkable]
	status.text = "%d connectors proposed; name them and save" % found.size()
	return found.size()

## Saves the module set the proposal makes, under the names typed and with the walkable ticks, as
## the rule file at `path`; returns whether it was saved. A refused naming says why in the output.
func save_to(path: String) -> bool:
	if library == null:
		status.text = "Propose connectors first"
		return false
	var names := {}
	var walkable := PackedStringArray()
	for proposed: String in rows:
		var typed: String = rows[proposed][0].text.strip_edges()
		if not typed.is_empty():
			names[proposed] = typed
		if rows[proposed][1].button_pressed:
			walkable.append(proposed)
	var text: String = ClassDB.class_call_static("WaveForgeWorld", "name_module_set", library, cell, names, walkable)
	if text.is_empty():
		status.text = "The names were refused; the output says why"
		return false
	var file := FileAccess.open(path, FileAccess.WRITE)
	if file == null:
		status.text = "Cannot write %s: %s" % [path, error_string(FileAccess.get_open_error())]
		return false
	file.store_string(text)
	file.close()
	status.text = "Saved the module set to %s" % path
	return true

func _on_propose() -> void:
	var kit := load(library_edit.text) as MeshLibrary
	if kit == null:
		status.text = "No MeshLibrary at %s" % library_edit.text
		return
	propose(kit, Vector3(cell_spins[0].value, cell_spins[1].value, cell_spins[2].value))

func _row(text: String, control: Control) -> HBoxContainer:
	var row := HBoxContainer.new()
	var label := Label.new()
	label.text = text
	row.add_child(label)
	row.add_child(control)
	return row
