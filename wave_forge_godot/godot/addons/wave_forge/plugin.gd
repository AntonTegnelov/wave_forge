@tool
extends EditorPlugin
## Wave Forge in the editor (docs/reference/godot.md, "Editor"): a dock of brushes that paint the
## selected WaveForgeStages node's world as edits, one undo action per stroke, and a preview that
## follows the editor's camera while the node's `preview_in_editor` is on. What a stroke does is the
## node's `paint`, so the editor and a game paint the same way; the plugin only turns the mouse into
## a path along the ground. While not painting, the candidate of the node's `candidates_stage`
## under the mouse, and what its stage's modifiers read there (`candidate_panel.gd`, N5). Below
## them, the world run of M1: the node's whole finite world run ahead of time into a directory
## (`world_run_panel.gd`).

const WorldRunPanel := preload("res://addons/wave_forge/world_run_panel.gd")
const CandidatePanel := preload("res://addons/wave_forge/candidate_panel.gd")

## The brushes of the dock, as `paint` names them, with how each is shown.
const BRUSHES := {
	"Raise": "raise",
	"Lower": "raise",
	"Smooth": "smooth",
	"Dig": "dig",
	"Fill": "fill",
	"Remove": "remove",
}
## Where the presets the dock lists lie: packs with a few parameters, shipped with the plugin.
const PRESETS := "res://addons/wave_forge/presets"
## How far along a ray the plugin looks for the ground, in world units, and in what steps.
const RAY_LENGTH := 2000.0
const RAY_STEP := 0.5

var dock: VBoxContainer
var preset_choice: OptionButton
var painting_toggle: CheckButton
var brush_choice: OptionButton
var stage_edit: LineEdit
var radius_spin: SpinBox
var strength_spin: SpinBox
var world_run: VBoxContainer
var candidate: VBoxContainer
## The node being edited, while one is selected.
var stages: Node
## The stroke under way: its path along the ground, and the edits before it, for undo.
var stroking := false
var path := PackedVector3Array()
var before := ""

func _enter_tree() -> void:
	dock = _make_dock()
	add_control_to_dock(DOCK_SLOT_RIGHT_UL, dock)

func _exit_tree() -> void:
	remove_control_from_docks(dock)
	dock.queue_free()

func _handles(object: Object) -> bool:
	return object.get_class() == "WaveForgeStages"

func _edit(object: Object) -> void:
	stages = object
	world_run.bind(object)

func _process(_delta: float) -> void:
	if stages == null or not is_instance_valid(stages) or not stages.preview_in_editor:
		return
	var camera := EditorInterface.get_editor_viewport_3d(0).get_camera_3d()
	if camera != null:
		stages.follow(camera.global_position)

func _forward_3d_gui_input(camera: Camera3D, event: InputEvent) -> int:
	if stages == null or not is_instance_valid(stages):
		return AFTER_GUI_INPUT_PASS
	if not painting_toggle.button_pressed:
		if event is InputEventMouseMotion and not String(stages.candidates_stage).is_empty():
			var ground = _ground_under(camera, event.position)
			# Within a cell of the mouse, along the ground.
			candidate.show_candidate({} if ground == null else stages.candidate_near(ground, stages.cell_size.x))
		return AFTER_GUI_INPUT_PASS
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT:
		if event.pressed:
			stroking = true
			path = PackedVector3Array()
			before = stages.edits_log()
			_add_point(camera, event.position)
		else:
			stroking = false
			_finish_stroke()
		return AFTER_GUI_INPUT_STOP
	if event is InputEventMouseMotion and stroking:
		_add_point(camera, event.position)
		return AFTER_GUI_INPUT_STOP
	return AFTER_GUI_INPUT_PASS

## Adds the ground under the mouse to the stroke's path, if the ray from the camera meets it.
func _add_point(camera: Camera3D, screen: Vector2) -> void:
	var ground = _ground_under(camera, screen)
	if ground != null:
		path.append(ground)

## Where the ray from the camera through `screen` first meets the ground, or null if it does not.
func _ground_under(camera: Camera3D, screen: Vector2) -> Variant:
	var from := camera.project_ray_origin(screen)
	var along := camera.project_ray_normal(screen)
	var travelled := 0.0
	while travelled < RAY_LENGTH:
		var at := from + along * travelled
		if at.y <= stages.ground_height(at):
			return at
		travelled += RAY_STEP
	return null

## Paints the stroke and records it as one undo action.
func _finish_stroke() -> void:
	if path.is_empty() or not stages.paint(_brush(), path):
		return
	var undo := get_undo_redo()
	undo.create_action("Wave Forge: %s stroke" % brush_choice.get_item_text(brush_choice.selected))
	undo.add_do_property(stages, "edits_text", stages.edits_log())
	undo.add_undo_property(stages, "edits_text", before)
	undo.commit_action(false)

## The brush the dock describes, as `paint` takes it.
func _brush() -> Dictionary:
	var shown := brush_choice.get_item_text(brush_choice.selected)
	var brush := {"brush": BRUSHES[shown], "radius": radius_spin.value}
	if shown == "Remove":
		brush["stages"] = PackedStringArray(stage_edit.text.split(",", false))
	else:
		brush["stage"] = stage_edit.text.strip_edges()
	if shown == "Lower":
		brush["strength"] = -strength_spin.value
	elif shown in ["Raise", "Smooth"]:
		brush["strength"] = strength_spin.value
	return brush

func _make_dock() -> VBoxContainer:
	var box := VBoxContainer.new()
	box.name = "Wave Forge"
	preset_choice = OptionButton.new()
	preset_choice.add_item("Choose a preset")
	for preset in presets():
		preset_choice.add_item(preset.get_file().trim_suffix(".world.ron"))
		preset_choice.set_item_metadata(preset_choice.item_count - 1, preset)
	preset_choice.item_selected.connect(_choose_preset)
	box.add_child(_labelled("Preset", preset_choice))
	painting_toggle = CheckButton.new()
	painting_toggle.text = "Paint"
	box.add_child(painting_toggle)
	brush_choice = OptionButton.new()
	for shown in BRUSHES:
		brush_choice.add_item(shown)
	box.add_child(_labelled("Brush", brush_choice))
	stage_edit = LineEdit.new()
	stage_edit.placeholder_text = "stage, or stages for Remove"
	box.add_child(_labelled("Stage", stage_edit))
	radius_spin = _spin(0.5, 64.0, 4.0)
	box.add_child(_labelled("Radius (cells)", radius_spin))
	strength_spin = _spin(0.0, 16.0, 1.0)
	box.add_child(_labelled("Strength", strength_spin))
	box.add_child(HSeparator.new())
	candidate = CandidatePanel.new()
	box.add_child(candidate)
	box.add_child(HSeparator.new())
	world_run = WorldRunPanel.new()
	box.add_child(world_run)
	return box

## The presets shipped with the plugin, by path.
static func presets() -> PackedStringArray:
	var paths := PackedStringArray()
	for file in DirAccess.get_files_at(PRESETS):
		if file.ends_with(".world.ron"):
			paths.append(PRESETS.path_join(file))
	return paths

## Makes the chosen preset the selected node's pack, as one undo action; its parameters then show
## in the inspector as sliders.
func _choose_preset(index: int) -> void:
	if index == 0 or stages == null or not is_instance_valid(stages):
		return
	var undo := get_undo_redo()
	undo.create_action("Wave Forge: %s preset" % preset_choice.get_item_text(index))
	undo.add_do_property(stages, "pack_file", preset_choice.get_item_metadata(index))
	undo.add_undo_property(stages, "pack_file", stages.pack_file)
	undo.add_do_method(stages, "notify_property_list_changed")
	undo.add_undo_method(stages, "notify_property_list_changed")
	undo.commit_action()
	preset_choice.select(0)

func _labelled(text: String, control: Control) -> HBoxContainer:
	var row := HBoxContainer.new()
	var label := Label.new()
	label.text = text
	row.add_child(label)
	control.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	row.add_child(control)
	return row

func _spin(low: float, high: float, value: float) -> SpinBox:
	var spin := SpinBox.new()
	spin.min_value = low
	spin.max_value = high
	spin.step = 0.1
	spin.value = value
	return spin
