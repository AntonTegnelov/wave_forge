@tool
extends EditorPlugin
## Wave Forge in the editor (docs/reference/godot.md, "Editor"): a dock of brushes that paint the
## selected WaveForgeStages node's world as edits, one undo action per stroke, and a preview that
## follows the editor's camera while the node's `preview_in_editor` is on. What a stroke does is the
## node's `paint`, so the editor and a game paint the same way; the plugin only turns the mouse into
## a path along the ground.

## The brushes of the dock, as `paint` names them, with how each is shown.
const BRUSHES := {
	"Raise": "raise",
	"Lower": "raise",
	"Smooth": "smooth",
	"Dig": "dig",
	"Fill": "fill",
	"Remove": "remove",
}
## How far along a ray the plugin looks for the ground, in world units, and in what steps.
const RAY_LENGTH := 2000.0
const RAY_STEP := 0.5

var dock: VBoxContainer
var painting_toggle: CheckButton
var brush_choice: OptionButton
var stage_edit: LineEdit
var radius_spin: SpinBox
var strength_spin: SpinBox
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

func _process(_delta: float) -> void:
	if stages == null or not is_instance_valid(stages) or not stages.preview_in_editor:
		return
	var camera := EditorInterface.get_editor_viewport_3d(0).get_camera_3d()
	if camera != null:
		stages.follow(camera.global_position)

func _forward_3d_gui_input(camera: Camera3D, event: InputEvent) -> int:
	if stages == null or not is_instance_valid(stages) or not painting_toggle.button_pressed:
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
	var from := camera.project_ray_origin(screen)
	var along := camera.project_ray_normal(screen)
	var travelled := 0.0
	while travelled < RAY_LENGTH:
		var at := from + along * travelled
		if at.y <= stages.ground_height(at):
			path.append(at)
			return
		travelled += RAY_STEP

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
	return box

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
