@tool
extends EditorPlugin
## Wave Forge in the editor (docs/reference/godot.md, "Editor"): a dock of brushes that paint the
## selected WaveForgeStages node's world as edits, one undo action per stroke, and a preview that
## follows the editor's camera while the node's `preview_in_editor` is on. What a stroke does is the
## node's `paint`, so the editor and a game paint the same way; the plugin only turns the mouse into
## a path along the ground. While not painting, the candidate of the node's `candidates_stage`
## under the mouse, and what its stage's modifiers read there (`candidate_panel.gd`, N5). Below
## them, the world run of M1: the node's whole finite world run ahead of time into a directory
## (`world_run_panel.gd`), and the kit import of N6: a MeshLibrary's connectors proposed, named
## by the artist and saved as a module set (`kit_import_panel.gd`). The dock is an `EditorDock`;
## Paint and each brush have a shortcut under `wave_forge/` that a user rebinds in the editor
## settings, taken while the 3D viewport has focus and a node is selected. Edit as a stack turns
## the node's pack_file into a `WaveForgeStack` the scene holds (`stack.gd`). Below it, the Rules
## panel lists the categories of the pack's Rules stages; a scene dragged from the FileSystem dock
## onto one adds a Scatter stage placing it there, bound and generated (N3, `rule_drop.gd`). The
## plugin also lists the place names of pack files in the editor's translation template generation
## (`translation_parser.gd`).

const WorldRunPanel := preload("res://addons/wave_forge/world_run_panel.gd")
const CandidatePanel := preload("res://addons/wave_forge/candidate_panel.gd")
const KitImportPanel := preload("res://addons/wave_forge/kit_import_panel.gd")
const Presets := preload("res://addons/wave_forge/presets.gd")
const Stack := preload("res://addons/wave_forge/stack.gd")
const TranslationParser := preload("res://addons/wave_forge/translation_parser.gd")
const RuleDrop := preload("res://addons/wave_forge/rule_drop.gd")

## The brushes of the dock, as `paint` names them, with how each is shown.
const BRUSHES := {
	"Raise": "raise",
	"Lower": "raise",
	"Smooth": "smooth",
	"Dig": "dig",
	"Fill": "fill",
	"Remove": "remove",
}
## Each shortcut's path in the editor settings, what it is called there, and its default key: Paint
## toggles painting, and each brush's picks it.
const SHORTCUTS := {
	"wave_forge/paint": ["Toggle painting", KEY_P],
	"wave_forge/brush_raise": ["Raise brush", KEY_1],
	"wave_forge/brush_lower": ["Lower brush", KEY_2],
	"wave_forge/brush_smooth": ["Smooth brush", KEY_3],
	"wave_forge/brush_dig": ["Dig brush", KEY_4],
	"wave_forge/brush_fill": ["Fill brush", KEY_5],
	"wave_forge/brush_remove": ["Remove brush", KEY_6],
}
## How far along a ray the plugin looks for the ground, in world units, and in what steps.
const RAY_LENGTH := 2000.0
const RAY_STEP := 0.5

var dock: EditorDock
var preset_choice: OptionButton
var painting_toggle: CheckButton
var brush_choice: OptionButton
var stage_edit: LineEdit
var radius_spin: SpinBox
var strength_spin: SpinBox
var world_run: VBoxContainer
var candidate: VBoxContainer
var kit_import: VBoxContainer
var rules_box: VBoxContainer
var translation_parser: EditorTranslationParserPlugin
## The node being edited, while one is selected.
var stages: Node
## The stroke under way: its path along the ground, and the edits before it, for undo.
var stroking := false
var path := PackedVector3Array()
var before := ""

func _enter_tree() -> void:
	var settings := EditorInterface.get_editor_settings()
	for path: String in SHORTCUTS:
		if not settings.has_shortcut(path):
			var key := InputEventKey.new()
			key.keycode = SHORTCUTS[path][1]
			var shortcut := Shortcut.new()
			shortcut.resource_name = SHORTCUTS[path][0]
			shortcut.events = [key]
			settings.add_shortcut(path, shortcut)
	translation_parser = TranslationParser.new()
	add_translation_parser_plugin(translation_parser)
	dock = EditorDock.new()
	dock.title = "Wave Forge"
	dock.layout_key = "wave_forge"
	dock.default_slot = EditorDock.DOCK_SLOT_RIGHT_UL
	dock.add_child(_make_dock())
	add_dock(dock)

func _exit_tree() -> void:
	remove_translation_parser_plugin(translation_parser)
	remove_dock(dock)
	dock.queue_free()

func _handles(object: Object) -> bool:
	return object.get_class() == "WaveForgeStages"

func _edit(object: Object) -> void:
	stages = object
	world_run.bind(object)
	# A node just added, with nothing set, takes the default preset (N1).
	if stages != null and Presets.is_fresh(stages) and FileAccess.file_exists(Presets.DEFAULT):
		_apply_preset(Presets.DEFAULT, "Wave Forge: the default preset")
	_refresh_rules()

func _process(_delta: float) -> void:
	if stages == null or not is_instance_valid(stages) or not stages.preview_in_editor:
		return
	var camera := EditorInterface.get_editor_viewport_3d(0).get_camera_3d()
	if camera != null:
		stages.follow(camera.global_position)

func _forward_3d_gui_input(camera: Camera3D, event: InputEvent) -> int:
	if event is InputEventKey and event.pressed and not event.echo and _take_shortcut(event):
		return AFTER_GUI_INPUT_STOP
	if stages == null or not is_instance_valid(stages):
		return AFTER_GUI_INPUT_PASS
	if not painting_toggle.button_pressed:
		if event is InputEventMouseMotion and not String(stages.candidates_stage).is_empty():
			candidate.show_candidate(stages.candidate_under(camera.project_ray_origin(event.position), camera.project_ray_normal(event.position)))
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

## Toggles painting or picks a brush if `event` is one of their shortcuts; whether it was.
func _take_shortcut(event: InputEvent) -> bool:
	var settings := EditorInterface.get_editor_settings()
	if settings.is_shortcut("wave_forge/paint", event):
		painting_toggle.button_pressed = not painting_toggle.button_pressed
		return true
	for index in brush_choice.item_count:
		var shown := brush_choice.get_item_text(index)
		if settings.is_shortcut("wave_forge/brush_" + shown.to_lower(), event):
			brush_choice.select(index)
			return true
	return false

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
	for preset in Presets.paths():
		preset_choice.add_item(preset.get_file().trim_suffix(".tscn"))
		preset_choice.set_item_metadata(preset_choice.item_count - 1, preset)
	preset_choice.item_selected.connect(_choose_preset)
	box.add_child(_labelled("Preset", preset_choice))
	var as_stack := Button.new()
	as_stack.text = "Edit as a stack"
	as_stack.tooltip_text = "Turns the node's pack_file into a stack of stages the scene holds and the inspector edits."
	as_stack.pressed.connect(_make_stack)
	box.add_child(as_stack)
	rules_box = VBoxContainer.new()
	box.add_child(rules_box)
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
	box.add_child(HSeparator.new())
	kit_import = KitImportPanel.new()
	box.add_child(kit_import)
	return box

func _choose_preset(index: int) -> void:
	if index == 0 or stages == null or not is_instance_valid(stages):
		return
	_apply_preset(preset_choice.get_item_metadata(index), "Wave Forge: %s preset" % preset_choice.get_item_text(index))
	preset_choice.select(0)

## Copies the preset at `path` onto the selected node's settings, as one undo action.
func _apply_preset(path: String, action: String) -> void:
	var undo := get_undo_redo()
	undo.create_action(action)
	var settings := Presets.settings(path)
	for property: String in settings:
		undo.add_do_property(stages, property, settings[property])
		undo.add_undo_property(stages, property, stages.get(property))
	undo.add_do_method(stages, "notify_property_list_changed")
	undo.add_undo_method(stages, "notify_property_list_changed")
	undo.commit_action()

## Gives the selected node a stack of its pack_file's pack in place of the file, as one undo action.
func _make_stack() -> void:
	if stages == null or not is_instance_valid(stages) or String(stages.pack_file).is_empty():
		return
	var stack := Stack.new()
	if not stack.read_pack_text(FileAccess.get_file_as_string(stages.pack_file)):
		return
	var undo := get_undo_redo()
	undo.create_action("Wave Forge: edit as a stack")
	undo.add_do_property(stages, "stack", stack)
	undo.add_do_property(stages, "pack_file", "")
	undo.add_undo_property(stages, "stack", stages.stack)
	undo.add_undo_property(stages, "pack_file", stages.pack_file)
	undo.add_do_method(self, "_refresh_rules")
	undo.add_undo_method(self, "_refresh_rules")
	undo.commit_action()

## The selected node's pack as a stack: a copy of its own stack, or one of its pack_file; null if
## it has neither.
func _stack_of_node() -> Stack:
	if stages == null or not is_instance_valid(stages):
		return null
	if stages.stack != null:
		return stages.stack.duplicate(true)
	var stack := Stack.new()
	if String(stages.pack_file).is_empty() or not stack.read_pack_text(FileAccess.get_file_as_string(stages.pack_file)):
		return null
	return stack

## Lists the categories of the selected node's Rules stages, each a place to drop a scene onto.
func _refresh_rules() -> void:
	for child in rules_box.get_children():
		child.queue_free()
	var stack := _stack_of_node()
	if stack == null:
		return
	var categories := stack.rule_categories()
	if categories.is_empty():
		return
	var header := Label.new()
	header.text = "Drop a scene onto a rule" if not String(stages.ground_stage).is_empty() else "Set ground_stage to drop scenes onto rules"
	rules_box.add_child(header)
	if String(stages.ground_stage).is_empty():
		return
	for rules: String in categories:
		for category: String in categories[rules]:
			var drop := RuleDrop.new()
			drop.stage = rules
			drop.category = category
			drop.text = "  %s: %s" % [rules, category]
			drop.dropped.connect(_drop_scene)
			rules_box.add_child(drop)

## Adds a Scatter stage placing the scene at `path` on `category` of the Rules stage `rules`,
## standing on the node's ground, bound to the scene and generated, as one undo action (N3). The
## node's pack becomes a stack if it was a pack_file.
func _drop_scene(rules: String, category: String, path: String) -> void:
	var stack := _stack_of_node()
	var scene := load(path) as PackedScene
	if stack == null or scene == null:
		return
	var stage := stack.add_scatter_on(rules, category, path.get_file().get_basename(), stages.ground_stage)
	var scenes: Dictionary = stages.scenes.duplicate()
	scenes[stage.name] = scene
	var targets := PackedStringArray(stages.targets)
	targets.append(stage.name)
	var undo := get_undo_redo()
	undo.create_action("Wave Forge: place %s on %s" % [path.get_file(), category])
	for property: String in ["stack", "pack_file", "scenes", "targets"]:
		undo.add_undo_property(stages, property, stages.get(property))
	undo.add_do_property(stages, "stack", stack)
	undo.add_do_property(stages, "pack_file", "")
	undo.add_do_property(stages, "scenes", scenes)
	undo.add_do_property(stages, "targets", targets)
	for method in ["_refresh_rules", "_restart_preview"]:
		undo.add_do_method(self, method)
		undo.add_undo_method(self, method)
	undo.commit_action()

## Starts the selected node again if it previews in the editor, so it shows its pack as changed.
func _restart_preview() -> void:
	if stages != null and is_instance_valid(stages) and stages.preview_in_editor:
		stages.start()

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
