@tool
extends Node
## The editor plugin's dock and shortcuts, checked in the editor itself, where editor classes
## exist: an autoload of the test project that does nothing outside the editor. Once the plugin has
## loaded, the editor holds the plugin's own `EditorDock` titled "Wave Forge", laid out under the
## key `wave_forge`, and the editor settings hold every shortcut the plugin registers, each taking
## its default key. `../verify.sh` opens the editor headless and fails without the line this prints.

const PATHS := ["wave_forge/paint", "wave_forge/brush_raise", "wave_forge/brush_lower",
	"wave_forge/brush_smooth", "wave_forge/brush_dig", "wave_forge/brush_fill",
	"wave_forge/brush_remove"]
const KEYS := [KEY_P, KEY_1, KEY_2, KEY_3, KEY_4, KEY_5, KEY_6]
## Frames to wait for the editor to load the plugin.
const WAIT_FRAMES := 60

var frames := 0

func _process(_delta: float) -> void:
	if not Engine.is_editor_hint():
		set_process(false)
		return
	frames += 1
	if frames < WAIT_FRAMES:
		return
	set_process(false)
	var docks := EditorInterface.get_base_control().find_children("*", "EditorDock", true, false)
	# Its own dock, whose layout the editor saves under its key, not a control the editor wrapped.
	if not docks.any(func(dock: EditorDock) -> bool: return dock.title == "Wave Forge" and dock.layout_key == "wave_forge"):
		push_error("verify_plugin: no EditorDock titled Wave Forge with the layout key wave_forge among %d docks" % docks.size())
		return
	var settings := EditorInterface.get_editor_settings()
	for i in PATHS.size():
		var key := InputEventKey.new()
		key.keycode = KEYS[i]
		key.pressed = true
		if not settings.has_shortcut(PATHS[i]) or not settings.is_shortcut(PATHS[i], key):
			push_error("verify_plugin: %s is not a shortcut taking its key" % PATHS[i])
			return
	print("verify_plugin: the dock is an EditorDock, and Paint and the six brushes have shortcuts in the editor settings")
