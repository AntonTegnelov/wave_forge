@tool
extends Label
## A category of a Rules stage in the dock's Rules panel, which a scene dragged from the FileSystem
## dock drops onto (N3): `dropped` names the stage, the category and the scene's path, and the
## plugin adds a Scatter stage placing the scene there.

signal dropped(stage: String, category: String, scene: String)

var stage := ""
var category := ""

func _init() -> void:
	mouse_filter = Control.MOUSE_FILTER_STOP

func _can_drop_data(_at: Vector2, data: Variant) -> bool:
	return not scene_of(data).is_empty()

func _drop_data(_at: Vector2, data: Variant) -> void:
	dropped.emit(stage, category, scene_of(data))

## The one scene file that `data`, a drag from the FileSystem dock, holds; empty for anything else.
static func scene_of(data: Variant) -> String:
	if not data is Dictionary or data.get("type") != "files" or data.get("files", []).size() != 1:
		return ""
	var path: String = data["files"][0]
	return path if path.get_extension() in ["tscn", "scn", "glb", "gltf"] else ""
