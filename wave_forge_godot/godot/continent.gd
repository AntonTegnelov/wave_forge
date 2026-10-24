@tool
extends Node3D
## The maximal preset's continent (docs/product/user-stories.md, M1) as a scene to bake in the
## editor: the child WaveForgeStages node holds the pack, its eight cultures and the targets an
## engine draws, and this gives it the history a game would simulate, `continent/history.json`,
## as its `settlements` table whenever it has started without one, unless it plays a baked world,
## which holds the history's towns already. With the node's Preview In Editor on, the Wave Forge
## dock's World run then bakes the whole continent into a directory.

const HISTORY := "res://continent/history.json"

## Whether the history was refused, which is not tried again until the scene is opened again: the
## error printed says why.
var refused := false

func _process(_delta: float) -> void:
	if refused:
		return
	var stages: Node = get_node("Continent")
	if stages.play_directory.is_empty() and not stages.stage_names().is_empty() \
			and stages.table_rows("settlements").is_empty():
		refused = not give_history(stages)

## Gives `stages` the continent's history; returns whether its table took it. A history that is not
## there or not a list of rows is reported, and so is one the table refuses.
static func give_history(stages: Node) -> bool:
	var rows = JSON.parse_string(FileAccess.get_file_as_string(HISTORY))
	if not rows is Array:
		push_error("continent: %s holds no list of settlements; wave_forge_godot/prepare.sh writes it" % HISTORY)
		return false
	return stages.give_table("settlements", rows)
