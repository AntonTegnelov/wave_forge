@tool
extends VBoxContainer
## The dock's candidate view (docs/reference/godot.md, "Editor"): what became of the Scatter
## candidate under the mouse, drawn by the selected WaveForgeStages node's `candidates_stage`, and
## what its stage's modifiers read there. The node finds the candidate (`candidate_near`); the panel
## only renders the Dictionary it gives.

var title: Label
var details: Label

func _init() -> void:
	name = "Candidate"
	title = Label.new()
	add_child(title)
	details = Label.new()
	details.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	add_child(details)
	show_candidate({})

## Shows `candidate`, as `candidate_near` gives it, or that none is under the mouse when empty.
func show_candidate(candidate: Dictionary) -> void:
	if candidate.is_empty():
		title.text = "Candidate: none under the mouse"
		title.remove_theme_color_override("font_color")
		details.text = ""
		return
	title.text = "Candidate: %s" % candidate["verdict"]
	title.add_theme_color_override("font_color", candidate["colour"])
	var lines := PackedStringArray(["height %.2f" % candidate["height"]])
	if candidate.has("slope"):
		lines.append("slope %.2f" % candidate["slope"])
	if candidate.has("water_depth"):
		lines.append("water depth %.2f" % candidate["water_depth"])
	var number := 0
	for condition: Dictionary in candidate["conditions"]:
		lines.append("condition %d: %.3f, %s" % [number, condition["value"], "holds" if condition["holds"] else "fails"])
		number += 1
	details.text = "\n".join(lines)
