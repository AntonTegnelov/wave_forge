@tool
class_name WaveForgeStack
extends Resource
## A pack as a stack of stages the inspector edits (docs/reference/godot.md, "Stacks"): its stages in
## order, each a `WaveForgeStackStage`, and the rest of the pack (its version, parameters, noises,
## water, tables and bound) as the plain data `WaveForgeStages.pack_dictionary` gives. A
## `WaveForgeStages` node given one in `stack` generates the pack it saves, so the stack holds
## nothing the library could not read. Scripts that run before the editor has listed the project's
## classes preload this script rather than naming it.

const StackStage := preload("res://addons/wave_forge/stack_stage.gd")

## The stages, top to bottom; a stage may read any other, above or below it.
@export var stages: Array[StackStage] = []
## The pack's other fields, by their names in a pack file: `version`, `params`, `noises`, `water`,
## `tables` and `bound`.
@export var rest: Dictionary = {"version": 1}

## Makes this the stack of the pack `text`, and returns whether it is a valid pack; if not, the
## reason is reported as an error and the stack is left as it was.
func read_pack_text(text: String) -> bool:
	var data: Dictionary = WaveForgeStages.pack_dictionary(text)
	if data.is_empty():
		return false
	stages.clear()
	for row: Dictionary in data["stages"]:
		var stage := StackStage.new()
		stage.read_data(row)
		stages.append(stage)
	data.erase("stages")
	rest = data
	return true

## The pack as plain data, in the shape `WaveForgeStages.pack_text` takes.
func to_pack_data() -> Dictionary:
	var data := rest.duplicate(true)
	data["stages"] = stages.map(func(stage: StackStage) -> Dictionary: return stage.to_data())
	return data

## The pack's text, as a pack file is written; empty, with the reason as an error, if the stack
## holds no valid pack.
func to_pack_text() -> String:
	return WaveForgeStages.pack_text(to_pack_data())
