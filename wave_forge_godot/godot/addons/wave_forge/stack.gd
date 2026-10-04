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

## Every category of the stack's Rules stages, by stage name in stack order, as a pack file
## orders them: each rule's category where it first appears, then the one otherwise takes.
func rule_categories() -> Dictionary:
	var categories := {}
	for stage: StackStage in stages:
		if stage.kind != "Rules":
			continue
		var names := PackedStringArray()
		for rule: Dictionary in stage.settings["rules"]:
			if not names.has(rule["category"]):
				names.append(rule["category"])
		if not names.has(stage.settings["otherwise"]):
			names.append(stage.settings["otherwise"])
		categories[stage.name] = names
	return categories

## Adds, at the bottom, a Scatter stage placing points of `kind` on the columns of `category` of
## the Rules stage `rules`, standing on the field stage `height`: what dropping a scene onto a rule
## makes (N3). Its points lie at least 2 cells apart, one candidate per 3 cells square, off ground
## steeper than 1 cell of height per cell. It is named after `kind`, with a number added if a stage
## has the name already, and its points' kind is its name, so a scene bound to it binds it alone.
## Returns the stage.
func add_scatter_on(rules: String, category: String, kind: String, height: String) -> StackStage:
	var taken := stages.map(func(stage: StackStage) -> String: return stage.name)
	var stage := StackStage.new()
	stage.name = kind
	var number := 2
	while taken.has(stage.name):
		stage.name = "%s_%d" % [kind, number]
		number += 1
	stage.kind = "Scatter"
	stage.settings = {
		"kind": stage.name,
		"height": height,
		"spacing": 3,
		"apart": 2,
		"max_slope": 1.0,
		"when": [{"Greater": [{"Is": [rules, [category]]}, {"Constant": 0.5}]}],
	}
	stages.append(stage)
	return stage

## The pack as plain data, in the shape `WaveForgeStages.pack_text` takes.
func to_pack_data() -> Dictionary:
	var data := rest.duplicate(true)
	data["stages"] = stages.map(func(stage: StackStage) -> Dictionary: return stage.to_data())
	return data

## The pack's text, as a pack file is written; empty, with the reason as an error, if the stack
## holds no valid pack.
func to_pack_text() -> String:
	return WaveForgeStages.pack_text(to_pack_data())
