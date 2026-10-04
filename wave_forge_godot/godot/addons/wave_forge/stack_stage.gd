@tool
class_name WaveForgeStackStage
extends Resource
## One stage of a `WaveForgeStack`: its name, its kind as a pack names it (`Field`, `Rules`,
## `Scatter` and the rest, docs/reference/packs.md, "Stages"), and the kind's fields as plain data
## in the shape a pack file writes them: a Field's expression, say
## `{"Add": [{"FastNoise": "hills"}, {"Constant": 1}]}`, or a Scatter's Dictionary of its fields.

## The name other stages read it by.
@export var name := ""
## Its kind, as a pack names it.
@export var kind := "Field"
## The kind's fields as plain data.
@export var settings: Variant = {"Constant": 0}
## How many of the WFC lattice's cells one of its columns spans: 1 for the finest level.
@export var scale := 1
## What a save keeps of it (docs/reference/packs.md, "Persistence and saves").
@export_enum("Pure", "Frozen", "Ephemeral") var persist := "Pure"

## Makes this the stage `data` is, the plain data of a pack's stage as `WaveForgeStages.pack_dictionary`
## gives it.
func read_data(data: Dictionary) -> void:
	name = data["name"]
	kind = data["kind"].keys()[0]
	settings = data["kind"][kind]
	scale = data["scale"]
	persist = data["persist"]

## The stage as the plain data a pack's stage is.
func to_data() -> Dictionary:
	return {"name": name, "kind": {kind: settings}, "scale": scale, "persist": persist}
