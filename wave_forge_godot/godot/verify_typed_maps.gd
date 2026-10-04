## The nodes' exported maps are typed Dictionaries, and a scene saved while they were untyped
## still loads into them.
##
## Run by `../verify.sh` after `verify_inspector.gd`. Each map's property says its key and value
## types, and a scene written with untyped Dictionaries for every map, as scenes were saved before
## the maps were typed, loads with each map holding its entries, typed.
extends SceneTree

const SCENE := "user://verify_typed_maps.tscn"
## Each map: its node, its property, the key and value types its hint names, and its entry in
## the untyped scene.
const MAPS := [
	["WaveForgeStages", "target_radii", "21:;2:"],
	["WaveForgeStages", "noises", "21:;24/17:FastNoiseLite"],
	["WaveForgeWorld", "sounds", "21:;24/17:AudioStream"],
	["WaveForgeWorld", "proxy_colours", "21:;20:"],
]
const UNTYPED := """[gd_scene format=3]

[sub_resource type="FastNoiseLite" id="noise"]
frequency = 0.05

[sub_resource type="AudioStreamWAV" id="sound"]

[node name="Root" type="Node3D"]

[node name="Stages" type="WaveForgeStages" parent="."]
target_radii = {
"far": 6
}
noises = {
"hills": SubResource("noise")
}

[node name="World" type="WaveForgeWorld" parent="."]
sounds = {
"fountain": SubResource("sound")
}
proxy_colours = {
"grass": Color(0.3, 0.6, 0.3, 1)
}
"""

func _initialize() -> void:
	var problem: String = _check()
	if problem != "ok":
		printerr("verify_typed_maps: " + problem)
		quit(1)
		return
	quit(0)

## What is wrong, or "ok". A script error gives null.
func _check() -> String:
	for map: Array in MAPS:
		var node: Node = ClassDB.instantiate(map[0])
		var found: Array = node.get_property_list().filter(func(property: Dictionary) -> bool: return property["name"] == map[1])
		node.free()
		if found.size() != 1 or found[0]["type"] != TYPE_DICTIONARY or found[0]["hint"] != PROPERTY_HINT_TYPE_STRING or found[0]["hint_string"] != map[2]:
			return "%s.%s is %s, not a Dictionary of %s" % [map[0], map[1], found, map[2]]
	print("verify_typed_maps: target_radii, noises, sounds and proxy_colours are typed Dictionaries")
	var file := FileAccess.open(SCENE, FileAccess.WRITE)
	file.store_string(UNTYPED)
	file.close()
	var scene: PackedScene = ResourceLoader.load(SCENE, "", ResourceLoader.CACHE_MODE_IGNORE)
	var root_node: Node = scene.instantiate()
	var stages: Node = root_node.get_node("Stages")
	var world: Node = root_node.get_node("World")
	var entries := [
		[stages.target_radii, &"far", "6"],
		[stages.noises, &"hills", "FastNoiseLite"],
		[world.sounds, &"fountain", "AudioStreamWAV"],
		[world.proxy_colours, &"grass", "(0.3, 0.6, 0.3, 1.0)"],
	]
	for entry: Array in entries:
		var map: Dictionary = entry[0]
		var value: Variant = map.get(entry[1])
		var shown: String = value.get_class() if value is Object else str(value)
		if not map.is_typed() or map.size() != 1 or shown != entry[2]:
			root_node.free()
			return "an untyped scene's map loaded as %s" % [map]
	root_node.free()
	print("verify_typed_maps: a scene saved with untyped maps loads with each map's entry, typed")
	return "ok"
