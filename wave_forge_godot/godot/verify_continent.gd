## The maximal preset's continent in a real Godot, as its editor scene sets it up.
##
## Run by `../verify.sh` after `verify_world.gd`. `continent.tscn` is instanced and its node started
## asking for towns alone; the scene's script gives it the history (`continent/history.json`),
## all its settlements. The first settlement is of the coastfolk, not the Solve stage's default
## culture: following it, its town arrives on its site, solved with the coastfolk's module set,
## and every module its placements name is one of that set's.
extends SceneTree

const TIMEOUT_S := 300.0
const CULTURE := "coastfolk"

var scene: Node
var stages: Node
var chunk := Vector3i.ZERO
var started_usec := 0
var followed := false

func _initialize() -> void:
	scene = (load("res://continent.tscn") as PackedScene).instantiate()
	stages = scene.get_node("Continent")
	stages.targets = PackedStringArray(["towns"])
	# Towns alone: no far ground, so no stage to colour it either.
	stages.far_ground_stage = ""
	stages.far_ground_material_stage = ""
	stages.view_radius = 0
	stages.collider_radius = -1
	root.add_child(scene)
	if not stages.start():
		_fail("the continent did not start")
		return
	stages.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("the town of %s had not arrived after %.0f s" % [chunk, TIMEOUT_S])
		return true
	if not followed:
		# The scene's script gives the history once the node has started.
		var rows: Array = stages.table_rows("settlements")
		if rows.is_empty():
			return false
		var first: Dictionary = rows[0]
		if first["culture"] != CULTURE:
			_fail("the first settlement is of %s, not %s" % [first["culture"], CULTURE])
			return true
		var at := Vector3(first["x"] * 2.0, 0.0, first["y"] * 2.0)
		chunk = Vector3i(floori(first["x"] / 8.0), floori(first["y"] / 8.0), 0)
		stages.follow(at)
		followed = true
		print("verify_continent: the scene gave %d settlements of the history; following the first, at %s" % [rows.size(), chunk])
		return false
	var town: Dictionary = stages.town("towns", chunk)
	if town.is_empty():
		return false
	if town["rules"] != CULTURE:
		_fail("the town of %s is of %s, not %s" % [chunk, town["rules"], CULTURE])
		return true
	var modules := _module_names("res://continent/cultures/%s.ron" % CULTURE)
	var placed := 0
	for placements: Dictionary in stages.town_instance_sets("towns", chunk, PackedStringArray()):
		if not modules.has(placements["name"]):
			_fail("the town places %s, which the %s do not have" % [placements["name"], CULTURE])
			return true
		placed += placements["ids"].size()
	print("verify_continent: the first settlement's town arrived after %.1f s, of the %s's module set, %d modules placed in its chunk" % [(Time.get_ticks_usec() - started_usec) / 1e6, CULTURE, placed])
	quit(0)
	return true

## The names of the modules a rule set's file lists.
func _module_names(path: String) -> Array[String]:
	var names: Array[String] = []
	var pattern := RegEx.create_from_string("name: \"([^\"]+)\"")
	for found in pattern.search_all(FileAccess.get_file_as_string(path)):
		names.append(found.get_string(1))
	return names

func _fail(message: String) -> void:
	printerr("verify_continent: " + message)
	quit(1)
