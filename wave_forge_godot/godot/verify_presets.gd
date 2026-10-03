## Every preset the plugin ships, taken as a node added with nothing set takes it (N1, N2).
##
## Run by `../verify.sh` after `verify_inspector.gd`, which also fails if this prints an error or
## a warning. For each preset: a fresh WaveForgeStages node is given the preset's settings and
## added to a scene with a sun and an environment, and it starts on its own, as on a first play.
## Once nothing is pending around the followed point it warns of nothing, stands ground there,
## has bodies and navigation, draws its sea when it has one, draws its trees as MultiMesh instances,
## and has a palette colour per material of its ground.
extends SceneTree

const Presets := preload("res://addons/wave_forge/presets.gd")
const TIMEOUT_S := 120.0
const AT := Vector3(1, 0, 1)

var paths: PackedStringArray
var index := -1
var scene: Node3D
var stages: Node
var started_usec := 0

func _initialize() -> void:
	paths = Presets.paths()
	if paths.is_empty():
		_fail("no preset in %s" % Presets.DIRECTORY)

func _process(_delta: float) -> bool:
	if stages == null:
		index += 1
		if index >= paths.size():
			print("verify_presets: %d presets each take, start, and stand a lit ground with bodies, navigation and trees, warning of nothing" % paths.size())
			quit(0)
			return true
		_take(paths[index])
		return false
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		return _fail("%s: not settled after %.0f s: %s" % [paths[index], TIMEOUT_S, stages.stats()])
	var stats: Dictionary = stages.stats()
	if stages.ground_chunks().is_empty() or stats["pending_grounds"] > 0 or stats["pending_colliders"] > 0 or stats["pending_placements"] > 0 or stats["navigation_baked"] == 0:
		return false
	var problem = _settled(paths[index], stats)
	if problem != "ok":
		return _fail(problem)
	print("verify_presets: %s: %d grounds, %d bodies, %d navigation regions, %d trees" % [paths[index].get_file(), stages.ground_chunks().size(), stages.collider_chunks().size(), stages.navigation_chunks().size(), stats["placed_instances"]])
	scene.queue_free()
	stages = null
	return false

## A fresh node given the preset at `path`, in a lit scene, started as on a first play.
func _take(path: String) -> void:
	scene = Node3D.new()
	scene.add_child(DirectionalLight3D.new())
	scene.add_child(WorldEnvironment.new())
	stages = ClassDB.instantiate("WaveForgeStages")
	var settings := Presets.settings(path)
	for property: String in settings:
		stages.set(property, settings[property])
	stages.generation_failed.connect(func(reason: String) -> void: _fail("%s: %s" % [path, reason]))
	scene.add_child(stages)
	root.add_child(scene)
	stages.follow(AT)
	started_usec = Time.get_ticks_usec()

## What is wrong with the settled preset at `path`, or "ok". A script error gives null.
func _settled(path: String, stats: Dictionary) -> String:
	var warned: PackedStringArray = stages.configuration_warnings()
	if not warned.is_empty():
		return "%s warns: %s" % [path, warned]
	if is_nan(stages.ground_height(AT)):
		return "%s stands no ground at %s" % [path, AT]
	if stages.collider_chunks().is_empty() or stages.navigation_chunks().is_empty():
		return "%s has %d bodies and %d navigation regions" % [path, stages.collider_chunks().size(), stages.navigation_chunks().size()]
	if stages.sea_material != null and not stages.water().is_empty() and not stats["sea_drawn"]:
		return "%s has a sea and water but draws no sea" % path
	if stats["placed_instances"] == 0:
		return "%s draws no trees as MultiMesh instances: %s" % [path, stats]
	var materials: PackedStringArray = stages.category_names(stages.ground_material_stage)
	if materials.size() != stages.ground_palette.size():
		return "%s has %d palette colours for the materials %s" % [path, stages.ground_palette.size(), materials]
	return "ok"

func _fail(message: String) -> bool:
	printerr("verify_presets: " + message)
	quit(1)
	return true
