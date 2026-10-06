## N1: a scene of the default preset, with no code, stands a world a player can walk.
##
## Run by `../verify.sh` after `verify_presets.gd`, which also fails if this prints an error or a
## warning. The scene holds a sun, an environment, a WaveForgeStages node given the default preset
## and the checks' own walker (`walker.tscn`, a stand-in for a game's player) over land; nothing
## calls `follow`, so the node follows the walker's camera. The node warns of nothing; the walker stands on the ground once its body is
## there, then walks forward for three seconds, travelling at least half its speed's distance and
## never falling below the ground.
extends SceneTree

const Presets := preload("res://addons/wave_forge/presets.gd")
const TIMEOUT_S := 90.0
const WALK_S := 3.0

var stages: Node
var walker: CharacterBody3D
var started_usec := 0
var walk_usec := 0
var from := Vector3.ZERO
var lowest := INF

func _initialize() -> void:
	var scene := Node3D.new()
	scene.add_child(DirectionalLight3D.new())
	scene.add_child(WorldEnvironment.new())
	stages = ClassDB.instantiate("WaveForgeStages")
	var settings := Presets.settings(Presets.DEFAULT)
	for property: String in settings:
		stages.set(property, settings[property])
	scene.add_child(stages)
	walker = (load("res://walker.tscn") as PackedScene).instantiate()
	scene.add_child(walker)
	root.add_child(scene)
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		return _fail("the walker did not stand and walk within %.0f s: standing %s, %s" % [TIMEOUT_S, walker.standing(), stages.stats()])
	if stages.stage_names().is_empty():
		return false
	if walker.position == Vector3.ZERO:
		var land = _land()
		if land == null:
			return _fail("no land within 200 units of the origin")
		walker.position = land + Vector3(0, 20, 0)
		return false
	if not walker.standing():
		return false
	if walk_usec == 0:
		var warned: PackedStringArray = stages.configuration_warnings()
		if not warned.is_empty():
			return _fail("the scene warns: %s" % warned)
		from = walker.global_position
		walk_usec = Time.get_ticks_usec()
		Input.action_press("wave_forge_forward")
		return false
	var ground: float = stages.ground_height(walker.global_position)
	if not is_nan(ground):
		lowest = minf(lowest, walker.global_position.y - ground)
	if (Time.get_ticks_usec() - walk_usec) / 1e6 < WALK_S:
		return false
	Input.action_release("wave_forge_forward")
	var travelled := Vector2(walker.global_position.x - from.x, walker.global_position.z - from.z).length()
	if travelled < walker.speed * WALK_S * 0.5:
		return _fail("the walker travelled %.1f units in %.0f s" % [travelled, WALK_S])
	if lowest < -0.5:
		return _fail("the walker fell %.2f units below the ground" % -lowest)
	print("verify_walk: the default preset and the walker, with no code: the node followed the walker's camera, the walker stood on the ground and walked %.1f units in %.0f s, never below it" % [travelled, WALK_S])
	quit(0)
	return true

## A point of land near the origin, sampled before any chunk generates, or null.
func _land() -> Variant:
	for ring in range(0, 50):
		for step in range(0, 8 * ring + 1):
			var angle := TAU * step / maxf(8 * ring, 1)
			var at := Vector3(cos(angle) * ring * 4.0, 0, sin(angle) * ring * 4.0)
			if stages.sample("height", at) > 2.0:
				return at
	return null

func _fail(message: String) -> bool:
	printerr("verify_walk: " + message)
	quit(1)
	return true
