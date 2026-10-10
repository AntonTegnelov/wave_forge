## Runs the addon's demo scene and checks it shows a lit world: a sun and a sky from the addon's
## sun_and_sky.tscn, and the hills preset generating its ground around the demo's camera.
##
## Run by `../verify.sh` after `verify_presets.gd`.
extends SceneTree

const TIMEOUT_S := 60.0

var demo: Node
var stages: Node
var started_usec := 0

func _initialize() -> void:
	demo = (load("res://addons/wave_forge/demo/demo.tscn") as PackedScene).instantiate()
	root.add_child(demo)
	stages = demo.get_node("Hills")
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	if demo.find_children("*", "DirectionalLight3D", true, false).size() != 1 or demo.find_children("*", "WorldEnvironment", true, false).size() != 1:
		return _fail("the demo has no single sun and sky")
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		return _fail("no ground arrived around the camera")
	var camera: Camera3D = demo.get_node("Camera")
	var under := Vector3(camera.global_position.x, 0, camera.global_position.z)
	if is_nan(stages.ground_height(under)):
		return false
	print("verify_demo: the demo lights the hills preset with the addon's sun and sky, and its ground arrived under the camera")
	quit(0)
	return true

func _fail(message: String) -> bool:
	printerr("verify_demo: FAILED: %s" % message)
	quit(1)
	return true
