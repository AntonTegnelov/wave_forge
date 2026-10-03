## What the editor's inspector shows on the nodes (#273): their configuration warnings, and the
## buttons of WaveForgeStages.
##
## Run by `../verify.sh` after `verify_continent.gd`. A WaveForgeStages node in a scene with no
## light, no environment, no camera to follow and no pack warns of all four, and of none once the
## scene has them and the node a pack. A target or a stage setting naming the wrong stage warns, and `start` refuses
## it with the same words. Colliders without Jolt, occluders without occlusion culling, and a
## WaveForgeWorld that starts with no rules or names an interior bus the project lacks warn too.
## Then the buttons: "Start or regenerate" starts the node, "Reroll seed" takes another seed and
## keeps it running, and "Bake the view" refuses with nothing followed, then saves a scene of the
## nine chunks of a view of radius 1.
extends SceneTree

const BAKED := "user://inspector_bake.tscn"
const TIMEOUT_S := 60.0

var scene: Node3D
var stages: Node
var phase := 0
var started_usec := 0

func _process(_delta: float) -> bool:
	# A script error aborts a check, which then gives null, so only "ok" passes.
	if phase == 0:
		# On the first frame, when nodes added to the root are inside the tree.
		var problem = _check()
		if problem != "ok":
			return _fail(problem)
		print("verify_inspector: a dark scene and a node without a pack warn, a lit scene with a pack does not; a target or setting naming the wrong stage warns and is refused at start; colliders without Jolt, occluders without culling, a start with no rules and a missing interior bus warn")
		var problem_buttons = _buttons()
		if problem_buttons != "ok":
			return _fail(problem_buttons)
		phase = 1
		started_usec = Time.get_ticks_usec()
		return false
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		return _fail("the view did not generate in %.0f s" % TIMEOUT_S)
	for y in range(-1, 2):
		for x in range(-1, 2):
			if stages.field_values("height", Vector3i(x, y, 0)).is_empty():
				return false
	var baked = _baked()
	if baked != "ok":
		return _fail(baked)
	print("verify_inspector: the buttons start the node, reroll its seed and keep it running, refuse to bake with nothing followed, and bake the view's nine chunks into a scene")
	quit(0)
	return true

func _fail(problem) -> bool:
	printerr("verify_inspector: %s" % [problem])
	quit(1)
	return true

## The buttons that act at once: regenerate and reroll, and baking with nothing followed.
func _buttons() -> String:
	stages.seed = 5
	stages.view_radius = 1
	stages.collider_radius = -1
	stages.occluder_radius = -1
	(stages.regenerate_button as Callable).call()
	if stages.stage_names().is_empty():
		return "Start or regenerate did not start the node"
	(stages.reroll_button as Callable).call()
	if stages.seed == 5 or stages.stage_names().is_empty():
		return "Reroll seed kept seed %d, or stopped the node" % stages.seed
	if FileAccess.file_exists(BAKED):
		DirAccess.remove_absolute(BAKED)
	(stages.bake_button as Callable).call()
	if FileAccess.file_exists(BAKED):
		return "Bake the view saved a scene with nothing followed"
	stages.bake_path = BAKED
	stages.follow(Vector3(1, 0, 1))
	return "ok"

## The scene "Bake the view" saves once the view has generated.
func _baked() -> String:
	(stages.bake_button as Callable).call()
	var packed := load(BAKED) as PackedScene
	if packed == null:
		return "Bake the view saved no scene at %s" % BAKED
	var baked := packed.instantiate()
	var chunks := baked.get_child_count()
	baked.free()
	if chunks != 9:
		return "the baked view holds %d chunks, not 9" % chunks
	return "ok"

func _check() -> String:
	scene = Node3D.new()
	root.add_child(scene)
	stages = ClassDB.instantiate("WaveForgeStages")
	stages.collider_radius = -1
	scene.add_child(stages)

	var warned := Array(stages.configuration_warnings())
	for expected in ["no light", "no WorldEnvironment", "No pack_file", "no Camera3D"]:
		if not _any(warned, expected):
			return "a dark scene without a pack does not warn of %s: %s" % [expected, warned]

	scene.add_child(DirectionalLight3D.new())
	scene.add_child(WorldEnvironment.new())
	scene.add_child(Camera3D.new())
	stages.pack_file = "res://islands.world.ron"
	stages.targets = PackedStringArray(["height", "trees"])
	if not Array(stages.configuration_warnings()).is_empty():
		return "a lit scene with a pack still warns: %s" % [stages.configuration_warnings()]

	stages.targets = PackedStringArray(["height", "forest"])
	stages.candidates_stage = "height"
	warned = Array(stages.configuration_warnings())
	if not _any(warned, "forest") or not _any(warned, "candidates_stage \"height\" is no Scatter stage"):
		return "an unknown target and a candidates_stage of the wrong kind do not warn: %s" % [warned]
	if stages.start():
		return "the node started with a candidates_stage of the wrong kind"
	stages.targets = PackedStringArray(["height", "trees"])
	stages.candidates_stage = ""

	stages.collider_radius = 1
	ProjectSettings.set_setting("physics/3d/physics_engine", "GodotPhysics3D")
	warned = Array(stages.configuration_warnings())
	ProjectSettings.set_setting("physics/3d/physics_engine", "Jolt Physics")
	if not _any(warned, "Jolt Physics"):
		return "colliders without Jolt do not warn: %s" % [warned]
	stages.occluder_radius = 1
	ProjectSettings.set_setting("rendering/occlusion_culling/use_occlusion_culling", false)
	warned = Array(stages.configuration_warnings())
	if not _any(warned, "occlusion culling is off"):
		return "occluders without culling do not warn: %s" % [warned]
	ProjectSettings.set_setting("rendering/occlusion_culling/use_occlusion_culling", true)
	if _any(Array(stages.configuration_warnings()), "occlusion"):
		return "occluders with culling on still warn"

	var world: Node = ClassDB.instantiate("WaveForgeWorld")
	world.collider_radius = -1
	world.start_on_ready = false
	world.audio_radius = 2
	world.interior_reverb_bus = "Nowhere"
	scene.add_child(world)
	warned = Array(world.configuration_warnings())
	if not _any(warned, "Nowhere"):
		return "an interior bus the project lacks does not warn: %s" % [warned]
	AudioServer.add_bus()
	AudioServer.set_bus_name(AudioServer.bus_count - 1, "Nowhere")
	if _any(Array(world.configuration_warnings()), "Nowhere"):
		return "an interior bus the project has still warns"
	world.start_on_ready = true
	if not _any(Array(world.configuration_warnings()), "rules_file is empty"):
		return "a start with no rules does not warn"
	return "ok"

## Whether one of `warnings` contains `text`.
func _any(warnings: Array, text: String) -> bool:
	return warnings.any(func(warning: String) -> bool: return warning.contains(text))
