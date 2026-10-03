## The nodes' configuration warnings, which the editor shows on them (#273).
##
## Run by `../verify.sh` after `verify_continent.gd`. A WaveForgeStages node in a scene with no
## light, no environment and no pack warns of all three, and of none once the scene has them and
## the node a pack. A target or a stage setting naming the wrong stage warns, and `start` refuses
## it with the same words. Colliders without Jolt, occluders without occlusion culling, and a
## WaveForgeWorld that starts with no rules or names an interior bus the project lacks warn too.
extends SceneTree

var scene: Node3D

# On the first frame, when nodes added to the root are inside the tree.
func _process(_delta: float) -> bool:
	# A script error aborts _check, which then gives null, so only "ok" passes.
	var problem = _check()
	if problem != "ok":
		printerr("verify_warnings: %s" % [problem])
		quit(1)
		return true
	print("verify_warnings: a dark scene and a node without a pack warn, a lit scene with a pack does not; a target or setting naming the wrong stage warns and is refused at start; colliders without Jolt, occluders without culling, a start with no rules and a missing interior bus warn")
	quit(0)
	return true

func _check() -> String:
	scene = Node3D.new()
	root.add_child(scene)
	var stages: Node = ClassDB.instantiate("WaveForgeStages")
	stages.collider_radius = -1
	scene.add_child(stages)

	var warned := Array(stages.configuration_warnings())
	for expected in ["no light", "no WorldEnvironment", "No pack_file"]:
		if not _any(warned, expected):
			return "a dark scene without a pack does not warn of %s: %s" % [expected, warned]

	scene.add_child(DirectionalLight3D.new())
	scene.add_child(WorldEnvironment.new())
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
