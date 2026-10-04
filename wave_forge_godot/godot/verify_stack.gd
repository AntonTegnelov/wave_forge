## A pack as a stack of stages: the plugin's `WaveForgeStack` and a node's `stack`.
##
## Run by `../verify.sh` after `verify_pack_data.gd`. The islands preset made a stack saves the
## preset's own pack, and saved as a resource and loaded back saves it still. A node given the stack
## and no `pack_file` starts and generates it, warning of no missing pack. Raising the stack's sea
## floor stage in place, and putting its stages in another order, a node started again generates the
## edited pack: its height is the edited one. A stack that holds no valid pack does not start, and a
## node given both a stack and a `pack_file` warns that the stack is generated. Last, as dropping a
## scene onto a rule in the dock does (N3): the stack lists its Rules stages' categories, the drop
## target takes one scene file and nothing else, and a Scatter stage added on the islands' grass is
## generated, every point on a grass column standing on the ground there, and drawn from its scene.
extends SceneTree

const ISLANDS := "res://addons/wave_forge/presets/islands.world.ron"
const SAVED := "user://verify_stack.tres"
const TIMEOUT_S := 30.0
const AT := Vector3(1, 0, 1)
const Stack := preload("res://addons/wave_forge/stack.gd")
const StackStage := preload("res://addons/wave_forge/stack_stage.gd")
const RuleDrop := preload("res://addons/wave_forge/rule_drop.gd")

var stack: Stack
var node: Node
var phase := "first"
var first_height := 0.0
var started_usec := 0

func _initialize() -> void:
	var problem: String = _make()
	if problem != "ok":
		_fail(problem)
		return
	node = _node()
	if not node.start():
		_fail("a node given the stack did not start")
		return
	node.follow(AT)
	started_usec = Time.get_ticks_usec()

## Makes the stack and checks what it saves, or says what is wrong. A script error gives null.
func _make() -> String:
	var text := FileAccess.get_file_as_string(ISLANDS)
	stack = Stack.new()
	if not stack.read_pack_text(text) or stack.stages.size() < 3:
		return "the islands preset made no stack"
	var own: String = WaveForgeStages.pack_text(WaveForgeStages.pack_dictionary(text))
	if stack.to_pack_text() != own:
		return "the stack saves another pack than the preset's"
	if ResourceSaver.save(stack, SAVED) != OK:
		return "the stack could not be saved"
	var loaded: Stack = ResourceLoader.load(SAVED, "", ResourceLoader.CACHE_MODE_IGNORE)
	if loaded == null or loaded.to_pack_text() != own:
		return "the stack saved as a resource and loaded back saves another pack"
	print("verify_stack: the islands preset made a stack of %d stages saves its own pack, as a resource too" % stack.stages.size())
	return "ok"

## A node generating `stack`'s height, with no pack_file, not started yet.
func _node() -> Node:
	var made: Node = ClassDB.instantiate("WaveForgeStages")
	made.stack = stack
	made.targets = PackedStringArray(["height"])
	made.view_radius = 1
	root.add_child(made)
	return made

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		return _fail("the %s stack's height had not arrived" % phase)
	if phase == "dropped":
		return _check_dropped()
	var values: PackedFloat32Array = node.field_values("height", Vector3i.ZERO)
	if values.is_empty():
		return false
	if phase == "first":
		var warned := Array(node.configuration_warnings())
		if warned.any(func(warning: String) -> bool: return warning.contains("pack_file")):
			return _fail("a node with a stack warns of its pack: %s" % warned)
		first_height = values[0]
		# Raise the height stage by 5 cells, and put the stack's stages in another order.
		var height: StackStage = stack.stages.filter(func(stage: StackStage) -> bool: return stage.name == "height")[0]
		height.settings = {"Add": [height.settings, {"Constant": 5.0}]}
		stack.stages.push_front(stack.stages.pop_back())
		node.queue_free()
		node = _node()
		if not node.start():
			return _fail("a node given the edited stack did not start")
		node.follow(AT)
		phase = "edited"
		started_usec = Time.get_ticks_usec()
		return false
	if not is_equal_approx(values[0], first_height + 5.0):
		return _fail("the edited stack's height is %.3f, the first was %.3f" % [values[0], first_height])
	print("verify_stack: a node given the stack generates it, and started again the stack as edited in place, reordered too")
	return _refusals()

## A stack of no valid pack does not start, and both a stack and a pack_file warn.
func _refusals() -> bool:
	var broken := Stack.new()
	var stage := StackStage.new()
	stage.name = "height"
	stage.settings = {"Input": "nothing"}
	broken.stages.append(stage)
	var refused: Node = ClassDB.instantiate("WaveForgeStages")
	refused.stack = broken
	refused.targets = PackedStringArray(["height"])
	root.add_child(refused)
	if refused.start():
		return _fail("a stack of no valid pack started")
	refused.pack_file = ISLANDS
	refused.stack = stack
	var warned := Array(refused.configuration_warnings())
	if not warned.any(func(warning: String) -> bool: return warning.contains("the stack is generated")):
		return _fail("a node with a stack and a pack_file warns of %s" % warned)
	print("verify_stack: a stack of no valid pack does not start, and a stack beside a pack_file warns that the stack is generated")
	refused.queue_free()
	return _dropped()

## As a scene dropped onto the islands' grass in the dock: a Scatter stage on grass, bound to a
## scene of a box and generated.
func _dropped() -> bool:
	var islands := Stack.new()
	islands.read_pack_text(FileAccess.get_file_as_string(ISLANDS))
	var categories := islands.rule_categories()
	if categories != {"surface": PackedStringArray(["sand", "rock", "grass"])}:
		return _fail("the islands' rule categories are %s" % categories)
	# Each drag from the FileSystem dock, or of nodes, with the scene it drops; none for most.
	var drags := [
		[{"type": "files", "files": PackedStringArray(["res://bush.tscn"])}, "res://bush.tscn"],
		[{"type": "files", "files": PackedStringArray(["res://bush.tscn", "res://other.tscn"])}, ""],
		[{"type": "files", "files": PackedStringArray(["res://notes.txt"])}, ""],
		[{"type": "nodes", "nodes": [NodePath("Bush")]}, ""],
	]
	for drag: Array in drags:
		if RuleDrop.scene_of(drag[0]) != drag[1]:
			return _fail("a drag of %s drops %s" % [drag[0], RuleDrop.scene_of(drag[0])])
	var stage: StackStage = islands.add_scatter_on("surface", "grass", "bush", "height")
	var again: StackStage = islands.add_scatter_on("surface", "grass", "bush", "height")
	if stage.name != "bush" or again.name != "bush_2" or again.settings["kind"] != "bush_2":
		return _fail("two drops of a bush made stages %s and %s" % [stage.name, again.name])
	islands.stages.pop_back()
	var box := MeshInstance3D.new()
	box.mesh = BoxMesh.new()
	var scene := PackedScene.new()
	scene.pack(box)
	box.free()
	node = ClassDB.instantiate("WaveForgeStages")
	node.stack = islands
	node.targets = PackedStringArray(["height", "surface", "bush"])
	node.scenes = {"bush": scene}
	node.ground_stage = "height"
	node.view_radius = 1
	root.add_child(node)
	if not node.start():
		return _fail("the stack with the dropped bush did not start")
	node.follow(AT)
	phase = "dropped"
	started_usec = Time.get_ticks_usec()
	return false

## Whether the dropped bushes are generated and drawn, every one on grass on the ground; fails if
## not once they have all arrived.
func _check_dropped() -> bool:
	# Every chunk around the followed one has its bushes, and every bush is drawn.
	var stats: Dictionary = node.stats()
	if not stats["stages"].has("bush") or stats["stages"]["bush"]["products"] < 9 or stats["pending_placements"] > 0:
		return false
	var grass: int = node.category_names("surface").find("grass")
	var bushes := 0
	for y in range(-1, 2):
		for x in range(-1, 2):
			var chunk := Vector3i(x, y, 0)
			var categories: PackedByteArray = node.categories("surface", chunk)
			for set: Dictionary in node.point_sets("bush", chunk):
				var transforms: PackedFloat32Array = set["transforms"]
				for i in range(0, transforms.size(), 12):
					var at := Vector3(transforms[i + 3], transforms[i + 7], transforms[i + 11])
					var column := Vector2i(floori(at.x) - x * 8, floori(at.z) - y * 8)
					if categories[column.y * 8 + column.x] != grass:
						return _fail("a bush off the grass at %s" % at)
					var ground: float = node.ground_height(at)
					if is_nan(ground) or absf(at.y - ground) > 0.5:
						return _fail("a bush at %s floats off the ground at %.3f" % [at, ground])
					bushes += 1
	if bushes == 0 or stats["placed_instances"] != bushes:
		return _fail("%d bushes generated, %d drawn" % [bushes, stats["placed_instances"]])
	print("verify_stack: a scene dropped onto the islands' grass places %d bushes, each on grass on the ground, drawn from it" % bushes)
	quit(0)
	return true

func _fail(message: String) -> bool:
	printerr("verify_stack: " + message)
	quit(1)
	return true
