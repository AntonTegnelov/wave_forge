## Binds a list of scenes to a kind as its variants and checks each point is drawn with its own.
##
## Run by `../verify.sh` after `verify_scenes.gd`. The pack (`variants.world.ron`) scatters oaks
## of two variants and birches of one; "oak" is bound to two lone meshes and "birch" to one, so a
## chunk draws one MultiMesh per kind and variant. A bake of a chunk holds one MultiMeshInstance3D
## for each, named for its kind and, past the first, its variant, of that variant's mesh, holding
## as many instances as the chunk has points of that kind and variant. With a real renderer, which
## keeps a MultiMesh's data, each instance also has a white colour and its point's random and
## dryness as the third and fourth floats of its custom data, to the half precision Compatibility
## keeps them at: `../render_ground.sh` runs it so, since the headless renderer of `../verify.sh`
## keeps none.
extends SceneTree

const CELLS := 8
const TIMEOUT_S := 30.0
const CHUNK := Vector3i(0, 0, 0)
## Each binding's mesh size, by MultiMeshInstance3D name.
const SIZES := {"oak": Vector3(1, 3, 1), "oak1": Vector3(1, 4, 1), "birch": Vector3(1, 5, 1)}

var world: Node
var started_usec := 0

## A scene whose root is a lone MeshInstance3D of a box of `size`.
func _lone(size: Vector3) -> PackedScene:
	var instance := MeshInstance3D.new()
	var box := BoxMesh.new()
	box.size = size
	instance.mesh = box
	var scene := PackedScene.new()
	scene.pack(instance)
	instance.free()
	return scene

func _initialize() -> void:
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://variants.world.ron"
	world.targets = PackedStringArray(["height", "dryness", "trees"])
	world.seed = 4
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.view_radius = 1
	world.collider_radius = -1
	world.ground_stage = "height"
	world.scenes = {"oak": [_lone(SIZES["oak"]), _lone(SIZES["oak1"])], "birch": _lone(SIZES["birch"])}
	world.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	root.add_child(world)
	if not world.start():
		_fail("the stages did not start")
		return
	world.follow(Vector3(1, 0, 1))
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("the trees were not placed")
		return true
	if world.stats()["pending_placements"] > 0 or world.point_sets("trees", CHUNK).is_empty() or not world.ground_chunks().has(CHUNK):
		return false
	return _check()

## The looks of the chunk's points, by the name its MultiMeshInstance3D takes in a bake.
func _expected() -> Dictionary:
	var looks := {}
	for set: Dictionary in world.point_sets("trees", CHUNK):
		for i in set["ids"].size():
			var variant: int = set["variants"][i]
			var name: String = set["kind"] if variant == 0 else "%s%d" % [set["kind"], variant]
			if not looks.has(name):
				looks[name] = []
			looks[name].append(set["looks"][i])
	return looks

func _check() -> bool:
	var expected := _expected()
	var headless := DisplayServer.get_name() == "headless"
	if not (expected.has("oak") and expected.has("oak1") and expected.has("birch")):
		return _fail("the chunk lacks a kind or variant: %s" % expected.keys())
	var scene: PackedScene = world.bake(CHUNK, CHUNK)
	if scene == null:
		return _fail("the bake failed")
	var baked := scene.instantiate()
	var drawn := {}
	for node in baked.find_children("*", "MultiMeshInstance3D", true, false):
		var multimesh: MultiMesh = node.multimesh
		var size: Vector3 = (multimesh.mesh as BoxMesh).size
		if not SIZES.has(String(node.name)) or SIZES[String(node.name)] != size:
			return _fail("%s draws a box of %s" % [node.name, size])
		var looks := []
		for i in multimesh.instance_count:
			if headless:
				looks.append(null)
				continue
			if multimesh.get_instance_color(i) != Color.WHITE:
				return _fail("%s's instance %d is coloured %s" % [node.name, i, multimesh.get_instance_color(i)])
			var custom := multimesh.get_instance_custom_data(i)
			looks.append(Vector2(custom.b, custom.a))
		if not headless:
			looks.sort()
		drawn[String(node.name)] = looks
	baked.free()
	for name: String in expected:
		var looks: Array = expected[name]
		if headless:
			looks = looks.map(func(_look: int) -> Variant: return null)
			if drawn.get(name, []) != looks:
				return _fail("%s draws %d instances for %d points" % [name, drawn.get(name, []).size(), looks.size()])
			continue
		# A look's random and dryness, in 255ths, against the floats drawn at half precision.
		looks = looks.map(func(look: int) -> Vector2: return Vector2((look & 255) / 255.0, ((look >> 8) & 255) / 255.0))
		looks.sort()
		var got: Array = drawn.get(name, [])
		if got.size() != looks.size():
			return _fail("%s draws %d instances for %d points" % [name, got.size(), looks.size()])
		for i in looks.size():
			if absf(got[i].x - looks[i].x) > 1.5 / 255.0 or absf(got[i].y - looks[i].y) > 1.5 / 255.0:
				return _fail("%s draws looks %s for %s" % [name, got, looks])
	if drawn.size() != expected.size():
		return _fail("the bake draws %s for %s" % [drawn.keys(), expected.keys()])
	print("verify_variants: %d oaks of the first variant, %d of the second and %d birches, each drawn with its variant's mesh%s" % [expected["oak"].size(), expected["oak1"].size(), expected["birch"].size(), "" if headless else " and its look"])
	quit(0)
	return true

func _fail(message: String) -> bool:
	printerr("verify_variants: " + message)
	quit(1)
	return true
