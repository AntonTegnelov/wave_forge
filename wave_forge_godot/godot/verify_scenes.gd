## Binds scenes to what a pack places and checks each stands where its stage put it.
##
## Run by `../verify.sh` after `verify_assemble.gd`. Trees are a lone mesh, so they are drawn as
## MultiMeshes; houses are a scene given as a PackedScene and streets one given by a path, loaded
## on Godot's loader threads, so both are placed as nodes. Every house and street piece the stage
## grew gets exactly one node, named by `instance_spawned`, standing at the piece's transform, and
## every tree is drawn once. With a promotion radius that puts some pieces inside and some outside,
## those outside are freed as nodes and drawn as MultiMeshes of their mesh instead, and become
## nodes again when promotion is turned off. Moving away frees them all.
extends SceneTree

const CELLS := 8
const RADIUS := 4
const CELL := Vector3(2, 1, 2)
const TIMEOUT_S := 30.0
const STREET_PATH := "user://wave_forge_verify_street.tscn"

var world: Node
var ready := {}
var spawned := {}
var started_usec := 0
var phase := "arrive"
var pieces := {}
var trees := 0
var radius := 0

func _initialize() -> void:
	var tree := MeshInstance3D.new()
	tree.mesh = BoxMesh.new()
	var trees := PackedScene.new()
	trees.pack(tree)
	tree.free()
	var houses := _scene_with_a_child()
	if ResourceSaver.save(_scene_with_a_child(), STREET_PATH) != OK:
		_fail("the street scene could not be saved")
		return
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://scenes.world.ron"
	world.targets = PackedStringArray(["village", "level", "trees"])
	world.seed = 21
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = CELL
	world.view_radius = RADIUS
	world.collider_radius = -1
	world.scenes = {"tree": trees, "house": houses, "street": STREET_PATH}
	root.add_child(world)
	world.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	world.stage_ready.connect(func(stage: String, chunk: Vector3i) -> void: ready[[stage, chunk]] = true)
	world.instance_spawned.connect(func(node: Node3D, chunk: Vector3i, id: int) -> void:
		if spawned.has([chunk, id]) and is_instance_valid(spawned[[chunk, id]]):
			_fail("%s %d was spawned twice" % [chunk, id])
		spawned[[chunk, id]] = node)
	if not world.start():
		_fail("the stages did not start")
		return
	world.follow(Vector3(1, 0, 1))
	started_usec = Time.get_ticks_usec()

## A Node3D with a mesh child, packed, so it is placed as nodes.
func _scene_with_a_child() -> PackedScene:
	var top := Node3D.new()
	var child := MeshInstance3D.new()
	child.mesh = BoxMesh.new()
	top.add_child(child)
	child.owner = top
	var scene := PackedScene.new()
	scene.pack(top)
	top.free()
	return scene

func _area() -> Array[Vector3i]:
	var chunks: Array[Vector3i] = []
	for y in range(-RADIUS, RADIUS + 1):
		for x in range(-RADIUS, RADIUS + 1):
			chunks.append(Vector3i(x, y, 0))
	return chunks

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("the %s phase timed out" % phase)
		return true
	var stats: Dictionary = world.stats()
	match phase:
		"arrive":
			for chunk in _area():
				for stage in ["village", "trees"]:
					if not ready.has([stage, chunk]):
						return false
			if stats["pending_signals"] > 0 or stats["pending_placements"] > 0:
				return false
			return _check(stats)
		"promote":
			if stats["pending_placements"] > 0:
				return false
			return _check_promoted(stats)
		"restore":
			if stats["pending_placements"] > 0:
				return false
			for key in pieces:
				if not is_instance_valid(spawned[key]):
					return false
			if stats["placed_nodes"] != pieces.size() or stats["placed_instances"] != trees:
				return false
			print("verify_scenes: without promotion, every piece is a node again")
			world.follow(Vector3(2000, 0, 2000))
			phase = "leave"
			started_usec = Time.get_ticks_usec()
			return false
		"leave":
			for node in spawned.values():
				if is_instance_valid(node):
					return false
			if stats["placed_nodes"] > 0 or stats["placed_instances"] > 0:
				return false
			print("verify_scenes: moving away freed every node and MultiMesh")
			quit(0)
			return true
	return false

func _chunk_of(at: Vector3) -> Vector3i:
	return Vector3i(floori(at.x / CELL.x / CELLS), floori(at.z / CELL.z / CELLS), 0)

func _check(stats: Dictionary) -> bool:
	var expected := {}
	# Every chunk the stages hold places its pieces, those generated for a stage that reads them
	# beyond the view included.
	var held: Array[Vector3i] = []
	for key in ready:
		if key[0] == "village":
			held.append(key[1])
	for chunk in held:
		for stamp: Dictionary in world.stamps("village", chunk):
			var transform: Transform3D = stamp["transform"]
			if stamp["piece"] in ["house", "street"] and _chunk_of(transform.origin) == chunk:
				expected[[chunk, stamp["id"]]] = transform
	for chunk in _area():
		for set: Dictionary in world.point_sets("trees", chunk):
			trees += set["ids"].size()
	if expected.size() < 5 or trees < 20:
		_fail("only %d pieces and %d trees to place" % [expected.size(), trees])
		return true
	if spawned.size() != expected.size() or stats["placed_nodes"] != expected.size():
		_fail("%d nodes spawned and %d placed for %d pieces" % [spawned.size(), stats["placed_nodes"], expected.size()])
		return true
	for key in expected:
		if not spawned.has(key):
			_fail("the piece %s has no node" % [key])
			return true
		var node: Node3D = spawned[key]
		if not node.global_transform.is_equal_approx(expected[key]):
			_fail("the node of %s stands at %s, not %s" % [key, node.global_transform, expected[key]])
			return true
	if stats["placed_instances"] != trees:
		_fail("%d trees drawn for %d points" % [stats["placed_instances"], trees])
		return true
	print("verify_scenes: %d pieces placed as nodes where the stage grew them, %d trees drawn as MultiMeshes; placing's slowest frame %.3f ms" % [expected.size(), trees, stats["slowest_frame_placements_ms"]])
	pieces = expected
	var distances: Array[int] = []
	for key in pieces:
		distances.append(maxi(absi(key[0].x), absi(key[0].y)))
	distances.sort()
	radius = distances[distances.size() / 2]
	world.promotion_radius = radius
	phase = "promote"
	started_usec = Time.get_ticks_usec()
	return false

## Nodes only within the promotion radius of the followed chunk, every other piece drawn by its
## mesh.
func _check_promoted(stats: Dictionary) -> bool:
	var near := 0
	for key in pieces:
		var chunk: Vector3i = key[0]
		var node = spawned[key]
		var inside: bool = maxi(absi(chunk.x), absi(chunk.y)) <= radius
		if inside:
			near += 1
		if inside != is_instance_valid(node):
			return false
	if stats["placed_nodes"] != near or stats["placed_instances"] != trees + pieces.size() - near:
		return false
	if near == 0 or near == pieces.size():
		_fail("a promotion radius of %d puts %d of %d pieces inside" % [radius, near, pieces.size()])
		return true
	print("verify_scenes: with a promotion radius of %d, %d pieces stay nodes and %d are drawn as MultiMeshes" % [radius, near, pieces.size() - near])
	world.promotion_radius = -1
	phase = "restore"
	started_usec = Time.get_ticks_usec()
	return false

func _fail(message: String) -> void:
	printerr("verify_scenes: " + message)
	quit(1)
