## Binds scenes to a cave level's rooms and to the enemies spawned in them.
##
## Run by `../verify.sh` after `verify_scenes.gd`. Caverns are a scene with a child, so each is
## placed as a node, once, by the chunk its id names, standing at its room's transform; grunts are
## a lone mesh, so they are drawn as MultiMeshes, one instance per spawned point.
extends SceneTree

const CELLS := 8
const RADIUS := 4
const CELL := Vector3(2, 1, 2)
const TIMEOUT_S := 30.0

var world: Node
var ready := {}
var spawned := {}
var started_usec := 0

func _initialize() -> void:
	var grunt := MeshInstance3D.new()
	grunt.mesh = BoxMesh.new()
	var grunts := PackedScene.new()
	grunts.pack(grunt)
	grunt.free()
	var top := Node3D.new()
	var child := MeshInstance3D.new()
	child.mesh = BoxMesh.new()
	top.add_child(child)
	child.owner = top
	var caverns := PackedScene.new()
	caverns.pack(top)
	top.free()
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://cave_rooms.world.ron"
	world.targets = PackedStringArray(["level", "enemies"])
	world.seed = 5
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = CELL
	world.view_radius = RADIUS
	world.collider_radius = -1
	world.scenes = {"cavern": caverns, "grunt": grunts}
	root.add_child(world)
	world.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	world.stage_ready.connect(func(stage: String, chunk: Vector3i) -> void: ready[[stage, chunk]] = true)
	world.instance_spawned.connect(func(node: Node3D, chunk: Vector3i, id: int) -> void:
		if spawned.has([chunk, id]):
			_fail("%s %d was spawned twice" % [chunk, id])
		spawned[[chunk, id]] = node)
	if not world.start():
		_fail("the stages did not start")
		return
	world.follow(Vector3(1, 0, 1))
	started_usec = Time.get_ticks_usec()

func _chunk_of(at: Vector3) -> Vector3i:
	return Vector3i(floori(at.x / CELL.x / CELLS), floori(at.z / CELL.z / CELLS), 0)

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("the cave did not arrive")
		return true
	var chunks: Array[Vector3i] = []
	for y in range(-RADIUS, RADIUS + 1):
		for x in range(-RADIUS, RADIUS + 1):
			chunks.append(Vector3i(x, y, 0))
	for chunk in chunks:
		for stage in ["level", "enemies"]:
			if not ready.has([stage, chunk]):
				return false
	var stats: Dictionary = world.stats()
	if stats["pending_signals"] > 0 or stats["pending_placements"] > 0:
		return false
	var rooms := {}
	var grunts := 0
	for key in ready:
		if key[0] == "level":
			for stamp: Dictionary in world.stamps("level", key[1]):
				var transform: Transform3D = stamp["transform"]
				if _chunk_of(transform.origin) == key[1]:
					rooms[[key[1], stamp["id"]]] = transform
	for chunk in chunks:
		for set: Dictionary in world.point_sets("enemies", chunk):
			grunts += set["ids"].size()
	if rooms.size() < 4 or grunts < 16:
		_fail("only %d rooms and %d grunts to place" % [rooms.size(), grunts])
		return true
	if spawned.size() != rooms.size() or stats["placed_nodes"] != rooms.size():
		_fail("%d nodes spawned and %d placed for %d rooms" % [spawned.size(), stats["placed_nodes"], rooms.size()])
		return true
	for key in rooms:
		if not spawned.has(key):
			_fail("the room %s has no node" % [key])
			return true
		var node: Node3D = spawned[key]
		if not node.global_transform.is_equal_approx(rooms[key]):
			_fail("the node of %s stands at %s, not %s" % [key, node.global_transform, rooms[key]])
			return true
	if stats["placed_instances"] != grunts:
		_fail("%d grunts drawn for %d spawned" % [stats["placed_instances"], grunts])
		return true
	print("verify_cave_scenes: %d rooms placed as nodes where the cave planned them, %d grunts drawn as MultiMeshes" % [rooms.size(), grunts])
	quit(0)
	return true

func _fail(message: String) -> void:
	printerr("verify_cave_scenes: " + message)
	quit(1)
