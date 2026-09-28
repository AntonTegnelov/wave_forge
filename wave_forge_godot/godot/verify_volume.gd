## Draws a Volume stage's surface and collides with it, overhangs included.
##
## Run by `../verify.sh` after `verify_ground.gd`. The pack's volume is ground solid below 6.3 cells
## with a cave under all of it from 1.8 to 4.2 cells up. Every chunk around the player gets its
## surface, a ray down from the sky lands on the ground's top, and from inside the cave a ray up
## meets its ceiling and a ray down its floor, each facing into the cave. A volume stage that is
## no Volume stage is refused.
extends SceneTree

const CELLS := 8
const CELL := Vector3(2, 1.5, 2)
const TIMEOUT_S := 30.0

var world: Node
var started_usec := 0
var settled_frames := 0

func _initialize() -> void:
	var wrong := _world("height")
	if wrong.start():
		_fail("a volume stage that is a field was taken")
		return
	wrong.queue_free()
	world = _world("cave")
	if not world.start():
		_fail("the stages did not start")
		return
	world.follow(Vector3(5, 0, 5))
	started_usec = Time.get_ticks_usec()

func _world(volume_stage: String) -> Node:
	var node: Node = ClassDB.instantiate("WaveForgeStages")
	node.pack_file = "res://volume.world.ron"
	node.targets = PackedStringArray(["cave"])
	node.seed = 3
	node.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	node.cell_size = CELL
	node.view_radius = 2
	node.collider_radius = 1
	node.volume_stage = volume_stage
	root.add_child(node)
	return node

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("the volume had not arrived: %d surfaces, %d bodies" % [world.volume_chunks().size(), world.collider_chunks().size()])
		return true
	var stats: Dictionary = world.stats()
	if stats["pending_volumes"] > 0 or world.volume_chunks().size() < 9 or not world.collider_chunks().has(Vector3i.ZERO):
		return false
	# The bodies join the physics space this frame; rays see them from the next.
	settled_frames += 1
	if settled_frames < 3:
		return false
	return _check()

func _check() -> bool:
	var space := root.get_world_3d().direct_space_state
	var x := 5.0
	var z := 7.0
	var rays := [
		["the ground's top from the sky", 50.0, -1.0, 6.3],
		["the cave's ceiling from inside it", 3.0 * CELL.y, 1.0, 4.2],
		["the cave's floor from inside it", 3.0 * CELL.y, -1.0, 1.8],
	]
	for ray in rays:
		var from := Vector3(x, ray[1], z)
		var query := PhysicsRayQueryParameters3D.create(from, from + Vector3.UP * ray[2] * 100.0)
		var hit := space.intersect_ray(query)
		if hit.is_empty():
			_fail("a ray to %s hit nothing" % ray[0])
			return true
		var expected: float = ray[3] * CELL.y
		if absf(hit["position"].y - expected) > 0.001:
			_fail("a ray to %s hit at height %.4f, expected %.4f" % [ray[0], hit["position"].y, expected])
			return true
		if hit["normal"].dot(Vector3.UP * -ray[2]) < 0.99:
			_fail("%s faces %s, not back along the ray" % [ray[0], hit["normal"]])
			return true
	print("verify_volume: %d chunks have their surface; rays meet the ground's top and the cave's ceiling and floor" % world.volume_chunks().size())
	quit(0)
	return true

func _fail(message: String) -> void:
	printerr("verify_volume: " + message)
	quit(1)
