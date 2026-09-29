## Draws a Volume stage's surface and collides with it, overhangs included.
##
## Run by `../verify.sh` after `verify_ground.gd`. The pack's volume is ground solid below 6.3 cells
## with a cave under all of it from 1.8 to 4.2 cells up. Every chunk around the player gets its
## surface, a ray down from the sky lands on the ground's top, and from inside the cave a ray up
## meets its ceiling and a ray down its floor, each facing into the cave. The ground's top is grass
## and the cave rock, and the drawn surface carries each vertex's colour from `volume_palette`. A
## ball dug where four chunks meet opens the cave to the sky: a ray from above then falls through
## the hole to the cave's floor, which every chunk around it has built again. Ore embedded in the
## rock and bound to a scene is placed as nodes, each inside the rock, none in the cave. A volume
## stage that is no Volume stage is refused, and so is digging a field.
extends SceneTree

const CELLS := 8
const CELL := Vector3(2, 1.5, 2)
const TIMEOUT_S := 30.0
const PALETTE := [Color(0.2, 0.7, 0.2), Color(0.5, 0.5, 0.5)]

var world: Node
var started_usec := 0
var settled_frames := 0
var digging := false
var ores := {}
## Where four chunks meet, in the middle of the cave's height.
var hole := Vector3(CELLS * CELL.x, 6.3 * CELL.y, CELLS * CELL.z)

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
	node.targets = PackedStringArray(["cave", "ore"])
	node.seed = 3
	node.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	node.cell_size = CELL
	node.view_radius = 2
	node.collider_radius = 1
	node.volume_stage = volume_stage
	node.volume_palette = PackedColorArray(PALETTE)
	node.scenes = {"ore": _ore_scene()}
	root.add_child(node)
	node.instance_spawned.connect(func(ore: Node3D, chunk: Vector3i, id: int) -> void: ores[[chunk, id]] = ore)
	return node

func _process(_delta: float) -> bool:
	if digging:
		return _check_hole()
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
	if not _check_materials() or not _check_ores():
		return true
	print("verify_volume: %d chunks have their surface; rays meet the ground's top and the cave's ceiling and floor" % world.volume_chunks().size())
	if world.dig("height", hole, 3.0):
		_fail("digging a field was taken")
		return true
	if not world.dig("cave", hole, 3.0):
		_fail("the dig was refused")
		return true
	digging = true
	started_usec = Time.get_ticks_usec()
	return false

## A Node3D with a mesh child, packed, so each ore is placed as a node.
func _ore_scene() -> PackedScene:
	var top := Node3D.new()
	var child := MeshInstance3D.new()
	child.mesh = BoxMesh.new()
	top.add_child(child)
	child.owner = top
	var scene := PackedScene.new()
	scene.pack(top)
	top.free()
	return scene

## Every ore node stands inside the rock, under the ground's top and above or below the cave.
func _check_ores() -> bool:
	var inside := 0
	for key in ores:
		var ore: Node3D = ores[key]
		if not is_instance_valid(ore):
			continue
		var cells := ore.global_position.y / CELL.y
		if cells >= 6.3 or (cells > 1.8 and cells < 4.2):
			_fail("an ore stands %.2f cells up, outside the rock" % cells)
			return false
		inside += 1
	if inside < 10:
		_fail("only %d ores were placed" % inside)
		return false
	print("verify_volume: %d ores embedded in the rock are placed as nodes, none in the cave or the sky" % inside)
	return true

## Waits for the hole: a ray down from the sky at its centre lands on the cave's floor.
func _check_hole() -> bool:
	var from := Vector3(hole.x, 50.0, hole.z)
	var query := PhysicsRayQueryParameters3D.create(from, from + Vector3.DOWN * 100.0)
	var hit := root.get_world_3d().direct_space_state.intersect_ray(query)
	if not hit.is_empty() and absf(hit["position"].y - 1.8 * CELL.y) < 0.001:
		print("verify_volume: a ball dug where four chunks meet opens the cave to the sky, %.1f s after the dig" % ((Time.get_ticks_usec() - started_usec) / 1e6))
		quit(0)
		return true
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("a ray into the hole still lands at %s" % (hit["position"] if not hit.is_empty() else "nothing"))
		return true
	return false

## Every vertex on the ground's top is grass and every one in the cave rock, and the drawn mesh has
## each vertex in its material's colour.
func _check_materials() -> bool:
	var surface: Dictionary = world.volume_surface(Vector3i.ZERO)
	var positions: PackedVector3Array = surface["positions"]
	var materials: PackedByteArray = surface["materials"]
	if materials.size() != positions.size():
		_fail("%d materials for %d vertices" % [materials.size(), positions.size()])
		return false
	var grass := 0
	for i in positions.size():
		var expected := 0 if positions[i].y > 5.0 * CELL.y else 1
		if materials[i] != expected:
			_fail("the vertex at %s is material %d, expected %d" % [positions[i], materials[i], expected])
			return false
		grass += 1 - expected
	var arrays := RenderingServer.mesh_surface_get_arrays(world.volume_mesh_of(Vector3i.ZERO), 0)
	var colours: PackedColorArray = arrays[Mesh.ARRAY_COLOR]
	# A mesh keeps its colours in 8 bits a channel.
	for i in colours.size():
		var wanted: Color = PALETTE[materials[i]]
		if absf(colours[i].r - wanted.r) > 1.0 / 255.0 or absf(colours[i].g - wanted.g) > 1.0 / 255.0 or absf(colours[i].b - wanted.b) > 1.0 / 255.0:
			_fail("the drawn vertex %d is %s, its material's colour is %s" % [i, colours[i], PALETTE[materials[i]]])
			return false
	if grass == 0 or grass == positions.size() or colours.size() != positions.size():
		_fail("%d of %d vertices are grass and %d are coloured" % [grass, positions.size(), colours.size()])
		return false
	print("verify_volume: %d of the origin's %d vertices are grass on top, the rest rock in the cave, each drawn in its colour" % [grass, positions.size()])
	return true

func _fail(message: String) -> void:
	printerr("verify_volume: " + message)
	quit(1)
