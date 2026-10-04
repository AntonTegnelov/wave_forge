## Draws a Volume stage's surface and collides with it, overhangs included.
##
## Run by `../verify.sh` after `verify_ground.gd`. The pack's volume is ground solid below 6.3 cells
## with a cave under all of it from 1.8 to 4.2 cells up. Every chunk around the player gets its
## surface, a ray down from the sky lands on the ground's top, where `ground_height` stands with no
## ground stage, and from inside the cave a ray up meets its ceiling and a ray down its floor, each
## facing into the cave. The ground's top is grass
## and the cave rock, and the drawn surface carries each vertex's colour from `volume_palette`. A
## ball dug where four chunks meet opens the cave to the sky: a ray from above then falls through
## the hole to the cave's floor, which every chunk around it has built again, and `ground_height`
## stands there too. Ore embedded in the
## rock and bound to a scene is placed as nodes, each inside the rock, none in the cave. Pools of an
## Aquifer stage drawn as `fluid_stage` fill the cave's floor up to their levels, water and lava in
## `fluid_palette`'s see-through colours, lava glowing as `fluid_glow` says, and the ray to the cave's floor passes through them, since
## fluid is never collided with. The navigation baked around the player walks on the ground's top
## and on the cave's floor, and the dig bakes the chunks around the hole again. A volume stage that
## is no Volume stage is refused, and so is digging a field.
extends SceneTree

const CELLS := 8
const CELL := Vector3(2, 1.5, 2)
const TIMEOUT_S := 30.0
const PALETTE := [Color(0.2, 0.7, 0.2), Color(0.5, 0.5, 0.5)]
const FLUIDS := [Color(1.0, 0.3, 0.0, 0.9), Color(0.1, 0.3, 0.9, 0.5)]
## Lava glows; water does not.
const GLOWS := [2.0, 0.0]

var world: Node
var started_usec := 0
var settled_frames := 0
var digging := false
var ores := {}
## The fluid's mesh beside the hole before the dig, which the dig makes it build again.
var fluid_before := RID()
## Where four chunks meet, in the middle of the cave's height.
var hole := Vector3(CELLS * CELL.x, 6.3 * CELL.y, CELLS * CELL.z)
## The chunks whose navigation went into the map, since the start or the dig.
var navigable := {}
## How many chunks a path crossed on the ground's top on their navigation_ready, and the chunks one
## did not.
var pathed_on_signal := 0
var missed_on_signal: Array[Vector3i] = []

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
	node.targets = PackedStringArray(["cave", "ore", "water"])
	node.seed = 3
	node.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	node.cell_size = CELL
	node.view_radius = 2
	node.collider_radius = 1
	node.navigation_radius = 1
	node.volume_stage = volume_stage
	node.volume_palette = PackedColorArray(PALETTE)
	node.fluid_stage = "water"
	node.fluid_palette = PackedColorArray(FLUIDS)
	node.fluid_glow = PackedFloat32Array(GLOWS)
	node.scenes = {"ore": _ore_scene()}
	root.add_child(node)
	node.instance_spawned.connect(func(ore: Node3D, chunk: Vector3i, id: int) -> void: ores[[chunk, id]] = ore)
	node.navigation_ready.connect(_on_navigation_ready.bind(node))
	return node

func _process(_delta: float) -> bool:
	if digging:
		return _check_hole()
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("the volume had not arrived: %d surfaces, %d bodies" % [world.volume_chunks().size(), world.collider_chunks().size()])
		return true
	var stats: Dictionary = world.stats()
	if stats["pending_volumes"] > 0 or world.volume_chunks().size() < 9 or world.fluid_chunks().size() < 9 or not world.collider_chunks().has(Vector3i.ZERO):
		return false
	if not world.navigation_chunks().has(Vector3i.ZERO):
		return false
	# The bodies join the physics space this frame, and rays see them from the next; the navigation
	# map takes the new regions in on its next synchronisation.
	settled_frames += 1
	if settled_frames < 30:
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
	var height: float = world.ground_height(Vector3(x, 0, z))
	if is_nan(height) or absf(height - 6.3 * CELL.y) > 0.001:
		_fail("the ground's height over the cave is %.4f, not its top at %.4f" % [height, 6.3 * CELL.y])
		return true
	if not _check_materials() or not _check_ores() or not _check_fluid() or not _check_navigation(x, z):
		return true
	print("verify_volume: %d chunks have their surface; rays meet the ground's top and the cave's ceiling and floor" % world.volume_chunks().size())
	if world.dig("height", hole, 3.0):
		_fail("digging a field was taken")
		return true
	fluid_before = world.fluid_mesh_of(Vector3i(1, 1, 0))
	if not fluid_before.is_valid():
		_fail("the fluid beside the hole is not drawn")
		return true
	if not world.dig("cave", hole, 3.0):
		_fail("the dig was refused")
		return true
	digging = true
	navigable.clear()
	started_usec = Time.get_ticks_usec()
	return false

## A path along the ground's top across the chunk `navigation_ready` names, asked for on the signal
## itself with no wait, since the signal comes once the map has taken the chunk's mesh in.
func _on_navigation_ready(chunk: Vector3i, node: Node) -> void:
	if node != world:
		return
	navigable[chunk] = true
	var corner := Vector3(chunk.x * CELLS * CELL.x, 6.3 * CELL.y, chunk.y * CELLS * CELL.z)
	var from := corner + Vector3(CELL.x, 0, CELLS * CELL.z / 2.0)
	var to := corner + Vector3((CELLS - 1) * CELL.x, 0, CELLS * CELL.z / 2.0)
	var path := NavigationServer3D.map_get_path(root.get_world_3d().navigation_map, from, to, true)
	if path.is_empty() or path[path.size() - 1].distance_to(to) > 0.5:
		missed_on_signal.append(chunk)
	else:
		pathed_on_signal += 1

## The navigation map holds walkable ground on the ground's top and on the cave's floor at (x, z).
func _check_navigation(x: float, z: float) -> bool:
	var map := root.get_world_3d().navigation_map
	for level in [["the ground's top", 6.3], ["the cave's floor", 1.8]]:
		var on: Vector3 = Vector3(x, level[1] * CELL.y, z)
		var nearest := NavigationServer3D.map_get_closest_point(map, on + Vector3.UP * 0.3)
		# A navigation mesh follows what it was baked from as closely as its detail sampling's error.
		if nearest.distance_to(on) > NavigationMesh.new().detail_sample_max_error:
			_fail("the navigation nearest %s is at %s" % [level[0], nearest])
			return false
	if not missed_on_signal.is_empty() or pathed_on_signal == 0:
		_fail("no path across %s on their navigation_ready, of %d chunks" % [missed_on_signal, missed_on_signal.size() + pathed_on_signal])
		return false
	print("verify_volume: the navigation walks on the ground's top and on the cave's floor, and a path crossed each of %d chunks on its navigation_ready" % pathed_on_signal)
	return true

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
	var fluid_after: RID = world.fluid_mesh_of(Vector3i(1, 1, 0))
	var baked_again := navigable.has(Vector3i.ZERO)
	var height: float = world.ground_height(hole)
	if not hit.is_empty() and absf(hit["position"].y - 1.8 * CELL.y) < 0.001 and absf(height - 1.8 * CELL.y) < 0.001 and fluid_after.is_valid() and fluid_after != fluid_before and baked_again:
		if not missed_on_signal.is_empty():
			_fail("no path across %s on their navigation_ready, of %d chunks" % [missed_on_signal, missed_on_signal.size() + pathed_on_signal])
			return true
		print("verify_volume: a ball dug where four chunks meet opens the cave to the sky and builds the fluid and the navigation beside it again, %.1f s after the dig; a path crossed each of %d chunks on its navigation_ready" % [(Time.get_ticks_usec() - started_usec) / 1e6, pathed_on_signal])
		quit(0)
		return true
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("a ray into the hole still lands at %s, the ground's height there is %.4f, the fluid beside it is %s, was %s, and its navigation baked again: %s" % [hit["position"] if not hit.is_empty() else "nothing", height, fluid_after, fluid_before, baked_again])
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

## Every fluid vertex lies in the cave, no higher than the highest pool's level; the pools are
## water and lava, each drawn in its see-through colour; and no fluid has a body.
func _check_fluid() -> bool:
	var seen := [0, 0]
	var vertices := 0
	for chunk in world.fluid_chunks():
		var surface: Dictionary = world.fluid_surface(chunk)
		var positions: PackedVector3Array = surface["positions"]
		var materials: PackedByteArray = surface["materials"]
		for i in positions.size():
			var cells := positions[i].y / CELL.y
			if cells < 1.3 or cells > 3.9:
				_fail("a fluid vertex of %s stands %.2f cells up, outside the cave's pools" % [chunk, cells])
				return false
			seen[materials[i]] += 1
		vertices += positions.size()
		if positions.size() > 0:
			var arrays := RenderingServer.mesh_surface_get_arrays(world.fluid_mesh_of(chunk), 0)
			var colours: PackedColorArray = arrays[Mesh.ARRAY_COLOR]
			var glows: PackedVector2Array = arrays[Mesh.ARRAY_TEX_UV]
			for i in colours.size():
				if absf(colours[i].a - FLUIDS[materials[i]].a) > 1.0 / 255.0:
					_fail("a fluid vertex is drawn with alpha %.3f, its fluid's is %.3f" % [colours[i].a, FLUIDS[materials[i]].a])
					return false
				if absf(glows[i].x - GLOWS[materials[i]]) > 0.01:
					_fail("a fluid vertex glows %.3f, its fluid %.3f" % [glows[i].x, GLOWS[materials[i]]])
					return false
	if seen[0] == 0 or seen[1] == 0:
		_fail("%d lava and %d water vertices" % seen)
		return false
	print("verify_volume: pools fill the cave's floor, %d glowing lava and %d water vertices drawn see-through in %d chunks, none collided with" % [seen[0], seen[1], world.fluid_chunks().size()])
	return true

func _fail(message: String) -> void:
	printerr("verify_volume: " + message)
	quit(1)
