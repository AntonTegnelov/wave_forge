## Draws the near ground and the far ground beyond it and checks the far ground is drawn exactly
## where some near ground is missing, and coloured by its own surface.
##
## Run by `../verify.sh` after `verify_ground.gd`. The far ground check's pack (`far.world.ron`) has
## the same ground at full detail and at a coarse scale of 8; the near ground reaches 7 chunks from
## the focus, one less than the fields it is built from, so it covers the coarse chunk at the
## origin whole, which gets no far ground, and part of the ring around it, which does. The far
## ground's surface (`far_surface`) colours each vertex with the palette's colour of the near
## surface's category of the same name, a far category without one a colour of its own. A far
## ground material stage at another scale than the far ground's is warned of and refused.
extends SceneTree

const CELLS := 8
const RADIUS := 8
const SCALE := 8
const FAR_RADIUS := 24
const TIMEOUT_S := 60.0
const PALETTE := [Color(0.3, 0.6, 0.2), Color(0.6, 0.55, 0.5)]

var world: Node
var started_usec := 0

func _initialize() -> void:
	var wrong: Node = ClassDB.instantiate("WaveForgeStages")
	wrong.pack_file = "res://far.world.ron"
	wrong.targets = PackedStringArray(["height", "far"])
	wrong.ground_stage = "height"
	wrong.far_ground_stage = "far"
	wrong.far_ground_material_stage = "surface"
	root.add_child(wrong)
	var warned := " ".join(wrong.configuration_warnings())
	if not warned.contains("far_ground_material_stage \"surface\" is no Rules stage of the pack at far_ground_stage's scale"):
		_fail("a far ground material stage at scale 1 does not warn: %s" % warned)
		return
	if wrong.start():
		_fail("the node started with a far ground material stage at scale 1")
		return
	wrong.free()
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://far.world.ron"
	world.targets = PackedStringArray(["height", "far", "surface", "far_surface"])
	var radii: Dictionary[StringName, int] = {&"far": FAR_RADIUS, &"far_surface": FAR_RADIUS}
	world.target_radii = radii
	world.seed = 3
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.view_radius = RADIUS
	world.collider_radius = -1
	world.ground_stage = "height"
	world.far_ground_stage = "far"
	world.ground_material_stage = "surface"
	world.ground_palette = PackedColorArray(PALETTE)
	world.far_ground_material_stage = "far_surface"
	root.add_child(world)
	if not world.start():
		_fail("the stages did not start")
		return
	world.follow(Vector3(1, 0, 1))
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("the ground had not arrived: %d near grounds, %d far" % [world.ground_chunks().size(), world.far_ground_chunks().size()])
		return true
	var stats: Dictionary = world.stats()
	var side := 2 * (RADIUS - 1) + 1
	# Every far chunk within its radius whose fields have arrived around it: those of the coarse
	# chunks two from the edge of the radius.
	if stats["pending_grounds"] > 0 or stats["pending_far_grounds"] > 0 \
			or world.ground_chunks().size() < side * side or world.far_ground_chunks().size() < 12:
		return false
	var near := {}
	for chunk: Vector3i in world.ground_chunks():
		near[chunk] = true
	var far: Array[Vector3i] = world.far_ground_chunks()
	for chunk in far:
		if _covered(chunk, near):
			_fail("the coarse chunk %s, whose every chunk has near ground, has far ground" % chunk)
			return true
	for chunk in [Vector3i(1, 0, 0), Vector3i(-1, 0, 0), Vector3i(0, 1, 0), Vector3i(0, -1, 0)]:
		if not far.has(chunk):
			_fail("the coarse chunk %s, partly beyond the near ground, has no far ground" % chunk)
			return true
	var coloured := 0
	for chunk in far:
		var wrong := _wrong_colour(chunk)
		if not wrong.is_empty():
			_fail(wrong)
			return true
		coloured += world.far_ground_surface(chunk)["colours"].size()
	print("verify_far: %d near grounds, %d far grounds, none on a coarse chunk the near ground covers; %d far vertices coloured by their surface" % [near.size(), far.size(), coloured])
	quit(0)
	return true

## Why a vertex of the far ground of `chunk` is not in the colour of its column's far surface; empty
## if every one is. A far category the near surface names takes that category's palette colour;
## one it does not takes a colour that is in no palette.
func _wrong_colour(chunk: Vector3i) -> String:
	var surface: Dictionary = world.far_ground_surface(chunk)
	var positions: PackedVector3Array = surface["positions"]
	var colours: PackedColorArray = surface["colours"]
	if colours.size() != positions.size() or positions.is_empty():
		return "the far ground of %s has %d colours for %d vertices" % [chunk, colours.size(), positions.size()]
	var categories: PackedByteArray = world.categories("far_surface", chunk)
	var far_names: PackedStringArray = world.category_names("far_surface")
	var near_names: PackedStringArray = world.category_names("surface")
	var column_size: Vector3 = world.cell_size * SCALE
	for i in positions.size():
		var column := Vector2i(clampi(floori(positions[i].x / column_size.x), 0, CELLS - 1), clampi(floori(positions[i].z / column_size.z), 0, CELLS - 1))
		var name := far_names[categories[column.y * CELLS + column.x]]
		var near := near_names.find(name)
		if near >= 0 and not colours[i].is_equal_approx(PALETTE[near]):
			return "a far vertex of %s over %s is %s, not %s's %s" % [chunk, name, colours[i], name, PALETTE[near]]
		if near < 0 and PALETTE.any(func(colour: Color) -> bool: return colours[i].is_equal_approx(colour)):
			return "a far vertex of %s over %s, which the near surface does not name, is a palette colour" % [chunk, name]
	return ""

## Whether every chunk of the WFC lattice the coarse `chunk` covers has near ground.
func _covered(chunk: Vector3i, near: Dictionary) -> bool:
	for y in SCALE:
		for x in SCALE:
			if not near.has(Vector3i(chunk.x * SCALE + x, chunk.y * SCALE + y, 0)):
				return false
	return true

func _fail(message: String) -> void:
	printerr("verify_far: " + message)
	quit(1)
