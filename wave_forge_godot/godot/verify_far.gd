## Draws the near ground and the far ground beyond it and checks the far ground is drawn exactly
## where some near ground is missing, and coloured by its own surface.
##
## Run by `../verify.sh` after `verify_ground.gd`. The far ground check's pack (`far.world.ron`) has
## the same ground at full detail and at a coarse scale of 8; the near ground fills the view, 7
## chunks from the focus, so it covers the coarse chunk at the origin whole, which gets no far
## ground, and part of the ring around it, which does. The far
## ground's surface (`far_surface`) colours each vertex with the palette's colour of the near
## surface's category of the same name, a far category without one a colour of its own. With the
## near ground drawn by a ground material, each group of `GROUP` by `GROUP` far chunks is drawn with
## a copy of it of its own, holding the category of every vertex of their grids of coarse columns
## side by side, a coarse column's width as its cell and its first vertex at a fine cell's centre,
## and the far categories' palette. A far
## ground material stage at another scale than the far ground's, or one generated nearer than the
## far ground, is warned of and refused.
extends SceneTree

const CELLS := 8
const RADIUS := 7
const SCALE := 8
const FAR_RADIUS := 24
const TIMEOUT_S := 60.0
const PALETTE := [Color(0.3, 0.6, 0.2), Color(0.6, 0.55, 0.5)]
## How many coarse chunks along each side one far mesh and material draw (`FAR_GROUP` in
## stages_node.rs).
const GROUP := 4

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
	wrong.far_ground_material_stage = "far_surface"
	wrong.targets = PackedStringArray(["height", "far", "far_surface"])
	var nearer: Dictionary[StringName, int] = {&"far": FAR_RADIUS}
	wrong.target_radii = nearer
	warned = " ".join(wrong.configuration_warnings())
	if not warned.contains("far_ground_material_stage \"far_surface\" needs far_ground_stage's radius in target_radii"):
		_fail("a far ground material stage nearer than the far ground does not warn: %s" % warned)
		return
	if wrong.start():
		_fail("the node started with a far ground material stage nearer than the far ground")
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
	var side := 2 * RADIUS + 1
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
		var wrong := _wrong_colour(chunk) + _wrong_material(chunk)
		if not wrong.is_empty():
			_fail(wrong)
			return true
		coloured += world.far_ground_surface(chunk)["colours"].size()
	print("verify_far: %d near grounds, %d far grounds, none on a coarse chunk the near ground covers; %d far vertices coloured by their surface, each group of far chunks drawn with its own copy of the ground material holding their categories" % [near.size(), far.size(), coloured])
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

## Why the far ground of `chunk` is not drawn with its group's copy of the ground material holding
## the category of every vertex of its grid where the chunk lies in the group's, a coarse column
## apart from the centre of the first fine cell, in the far categories' palette; empty if it is.
func _wrong_material(chunk: Vector3i) -> String:
	var material: ShaderMaterial = world.far_ground_material_of(chunk)
	if material == null:
		return "the far ground of %s has no ground material of its own" % chunk
	var categories: PackedByteArray = world.categories("far_surface", chunk)
	var ids: Image = material.get_shader_parameter("wave_forge_materials").get_image()
	# One far square per chunk of the lattice: a vertex at every one's first column, SCALE + 1 a side
	# for a coarse chunk, its group's side by side and sharing their edges.
	if ids.get_size() != Vector2i(GROUP * SCALE + 1, GROUP * SCALE + 1):
		return "the far material of %s holds %s categories" % [chunk, ids.get_size()]
	var at := Vector2i(posmod(chunk.x, GROUP), posmod(chunk.y, GROUP)) * SCALE
	# Along the chunk's far edges the next chunk's first vertex is held, and its own column only
	# where the next chunk has no far ground.
	for j in SCALE:
		for i in SCALE:
			var expected := categories[mini(j, CELLS - 1) * CELLS + mini(i, CELLS - 1)]
			var held := roundi(ids.get_pixel(at.x + i, at.y + j).r * 255.0)
			if held != expected:
				return "vertex (%d, %d) of the far material of %s holds category %d, not %d" % [i, j, chunk, held, expected]
	var column_size: Vector3 = world.cell_size * SCALE
	var cell: Vector2 = material.get_shader_parameter("wave_forge_cell")
	var first: Vector2 = material.get_shader_parameter("wave_forge_first")
	if not cell.is_equal_approx(Vector2(column_size.x, column_size.z)) or not first.is_equal_approx(Vector2(0.5, 0.5) / SCALE):
		return "the far material of %s has cells of %s with its first vertex at %s" % [chunk, cell, first]
	var palette: Image = material.get_shader_parameter("wave_forge_palette").get_image()
	var far_names: PackedStringArray = world.category_names("far_surface")
	var near_names: PackedStringArray = world.category_names("surface")
	for index in far_names.size():
		var near := near_names.find(far_names[index])
		if near < 0:
			continue
		var held := palette.get_pixel(index, 0)
		var colour: Color = PALETTE[near]
		# The palette holds a byte a channel.
		if maxf(maxf(absf(held.r - colour.r), absf(held.g - colour.g)), absf(held.b - colour.b)) > 1.0 / 255.0:
			return "the far palette of %s gives %s %s, not %s" % [chunk, far_names[index], held, colour]
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
