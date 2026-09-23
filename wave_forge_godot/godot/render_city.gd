## Renders a generated city with the module models, to look at (developer tool).
##
## Run through `../render_city.sh`, which builds the extension, exports the models and starts Godot
## with a display. The city is generated around the origin; each chunk is drawn as one MultiMesh
## per module, and a picture is saved once every chunk in view is there.
##
## It draws one of two ways, named after `--` on the command line, and times each chunk on Godot's
## thread:
## - `server` (the default): the extension's `instance_sets` handed to RenderingServer, no nodes.
##   The first chunk's instances are read back and compared with `tile_basis` and `cell_position`,
##   which checks the buffer layout against a renderer that stores it.
## - `nodes`: transforms computed in GDScript, as MultiMeshInstance3D nodes.
extends SceneTree

const CELLS := 8
const CELL_SIZE := 2.0
const RADIUS := 1
const OUT := "user://city.png"

var world: Node
var models := {}
var frames := 0
## The frame on which every chunk in view was first there, or -1 before.
var complete_at := -1
var placed := {}
## How this run draws: "server" or "nodes".
var mode := "server"
## Milliseconds of Godot's thread each chunk took to draw, and the parts of it: asking the
## extension for the chunk's placements, and handing them to the renderer.
var place_ms := PackedFloat64Array()
var ask_ms := PackedFloat64Array()
var hand_ms := PackedFloat64Array()
## The modules that have a model, found once.
var drawn_names := PackedStringArray()
## RenderingServer resources per chunk, freed when it is drawn again.
var server_rids := {}
var checked_layout := false

func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	if not args.is_empty():
		mode = args[0]
	if mode != "server" and mode != "nodes":
		_fail("draw with server or nodes, not %s" % mode)
		return
	world = ClassDB.instantiate("WaveForgeWorld")
	world.seed = 11
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = Vector3(CELL_SIZE, CELL_SIZE, CELL_SIZE)
	world.view_radius = RADIUS
	world.evict_margin = 0
	if not world.load_rules(FileAccess.get_file_as_string("res://city.ron")):
		_fail("res://city.ron could not be loaded")
		return
	# Street level at the bottom, open storeys, and air on top so every building gets its roof.
	var layers: Array[PackedInt32Array] = [world.tiles_tagged("street_level")]
	for storey in CELLS - 2:
		layers.append(PackedInt32Array())
	layers.append(world.tiles_named("air"))
	world.set_layer_tiles(layers)
	root.add_child(world)
	world.chunk_updated.connect(_place)
	world.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	if not world.start():
		_fail("generation did not start")
		return
	world.follow(Vector3.ZERO)
	_scene()

func _scene() -> void:
	var span := float(CELLS) * CELL_SIZE * (2 * RADIUS + 1)
	var camera := Camera3D.new()
	camera.current = true
	camera.fov = 50.0
	root.add_child(camera)
	camera.look_at_from_position(Vector3(span * 0.55, span * 0.45, span * 0.55), Vector3(0, 2, 0))
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-55, 35, 0)
	sun.shadow_enabled = true
	root.add_child(sun)
	var environment := WorldEnvironment.new()
	environment.environment = Environment.new()
	environment.environment.background_mode = Environment.BG_COLOR
	environment.environment.background_color = Color(0.62, 0.74, 0.86)
	environment.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.environment.ambient_light_color = Color(0.55, 0.55, 0.6)
	root.add_child(environment)

## The mesh of a module's model, loaded once; null for a module with nothing to draw.
func _model(module: String) -> Mesh:
	if models.has(module):
		return models[module]
	var doc := GLTFDocument.new()
	var state := GLTFState.new()
	var mesh: Mesh = null
	if doc.append_from_file("res://models/%s.glb" % module, state) == OK:
		var scene := doc.generate_scene(state)
		var found := scene.find_children("*", "MeshInstance3D", true, false)
		if not found.is_empty():
			mesh = (found[0] as MeshInstance3D).mesh
		scene.free()
	models[module] = mesh
	return mesh

func _place(chunk: Vector3i) -> void:
	var started := Time.get_ticks_usec()
	if mode == "server":
		_place_server(chunk)
	else:
		_place_nodes(chunk)
	place_ms.append((Time.get_ticks_usec() - started) / 1000.0)

## The extension's instance sets as RenderingServer multimeshes, without nodes.
func _place_server(chunk: Vector3i) -> void:
	if drawn_names.is_empty():
		for module: String in _modules_with_models():
			drawn_names.append(module)
	var rids := []
	var scenario := root.get_world_3d().scenario
	var asked := Time.get_ticks_usec()
	var sets: Array = world.instance_sets(chunk, drawn_names)
	var handing := Time.get_ticks_usec()
	ask_ms.append((handing - asked) / 1000.0)
	for set: Dictionary in sets:
		var multimesh := RenderingServer.multimesh_create()
		var transforms: PackedFloat32Array = set["transforms"]
		RenderingServer.multimesh_set_mesh(multimesh, _model(set["name"]).get_rid())
		RenderingServer.multimesh_allocate_data(multimesh, transforms.size() / 12, RenderingServer.MULTIMESH_TRANSFORM_3D)
		RenderingServer.multimesh_set_buffer(multimesh, transforms)
		var instance := RenderingServer.instance_create2(multimesh, scenario)
		rids.append(instance)
		rids.append(multimesh)
		if not checked_layout:
			_check_layout(chunk, multimesh, set)
	hand_ms.append((Time.get_ticks_usec() - handing) / 1000.0)
	if checked_layout == false:
		checked_layout = true
		print("render_city: every instance of chunk %s sits where tile_basis and cell_position say" % chunk)
	if server_rids.has(chunk):
		for rid: RID in server_rids[chunk]:
			RenderingServer.free_rid(rid)
	server_rids[chunk] = rids
	placed[chunk] = true

## The multimesh's instances, read back from the renderer, are where the tile catalogue says.
func _check_layout(chunk: Vector3i, multimesh: RID, set: Dictionary) -> void:
	var tiles: PackedInt32Array = world.tiles_at(chunk)
	var ids: PackedInt64Array = set["ids"]
	for i in ids.size():
		var cell := ids[i] & 0xFFFFFFFF
		var tile := tiles[cell]
		var expected := Transform3D(world.tile_basis(tile).scaled(Vector3.ONE * CELL_SIZE), world.cell_position(chunk, cell))
		var got := RenderingServer.multimesh_instance_get_transform(multimesh, i)
		if not got.is_equal_approx(expected):
			_fail("%s at cell %d is at %s, the catalogue says %s" % [set["name"], cell, got, expected])
			return

## The modules that have a model to draw.
func _modules_with_models() -> Array:
	var names := []
	for tile in world.tile_count():
		var module: String = world.tile_name(tile)
		if not names.has(module) and _model(module) != null:
			names.append(module)
	return names

## Draws a chunk from GDScript: every cell's module at the cell's centre, scaled to the cell and
## turned by the tile's rotation, one MultiMeshInstance3D per module.
func _place_nodes(chunk: Vector3i) -> void:
	var tiles: PackedInt32Array = world.tiles_at(chunk)
	var by_module := {}
	for cell in tiles.size():
		var module: String = world.tile_name(tiles[cell])
		if _model(module) == null:
			continue
		var at := Transform3D(world.tile_basis(tiles[cell]).scaled(Vector3.ONE * CELL_SIZE), world.cell_position(chunk, cell))
		by_module.get_or_add(module, []).append(at)
	var holder := Node3D.new()
	for module: String in by_module:
		var multimesh := MultiMesh.new()
		multimesh.transform_format = MultiMesh.TRANSFORM_3D
		multimesh.mesh = _model(module)
		multimesh.instance_count = by_module[module].size()
		for i in multimesh.instance_count:
			multimesh.set_instance_transform(i, by_module[module][i])
		var instance := MultiMeshInstance3D.new()
		instance.multimesh = multimesh
		holder.add_child(instance)
	if placed.has(chunk):
		placed[chunk].queue_free()
	placed[chunk] = holder
	root.add_child(holder)

func _process(_delta: float) -> bool:
	frames += 1
	var wanted := 0
	for x in range(-RADIUS, RADIUS + 1):
		for y in range(-RADIUS, RADIUS + 1):
			if placed.has(Vector3i(x, y, 0)):
				wanted += 1
	if wanted < (2 * RADIUS + 1) * (2 * RADIUS + 1):
		if Time.get_ticks_msec() > 180_000:
			_fail("the view was not generated after three minutes")
			return true
		return false
	# A few frames more, so shadows and the last MultiMeshes are drawn.
	if complete_at < 0:
		complete_at = frames
	if frames < complete_at + 30:
		return false
	var image := root.get_texture().get_image()
	image.save_png(OUT)
	var sorted := place_ms.duplicate()
	sorted.sort()
	print("render_city: %d chunks drawn with %s, Godot's thread per chunk p50 %.2f ms, max %.2f ms; saved %s" % [
		placed.size(), mode, sorted[sorted.size() / 2], sorted[sorted.size() - 1], ProjectSettings.globalize_path(OUT)])
	if mode == "server":
		ask_ms.sort()
		hand_ms.sort()
		print("render_city: of which instance_sets p50 %.3f ms, RenderingServer p50 %.3f ms" % [ask_ms[ask_ms.size() / 2], hand_ms[hand_ms.size() / 2]])
	for rids: Array in server_rids.values():
		for rid: RID in rids:
			RenderingServer.free_rid(rid)
	quit(0)
	return true

func _fail(message: String) -> void:
	printerr("render_city: " + message)
	quit(1)
