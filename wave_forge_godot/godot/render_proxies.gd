## Draws the city near and far, handing each chunk from its modules to its proxy (developer tool).
##
## Run through `../render_proxies.sh`. The city is generated around the origin with proxies from
## `proxy_distance` on, and drawn as a game would: one MultiMesh per module and chunk, each with its
## chunk's proxy as its `visibility_parent`. From near, inside the distance, only the modules may
## draw; from far, beyond it, only the proxies, one object a chunk. It saves a picture from each and
## prints the objects and primitives drawn.
extends SceneTree

const CELLS := 8
const CELL_SIZE := 2.0
const RADIUS := 2
const DISTANCE := 150.0
const TIMEOUT_S := 120.0
const COLOURS := {
	"building_base": Color(0.76, 0.68, 0.58), "building_door": Color(0.76, 0.68, 0.58),
	"building_floor": Color(0.72, 0.64, 0.56), "building_balcony": Color(0.72, 0.64, 0.56),
	"building_passage": Color(0.72, 0.64, 0.56), "building_arcade": Color(0.72, 0.64, 0.56),
	"roof_pyramid": Color(0.62, 0.25, 0.2), "roof_tower": Color(0.62, 0.25, 0.2),
	"roof_flat": Color(0.5, 0.5, 0.52), "roof_flat_edge": Color(0.5, 0.5, 0.52),
	"roof_flat_corner": Color(0.5, 0.5, 0.52), "roof_flat_strip": Color(0.5, 0.5, 0.52),
	"roof_flat_end": Color(0.5, 0.5, 0.52), "grass": Color(0.35, 0.55, 0.3),
	"plaza": Color(0.7, 0.7, 0.68), "road_straight": Color(0.3, 0.3, 0.32),
	"road_corner": Color(0.3, 0.3, 0.32), "road_t": Color(0.3, 0.3, 0.32),
	"road_cross": Color(0.3, 0.3, 0.32), "road_end": Color(0.3, 0.3, 0.32),
}

var world: Node
var camera: Camera3D
var models := {}
var started_usec := 0
var phase := "arrive"
var waited := 0
var last_count := -1
var steady := 0
var drawn := {}
var results := PackedStringArray()

func _initialize() -> void:
	world = ClassDB.instantiate("WaveForgeWorld")
	world.seed = 11
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = Vector3(CELL_SIZE, CELL_SIZE, CELL_SIZE)
	world.view_radius = RADIUS
	world.collider_radius = -1
	world.proxy_distance = DISTANCE
	world.proxy_colours = COLOURS
	if not world.load_rules(FileAccess.get_file_as_string("res://city.ron")):
		_fail("res://city.ron could not be loaded")
		return
	var layers: Array[PackedInt32Array] = [world.tiles_tagged("street_level")]
	for storey in CELLS - 2:
		layers.append(PackedInt32Array())
	layers.append(world.tiles_named("air"))
	world.set_layer_tiles(layers)
	root.add_child(world)
	world.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	if not world.start():
		_fail("generation did not start")
		return
	world.follow(Vector3(CELLS, 0, CELLS))
	camera = Camera3D.new()
	camera.current = true
	camera.far = 2000
	root.add_child(camera)
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-55, 35, 0)
	root.add_child(sun)
	var environment := WorldEnvironment.new()
	environment.environment = Environment.new()
	environment.environment.background_mode = Environment.BG_COLOR
	environment.environment.background_color = Color(0.62, 0.74, 0.86)
	environment.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.environment.ambient_light_color = Color(0.55, 0.55, 0.6)
	root.add_child(environment)
	started_usec = Time.get_ticks_usec()

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

## Draws every generated chunk's modules, each chunk's under its proxy.
func _draw_modules() -> void:
	var names := PackedStringArray()
	for tile in world.tile_count():
		var module: String = world.tile_name(tile)
		if not names.has(module) and _model(module) != null:
			names.append(module)
	var scenario := root.get_world_3d().scenario
	for chunk: Vector3i in world.generated_chunks():
		var parent: RID = world.proxy_instance(chunk)
		var rids := []
		for set: Dictionary in world.instance_sets(chunk, names):
			var multimesh := RenderingServer.multimesh_create()
			var transforms: PackedFloat32Array = set["transforms"]
			RenderingServer.multimesh_set_mesh(multimesh, _model(set["name"]).get_rid())
			RenderingServer.multimesh_allocate_data(multimesh, transforms.size() / 12, RenderingServer.MULTIMESH_TRANSFORM_3D)
			RenderingServer.multimesh_set_buffer(multimesh, transforms)
			var instance := RenderingServer.instance_create2(multimesh, scenario)
			if parent.is_valid():
				RenderingServer.instance_set_visibility_parent(instance, parent)
			rids.append(instance)
			rids.append(multimesh)
		drawn[chunk] = rids

func _free_modules() -> void:
	for chunk: Vector3i in drawn:
		for rid: RID in drawn[chunk]:
			RenderingServer.free_rid(rid)
	drawn.clear()

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("timed out in %s" % phase)
		return true
	match phase:
		"arrive":
			var count: int = world.generated_chunks().size()
			steady = steady + 1 if count == last_count and count > 0 else 0
			last_count = count
			if steady < 60:
				return false
			_draw_modules()
			camera.look_at_from_position(Vector3(-30, 40, -30), Vector3(CELLS, 0, CELLS))
			phase = "near"
			waited = 0
		"near", "far":
			waited += 1
			if waited < 10:
				return false
			var objects := RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_TOTAL_OBJECTS_IN_FRAME)
			var primitives := RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_TOTAL_PRIMITIVES_IN_FRAME)
			var path := "user://proxies_%s.png" % phase
			root.get_viewport().get_texture().get_image().save_png(path)
			results.append("%s (%.0f units): %d objects, %d primitives" % [phase, camera.global_position.distance_to(Vector3(CELLS, 0, CELLS)), objects, primitives])
			if phase == "near":
				if objects <= world.generated_chunks().size():
					_fail("from near only %d objects were drawn: the proxies, not the modules" % objects)
					return true
				camera.look_at_from_position(Vector3(-120, 150, -120), Vector3(CELLS, 0, CELLS))
				phase = "far"
				waited = 0
				return false
			if objects > world.generated_chunks().size():
				_fail("from far %d objects were drawn for %d chunks: the modules too" % [objects, world.generated_chunks().size()])
				return true
			print("render_proxies: %s; %d chunks; saved user://proxies_near.png and user://proxies_far.png in %s" % ["; ".join(results), world.generated_chunks().size(), ProjectSettings.globalize_path("user://")])
			_free_modules()
			quit(0)
			return true
	return false

func _fail(message: String) -> void:
	printerr("render_proxies: " + message)
	_free_modules()
	quit(1)
