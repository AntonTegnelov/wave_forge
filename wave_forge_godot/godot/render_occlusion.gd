## Measures occluders in the city: what they cull and what rebuilding them costs (developer tool).
##
## Run through `../render_occlusion.sh`, from `above` the roofs and on a `street`, named after `--`
## on the command line. The city is generated around the origin and drawn as one
## MultiMesh per module and chunk, and the node gives the chunks near the camera occluders of their
## solid cells. With vsync off and the camera still, it times frames and counts the objects drawn in
## three phases, with Godot's own CPU time for the viewport and for setting up the frame: occlusion
## culling off; on, the occluders standing; and on while three chunks'
## occluders are freed and built again every frame, the most the node builds in a frame while the
## player streams through the city, which is what rebuilding them costs Godot's culler.
extends SceneTree

const CELLS := 8
const CELL_SIZE := 2.0
const VIEW_RADIUS := 3
const OCCLUDER_RADIUS := 2
const LOAD_TIMEOUT_S := 120.0
const FRAMES := 400
## Chunks whose occluders are rebuilt every frame of the last phase: the node's budget.
const CHURN := 3
const PHASES := ["off", "on", "churn"]

var world: Node
var camera: Camera3D
## Where the camera looks from: `above` the roofs, or on a `street` looking down it.
var view := "above"
var models := {}
var drawn_names := PackedStringArray()
var server_rids := {}
var started_usec := 0
var ready := false
var phase_at := 0
var frame := -30
var last_usec := 0
var frame_ms := PackedFloat64Array()
## Godot's CPU time rendering the viewport each frame, culling included, and setting up the frame.
var render_ms := PackedFloat64Array()
var setup_ms := PackedFloat64Array()
var objects := 0
var results := PackedStringArray()
## The chunks with occluders, and the occluders the churn phase builds itself in their stead.
var churned: Array[Vector3i] = []
var churn_nodes := {}
var churn_next := 0

func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	if not args.is_empty():
		view = args[0]
	if view != "above" and view != "street":
		_fail("look from above or street, not %s" % view)
		return
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0
	world = ClassDB.instantiate("WaveForgeWorld")
	world.seed = 11
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = Vector3(CELL_SIZE, CELL_SIZE, CELL_SIZE)
	world.view_radius = VIEW_RADIUS
	world.collider_radius = -1
	world.occluder_radius = OCCLUDER_RADIUS
	if not world.load_rules(FileAccess.get_file_as_string("res://city.ron")):
		_fail("res://city.ron could not be loaded")
		return
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
	world.follow(Vector3(CELLS, 0, CELLS))
	camera = Camera3D.new()
	camera.current = true
	camera.far = 400
	root.add_child(camera)
	camera.look_at_from_position(Vector3(-40, 22, CELLS), Vector3(60, 0, CELLS))
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-55, 35, 0)
	root.add_child(sun)
	var environment := WorldEnvironment.new()
	environment.environment = Environment.new()
	environment.environment.background_mode = Environment.BG_COLOR
	environment.environment.background_color = Color(0.62, 0.74, 0.86)
	root.add_child(environment)
	started_usec = Time.get_ticks_usec()

## Stands the camera on the first straight road of the chunk at the origin, at eye height, looking
## down the road the longer way.
func _stand_on_a_street() -> bool:
	var tiles: PackedInt32Array = world.tiles_at(Vector3i.ZERO)
	for cell in CELLS * CELLS:
		if world.tile_name(tiles[cell]) != "road_straight":
			continue
		var at: Vector3 = world.cell_position(Vector3i.ZERO, cell)
		var along: Vector3 = world.tile_basis(tiles[cell]) * Vector3.RIGHT
		var eye := Vector3(at.x, 1.6, at.z)
		# Of the road's two directions, the one with more city ahead.
		if along.dot(Vector3(CELLS, 0, CELLS) - eye) < 0:
			along = -along
		camera.look_at_from_position(eye, eye + along * 10)
		return true
	return false

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
	if drawn_names.is_empty():
		for tile in world.tile_count():
			var module: String = world.tile_name(tile)
			if not drawn_names.has(module) and _model(module) != null:
				drawn_names.append(module)
	_free_chunk(chunk)
	var rids := []
	var scenario := root.get_world_3d().scenario
	for set: Dictionary in world.instance_sets(chunk, drawn_names):
		var multimesh := RenderingServer.multimesh_create()
		var transforms: PackedFloat32Array = set["transforms"]
		RenderingServer.multimesh_set_mesh(multimesh, _model(set["name"]).get_rid())
		RenderingServer.multimesh_allocate_data(multimesh, transforms.size() / 12, RenderingServer.MULTIMESH_TRANSFORM_3D)
		RenderingServer.multimesh_set_buffer(multimesh, transforms)
		rids.append(RenderingServer.instance_create2(multimesh, scenario))
		rids.append(multimesh)
	server_rids[chunk] = rids

func _free_chunk(chunk: Vector3i) -> void:
	if server_rids.has(chunk):
		for rid: RID in server_rids[chunk]:
			RenderingServer.free_rid(rid)
		server_rids.erase(chunk)

## An occluder like the node's for `chunk`: its boxes, eight corners and twelve triangles each.
func _occluder(chunk: Vector3i) -> OccluderInstance3D:
	var vertices := PackedVector3Array()
	var indices := PackedInt32Array()
	for box: AABB in world.occluders(chunk):
		var first := vertices.size()
		for corner in 8:
			vertices.append(box.position + box.size * Vector3(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1))
		for triangle in [[0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5]]:
			for corner: int in triangle:
				indices.append(first + corner)
	var mesh := ArrayOccluder3D.new()
	mesh.set_arrays(vertices, indices)
	var instance := OccluderInstance3D.new()
	instance.occluder = mesh
	return instance

func _process(_delta: float) -> bool:
	var now := Time.get_ticks_usec()
	if not ready:
		if (now - started_usec) / 1e6 > LOAD_TIMEOUT_S:
			_fail("the city had not arrived")
			return true
		var side := 2 * VIEW_RADIUS + 1
		if world.generated_chunks().size() < side * side or server_rids.size() < side * side:
			return false
		var occluded := world.get_children().filter(func(node: Node) -> bool: return node is OccluderInstance3D)
		if occluded.size() < (2 * OCCLUDER_RADIUS + 1) * (2 * OCCLUDER_RADIUS + 1):
			return false
		ready = true
		if view == "street" and not _stand_on_a_street():
			_fail("the chunk at the origin has no road to stand on")
			return true
		root.use_occlusion_culling = false
		RenderingServer.viewport_set_measure_render_time(root.get_viewport_rid(), true)
		last_usec = now
		return false
	frame += 1
	if frame > 0:
		frame_ms.append((now - last_usec) / 1000.0)
		objects += RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_TOTAL_OBJECTS_IN_FRAME)
		render_ms.append(RenderingServer.viewport_get_measured_render_time_cpu(root.get_viewport_rid()))
		setup_ms.append(RenderingServer.get_frame_setup_time_cpu())
	last_usec = now
	if PHASES[phase_at] == "churn" and frame > -30:
		_churn()
	if frame < FRAMES:
		return false
	_report()
	phase_at += 1
	if phase_at == PHASES.size():
		print("render_occlusion %s: %s; on %s" % [view, "; ".join(results), RenderingServer.get_video_adapter_name()])
		for chunk: Vector3i in server_rids.keys():
			_free_chunk(chunk)
		quit(0)
		return true
	root.use_occlusion_culling = true
	if PHASES[phase_at] == "churn":
		# The node's occluders stand aside; the phase frees and builds copies of them in turn.
		world.occluder_radius = -1
		for y in range(-OCCLUDER_RADIUS, OCCLUDER_RADIUS + 1):
			for x in range(-OCCLUDER_RADIUS, OCCLUDER_RADIUS + 1):
				var chunk := Vector3i(1 + x, 1 + y, 0)
				churned.append(chunk)
				churn_nodes[chunk] = _occluder(chunk)
				root.add_child(churn_nodes[chunk])
	frame = -30
	frame_ms = PackedFloat64Array()
	render_ms = PackedFloat64Array()
	setup_ms = PackedFloat64Array()
	objects = 0
	return false

## Frees and builds again the occluders of the next few chunks.
func _churn() -> void:
	for i in CHURN:
		var chunk: Vector3i = churned[churn_next % churned.size()]
		churn_next += 1
		churn_nodes[chunk].queue_free()
		churn_nodes[chunk] = _occluder(chunk)
		root.add_child(churn_nodes[chunk])

func _report() -> void:
	var p := func(values: PackedFloat64Array, share: float) -> float:
		var sorted := values.duplicate()
		sorted.sort()
		return sorted[int((sorted.size() - 1) * share)]
	results.append("%s: frame p50 %.2f ms, p99 %.2f; viewport CPU p50 %.2f, p99 %.2f; frame setup CPU p50 %.2f, p99 %.2f; %.0f objects drawn" % [
		PHASES[phase_at], p.call(frame_ms, 0.5), p.call(frame_ms, 0.99), p.call(render_ms, 0.5), p.call(render_ms, 0.99),
		p.call(setup_ms, 0.5), p.call(setup_ms, 0.99), float(objects) / frame_ms.size()])

func _fail(message: String) -> void:
	printerr("render_occlusion: " + message)
	quit(1)
