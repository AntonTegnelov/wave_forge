## Frame times while the city streams in around a walker, drawn with its module models on a real
## renderer, for #39 and the P stories on a desktop (docs/guides/desktop-measurements.md).
##
## Run with a display and the renderer to measure, never headless:
##     godot --path . --rendering-driver vulkan --script measure_city.gd -- --speed 4.2 --seconds 20 --out results.txt
## Each chunk is drawn one of two ways, named by `--draw`, and freed when the chunk is dropped:
## - `multimesh` (the default): one RenderingServer MultiMesh per module, from `instance_sets`;
## - `merged`: one mesh per chunk, a surface per module holding every instance's copy of the
##   module's model where `instance_sets` puts it, built with SurfaceTool, under one instance.
## A camera follows the walker and a sun casts shadows; vsync is off.
## Once the first view is drawn it measures two phases of `--seconds` each: `idle`, standing still
## with nothing to generate, which is the cost of drawing alone, and `streaming`, walking along +x at
## `--speed` units a second. Each prints one line of `key=value` fields (frames, median, 99th
## percentile and slowest frame in milliseconds, chunks drawn, the node's own process time at the
## 99th percentile, and how long drawing a chunk took Godot's thread, at the median and at worst),
## appended to `--out` as well when it is given. The two ways of drawing are #203's choice.
extends SceneTree

const CELLS := 8
const CELL_SIZE := 2.0
const RADIUS := 4
const LOAD_TIMEOUT_S := 300.0

var world: Node
var camera: Camera3D
var models := {}
var drawn_names := PackedStringArray()
var server_rids := {}
## Each merged chunk's mesh, by its instance's RID, so the mesh lives as long as its instance.
var merged_meshes := {}
var speed := 4.2
var seconds := 20.0
var out := ""
var phase := "load"
var walker := Vector3.ZERO
var phase_started_usec := 0
var last_usec := 0
var frames := PackedFloat64Array()
var drawn := 0
## How this run draws a chunk: "multimesh" or "merged".
var draw := "multimesh"
## Milliseconds of Godot's thread each chunk took to draw in the phase.
var draw_ms := PackedFloat64Array()

func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	var i := 0
	while i < args.size():
		var name := args[i]
		if i + 1 >= args.size():
			_fail("%s needs a value" % name)
			return
		var value := args[i + 1]
		match name:
			"--speed": speed = value.to_float()
			"--seconds": seconds = value.to_float()
			"--out": out = value
			"--draw": draw = value
			_:
				_fail("unknown option %s" % name)
				return
		i += 2
	if draw != "multimesh" and draw != "merged":
		_fail("--draw multimesh or merged, not %s" % draw)
		return
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0
	world = ClassDB.instantiate("WaveForgeWorld")
	world.seed = 11
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = Vector3(CELL_SIZE, CELL_SIZE, CELL_SIZE)
	world.view_radius = RADIUS
	world.collider_radius = -1
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
	world.chunk_updated.connect(_draw)
	world.chunk_evicted.connect(_forget)
	world.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	if not world.start():
		_fail("generation did not start")
		return
	world.follow(walker)
	_scene()
	phase_started_usec = Time.get_ticks_usec()

func _scene() -> void:
	camera = Camera3D.new()
	camera.current = true
	camera.fov = 60.0
	camera.far = 2000.0
	root.add_child(camera)
	_follow_camera()
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

func _follow_camera() -> void:
	camera.look_at_from_position(walker + Vector3(-40, 60, 0), walker + Vector3(0, 2, 0))

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

func _draw(chunk: Vector3i) -> void:
	if drawn_names.is_empty():
		for tile in world.tile_count():
			var module: String = world.tile_name(tile)
			if not drawn_names.has(module) and _model(module) != null:
				drawn_names.append(module)
		if drawn_names.is_empty():
			_fail("no module has a model; run prepare first")
			return
	var started := Time.get_ticks_usec()
	_forget(chunk)
	var scenario := root.get_world_3d().scenario
	var sets: Array = world.instance_sets(chunk, drawn_names)
	server_rids[chunk] = _multimeshes(sets, scenario) if draw == "multimesh" else _merged(sets, scenario)
	draw_ms.append((Time.get_ticks_usec() - started) / 1000.0)
	drawn += 1

## A chunk's instance sets as a RenderingServer MultiMesh per module: the RIDs to free.
func _multimeshes(sets: Array, scenario: RID) -> Array:
	var rids := []
	for set: Dictionary in sets:
		var multimesh := RenderingServer.multimesh_create()
		var transforms: PackedFloat32Array = set["transforms"]
		RenderingServer.multimesh_set_mesh(multimesh, _model(set["name"]).get_rid())
		RenderingServer.multimesh_allocate_data(multimesh, transforms.size() / 12, RenderingServer.MULTIMESH_TRANSFORM_3D)
		RenderingServer.multimesh_set_buffer(multimesh, transforms)
		rids.append(RenderingServer.instance_create2(multimesh, scenario))
		rids.append(multimesh)
	return rids

## A chunk's instance sets as one mesh, a surface per module holding a copy of the module's model
## at each of its instances, under one RenderingServer instance: the RIDs to free. The mesh is kept
## alive by the RIDs' owner, the ArrayMesh in `merged_meshes`.
func _merged(sets: Array, scenario: RID) -> Array:
	var mesh := ArrayMesh.new()
	for set: Dictionary in sets:
		var model := _model(set["name"])
		var transforms: PackedFloat32Array = set["transforms"]
		var tool := SurfaceTool.new()
		tool.begin(Mesh.PRIMITIVE_TRIANGLES)
		for i in range(0, transforms.size(), 12):
			var at := Transform3D(
				Vector3(transforms[i], transforms[i + 4], transforms[i + 8]),
				Vector3(transforms[i + 1], transforms[i + 5], transforms[i + 9]),
				Vector3(transforms[i + 2], transforms[i + 6], transforms[i + 10]),
				Vector3(transforms[i + 3], transforms[i + 7], transforms[i + 11]))
			for surface in model.get_surface_count():
				tool.append_from(model, surface, at)
		tool.set_material(model.surface_get_material(0))
		tool.commit(mesh)
	var instance := RenderingServer.instance_create2(mesh.get_rid(), scenario)
	merged_meshes[instance] = mesh
	return [instance]

func _forget(chunk: Vector3i) -> void:
	for rid: RID in server_rids.get(chunk, []):
		RenderingServer.free_rid(rid)
		merged_meshes.erase(rid)
	server_rids.erase(chunk)

## Whether every chunk within the radius of the walker's chunk is drawn.
func _view_drawn() -> bool:
	var centre := Vector3i(floori(walker.x / (CELLS * CELL_SIZE)), floori(walker.z / (CELLS * CELL_SIZE)), 0)
	for y in range(-RADIUS, RADIUS + 1):
		for x in range(-RADIUS, RADIUS + 1):
			if not server_rids.has(centre + Vector3i(x, y, 0)):
				return false
	return true

func _process(_delta: float) -> bool:
	var now := Time.get_ticks_usec()
	var frame_ms := (now - last_usec) / 1000.0 if last_usec > 0 else 0.0
	last_usec = now
	var elapsed := (now - phase_started_usec) / 1e6
	match phase:
		"load":
			if elapsed > LOAD_TIMEOUT_S:
				_fail("the first view was not drawn in %d s" % LOAD_TIMEOUT_S)
				return true
			if _view_drawn():
				_start("idle")
		"idle":
			frames.append(frame_ms)
			if elapsed >= seconds:
				_report("idle")
				_start("streaming")
		"streaming":
			frames.append(frame_ms)
			walker.x += speed * frame_ms / 1000.0
			world.follow(walker)
			_follow_camera()
			if elapsed >= seconds:
				_report("streaming")
				for chunk in server_rids.keys():
					_forget(chunk)
				quit(0)
				return true
	return false

func _start(next: String) -> void:
	phase = next
	frames = PackedFloat64Array()
	drawn = 0
	draw_ms = PackedFloat64Array()
	phase_started_usec = Time.get_ticks_usec()

func _report(name: String) -> void:
	var sorted := frames.duplicate()
	sorted.sort()
	var at := func(fraction: float) -> float: return sorted[roundi((sorted.size() - 1) * fraction)]
	var stats: Dictionary = world.stats()
	var drawing := draw_ms.duplicate()
	drawing.sort()
	var draw_p50 := drawing[drawing.size() / 2] if not drawing.is_empty() else 0.0
	var draw_max := drawing[drawing.size() - 1] if not drawing.is_empty() else 0.0
	var line := "measure_city adapter=\"%s\" driver=%s method=%s draw=%s speed=%s phase=%s frames=%d p50_ms=%.2f p99_ms=%.2f max_ms=%.2f chunks=%d node_p99_ms=%.2f draw_p50_ms=%.3f draw_max_ms=%.3f" % [
		RenderingServer.get_video_adapter_name(), RenderingServer.get_current_rendering_driver_name(),
		RenderingServer.get_current_rendering_method(), draw, speed, name, sorted.size(), at.call(0.5), at.call(0.99),
		sorted[sorted.size() - 1], drawn, stats.get("process_ms_p99", -1.0), draw_p50, draw_max]
	print(line)
	if out != "":
		var file := FileAccess.open(out, FileAccess.READ_WRITE) if FileAccess.file_exists(out) else FileAccess.open(out, FileAccess.WRITE)
		if file == null:
			_fail("%s cannot be written" % out)
			return
		file.seek_end()
		file.store_line(line)

func _fail(message: String) -> void:
	printerr("measure_city: " + message)
	quit(1)
