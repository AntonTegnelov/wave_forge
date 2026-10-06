## Frame times while a camera walks into the small city preset, a stage world with its town drawn
## from the city kit, its grass, trees, bodies and navigation, for P1 on a desktop
## (docs/guides/desktop-measurements.md). Run with a display and the renderer to measure, never
## headless:
##     godot --path . --rendering-driver vulkan --script measure_city_preset.gd -- --seconds 20 --out results.txt
## It instances the addon's `presets/city.tscn` at a view of 6 chunks with navigation 4 out, in a
## window of 1920 by 1080 with vsync off and a sun, and once the city's site and every chunk of it
## have navigation it walks a camera from the world's origin towards the city's middle and on
## through it, at 4.2 m/s for `--seconds`, at eye height over the ground. It prints one line of
## `key=value` fields (the window's size, frames, median, 99th percentile and slowest frame in
## milliseconds, and the node's own time per frame of the walk at the median, the 99th percentile
## and its slowest), appended to `--out` as well when it is given.
extends SceneTree

const SPEED := 4.2
const EYE_M := 1.7
const LOAD_TIMEOUT_S := 300.0

var stages: Node
var camera: Camera3D
var seconds := 20.0
var out := ""
var position := Vector3.ZERO
var heading := Vector3.ZERO
var ground := 0.0
var started_usec := 0
var last_usec := 0
var frames := PackedFloat64Array()
var node_frames := PackedFloat64Array()

func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	var i := 0
	while i < args.size():
		if i + 1 >= args.size():
			_fail("%s needs a value" % args[i])
			return
		match args[i]:
			"--seconds": seconds = args[i + 1].to_float()
			"--out": out = args[i + 1]
			_:
				_fail("unknown option %s" % args[i])
				return
		i += 2
	DisplayServer.window_set_size(Vector2i(1920, 1080))
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0
	stages = (load("res://addons/wave_forge/presets/city.tscn") as PackedScene).instantiate()
	stages.view_radius = 6
	stages.navigation_radius = 4
	root.add_child(stages)
	stages.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	camera = Camera3D.new()
	camera.current = true
	camera.far = 2000.0
	root.add_child(camera)
	camera.look_at_from_position(Vector3(0.0, EYE_M, 0.0), Vector3(-1.0, EYE_M, -1.0))
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
	started_usec = Time.get_ticks_usec()

## The middle of the city's site on the ground plane, once its site is generated and every chunk
## of it has navigation; null before.
func _city_middle() -> Variant:
	var navigable: Array[Vector3i] = stages.navigation_chunks()
	for chunk in navigable:
		for site: Dictionary in stages.sites("places", chunk):
			if site["kind"] != "city":
				continue
			var low: Vector2i = site["min"]
			var high: Vector2i = site["max"]
			for y in range(low.y, high.y):
				for x in range(low.x, high.x):
					if not navigable.has(Vector3i(x, y, 0)):
						return null
			var chunk_size: Vector3 = Vector3(stages.chunk_cells) * stages.cell_size
			return Vector2(low + high) * 0.5 * Vector2(chunk_size.x, chunk_size.z)
	return null

func _process(_delta: float) -> bool:
	var now := Time.get_ticks_usec()
	if heading == Vector3.ZERO:
		if (now - started_usec) / 1e6 > LOAD_TIMEOUT_S:
			_fail("the city had no navigation after %d s" % LOAD_TIMEOUT_S)
			return true
		var middle: Variant = _city_middle()
		if middle == null:
			return false
		heading = Vector3(middle.x, 0.0, middle.y).normalized()
		started_usec = now
		last_usec = now
		return false
	frames.append((now - last_usec) / 1000.0)
	last_usec = now
	node_frames.append(stages.last_frame_ms())
	position += heading * SPEED * frames[frames.size() - 1] / 1000.0
	var height: float = stages.ground_height(position)
	if not is_nan(height):
		ground = height
	camera.look_at_from_position(position + Vector3(0.0, ground + EYE_M, 0.0), position + heading + Vector3(0.0, ground + EYE_M, 0.0))
	if (now - started_usec) / 1e6 < seconds:
		return false
	_report()
	quit(0)
	return true

## `values` sorted, and its value at `fraction` of the way from the smallest to the largest.
func _at(values: PackedFloat64Array, fraction: float) -> float:
	var sorted := values.duplicate()
	sorted.sort()
	return sorted[roundi((sorted.size() - 1) * fraction)]

func _report() -> void:
	var size := root.get_visible_rect().size
	var line := "measure_city_preset adapter=\"%s\" driver=%s method=%s size=%dx%d speed=%s frames=%d p50_ms=%.2f p99_ms=%.2f max_ms=%.2f node_p50_ms=%.2f node_p99_ms=%.2f node_max_ms=%.2f" % [
		RenderingServer.get_video_adapter_name(), RenderingServer.get_current_rendering_driver_name(),
		RenderingServer.get_current_rendering_method(), size.x, size.y, SPEED, frames.size(),
		_at(frames, 0.5), _at(frames, 0.99), _at(frames, 1.0),
		_at(node_frames, 0.5), _at(node_frames, 0.99), _at(node_frames, 1.0)]
	print(line)
	if out != "":
		var file := FileAccess.open(out, FileAccess.READ_WRITE) if FileAccess.file_exists(out) else FileAccess.open(out, FileAccess.WRITE)
		if file == null:
			_fail("%s cannot be written" % out)
			return
		file.seek_end()
		file.store_line(line)

func _fail(message: String) -> void:
	printerr("measure_city_preset: " + message)
	quit(1)
