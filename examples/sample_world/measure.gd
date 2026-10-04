## Frame times in the sample world, for N10's frame rate on a desktop (P1's bars,
## docs/guides/desktop-measurements.md). Run with a display and the renderer to measure, never
## headless:
##     godot --path . --rendering-driver vulkan --script measure.gd -- --seconds 20 --out results.txt
## It runs `main.tscn` with vsync off, and once the townsfolk are out it walks the walker from where
## it starts towards the city's middle and on through it, at 4.2 m/s for `--seconds`, at eye height
## over the ground. It prints one line of `key=value` fields (the window's size, frames, median,
## 99th percentile and slowest frame in milliseconds, and the node's own time per frame of the walk
## at the median, the 99th percentile and its slowest), appended to `--out` as well when it is
## given.
extends SceneTree

const SPEED := 4.2
const LOAD_TIMEOUT_S := 300.0

var main: Node
var walker: Node3D
var seconds := 20.0
var out := ""
var heading := Vector3.ZERO
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
	main = (load("res://main.tscn") as PackedScene).instantiate()
	root.add_child(main)
	walker = main.get_node("Walker")
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	var now := Time.get_ticks_usec()
	var townsfolk: Node3D = main.get_node("Townsfolk")
	if heading == Vector3.ZERO:
		if (now - started_usec) / 1e6 > LOAD_TIMEOUT_S:
			_fail("the townsfolk were not out after %d s" % LOAD_TIMEOUT_S)
			return true
		if townsfolk.get_child_count() < townsfolk.count:
			return false
		# The walk is the script's: the walker's own physics stays out of it.
		walker.set_physics_process(false)
		var middle: Vector2 = townsfolk.streets.get_center()
		heading = (Vector3(middle.x, 0.0, middle.y) - Vector3(walker.global_position.x, 0.0, walker.global_position.z)).normalized()
		walker.look_at(walker.global_position + heading, Vector3.UP)
		started_usec = now
		last_usec = now
		return false
	frames.append((now - last_usec) / 1000.0)
	last_usec = now
	node_frames.append(main.get_node("City").stats()["last_frame_ms"])
	var step := heading * SPEED * frames[frames.size() - 1] / 1000.0
	var at := walker.global_position + step
	var ground: float = main.get_node("City").ground_height(at)
	if not is_nan(ground):
		at.y = ground
	walker.global_position = at
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
	var line := "measure_sample adapter=\"%s\" driver=%s method=%s size=%dx%d speed=%s frames=%d p50_ms=%.2f p99_ms=%.2f max_ms=%.2f node_p50_ms=%.2f node_p99_ms=%.2f node_max_ms=%.2f" % [
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
	printerr("measure: " + message)
	quit(1)
