## Frame times while a camera flies over the hills preset generated live with its far ground out to
## 2 km, and the chunks per second each level generates, for P2 on a desktop
## (docs/guides/desktop-measurements.md). Run with a display and the renderer to measure, never
## headless:
##     godot --path . --rendering-driver vulkan --script measure_view.gd -- --seconds 20 --out results.txt
## It instances the addon's `presets/hills.tscn` at its own settings, but with every target that has
## a radius of its own (its far levels) reaching `REACH_M`, under the reference look, in a
## window of 1920 by 1080 with vsync off, and once everything around the start has generated it
## flies the camera 60 m over the ground along +x at 30 m/s for `--seconds`. It prints one line of
## `key=value` fields: how far the far ground reaches in metres, frames, median, 99th percentile
## and slowest frame in milliseconds, the node's own time per frame at the 99th percentile and its
## slowest, and for the near level (`ground_stage`) and the far one (`far_ground_stage`) the chunks
## each generated during the flight a second, of the flight's time and of the level's own;
## appended to `--out` as well when it is given.
extends SceneTree

const ReferenceLook := preload("res://reference_look.gd")
const Generated := preload("res://generated.gd")
const SPEED := 30.0
const ABOVE_M := 60.0
## How far the far levels reach, P2's view distance.
const REACH_M := 2000.0
const TIMEOUT_S := 300.0
## Frames everything has to stay generated for before the flight starts.
const SETTLE := 30

var stages: Node
var camera: Camera3D
var seconds := 20.0
var out := ""
var position := Vector3(1.0, 0.0, 1.0)
var ground := 0.0
var started_usec := 0
var flight_usec := 0
var last_usec := 0
var settled := 0
var last := {}
var before := {}
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
	var scene := Node3D.new()
	ReferenceLook.add_to(scene)
	root.add_child(scene)
	camera = Camera3D.new()
	camera.current = true
	camera.far = 2500.0
	scene.add_child(camera)
	stages = (load("res://addons/wave_forge/presets/hills.tscn") as PackedScene).instantiate()
	var chunk_m: float = stages.chunk_cells.x * stages.cell_size.x
	var radii: Dictionary = stages.target_radii.duplicate()
	for target: StringName in radii:
		radii[target] = ceili(REACH_M / chunk_m)
	stages.target_radii = radii
	stages.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	_place_camera()
	scene.add_child(stages)
	started_usec = Time.get_ticks_usec()

## Puts the camera `ABOVE_M` over the ground at `position`, looking ahead along +x and down a little.
func _place_camera() -> void:
	var height: float = stages.ground_height(position) if stages.is_inside_tree() else NAN
	if not is_nan(height):
		ground = height
	var eye := position + Vector3(0.0, ground + ABOVE_M, 0.0)
	camera.look_at_from_position(eye, eye + Vector3(1.0, -0.15, 0.0))

## Each level's products and milliseconds so far, by stage.
func _levels() -> Dictionary:
	var ran: Dictionary = stages.stats()["stages"]
	var levels := {}
	for stage: String in [stages.ground_stage, stages.far_ground_stage]:
		levels[stage] = ran[stage] if ran.has(stage) else {"products": 0, "ms": 0.0}
	return levels

func _process(_delta: float) -> bool:
	var now := Time.get_ticks_usec()
	if flight_usec == 0:
		if (now - started_usec) / 1e6 > TIMEOUT_S:
			_fail("the view around the start had not generated after %d s" % TIMEOUT_S)
			return true
		_place_camera()
		var stats: Dictionary = stages.stats()
		if not Generated.all_generated(stages, stats, last) or stats["pending_grounds"] > 0 or stats["pending_placements"] > 0:
			settled = 0
			return false
		settled += 1
		if settled < SETTLE:
			return false
		before = _levels()
		flight_usec = now
		last_usec = now
		return false
	frames.append((now - last_usec) / 1000.0)
	node_frames.append(stages.last_frame_ms())
	position.x += SPEED * frames[frames.size() - 1] / 1000.0
	last_usec = now
	_place_camera()
	if (now - flight_usec) / 1e6 < seconds:
		return false
	_report((now - flight_usec) / 1e6)
	quit(0)
	return true

## `values` sorted, and its value at `fraction` of the way from the smallest to the largest.
func _at(values: PackedFloat64Array, fraction: float) -> float:
	var sorted := values.duplicate()
	sorted.sort()
	return sorted[roundi((sorted.size() - 1) * fraction)]

func _report(flown_s: float) -> void:
	var after := _levels()
	var rates := ""
	for stage: String in after:
		var products: int = after[stage]["products"] - before[stage]["products"]
		var ms: float = after[stage]["ms"] - before[stage]["ms"]
		rates += " %s_chunks=%d %s_per_s=%.0f %s_per_own_s=%.0f" % [stage, products, stage,
			products / flown_s, stage, products / (ms / 1000.0) if ms > 0.0 else 0.0]
	var reach_m: float = stages.target_radii[stages.far_ground_stage] * stages.chunk_cells.x * stages.cell_size.x
	var size := root.get_visible_rect().size
	var line := "measure_view adapter=\"%s\" driver=%s method=%s size=%dx%d reach_m=%.0f speed=%s frames=%d p50_ms=%.2f p99_ms=%.2f max_ms=%.2f node_p99_ms=%.2f node_max_ms=%.2f%s" % [
		RenderingServer.get_video_adapter_name(), RenderingServer.get_current_rendering_driver_name(),
		RenderingServer.get_current_rendering_method(), size.x, size.y, reach_m, SPEED, frames.size(),
		_at(frames, 0.5), _at(frames, 0.99), _at(frames, 1.0), _at(node_frames, 0.99), _at(node_frames, 1.0), rates]
	print(line)
	if out != "":
		var file := FileAccess.open(out, FileAccess.READ_WRITE) if FileAccess.file_exists(out) else FileAccess.open(out, FileAccess.WRITE)
		if file == null:
			_fail("%s cannot be written" % out)
			return
		file.seek_end()
		file.store_line(line)

func _fail(message: String) -> void:
	printerr("measure_view: " + message)
	quit(1)
