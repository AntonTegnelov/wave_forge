## Frame times while the maximal preset's continent plays from a directory its world run baked, for
## M1 on a desktop (docs/guides/desktop-measurements.md). Bake the directory first with
## `measure_world_run.gd`, then run with a display and the renderer to measure, never headless:
##     godot --path . --rendering-driver vulkan --script measure_playback.gd -- --directory <dir> --seconds 20 --out results.txt
## `continent.tscn`'s node plays `--directory` around a camera in a window of 1920 by 1080 with vsync
## off, its far ground reaching 2 km. Once the view around the first settlement is drawn it measures
## two phases of `--seconds` each: `walk`, at eye height along +x at 4.2 m/s, and `fly`, 60 m above
## the ground along +x at 30 m/s. Each prints one line of `key=value` fields (the window's size,
## frames, median, 99th percentile and slowest frame in milliseconds, the most memory the process
## has held, and the stages that ran, which for a played world is none), appended to `--out` as well
## when it is given. A stage that ran fails the run.
extends SceneTree

const PeakMemory := preload("res://peak_memory.gd")
const LOAD_TIMEOUT_S := 300.0
## The first settlement's town (chunk (83, 116) in chunks of 16 m).
const START := Vector3(1336.0, 0.0, 1864.0)
const EYE_M := 1.7
const FLIGHT_M := 60.0
const PHASES := {"walk": 4.2, "fly": 30.0}

var stages: Node
var camera: Camera3D
var directory := ""
var seconds := 20.0
var out := ""
var phase := "load"
var position := START
var phase_started_usec := 0
var last_usec := 0
var frames := PackedFloat64Array()
## The ground's height under the camera when last known: the played world draws the ground a
## moment after the camera reaches it, and before the first view none is known.
var ground := 0.0

func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	var i := 0
	while i < args.size():
		if i + 1 >= args.size():
			_fail("%s needs a value" % args[i])
			return
		match args[i]:
			"--directory": directory = args[i + 1]
			"--seconds": seconds = args[i + 1].to_float()
			"--out": out = args[i + 1]
			_:
				_fail("unknown option %s" % args[i])
				return
		i += 2
	if not DirAccess.dir_exists_absolute(directory):
		_fail("--directory names no directory: %s" % directory)
		return
	DisplayServer.window_set_size(Vector2i(1920, 1080))
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	Engine.max_fps = 0
	var scene: Node = (load("res://continent.tscn") as PackedScene).instantiate()
	stages = scene.get_node("Continent")
	stages.play_directory = directory
	stages.collider_radius = -1
	root.add_child(scene)
	stages.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	_scene()
	if not stages.start():
		_fail("the continent did not start")
		return
	phase_started_usec = Time.get_ticks_usec()

func _scene() -> void:
	camera = Camera3D.new()
	camera.current = true
	camera.fov = 70.0
	camera.far = 2500.0
	root.add_child(camera)
	_place_camera(EYE_M)
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

## Puts the camera `above` metres over the ground at `position`, looking ahead along +x.
func _place_camera(above: float) -> void:
	var height: float = stages.ground_height(position)
	if not is_nan(height):
		ground = height
	var eye := Vector3(position.x, ground + above, position.z)
	camera.look_at_from_position(eye, eye + Vector3(1.0, -0.15, 0.0))

## Whether the near ground around the camera and the far ground beyond it are drawn, with nothing
## of either still to build.
func _view_drawn() -> bool:
	var stats: Dictionary = stages.stats()
	return stats["pending_grounds"] == 0 and stats["pending_far_grounds"] == 0 \
		and stats["pending_placements"] == 0 and stats["pending_volumes"] == 0 \
		and stages.ground_chunks().size() >= (2 * stages.view_radius + 1) ** 2 \
		and not stages.far_ground_chunks().is_empty()

func _process(_delta: float) -> bool:
	var now := Time.get_ticks_usec()
	var frame_ms := (now - last_usec) / 1000.0 if last_usec > 0 else 0.0
	last_usec = now
	var elapsed := (now - phase_started_usec) / 1e6
	if phase == "load":
		if elapsed > LOAD_TIMEOUT_S:
			_fail("the first view was not drawn in %d s" % LOAD_TIMEOUT_S)
			return true
		if _view_drawn():
			_start("walk")
		return false
	frames.append(frame_ms)
	position.x += PHASES[phase] * frame_ms / 1000.0
	_place_camera(EYE_M if phase == "walk" else FLIGHT_M)
	if elapsed < seconds:
		return false
	if not _report():
		return true
	if phase == "walk":
		_start("fly")
		return false
	quit(0)
	return true

func _start(next: String) -> void:
	phase = next
	frames = PackedFloat64Array()
	phase_started_usec = Time.get_ticks_usec()

## Prints the phase's line; returns whether the run goes on.
func _report() -> bool:
	var sorted := frames.duplicate()
	sorted.sort()
	var at := func(fraction: float) -> float: return sorted[roundi((sorted.size() - 1) * fraction)]
	var peak_mb := PeakMemory.peak_resident_mb()
	if peak_mb < 0.0:
		_fail("the system did not say how much memory the process held")
		return false
	var ran: Dictionary = stages.stats()["stages"]
	var size := root.get_visible_rect().size
	var line := "measure_playback adapter=\"%s\" driver=%s method=%s size=%dx%d phase=%s speed=%s frames=%d p50_ms=%.2f p99_ms=%.2f max_ms=%.2f peak_mb=%.0f stages_ran=%d" % [
		RenderingServer.get_video_adapter_name(), RenderingServer.get_current_rendering_driver_name(),
		RenderingServer.get_current_rendering_method(), size.x, size.y, phase, PHASES[phase], sorted.size(),
		at.call(0.5), at.call(0.99), sorted[sorted.size() - 1], peak_mb, ran.size()]
	print(line)
	if out != "":
		var file := FileAccess.open(out, FileAccess.READ_WRITE) if FileAccess.file_exists(out) else FileAccess.open(out, FileAccess.WRITE)
		if file == null:
			_fail("%s cannot be written" % out)
			return false
		file.seek_end()
		file.store_line(line)
	if not ran.is_empty():
		_fail("a stage ran in the played world: %s" % ran.keys())
		return false
	return true

func _fail(message: String) -> void:
	printerr("measure_playback: " + message)
	quit(1)
