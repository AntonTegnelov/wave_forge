## Memory over a long walk through the small city preset, for P4 (docs/product/user-stories.md):
## process and GPU memory stay within 5 % of their level after the first minute. Run with a display
## and the renderer, never headless:
##     godot --path . --rendering-driver vulkan --script measure_memory.gd -- --minutes 30 --out memory.txt
## It instances the addon's `presets/city.tscn` at a view of 6 chunks with navigation 4 out, and walks
## a camera from the world's origin along a straight line at 4.2 m/s at eye height over the ground,
## so new ground, towns, grass, bodies and navigation keep arriving and old ones keep leaving. Every
## 10 s it samples the process's resident memory (`VmRSS` in `/proc/self/status`, so Linux only),
## Godot's static memory and the video memory the renderer reports in use, with the node's chunks of
## ground, navigation and far ground, the navigation bakes it has started and its placed nodes and
## instances, one line each, appended to
## `--out` when it is given. At the end it prints each measure's level after the first minute and its
## lowest and highest after, and fails if any strays more than 5 % from that level.
extends SceneTree

const SPEED := 4.2
const EYE_M := 1.7
const SAMPLE_S := 10.0
const SETTLE_S := 60.0
const BOUND := 0.05

var stages: Node
var camera: Camera3D
var minutes := 30.0
var out := ""
var position := Vector3.ZERO
var heading := Vector3(1.0, 0.0, 0.37).normalized()
var ground := 0.0
var started_usec := 0
var last_usec := 0
var next_sample_s := 0.0
## Each sample's time in seconds and measures, by name.
var samples: Array[Dictionary] = []

func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	var i := 0
	while i < args.size():
		if i + 1 >= args.size():
			_fail("%s needs a value" % args[i])
			return
		match args[i]:
			"--minutes": minutes = args[i + 1].to_float()
			"--out": out = args[i + 1]
			_:
				_fail("unknown option %s" % args[i])
				return
		i += 2
	stages = (load("res://addons/wave_forge/presets/city.tscn") as PackedScene).instantiate()
	stages.view_radius = 6
	stages.navigation_radius = 4
	root.add_child(stages)
	stages.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	camera = Camera3D.new()
	camera.current = true
	camera.far = 2000.0
	root.add_child(camera)
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-55, 35, 0)
	sun.shadow_enabled = true
	root.add_child(sun)
	started_usec = Time.get_ticks_usec()
	last_usec = started_usec

func _process(_delta: float) -> bool:
	var now := Time.get_ticks_usec()
	var elapsed := (now - started_usec) / 1e6
	position += heading * SPEED * (now - last_usec) / 1e6
	last_usec = now
	var height: float = stages.ground_height(position)
	if not is_nan(height):
		ground = height
	var eye := position + Vector3(0.0, ground + EYE_M, 0.0)
	camera.look_at_from_position(eye, eye + heading)
	stages.follow(position)
	if elapsed >= next_sample_s:
		_sample(elapsed)
		next_sample_s += SAMPLE_S
	if elapsed < minutes * 60.0:
		return false
	_report()
	return true

## The process's resident memory in bytes, from `/proc/self/status`.
func _resident() -> float:
	var file := FileAccess.open("/proc/self/status", FileAccess.READ)
	if file == null:
		_fail("/proc/self/status cannot be read")
		return NAN
	# A file under /proc reports no length, so it is read line by line.
	while not file.eof_reached():
		var line := file.get_line()
		if line.begins_with("VmRSS:"):
			return line.split(":")[1].strip_edges().split(" ")[0].to_float() * 1024.0
	_fail("/proc/self/status has no VmRSS")
	return NAN

func _sample(elapsed: float) -> void:
	var stats: Dictionary = stages.stats()
	var sample := {
		"s": elapsed,
		"resident_mb": _resident() / 1e6,
		"static_mb": Performance.get_monitor(Performance.MEMORY_STATIC) / 1e6,
		"video_mb": Performance.get_monitor(Performance.RENDER_VIDEO_MEM_USED) / 1e6,
		"grounds": stages.ground_chunks().size(),
		"navigation": stages.navigation_chunks().size(),
		"far": stages.far_ground_chunks().size(),
		"nodes": stats["placed_nodes"],
		"instances": stats["placed_instances"],
		"baked": stats["navigation_baked"],
	}
	samples.append(sample)
	var line := "measure_memory s=%.0f walked_m=%.0f resident_mb=%.1f static_mb=%.1f video_mb=%.1f grounds=%d navigation=%d baked=%d far=%d nodes=%d instances=%d" % [
		elapsed, position.length(), sample["resident_mb"], sample["static_mb"], sample["video_mb"],
		sample["grounds"], sample["navigation"], sample["baked"], sample["far"], sample["nodes"], sample["instances"]]
	print(line)
	_write(line)

func _report() -> void:
	var failed := false
	for measure in ["resident_mb", "static_mb", "video_mb"]:
		var level := NAN
		var low := INF
		var high := -INF
		for sample in samples:
			if is_nan(level) and sample["s"] >= SETTLE_S:
				level = sample[measure]
			elif not is_nan(level):
				low = minf(low, sample[measure])
				high = maxf(high, sample[measure])
		var spread := maxf(high / level - 1.0, 1.0 - low / level)
		var line := "measure_memory %s level=%.1f low=%.1f high=%.1f spread=%.1f%%" % [measure, level, low, high, spread * 100.0]
		print(line)
		_write(line)
		failed = failed or spread > BOUND
	if failed:
		_fail("memory strayed more than %d %% from its level after the first minute" % roundi(BOUND * 100.0))
		return
	quit(0)

func _write(line: String) -> void:
	if out == "":
		return
	var file := FileAccess.open(out, FileAccess.READ_WRITE) if FileAccess.file_exists(out) else FileAccess.open(out, FileAccess.WRITE)
	if file == null:
		_fail("%s cannot be written" % out)
		return
	file.seek_end()
	file.store_line(line)

func _fail(message: String) -> void:
	printerr("measure_memory: " + message)
	quit(1)
