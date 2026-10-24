## The first seconds of the addon's demo scene in an exported game, for #277: how long its first
## frames take, which is where shaders compile when an export carries none baked. An export's main
## scene (`first_frames.tscn`, through the `first_frames` feature of `export_presets.cfg`); the
## desktop script exports the project with Godot's shader baker on and off and runs each export:
##     <export> -- --label baked --out results.txt
## After `--seconds` (default 15) it prints one line of `key=value` fields: the label, how many
## frames there were, the first frame's time, the slowest frame and when it came, how many frames
## took over 33 ms, the first 120 frames' total, and when the ground arrived under the camera; all
## times in milliseconds but `ground_s`. A frame's time runs from one process to the next, which
## holds the draw between them; Godot processes before it draws, so the first frame runs from the
## first process to the second, holding the first draw. It appends the line to `--out` too when
## given, then quits.
extends Node

const SLOW_MS := 33.0
const FIRST_FRAMES := 120

var label := "unlabelled"
var out := ""
var seconds := 15.0
var stages: Node
var camera: Camera3D
var started_usec := 0
var last_usec := 0
var frames: PackedFloat64Array = []
var ground_s := -1.0

func _ready() -> void:
	var args := OS.get_cmdline_user_args()
	var i := 0
	while i < args.size():
		if i + 1 >= args.size():
			_fail("%s needs a value" % args[i])
			return
		match args[i]:
			"--label": label = args[i + 1]
			"--out": out = args[i + 1]
			"--seconds": seconds = float(args[i + 1])
			_:
				_fail("unknown option %s" % args[i])
				return
		i += 2
	var demo := (load("res://addons/wave_forge/demo/demo.tscn") as PackedScene).instantiate()
	add_child(demo)
	stages = demo.get_node("Hills")
	camera = demo.get_node("Camera")
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> void:
	var now := Time.get_ticks_usec()
	if last_usec > 0:
		frames.append((now - last_usec) / 1000.0)
	last_usec = now
	if ground_s < 0.0:
		var under := Vector3(camera.global_position.x, 0, camera.global_position.z)
		if not is_nan(stages.ground_height(under)):
			ground_s = (now - started_usec) / 1e6
	if (now - started_usec) / 1e6 < seconds:
		return
	var slowest := 0
	var over := 0
	var first_total := 0.0
	for index in frames.size():
		if frames[index] > frames[slowest]:
			slowest = index
		if frames[index] > SLOW_MS:
			over += 1
		if index < FIRST_FRAMES:
			first_total += frames[index]
	var line := "first_frames label=%s renderer=%s frames=%d first_ms=%.1f max_ms=%.1f max_at_frame=%d over_33ms=%d first_120_ms=%.1f ground_s=%.2f" % [
		label, RenderingServer.get_current_rendering_method(), frames.size(), frames[0],
		frames[slowest], slowest, over, first_total, ground_s]
	print(line)
	if not out.is_empty():
		var file := FileAccess.open(out, FileAccess.READ_WRITE if FileAccess.file_exists(out) else FileAccess.WRITE)
		file.seek_end()
		file.store_line(line)
		file.close()
	get_tree().quit(0)
	set_process(false)

func _fail(message: String) -> void:
	printerr("first_frames: " + message)
	get_tree().quit(1)
