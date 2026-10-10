## The grass's cost at a radius, with its blades thinning with distance or not (developer tool;
## docs/research/measurements.md, E65).
##
## Run with a display and the renderer to measure:
## `godot --path . --script measure_grass.gd -- <radius> <blades per column> <thin|full>`.
## The hills preset's node, with its grass at that radius and that many blades a column, drawn by
## the reference grass shader thinning its blades with distance (`thin`) or not (`full`), stands
## under the reference look; the camera stands at eye level on its meadow, looking along it. Once
## everything has arrived, with vsync off, it times the viewport's GPU over 200 frames with the
## grass and 200 without, four times over, alternating so the renderer's drift falls on both, and
## prints the grass's cost, the mean of the four differences, and their spread; and the primitives
## the grass adds to a frame, its blades drawn, collapsed or not, which is what thinning cuts and
## does not drift. Under Xvfb through Mesa's D3D12 layer the GPU time drifts by milliseconds, so
## there the primitives are the measure and the time is for a desktop (docs/guides/desktop-measurements.md).
extends SceneTree

const ReferenceLook := preload("res://reference_look.gd")
const Generated := preload("res://generated.gd")
const AT := Vector3(-14, 0, 6)
const FRAMES := 200
const ROUNDS := 4
const TIMEOUT_S := 300.0

var stages: Node
var camera: Camera3D
var frames := 0
var settled := 0
var last := {}
var phase := "arrive"
var gpu_ms := 0.0
var with_grass := 0.0
var radius := 0
## The grass's cost in each round, with less without.
var costs: Array[float] = []
## The primitives in a frame with the grass, and without.
var primitives_with := 0
var primitives_without := 0
## How many chunks had grass when last looked at, while it grows back.
var grown := -1
var started_usec := 0

func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	if args.size() != 3 or not args[2] in ["thin", "full"]:
		printerr("measure_grass: give a radius, blades per column and thin or full after --")
		quit(1)
		return
	ReferenceLook.add_to(root)
	stages = (load("res://addons/wave_forge/presets/hills.tscn") as PackedScene).instantiate()
	stages.follow_camera = false
	stages.collider_radius = -1
	stages.navigation_radius = -1
	radius = int(args[0])
	stages.grass_radius = radius
	stages.grass_per_cell = int(args[1])
	# Grass reaches as far as the view does, at the least.
	stages.view_radius = maxi(stages.view_radius, int(args[0]) + 1)
	if args[2] == "full":
		var material := ShaderMaterial.new()
		material.shader = load("res://addons/wave_forge/shaders/grass.gdshader")
		material.set_shader_parameter("thin_keep", 1.0)
		stages.grass_material = material
	camera = Camera3D.new()
	camera.far = 2000
	root.add_child(camera)
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	frames += 1
	if frames == 1:
		root.add_child(stages)
		stages.follow(AT)
		return false
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		printerr("measure_grass: the %s phase timed out" % phase)
		quit(1)
		return true
	match phase:
		"arrive":
			var ground: float = stages.ground_height(AT)
			if not is_nan(ground):
				var eye := AT + Vector3(0, ground + 1.7, 0)
				camera.look_at_from_position(eye, eye + Vector3(1, -0.12, 0.6))
			var stats: Dictionary = stages.stats()
			if is_nan(ground) or not Generated.all_generated(stages, stats, last) or stats["pending_grounds"] > 0 or stats["pending_placements"] > 0:
				settled = 0
				return false
			settled += 1
			if settled < 30:
				return false
			DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
			Engine.max_fps = 0
			RenderingServer.viewport_set_measure_render_time(root.get_viewport().get_viewport_rid(), true)
			phase = "with grass"
			frames = -10
		"with grass", "without grass":
			return _time()
	return false

## Times the viewport's GPU over `FRAMES` frames once it has settled and the grass has grown back,
## a few chunks a frame, then moves on.
func _time() -> bool:
	if phase == "with grass" and frames == -1:
		if stages.grass_chunks().size() != grown:
			grown = stages.grass_chunks().size()
			frames = -10
			return false
	if frames == 0:
		gpu_ms = 0.0
		var primitives := RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_TOTAL_PRIMITIVES_IN_FRAME)
		if phase == "with grass":
			primitives_with = primitives
		else:
			primitives_without = primitives
	elif frames > 0:
		gpu_ms += RenderingServer.viewport_get_measured_render_time_gpu(root.get_viewport().get_viewport_rid())
	if frames < FRAMES:
		return false
	gpu_ms /= FRAMES
	frames = -10
	if phase == "with grass":
		with_grass = gpu_ms
		stages.grass_radius = -1
		phase = "without grass"
		return false
	costs.append(with_grass - gpu_ms)
	if costs.size() < ROUNDS:
		stages.grass_radius = radius
		phase = "with grass"
		grown = -1
		return false
	var mean := 0.0
	for cost in costs:
		mean += cost / costs.size()
	var args := OS.get_cmdline_user_args()
	print("measure_grass: radius %s, %s blades a column, %s: the grass %d primitives a frame; %.3f ms a frame on the GPU, from %.3f to %.3f over %d rounds, the frame without it %.3f, on %s, %s" % [args[0], args[1], args[2], primitives_with - primitives_without, mean, costs.min(), costs.max(), ROUNDS, gpu_ms, RenderingServer.get_video_adapter_name(), RenderingServer.get_current_rendering_method()])
	quit(0)
	return true
