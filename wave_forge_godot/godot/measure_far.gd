## What drawing the hills preset costs the GPU from the reference views that see most of its far
## ground (developer tool; docs/research/measurements.md, E67).
##
## Run with a display and the renderer to measure:
## `godot --path . --script measure_far.gd -- <oblique|top>`.
## The hills preset's node at its defaults stands under the reference look, the camera where the
## preset sheets' oblique or top-down view puts it (`render_presets.gd`), so the far ground fills
## most of the picture. Once everything has arrived, with vsync off, it times the viewport's GPU
## over 300 frames and prints the mean and the primitives a frame. Run it once with each build to
## compare; under Xvfb through Mesa's D3D12 layer the GPU time drifts by a few tenths of a
## millisecond between runs, so alternate the builds. Seen from the top, it also prints the median
## luminance of the picture just inside and just outside the near ground's edge, 3 units either side
## of it all the way round, for how well the far ground matches the near ground where they meet;
## with `-- top noshadow` the sun casts no shadow, so the near ground's trees darken neither.
extends SceneTree

const ReferenceLook := preload("res://reference_look.gd")
const Generated := preload("res://generated.gd")
const AT := Vector3(1, 0, 1)
const FRAMES := 300
const TIMEOUT_S := 300.0

var stages: Node
var camera: Camera3D
var view := ""
var frames := 0
var settled := 0
var last := {}
var timing := false
var gpu_ms := 0.0
var started_usec := 0

func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	if args.is_empty() or not args[0] in ["oblique", "top"]:
		printerr("measure_far: give oblique or top after --")
		quit(1)
		return
	view = args[0]
	var scene := Node3D.new()
	var sun := ReferenceLook.add_to(scene)
	root.add_child(scene)
	stages = (load("res://addons/wave_forge/presets/hills.tscn") as PackedScene).instantiate()
	stages.follow_camera = false
	stages.collider_radius = -1
	stages.navigation_radius = -1
	sun.directional_shadow_max_distance = 3.0 * stages.view_radius * stages.chunk_cells.x * stages.cell_size.x
	sun.shadow_enabled = not "noshadow" in args
	camera = Camera3D.new()
	camera.far = 2000
	scene.add_child(camera)
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	frames += 1
	if frames == 1:
		root.add_child(stages)
		stages.follow(AT)
		return false
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		printerr("measure_far: the view did not settle: %s" % stages.stats())
		quit(1)
		return true
	if timing:
		return _time()
	var ground: float = stages.ground_height(AT)
	if not is_nan(ground):
		var at := AT + Vector3(0, ground, 0)
		if view == "oblique":
			camera.look_at_from_position(at + Vector3(-30, 40, 30), at)
		else:
			var reach: float = stages.view_radius * stages.chunk_cells.x * stages.cell_size.x
			camera.look_at_from_position(at + Vector3(0, reach * 2.0, 0), at, Vector3(0, 0, -1))
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
	timing = true
	frames = -10
	return false

func _time() -> bool:
	if frames > 0:
		gpu_ms += RenderingServer.viewport_get_measured_render_time_gpu(root.get_viewport().get_viewport_rid())
	if frames < FRAMES:
		return false
	var primitives := RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_TOTAL_PRIMITIVES_IN_FRAME)
	print("measure_far: %s, %.3f ms a frame on the GPU, %d primitives, %d far chunks, on %s, %s" % [view, gpu_ms / FRAMES, primitives, stages.far_ground_chunks().size(), RenderingServer.get_video_adapter_name(), RenderingServer.get_current_rendering_method()])
	if view == "top":
		_seam()
	quit(0)
	return true

## The median luminance just inside and just outside the near ground's edge in the picture.
func _seam() -> void:
	var image := root.get_viewport().get_texture().get_image()
	var span: float = stages.chunk_cells.x * stages.cell_size.x
	# The near ground covers the view's chunks around the followed one, which holds AT.
	var low: float = -stages.view_radius * span
	var high: float = (stages.view_radius + 1) * span
	var near: Array[float] = []
	var far: Array[float] = []
	for k in 200:
		var t := low + (high - low) * (k + 0.5) / 200.0
		for edge in [[Vector2(low, t), Vector2(-1, 0)], [Vector2(high, t), Vector2(1, 0)], [Vector2(t, low), Vector2(0, -1)], [Vector2(t, high), Vector2(0, 1)]]:
			for side in [-1.0, 1.0]:
				var at: Vector2 = edge[0] + edge[1] * side * 3.0
				var point := Vector3(at.x, 0, at.y)
				var height: float = stages.ground_height(point)
				point.y = height if not is_nan(height) else stages.ground_height(AT)
				var pixel := Vector2i(camera.unproject_position(point))
				if pixel.x < 0 or pixel.y < 0 or pixel.x >= image.get_width() or pixel.y >= image.get_height():
					continue
				var colour := image.get_pixelv(pixel)
				(far if side > 0.0 else near).append(0.2126 * colour.r + 0.7152 * colour.g + 0.0722 * colour.b)
	near.sort()
	far.sort()
	print("measure_far: at the near ground's edge the near ground's median luminance %.3f, the far ground's %.3f, over %d points each" % [near[near.size() / 2], far[far.size() / 2], near.size()])
