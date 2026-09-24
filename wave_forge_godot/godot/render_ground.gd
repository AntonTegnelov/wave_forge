## Renders a pack's ground with a material per category and saves the picture (developer tool).
##
## Run through `../render_ground.sh`. The ground check's pack draws sand, grass and rock through
## the reference ground shader with a palette of those three colours, and grass through the
## reference grass shader, lit from above, seen from above at an angle; the picture shows whether
## materials blend smoothly across triangles and chunks and where grass grows. It lands in Godot's
## user directory, and the script prints where. Then, with vsync off, it times frames with the
## grass and without it, and prints both: grass's cost on this renderer.
extends SceneTree

const CELLS := 8
const RADIUS := 3
const CELL := Vector3(2, 1, 2)
const TIMEOUT_S := 60.0
const TIMED_FRAMES := 300

var world: Node
var started_usec := 0
var waited_frames := 0
var phase := "arrive"
var frame_usec := 0
var frames := 0
var timed_usec := 0
var with_grass_ms := 0.0

func _initialize() -> void:
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://ground.world.ron"
	world.targets = PackedStringArray(["height", "surface"])
	world.seed = 3
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = CELL
	world.view_radius = RADIUS
	world.collider_radius = -1
	world.ground_stage = "height"
	world.ground_material_stage = "surface"
	world.ground_palette = PackedColorArray([Color(0.85, 0.78, 0.55), Color(0.52, 0.5, 0.48), Color(0.3, 0.55, 0.25)])
	world.targets = PackedStringArray(["height", "surface", "cover"])
	world.grass_stage = "cover"
	world.grass_radius = RADIUS - 1
	root.add_child(world)
	var camera := Camera3D.new()
	root.add_child(camera)
	camera.look_at_from_position(Vector3(0, 30, -30), Vector3(0, 0, 0))
	camera.far = 500
	var sun := DirectionalLight3D.new()
	root.add_child(sun)
	sun.rotation_degrees = Vector3(-60, 30, 0)
	var environment := WorldEnvironment.new()
	environment.environment = Environment.new()
	environment.environment.background_mode = Environment.BG_COLOR
	environment.environment.background_color = Color(0.6, 0.75, 0.9)
	environment.environment.ambient_light_color = Color(0.5, 0.5, 0.5)
	root.add_child(environment)
	if not world.start():
		printerr("render_ground: the stages did not start")
		quit(1)
		return
	world.follow(Vector3(1, 0, 1))
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		printerr("render_ground: the ground had not arrived")
		quit(1)
		return true
	match phase:
		"arrive":
			if world.stats()["pending_grounds"] > 0 or world.ground_chunks().size() < 25 or world.grass_chunks().size() < 25:
				return false
			# A few frames for the renderer to draw what just arrived.
			waited_frames += 1
			if waited_frames < 10:
				return false
			var path := "user://ground.png"
			root.get_viewport().get_texture().get_image().save_png(path)
			print("render_ground: saved %s, %d chunks of ground, %d with grass" % [ProjectSettings.globalize_path(path), world.ground_chunks().size(), world.grass_chunks().size()])
			phase = "close"
			waited_frames = 0
			var camera: Camera3D = root.get_viewport().get_camera_3d()
			camera.look_at_from_position(_grassy() + Vector3(0, 2.5, -5), _grassy())
		"close":
			waited_frames += 1
			if waited_frames < 10:
				return false
			var close := "user://grass.png"
			root.get_viewport().get_texture().get_image().save_png(close)
			print("render_ground: saved %s, grass close up" % ProjectSettings.globalize_path(close))
			DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
			Engine.max_fps = 0
			phase = "with grass"
			frames = -10
		"with grass", "without grass":
			return _time()
	return false

## Where the first fully covered column of the chunk at the origin stands, in world space.
func _grassy() -> Vector3:
	var cover: PackedFloat32Array = world.field_values("cover", Vector3i.ZERO)
	var heights: PackedFloat32Array = world.field_values("height", Vector3i.ZERO)
	for i in cover.size():
		if cover[i] >= 1.0:
			return Vector3((i % CELLS + 0.5) * CELL.x, heights[i] * CELL.y, (i / CELLS + 0.5) * CELL.z)
	return Vector3.ZERO

## Times frames once the renderer has settled, then moves on.
func _time() -> bool:
	frames += 1
	if frames == 0:
		timed_usec = Time.get_ticks_usec()
	if frames < TIMED_FRAMES:
		return false
	var ms := (Time.get_ticks_usec() - timed_usec) / 1000.0 / TIMED_FRAMES
	if phase == "with grass":
		with_grass_ms = ms
		world.grass_radius = -1
		phase = "without grass"
		frames = -10
		return false
	var blades: int = CELLS * CELLS * world.grass_per_cell * 25
	print("render_ground: %.2f ms a frame with grass (%d chunks, %d blades drawn, most collapsed where there is no cover), %.2f ms without, on %s" % [with_grass_ms, 25, blades, ms, RenderingServer.get_video_adapter_name()])
	quit(0)
	return true
