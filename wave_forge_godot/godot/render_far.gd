## Draws the near ground and the far ground beyond it and checks no view sees between them
## (developer tool).
##
## Run through `../render_ground.sh`. The far ground check's pack has the same ground at full detail
## (`height`) and at a coarse scale of 8 (`far`), asked for 40 chunks out. Each view is filled with
## ground over a magenta background, so a magenta pixel is a gap: from above at an angle, across
## the boundary between near and far ground from low down, and from high above. It prints what
## each view drew and how many chunks each level generated per second of its stage's time.
extends SceneTree

const CELLS := 8
const RADIUS := 3
const FAR_RADIUS := 40
const CELL := Vector3(2, 1, 2)
const TIMEOUT_S := 120.0
const GAP := Color(1, 0, 1)
## Where each view looks from and at, and its vertical field of view.
const VIEWS := [
	[Vector3(8, 80, -40), Vector3(8, 0, 40), 60.0],
	[Vector3(8, 25, 20), Vector3(8, 0, 70), 40.0],
	[Vector3(8, 250, -80), Vector3(8, 0, 11), 50.0],
]

var world: Node
var camera: Camera3D
var started_usec := 0
var waited_frames := 0
var arrived := false
var view_at := 0
var lines := PackedStringArray()
var gaps := 0

func _initialize() -> void:
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://far.world.ron"
	world.targets = PackedStringArray(["height", "far"])
	world.target_radii = {"far": FAR_RADIUS}
	world.seed = 3
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = CELL
	world.view_radius = RADIUS
	world.collider_radius = -1
	world.ground_stage = "height"
	world.far_ground_stage = "far"
	root.add_child(world)
	camera = Camera3D.new()
	camera.far = 2000
	root.add_child(camera)
	var sun := DirectionalLight3D.new()
	root.add_child(sun)
	sun.rotation_degrees = Vector3(-60, 30, 0)
	var environment := WorldEnvironment.new()
	environment.environment = Environment.new()
	environment.environment.background_mode = Environment.BG_COLOR
	environment.environment.background_color = GAP
	environment.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.environment.ambient_light_color = Color(0.5, 0.5, 0.5)
	root.add_child(environment)
	if not world.start():
		printerr("render_far: the stages did not start")
		quit(1)
		return
	world.follow(Vector3(8, 0, 8))
	started_usec = Time.get_ticks_usec()

## Whether the near ground around the focus and the far ground around it are all drawn.
func _drawn() -> bool:
	var side := 2 * (RADIUS - 1) + 1
	var stats: Dictionary = world.stats()
	return stats["pending_grounds"] == 0 and stats["pending_far_grounds"] == 0 \
		and world.ground_chunks().size() >= side * side and world.far_ground_chunks().size() >= 25

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		printerr("render_far: the ground had not arrived")
		quit(1)
		return true
	if not arrived:
		if not _drawn():
			return false
		arrived = true
		_look(0)
	# A few frames for the renderer to draw the view.
	waited_frames += 1
	if waited_frames < 10:
		return false
	waited_frames = 0
	var image := root.get_viewport().get_texture().get_image()
	var count := _gaps(image)
	gaps += count
	lines.append("view %d: %d primitives, %d gap pixels" % [view_at, RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_TOTAL_PRIMITIVES_IN_FRAME), count])
	image.save_png("user://far_%d.png" % view_at)
	view_at += 1
	if view_at < VIEWS.size():
		_look(view_at)
		return false
	var stages: Dictionary = world.stats()["stages"]
	for stage in ["height", "far"]:
		var cost: Dictionary = stages[stage]
		lines.append("%s: %d chunks in %.1f ms, %.0f a second" % [stage, cost["products"], cost["ms"], cost["products"] / max(cost["ms"], 0.001) * 1000.0])
	print("render_far: %d near grounds, %d far grounds; %s; pictures in %s" % [world.ground_chunks().size(), world.far_ground_chunks().size(), "; ".join(lines), ProjectSettings.globalize_path("user://")])
	if gaps > 0:
		printerr("render_far: FAILED")
		quit(1)
		return true
	quit(0)
	return true

func _look(view: int) -> void:
	camera.fov = VIEWS[view][2]
	camera.look_at_from_position(VIEWS[view][0], VIEWS[view][1])

## How many pixels show the background, which the ground fills unless it has a gap.
func _gaps(image: Image) -> int:
	var count := 0
	for y in image.get_height():
		for x in image.get_width():
			var pixel := image.get_pixel(x, y)
			if pixel.r > 0.8 and pixel.g < 0.2 and pixel.b > 0.8:
				count += 1
	return count
