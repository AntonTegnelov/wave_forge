## Draws a wide view of the ground at its levels of detail and checks their skirts close every gap
## (developer tool).
##
## Run through `../render_ground.sh`. The ground check's pack is drawn over 17 by 17 chunks, seen
## from above at an angle that fills the whole picture with ground, over a magenta background, so a
## magenta pixel is a gap between chunks. It counts the primitives drawn with levels of detail off
## and at several thresholds, from Godot's default of a pixel to 16, each of which puts neighbours
## at different levels somewhere in view; no picture may have a gap. The last one is saved.
extends SceneTree

const CELLS := 8
const RADIUS := 8
const CELL := Vector3(2, 1, 2)
const TIMEOUT_S := 120.0
const GAP := Color(1, 0, 1)
## Pixels of error a level may show before a finer one is drawn: off, Godot's default, and coarser.
const THRESHOLDS := [0.0, 1.0, 4.0, 16.0]

var world: Node
var started_usec := 0
var waited_frames := 0
var arrived := false
var threshold_at := 0
var primitives: Array[int] = []
var gaps: Array[int] = []

func _initialize() -> void:
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://ground.world.ron"
	world.targets = PackedStringArray(["height"])
	world.seed = 3
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = CELL
	world.view_radius = RADIUS
	world.collider_radius = -1
	world.ground_stage = "height"
	root.add_child(world)
	var camera := Camera3D.new()
	root.add_child(camera)
	camera.look_at_from_position(Vector3(8, 40, -10), Vector3(8, 0, 9))
	camera.far = 1000
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
		printerr("render_lods: the stages did not start")
		quit(1)
		return
	world.follow(Vector3(8, 0, 8))
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		printerr("render_lods: the ground had not arrived")
		quit(1)
		return true
	if not arrived:
		var side := 2 * (RADIUS - 1) + 1
		if world.stats()["pending_grounds"] > 0 or world.ground_chunks().size() < side * side:
			return false
		arrived = true
		root.get_viewport().mesh_lod_threshold = THRESHOLDS[0]
	# A few frames for the renderer to draw what arrived, or at its new threshold.
	waited_frames += 1
	if waited_frames < 10:
		return false
	waited_frames = 0
	primitives.append(RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_TOTAL_PRIMITIVES_IN_FRAME))
	var image := root.get_viewport().get_texture().get_image()
	gaps.append(_gaps(image))
	threshold_at += 1
	if threshold_at < THRESHOLDS.size():
		root.get_viewport().mesh_lod_threshold = THRESHOLDS[threshold_at]
		return false
	var path := "user://lods.png"
	image.save_png(path)
	var lines := PackedStringArray()
	for i in THRESHOLDS.size():
		lines.append("%s px: %d primitives, %d gap pixels" % [THRESHOLDS[i], primitives[i], gaps[i]])
	print("render_lods: %d chunks; %s; saved %s" % [world.ground_chunks().size(), "; ".join(lines), ProjectSettings.globalize_path(path)])
	if gaps.max() > 0 or primitives[-1] >= primitives[0]:
		printerr("render_lods: FAILED")
		quit(1)
		return true
	quit(0)
	return true

## How many pixels show the background, which the ground fills unless it has a gap.
func _gaps(image: Image) -> int:
	var count := 0
	for y in image.get_height():
		for x in image.get_width():
			var pixel := image.get_pixel(x, y)
			if pixel.r > 0.8 and pixel.g < 0.2 and pixel.b > 0.8:
				count += 1
	return count
