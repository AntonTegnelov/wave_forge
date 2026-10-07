## Draws the water checks' valley with its lakes and river and checks that the water shows and no
## view sees between it and its banks (developer tool).
##
## Run through `../render_ground.sh`. The valley's ground fills each view over a magenta
## background, so a magenta pixel is a gap, and the water is drawn unshaded in a blue nothing else
## has, so its pixels can be counted: along the river from above at an angle, across it from low
## down, and over the lake in the hollow. It prints what each view drew, saves the pictures in
## Godot's user directory, and fails on a gap or a view without water.
extends SceneTree

const CELLS := 8
const CELL := Vector3(2, 1, 2)
const TIMEOUT_S := 60.0
const GAP := Color(1, 0, 1)
const WATER := Color(0.1, 0.35, 0.9)
## Where each view looks from and at.
const VIEWS := [
	[Vector3(36, 45, 44), Vector3(60, 0, 64)],
	[Vector3(48, 8, 52), Vector3(48, 0, 66)],
	[Vector3(66, 40, 24), Vector3(80, -2, 40)],
]

var world: Node
var camera: Camera3D
var started_usec := 0
var waited_frames := 0
var arrived := false
var view_at := 0
var lines := PackedStringArray()
var failures := PackedStringArray()

func _initialize() -> void:
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://water.world.ron"
	world.targets = PackedStringArray(["ground", "water"])
	world.seed = 9
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = CELL
	world.view_radius = 6
	world.collider_radius = -1
	world.ground_stage = "ground"
	world.water_stage = "water"
	var water := StandardMaterial3D.new()
	water.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	water.albedo_color = WATER
	world.water_material = water
	root.add_child(world)
	camera = Camera3D.new()
	camera.fov = 40
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
		printerr("render_water: the stages did not start")
		quit(1)
		return
	var river := {"id": 1, "x0": 2.0, "y0": 32.0, "x1": 62.0, "y1": 32.0, "width": 1.5}
	if not world.give_table("rivers", [river]):
		printerr("render_water: the river was not taken")
		quit(1)
		return
	world.follow(Vector3(56, 0, 56))
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		printerr("render_water: the water had not arrived")
		quit(1)
		return true
	if not arrived:
		if world.stats()["pending_grounds"] > 0 or world.ground_chunks().size() < 121 or world.water_chunks().size() < 6:
			return false
		arrived = true
		_look(0)
	# A few frames for the renderer to draw the view.
	waited_frames += 1
	if waited_frames < 10:
		return false
	waited_frames = 0
	var image := root.get_viewport().get_texture().get_image()
	var gaps := 0
	var water := 0
	for y in image.get_height():
		for x in image.get_width():
			var pixel := image.get_pixel(x, y)
			if pixel.r > 0.8 and pixel.g < 0.2 and pixel.b > 0.8:
				gaps += 1
			elif absf(pixel.r - WATER.r) < 0.05 and absf(pixel.g - WATER.g) < 0.05 and absf(pixel.b - WATER.b) < 0.05:
				water += 1
	lines.append("view %d: %d water pixels, %d gap pixels" % [view_at, water, gaps])
	if gaps > 0 or water == 0:
		failures.append("view %d" % view_at)
	image.save_png("user://water_%d.png" % view_at)
	view_at += 1
	if view_at < VIEWS.size():
		_look(view_at)
		return false
	print("render_water: %d chunks of water; %s; pictures in %s" % [world.water_chunks().size(), "; ".join(lines), ProjectSettings.globalize_path("user://")])
	if not failures.is_empty():
		printerr("render_water: FAILED: %s" % ", ".join(failures))
		quit(1)
		return true
	quit(0)
	return true

func _look(view: int) -> void:
	camera.look_at_from_position(VIEWS[view][0], VIEWS[view][1])
