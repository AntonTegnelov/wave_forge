## Draws the tables check's road with a red decal and checks it shows on the road, at a joint
## between two pieces too, and nowhere beside it (developer tool). Run through `../render_ground.sh`, on Forward+ and on Compatibility; the
## Compatibility renderer draws decals from Godot 4.8, so before it that run fails. The picture lands
## in Godot's user directory, and the script prints where.
extends SceneTree

const CELLS := 8
const CELL := Vector3(2, 1, 2)
const TIMEOUT_S := 60.0
## The road, in cells: along x at row 12, from column 1, with a radius of 1.
const ROAD_ROW := 12.0
## Where the camera looks, in world units: the road's middle within the decal radius.
const CENTRE := Vector3(16, 0, 24)
## How many world units the camera sees from top to bottom.
const VIEW := 24.0

var world: Node
var started_usec := 0
var settled := 0

func _initialize() -> void:
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://facts.world.ron"
	world.targets = PackedStringArray(["roads", "paved"])
	world.ground_stage = "paved"
	world.seed = 5
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = CELL
	world.view_radius = 2
	world.collider_radius = -1
	world.decal_radius = 2
	var ground := StandardMaterial3D.new()
	ground.albedo_color = Color(0.6, 0.6, 0.6)
	world.ground_material = ground
	var decals: Dictionary[StringName, PackedScene] = {&"roads": _red_decal()}
	world.curve_decals = decals
	root.add_child(world)
	world.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	if not world.start():
		_fail("the stages did not start")
		return
	if not world.give_table("roads", [{"id": 1, "x0": 1, "y0": ROAD_ROW, "x1": 60, "y1": ROAD_ROW, "width": 1}]):
		_fail("the road was refused")
		return
	world.follow(CENTRE)
	var environment := Environment.new()
	environment.background_mode = Environment.BG_COLOR
	environment.background_color = Color.BLACK
	environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.ambient_light_color = Color.WHITE
	environment.reflected_light_source = Environment.REFLECTION_SOURCE_DISABLED
	environment.tonemap_mode = Environment.TONE_MAPPER_LINEAR
	var world_environment := WorldEnvironment.new()
	world_environment.environment = environment
	root.add_child(world_environment)
	var camera := Camera3D.new()
	camera.projection = Camera3D.PROJECTION_ORTHOGONAL
	camera.size = VIEW
	camera.far = 400.0
	root.add_child(camera)
	camera.look_at_from_position(CENTRE + Vector3(0, 100, 0), CENTRE, Vector3(0, 0, -1))
	started_usec = Time.get_ticks_usec()

func _red_decal() -> PackedScene:
	var image := Image.create(64, 64, false, Image.FORMAT_RGBA8)
	image.fill(Color.RED)
	var decal := Decal.new()
	decal.size = Vector3(1, 6, 1)
	decal.texture_albedo = ImageTexture.create_from_image(image)
	var scene := PackedScene.new()
	scene.pack(decal)
	decal.free()
	return scene

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("timed out with %d decals" % world.find_children("*", "Decal", false, false).size())
		return true
	if world.find_children("*", "Decal", false, false).is_empty():
		return false
	# A few frames for the ground and the decals to be drawn.
	settled += 1
	if settled < 30:
		return false
	var picture := root.get_viewport().get_texture().get_image()
	var path := "user://render_decals_%s.png" % RenderingServer.get_current_rendering_method()
	picture.save_png(path)
	var pixels_per_unit := picture.get_height() / VIEW
	var middle := Vector2i(picture.get_width() / 2, picture.get_height() / 2)
	var on_road := picture.get_pixelv(middle)
	# Pieces two cells long from column 1 meet at column 7, a cell left of the middle: the reddest
	# and the least red pixel within a few of it.
	var joint := Color.RED
	var at_joint := middle.x - int(CELL.x * pixels_per_unit)
	for x in range(at_joint - 4, at_joint + 5):
		var pixel := picture.get_pixel(x, middle.y)
		if pixel.r - pixel.g < joint.r - joint.g:
			joint = pixel
	# Four cells beside the road, well past its radius of one.
	var beside := picture.get_pixelv(middle + Vector2i(0, int(4.0 * CELL.z * pixels_per_unit)))
	print("render_decals: %s, on the road %s, at a joint %s, beside it %s, saved to %s" % [RenderingServer.get_current_rendering_method(), on_road, joint, beside, ProjectSettings.globalize_path(path)])
	# Red is what the decal draws: a hairline between pieces lets the grey ground through.
	var red := func(pixel: Color) -> bool: return pixel.r > 0.5 and pixel.g < 0.15 and pixel.b < 0.15
	if not red.call(on_road) or not red.call(joint) or absf(beside.r - beside.g) > 0.05 or beside.g < 0.1:
		_fail("the road should be red and the ground beside it grey")
		return true
	quit(0)
	return true

func _fail(message: String) -> void:
	printerr("render_decals: " + message)
	quit(1)
