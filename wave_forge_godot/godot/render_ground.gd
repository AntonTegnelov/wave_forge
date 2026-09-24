## Renders a pack's ground with a material per category and saves the picture (developer tool).
##
## Run through `../render_ground.sh`. The ground check's pack draws sand, grass and rock through
## the reference ground shader with a palette of those three colours, lit from above, seen from
## above at an angle; the picture shows whether materials blend smoothly across triangles and
## chunks. It lands in Godot's user directory, and the script prints where.
extends SceneTree

const CELLS := 8
const RADIUS := 3
const CELL := Vector3(2, 1, 2)
const TIMEOUT_S := 60.0

var world: Node
var started_usec := 0
var waited_frames := 0

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
	root.add_child(world)
	var camera := Camera3D.new()
	root.add_child(camera)
	camera.look_at_from_position(Vector3(0, 60, -35), Vector3(0, 0, 0))
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
	if world.stats()["pending_grounds"] > 0 or world.ground_chunks().size() < 25:
		return false
	# A few frames for the renderer to draw what just arrived.
	waited_frames += 1
	if waited_frames < 10:
		return false
	var path := "user://ground.png"
	root.get_viewport().get_texture().get_image().save_png(path)
	print("render_ground: saved %s, %d chunks of ground" % [ProjectSettings.globalize_path(path), world.ground_chunks().size()])
	quit(0)
	return true
