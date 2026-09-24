## Draws trees with the reference vegetation shader and checks they move in the wind (developer tool).
##
## Run through `../render_ground.sh`, after `render_ground.gd`. The scenes check's pack scatters
## trees; each is a lone cylinder whose material is the vegetation shader, so the stages node draws
## them as MultiMeshes carrying each tree's phase and stiffness. Two frames half a second apart must
## differ while the wind blows, and must not once its strength is zero. It saves the first frame
## and prints how many pixels changed each time.
extends SceneTree

const CELLS := 8
const CELL := Vector3(2, 1, 2)
const TIMEOUT_S := 60.0

var world: Node
var started_usec := 0
var phase := "arrive"
var mark_usec := 0
var first: Image

func _initialize() -> void:
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://scenes.world.ron"
	world.targets = PackedStringArray(["level", "trees"])
	world.seed = 21
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = CELL
	world.view_radius = 1
	world.collider_radius = -1
	var trunk := CylinderMesh.new()
	trunk.height = 6.0
	trunk.top_radius = 0.2
	trunk.bottom_radius = 0.5
	var material := ShaderMaterial.new()
	var shader := Shader.new()
	shader.code = world.vegetation_shader_code()
	material.shader = shader
	trunk.material = material
	var tree := MeshInstance3D.new()
	tree.mesh = trunk
	var scene := PackedScene.new()
	scene.pack(tree)
	tree.free()
	world.scenes = {"tree": scene}
	root.add_child(world)
	var camera := Camera3D.new()
	root.add_child(camera)
	camera.look_at_from_position(Vector3(8, 14, -14), Vector3(8, 0, 8))
	var sun := DirectionalLight3D.new()
	root.add_child(sun)
	sun.rotation_degrees = Vector3(-50, 30, 0)
	if not world.start():
		_fail("the stages did not start")
		return
	RenderingServer.global_shader_parameter_set("wave_forge_wind", Vector4(1, 0, 1.0, 3.0))
	world.follow(Vector3(8, 0, 8))
	started_usec = Time.get_ticks_usec()

func _changed(a: Image, b: Image) -> int:
	var changed := 0
	for y in range(0, a.get_height(), 2):
		for x in range(0, a.get_width(), 2):
			if not a.get_pixel(x, y).is_equal_approx(b.get_pixel(x, y)):
				changed += 1
	return changed

func _frame() -> Image:
	return root.get_viewport().get_texture().get_image()

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("the %s phase timed out" % phase)
		return true
	var waited := (Time.get_ticks_usec() - mark_usec) / 1e6
	match phase:
		"arrive":
			if world.stats()["placed_instances"] == 0 or world.stats()["pending_placements"] > 0:
				return false
			mark_usec = Time.get_ticks_usec()
			phase = "settle"
		"settle":
			if waited < 0.5:
				return false
			first = _frame()
			first.save_png("user://wind.png")
			mark_usec = Time.get_ticks_usec()
			phase = "blow"
		"blow":
			if waited < 0.5:
				return false
			var moving := _changed(first, _frame())
			print("render_wind: %d pixels changed in half a second of wind, saved %s" % [moving, ProjectSettings.globalize_path("user://wind.png")])
			if moving == 0:
				_fail("the trees did not move in the wind")
				return true
			RenderingServer.global_shader_parameter_set("wave_forge_wind", Vector4(1, 0, 0, 3.0))
			mark_usec = Time.get_ticks_usec()
			phase = "calm"
		"calm":
			if waited < 0.3:
				return false
			first = _frame()
			mark_usec = Time.get_ticks_usec()
			phase = "still"
		"still":
			if waited < 0.5:
				return false
			var still := _changed(first, _frame())
			print("render_wind: %d pixels changed in half a second without wind" % still)
			if still != 0:
				_fail("the trees moved without wind")
				return true
			quit(0)
			return true
	return false

func _fail(message: String) -> void:
	printerr("render_wind: " + message)
	quit(1)
