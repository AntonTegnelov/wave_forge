## Assigns a Godot FastNoiseLite resource to a pack's noise and checks that the field reading it
## holds exactly what the resource gives (story N7).
##
## Run by `../verify.sh` after `verify_tables.gd`. The resource is far from Godot's defaults
## (cellular, ridged, with a domain warp and an offset), so every part of the conversion and of
## the library's port is exercised. Each column of the field, and a sample between chunks, must
## equal `get_noise_2d` at the column's centre in cells.
extends SceneTree

const CELLS := 8
const TIMEOUT_S := 30.0

var world: Node
var noise := FastNoiseLite.new()
var started_usec := 0
var arrived := false

func _initialize() -> void:
	noise.noise_type = FastNoiseLite.TYPE_CELLULAR
	noise.seed = -4242
	noise.frequency = 0.07
	noise.offset = Vector3(13.5, -7.25, 0)
	noise.fractal_type = FastNoiseLite.FRACTAL_RIDGED
	noise.fractal_octaves = 4
	noise.fractal_weighted_strength = 0.3
	noise.cellular_distance_function = FastNoiseLite.DISTANCE_MANHATTAN
	noise.cellular_return_type = FastNoiseLite.RETURN_DISTANCE2_SUB
	noise.cellular_jitter = 0.8
	noise.domain_warp_enabled = true
	noise.domain_warp_type = FastNoiseLite.DOMAIN_WARP_BASIC_GRID
	noise.domain_warp_amplitude = 12.0
	noise.domain_warp_fractal_type = FastNoiseLite.DOMAIN_WARP_FRACTAL_INDEPENDENT
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://noise.world.ron"
	world.noises = {"hills": noise}
	world.targets = PackedStringArray(["height"])
	world.seed = 99
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.view_radius = 1
	world.collider_radius = -1
	root.add_child(world)
	world.stage_ready.connect(func(stage: String, chunk: Vector3i) -> void:
		if stage == "height" and chunk == Vector3i(-1, 2, 0):
			arrived = true)
	world.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	if not world.start():
		_fail("the stages did not start")
		return
	world.follow(Vector3(-4, 0, 20))
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	if not arrived:
		if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
			_fail("the height of (-1, 2) had not arrived")
			return true
		return false
	var chunk := Vector3i(-1, 2, 0)
	var values: PackedFloat32Array = world.field_values("height", chunk)
	for i in values.size():
		var x := chunk.x * CELLS + i % CELLS + 0.5
		var y := chunk.y * CELLS + i / CELLS + 0.5
		if values[i] != noise.get_noise_2d(x, y):
			_fail("column (%s, %s) holds %s; the resource gives %s" % [x, y, values[i], noise.get_noise_2d(x, y)])
			return true
	var sampled: float = world.sample("height", Vector3(-3.5, 0, 17.5))
	if sampled != noise.get_noise_2d(-3.5, 17.5):
		_fail("a sample gives %s; the resource gives %s" % [sampled, noise.get_noise_2d(-3.5, 17.5)])
		return true
	print("verify_noise: a FastNoiseLite resource's noise is what the field holds, column for column, and what a sample gives")
	quit(0)
	return true

func _fail(message: String) -> void:
	printerr("verify_noise: " + message)
	quit(1)
