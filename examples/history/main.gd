## The example: a continent, a toy history over it (history.gd), and a player walking through what
## the history left. The history runs once and is saved; later runs restore the same world from the
## save alone, since the world is a function of the pack, the seed and the facts. Press N for a new
## history: the stages regenerate only what it changed, around the player.
extends Node3D

const SAVE := "user://history.json"
const CELLS := 8
const CELL_SIZE := 2.0
const SEED := 7
const HISTORY := preload("res://history.gd")

var world: Node
var player := preload("res://player.gd").new()
## Each module's mesh, loaded once, by module name; null for a module with nothing to draw.
var models := {}
## Each town chunk's MultiMesh nodes, by chunk, freed when the chunk goes.
var drawn := {}

func _ready() -> void:
	_add_light_and_sea()
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://continent.world.ron"
	world.rules_files = {"city": "res://city.ron", "ruins": "res://ruins.ron"}
	world.targets = PackedStringArray(["level", "towns"])
	world.seed = SEED
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = Vector3.ONE * CELL_SIZE
	world.view_radius = 4
	world.ground_stage = "level"
	world.ground_material = _material(Color(0.4, 0.55, 0.3))
	world.collider_radius = 2
	world.kernel_cache = "user://kernels"
	add_child(world)
	world.stage_ready.connect(_on_stage_ready)
	world.stage_dropped.connect(_on_stage_dropped)
	world.generation_failed.connect(func(reason: String) -> void: push_error(reason))
	if not world.start():
		return
	# Every town module above the street is solid to walk into; the street is the ground itself.
	var box := BoxShape3D.new()
	box.size = Vector3.ONE * CELL_SIZE
	for rules in ["city", "ruins"]:
		for tag in ["building", "roof", "walkway", "pillar", "stair"]:
			for module in world.modules_tagged(rules, tag):
				world.set_collision_shape(module, box)
	var history := _load()
	if history.is_empty():
		history = HISTORY.new().run(world, SEED)
		_save(history)
	_give(history)
	# The player starts in the air just south of the first village's site, facing north into the
	# town, and lands once the ground under it has its collider.
	var village: Dictionary = history["villages"][0]
	var at := Vector2(village["x"], village["y"] + 13.0)
	var ground: float = world.sample("height", Vector3(at.x, 0, at.y) * CELL_SIZE)
	player.position = Vector3(at.x, ground + 4.0, at.y) * CELL_SIZE
	add_child(player)

func _process(_delta: float) -> void:
	world.follow(player.position)
	if not player.landed_allowed:
		var here := Vector3i(floori(player.position.x / (CELLS * CELL_SIZE)), floori(player.position.z / (CELLS * CELL_SIZE)), 0)
		player.landed_allowed = world.collider_chunks().has(here)

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and event.physical_keycode == KEY_N:
		var history: Dictionary = HISTORY.new().run(world, randi())
		_save(history)
		_give(history)

func _give(history: Dictionary) -> void:
	for table: String in history:
		world.give_table(table, history[table])

func _save(history: Dictionary) -> void:
	FileAccess.open(SAVE, FileAccess.WRITE).store_string(JSON.stringify(history))

func _load() -> Dictionary:
	if not FileAccess.file_exists(SAVE):
		return {}
	return JSON.parse_string(FileAccess.get_file_as_string(SAVE))

## A town chunk is drawn as one MultiMesh per module, from the placements the node gives.
func _on_stage_ready(stage: String, chunk: Vector3i) -> void:
	if stage != "towns":
		return
	_on_stage_dropped(stage, chunk)
	var nodes := []
	for placements: Dictionary in world.town_instance_sets("towns", chunk, PackedStringArray()):
		var mesh := _model(placements["name"])
		if mesh == null:
			continue
		var multimesh := MultiMesh.new()
		multimesh.transform_format = MultiMesh.TRANSFORM_3D
		multimesh.mesh = mesh
		var transforms: PackedFloat32Array = placements["transforms"]
		multimesh.instance_count = transforms.size() / 12
		multimesh.buffer = transforms
		var instance := MultiMeshInstance3D.new()
		instance.multimesh = multimesh
		add_child(instance)
		nodes.append(instance)
	drawn[chunk] = nodes

func _on_stage_dropped(stage: String, chunk: Vector3i) -> void:
	if stage == "towns" and drawn.has(chunk):
		for node: Node in drawn[chunk]:
			node.queue_free()
		drawn.erase(chunk)

func _model(module: String) -> Mesh:
	if not models.has(module):
		models[module] = null
		var document := GLTFDocument.new()
		var state := GLTFState.new()
		if document.append_from_file("res://models/%s.glb" % module, state) == OK:
			var scene := document.generate_scene(state)
			var meshes := scene.find_children("*", "MeshInstance3D", true, false)
			if not meshes.is_empty():
				models[module] = (meshes[0] as MeshInstance3D).mesh
			scene.free()
	return models[module]

func _add_light_and_sea() -> void:
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-50, 30, 0)
	sun.shadow_enabled = true
	add_child(sun)
	var environment := WorldEnvironment.new()
	environment.environment = Environment.new()
	environment.environment.background_mode = Environment.BG_COLOR
	environment.environment.background_color = Color(0.62, 0.74, 0.86)
	environment.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.environment.ambient_light_color = Color(0.5, 0.5, 0.55)
	add_child(environment)
	var sea := MeshInstance3D.new()
	var plane := PlaneMesh.new()
	plane.size = Vector2(4000, 4000)
	plane.material = _material(Color(0.2, 0.35, 0.55))
	sea.mesh = plane
	add_child(sea)

func _material(colour: Color) -> StandardMaterial3D:
	var material := StandardMaterial3D.new()
	material.albedo_color = colour
	return material
