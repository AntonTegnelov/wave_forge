## Draws a pack's ground with a material per category and checks what each chunk's material holds.
##
## Run by `../verify.sh` after `verify_scenes.gd`. The ground's materials come from a Rules stage:
## every chunk's ground gets its own copy of the reference ground shader's material, whose id
## texture holds the category of every vertex of the ground, the chunks beyond the far edges
## included, and whose cell is the node's. A material stage that is no Rules or Area stage is
## refused. Grass grows on every chunk of ground within its radius, each chunk's grass material
## holding the chunk's cover per column and the ground's height per vertex.
extends SceneTree

const CELLS := 8
const RADIUS := 2
const CELL := Vector3(2, 1, 3)
const TIMEOUT_S := 30.0

var world: Node
var ready := {}
var started_usec := 0

func _initialize() -> void:
	var wrong := _world("height")
	if wrong.start():
		_fail("a material stage that is a field was taken")
		return
	wrong.queue_free()
	world = _world("surface")
	world.stage_ready.connect(func(stage: String, chunk: Vector3i) -> void: ready[[stage, chunk]] = true)
	if not world.start():
		_fail("the stages did not start")
		return
	world.follow(Vector3(1, 0, 1))
	if world.water() != {"level": 3.0}:
		_fail("the pack's water is %s" % world.water())
		return
	started_usec = Time.get_ticks_usec()

func _world(material_stage: String) -> Node:
	var node: Node = ClassDB.instantiate("WaveForgeStages")
	node.pack_file = "res://ground.world.ron"
	node.targets = PackedStringArray(["height", "surface", "cover"])
	node.seed = 3
	node.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	node.cell_size = CELL
	node.view_radius = RADIUS
	node.collider_radius = -1
	node.ground_stage = "height"
	node.ground_material_stage = material_stage
	node.grass_stage = "cover"
	node.grass_radius = 1
	root.add_child(node)
	return node

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("the ground had not arrived: %d grounds, %d grass" % [world.ground_chunks().size(), world.grass_chunks().size()])
		return true
	var stats: Dictionary = world.stats()
	if stats["pending_grounds"] > 0 or world.ground_chunks().size() < 9 or world.grass_chunks().size() < 9:
		return false
	return _check() or _check_grass()

## The category of the world column `column`, from the chunk that holds it.
func _category(column: Vector2i) -> int:
	var chunk := Vector3i(floori(float(column.x) / CELLS), floori(float(column.y) / CELLS), 0)
	var values: PackedByteArray = world.categories("surface", chunk)
	return values[posmod(column.y, CELLS) * CELLS + posmod(column.x, CELLS)]

func _check() -> bool:
	var seen := {}
	for chunk: Vector3i in world.ground_chunks():
		var material: ShaderMaterial = world.ground_material_of(chunk)
		if material == null:
			_fail("the ground of %s has no material of its own" % chunk)
			return true
		if material.shader.code != world.ground_shader_code():
			_fail("the ground of %s is not drawn with the reference shader" % chunk)
			return true
		if material.get_shader_parameter("wave_forge_cell") != Vector2(CELL.x, CELL.z):
			_fail("the ground of %s has cells of %s" % [chunk, material.get_shader_parameter("wave_forge_cell")])
			return true
		var image: Image = material.get_shader_parameter("wave_forge_materials").get_image()
		if image.get_size() != Vector2i(CELLS + 1, CELLS + 1):
			_fail("the materials of %s are %s texels" % [chunk, image.get_size()])
			return true
		var data := image.get_data()
		for j in CELLS + 1:
			for i in CELLS + 1:
				var expected := _category(Vector2i(chunk.x * CELLS + i, chunk.y * CELLS + j))
				if data[j * (CELLS + 1) + i] != expected:
					_fail("vertex (%d, %d) of %s holds %d, not %d" % [i, j, chunk, data[j * (CELLS + 1) + i], expected])
					return true
				seen[expected] = true
	if seen.size() < 2:
		_fail("only %d materials on the ground" % seen.size())
		return true
	print("verify_ground: %d chunks of ground, each with the categories of its %d vertices, %d materials in all" % [world.ground_chunks().size(), (CELLS + 1) * (CELLS + 1), seen.size()])
	return false

## The height field's value at the world column `column`, in world units.
func _height(column: Vector2i) -> float:
	var chunk := Vector3i(floori(float(column.x) / CELLS), floori(float(column.y) / CELLS), 0)
	var values: PackedFloat32Array = world.field_values("height", chunk)
	return values[posmod(column.y, CELLS) * CELLS + posmod(column.x, CELLS)] * CELL.y

func _check_grass() -> bool:
	var covered := 0
	for chunk: Vector3i in world.grass_chunks():
		if maxi(absi(chunk.x), absi(chunk.y)) > 1:
			_fail("grass on %s, beyond its radius" % chunk)
			return true
		var material: ShaderMaterial = world.grass_material_of(chunk)
		if material.shader.code != world.grass_shader_code():
			_fail("the grass of %s is not drawn with the reference shader" % chunk)
			return true
		var cover: PackedByteArray = material.get_shader_parameter("wave_forge_cover").get_image().get_data()
		var field: PackedFloat32Array = world.field_values("cover", chunk)
		for i in field.size():
			if cover[i] != roundi(clampf(field[i], 0.0, 1.0) * 255.0):
				_fail("column %d of %s has cover %d for %f" % [i, chunk, cover[i], field[i]])
				return true
			if cover[i] > 0:
				covered += 1
		var heights: Image = material.get_shader_parameter("wave_forge_heights").get_image()
		for j in CELLS + 1:
			for i in CELLS + 1:
				var expected := _height(Vector2i(chunk.x * CELLS + i, chunk.y * CELLS + j))
				if not is_equal_approx(heights.get_pixel(i, j).r, expected):
					_fail("vertex (%d, %d) of %s holds height %f, not %f" % [i, j, chunk, heights.get_pixel(i, j).r, expected])
					return true
	if covered == 0:
		_fail("no column has grass")
		return true
	print("verify_ground: grass on %d chunks, %d columns covered, each chunk with its cover and the ground's heights" % [world.grass_chunks().size(), covered])
	quit(0)
	return true

func _fail(message: String) -> void:
	printerr("verify_ground: " + message)
	quit(1)
