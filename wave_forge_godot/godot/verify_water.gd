## Draws a pack's lakes and rivers and checks what each chunk's water holds.
##
## Run by `../verify.sh` after `verify_ground.gd`. The water check's pack has a valley with a hollow
## in it and a river the script gives as a table row: each chunk of drawn ground along the river
## gets its water, every vertex where the water stands above the ground at the water's level and
## every other one under the ground, so the water meets its banks with no gap, each vertex as deep
## as the water stands over its ground and a dry one not at all, and every triangle facing up as
## Godot draws its front faces; a chunk on the valley's dry side gets none. The ground's wetness is
## whole under the water, partial on its banks and none on the dry side. A water stage that is no
## field is refused.
extends SceneTree

const CELLS := 8
const CELL := Vector3(2, 1, 2)
const TIMEOUT_S := 30.0
const WET := 0.05
## The river's chunks, and one on the dry valley side.
const RIVER := [Vector3i(2, 4, 0), Vector3i(3, 4, 0), Vector3i(4, 3, 0)]
const DRY := Vector3i(3, 2, 0)

var world: Node
var started_usec := 0

func _initialize() -> void:
	var wrong := _world("rivers")
	if wrong.start():
		_fail("a water stage that is no field was taken")
		return
	wrong.queue_free()
	world = _world("water")
	if not world.start():
		_fail("the stages did not start")
		return
	var river := {"id": 1, "x0": 2.0, "y0": 32.0, "x1": 62.0, "y1": 32.0, "width": 1.5}
	if not world.give_table("rivers", [river]):
		_fail("the river was not taken")
		return
	world.follow(Vector3(24 * CELL.x, 0, 28 * CELL.z))
	started_usec = Time.get_ticks_usec()

func _world(water_stage: String) -> Node:
	var node: Node = ClassDB.instantiate("WaveForgeStages")
	node.pack_file = "res://water.world.ron"
	node.targets = PackedStringArray(["ground", "water", "surface"])
	node.seed = 9
	node.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	node.cell_size = CELL
	node.view_radius = 3
	node.collider_radius = -1
	node.ground_stage = "ground"
	node.water_stage = water_stage
	node.ground_material_stage = "surface"
	node.ground_palette = PackedColorArray([Color(0.376, 0.384, 0.231)])
	node.water_material = StandardMaterial3D.new()
	root.add_child(node)
	return node

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("the water had not arrived: %s" % [world.water_chunks()])
		return true
	if world.stats()["pending_grounds"] > 0:
		return false
	var drawn: Array = world.ground_chunks()
	for chunk: Vector3i in RIVER + [DRY]:
		if not drawn.has(chunk):
			return false
	if world.water_chunks().has(DRY):
		_fail("water on %s, the valley's dry side" % DRY)
		return true
	var wet := 0
	var banks := 0
	for chunk: Vector3i in RIVER:
		var surface: Dictionary = world.water_surface_of(chunk)
		if surface.is_empty() or surface["indices"].is_empty():
			_fail("no water drawn on %s, along the river" % chunk)
			return true
		var positions: PackedVector3Array = surface["positions"]
		# Drawn as Godot's front faces, clockwise seen from above, so the water shows from above.
		var indices: PackedInt32Array = surface["indices"]
		for t in range(0, indices.size(), 3):
			var a := positions[indices[t]]
			var normal := (positions[indices[t + 1]] - a).cross(positions[indices[t + 2]] - a)
			if normal.y >= 0.0:
				_fail("a triangle of %s's water faces down from above: %s" % [chunk, normal])
				return true
		for j in CELLS + 1:
			for i in CELLS + 1:
				var column := Vector2i(chunk.x * CELLS + i, chunk.y * CELLS + j)
				var level := _value("water", column)
				var ground := _value("ground", column)
				var height := positions[j * (CELLS + 1) + i].y
				var depth: float = surface["depths"][j * (CELLS + 1) + i]
				var wetness := _wetness(chunk, i, j)
				if level - ground > WET:
					wet += 1
					if absf(height - level * CELL.y) > 1e-4:
						_fail("vertex (%d, %d) of %s stands at %f, not the water's %f" % [i, j, chunk, height, level * CELL.y])
						return true
					if absf(depth - (level - ground) * CELL.y) > 1e-4 or wetness != 1.0:
						_fail("vertex (%d, %d) of %s is %f deep and its ground %f wet, its water %f over the ground" % [i, j, chunk, depth, wetness, (level - ground) * CELL.y])
						return true
				elif height >= ground * CELL.y:
					_fail("vertex (%d, %d) of %s stands at %f, over its dry ground %f" % [i, j, chunk, height, ground * CELL.y])
					return true
				elif depth != 0.0:
					_fail("dry vertex (%d, %d) of %s is %f deep" % [i, j, chunk, depth])
					return true
				elif wetness > 0.0 and wetness < 1.0:
					banks += 1
	if wet < 10:
		_fail("only %d wet vertices along the river" % wet)
		return true
	if banks < 10:
		_fail("only %d vertices of wet bank along the river" % banks)
		return true
	for j in CELLS + 1:
		for i in CELLS + 1:
			if _wetness(DRY, i, j) != 0.0:
				_fail("vertex (%d, %d) of %s, the valley's dry side, is %f wet" % [i, j, DRY, _wetness(DRY, i, j)])
				return true
	print("verify_water: water on %d chunks; along the river %d wet vertices at the water's level and as deep as it stands, the rest under the ground; %d vertices of wet bank, none on the dry side" % [world.water_chunks().size(), wet, banks])
	quit(0)
	return true

## How wet the ground of `chunk` is at its vertex (`i`, `j`), as its material's channels hold it.
func _wetness(chunk: Vector3i, i: int, j: int) -> float:
	var material: ShaderMaterial = world.ground_material_of(chunk)
	return material.get_shader_parameter("wave_forge_channels").get_image().get_pixel(i, j).g

## The value of field stage `stage` at the world column `column`.
func _value(stage: String, column: Vector2i) -> float:
	var chunk := Vector3i(floori(float(column.x) / CELLS), floori(float(column.y) / CELLS), 0)
	var values: PackedFloat32Array = world.field_values(stage, chunk)
	return values[posmod(column.y, CELLS) * CELLS + posmod(column.x, CELLS)]

func _fail(message: String) -> void:
	printerr("verify_water: FAILED: %s" % message)
	quit(1)
