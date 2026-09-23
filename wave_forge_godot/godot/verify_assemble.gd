## Places an Assemble stage's pieces from GDScript and checks they stand where the stage grew them.
##
## Run by `../verify.sh` after `verify_edits.gd`. Cells are twice as wide as they are tall, so an
## axis mixed up between the lattice and Godot's world shows. Each piece's transform stands at the
## centre of the cells it covers, on the levelled ground; a street, authored one cell by four,
## covers its cells once turned; and a house's door, authored in the middle of its south side,
## opens onto a street once the house is placed by its transform.
extends SceneTree

const CELLS := 8
const RADIUS := 4
const CELL := Vector3(2, 1, 2)
const TIMEOUT_S := 30.0

var world: Node
var ready := {}
var started_usec := 0

func _initialize() -> void:
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://assemble.world.ron"
	world.targets = PackedStringArray(["village", "level"])
	world.seed = 21
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = CELL
	world.view_radius = RADIUS
	world.collider_radius = -1
	root.add_child(world)
	world.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	world.stage_ready.connect(func(stage: String, chunk: Vector3i) -> void: ready[[stage, chunk]] = true)
	if not world.start():
		_fail("the stages did not start")
		return
	world.follow(Vector3(1, 0, 1))
	started_usec = Time.get_ticks_usec()

func _area() -> Array[Vector3i]:
	var chunks: Array[Vector3i] = []
	for y in range(-RADIUS, RADIUS + 1):
		for x in range(-RADIUS, RADIUS + 1):
			chunks.append(Vector3i(x, y, 0))
	return chunks

func _process(_delta: float) -> bool:
	for chunk in _area():
		if not (ready.has(["village", chunk]) and ready.has(["level", chunk])):
			if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
				_fail("chunk %s had not arrived" % chunk)
				return true
			return false
	return _check()

## The column of a point in Godot's world space.
func _column(at: Vector3) -> Vector2i:
	return Vector2i(floori(at.x / CELL.x), floori(at.z / CELL.z))

func _covers(stamp: Dictionary, column: Vector2i) -> bool:
	var low: Vector2i = stamp["min"]
	var high: Vector2i = stamp["max"]
	return column.x >= low.x and column.x < high.x and column.y >= low.y and column.y < high.y

## The levelled ground's height at a column, or NAN outside the chunks asked for.
func _level(column: Vector2i) -> float:
	var chunk := Vector3i(floori(float(column.x) / CELLS), floori(float(column.y) / CELLS), 0)
	var values: PackedFloat32Array = world.field_values("level", chunk)
	if values.is_empty():
		return NAN
	return values[posmod(column.y, CELLS) * CELLS + posmod(column.x, CELLS)]

func _check() -> bool:
	var stamps := {}
	for chunk in _area():
		for stamp: Dictionary in world.stamps("village", chunk):
			stamps[[chunk, stamp["id"]]] = stamp
	var pieces: Array = stamps.values()
	var houses := 0
	for stamp: Dictionary in pieces:
		var transform: Transform3D = stamp["transform"]
		var low: Vector2i = stamp["min"]
		var high: Vector2i = stamp["max"]
		var centre := Vector3((low.x + high.x) * 0.5 * CELL.x, 0, (low.y + high.y) * 0.5 * CELL.z)
		if not is_equal_approx(transform.origin.x, centre.x) or not is_equal_approx(transform.origin.z, centre.z):
			_fail("%s stands at %s, not at the centre of its cells %s" % [stamp["piece"], transform.origin, centre])
			return true
		var ground := _level(_column(transform.origin))
		if not is_nan(ground) and not is_equal_approx(transform.origin.y, ground * CELL.y):
			_fail("%s stands at height %f, not on the levelled ground" % [stamp["piece"], transform.origin.y])
			return true
		if stamp["piece"] == "street":
			var extent := transform.basis * Vector3(1 * CELL.x, 0, 4 * CELL.z)
			var covered := Vector2(absf(extent.x), absf(extent.z))
			if not covered.is_equal_approx(Vector2((high.x - low.x) * CELL.x, (high.y - low.y) * CELL.z)):
				_fail("a street turned covers %s, not its cells from %s to %s" % [covered, low, high])
				return true
		if stamp["piece"] == "house":
			var door := transform * Vector3(0, 0, -1 * CELL.z)
			var outside := transform * Vector3(0, 0, -2 * CELL.z)
			if not _covers(stamp, _column(door)):
				_fail("a house's door at %s lies outside it" % door)
				return true
			var column := _column(outside)
			var edge := RADIUS * CELLS
			if column.x < -edge or column.x >= edge + CELLS or column.y < -edge or column.y >= edge + CELLS:
				continue
			if not pieces.any(func(other: Dictionary) -> bool: return other["piece"] == "street" and _covers(other, column)):
				_fail("a house's door at %s opens onto no street" % door)
				return true
			houses += 1
	if houses < 3:
		_fail("only %d houses were checked" % houses)
		return true
	print("verify_assemble: %d pieces stand at their cells on the levelled ground; %d houses open onto streets" % [pieces.size(), houses])
	quit(0)
	return true

func _fail(message: String) -> void:
	printerr("verify_assemble: " + message)
	quit(1)
