## Gives a pack a painted map from GDScript and checks that its stage reads it.
##
## Run by `../verify.sh` after `verify_tables.gd`. This is the seam a map the game paints uses
## (stories G4 and G5): an RGB image with mipmaps, two pixels by two, one pixel per chunk, laid on
## the ground with its x along +x and its rows along +z. A colour the legend does not name is
## refused and changes nothing; a new painting drops and generates again the chunks that read it.
extends SceneTree

const CELLS := 8
const TIMEOUT_S := 30.0
const GREEN := Color8(0, 128, 0)
const BLUE := Color8(0, 0, 255)

var world: Node
var ready := {}
var dropped := {}
var started_usec := 0
var repainted := false

func _initialize() -> void:
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://painted.world.ron"
	world.targets = PackedStringArray(["biome"])
	world.seed = 1
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.view_radius = 1
	world.collider_radius = -1
	root.add_child(world)
	world.stage_ready.connect(func(stage: String, chunk: Vector3i) -> void: ready[[stage, chunk]] = true)
	world.stage_dropped.connect(func(stage: String, chunk: Vector3i) -> void: dropped[[stage, chunk]] = true)
	world.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	if not world.start():
		_fail("the stages did not start")
		return
	if world.give_painting("map", _painting([GREEN, BLUE, Color8(250, 0, 0), GREEN])):
		_fail("a colour the legend does not name was taken")
		return
	print("verify_painted: a colour the legend does not name is refused")
	if not world.give_painting("map", _painting([GREEN, BLUE, BLUE, GREEN])):
		_fail("the painting was refused")
		return
	world.follow(Vector3(4, 0, 4))
	started_usec = Time.get_ticks_usec()

## Two pixels by two in RGB with mipmaps: `colours` row by row.
func _painting(colours: Array) -> Image:
	var image := Image.create_empty(2, 2, false, Image.FORMAT_RGB8)
	for i in colours.size():
		image.set_pixel(i % 2, i / 2, colours[i])
	image.generate_mipmaps()
	return image

## Whether every column of `chunk` holds `category`.
func _all(chunk: Vector3i, category: String) -> bool:
	var names: PackedStringArray = world.category_names("biome")
	var values: PackedByteArray = world.categories("biome", chunk)
	if values.is_empty():
		return false
	for value in values:
		if names[value] != category:
			return false
	return true

func _process(_delta: float) -> bool:
	var waited := (Time.get_ticks_usec() - started_usec) / 1e6
	if waited > TIMEOUT_S:
		_fail("the painted chunks had not arrived after %.0f s" % waited)
		return true
	var expected := {
		Vector3i(0, 0, 0): "sea" if repainted else "forest",
		Vector3i(1, 0, 0): "sea",
		Vector3i(0, 1, 0): "sea",
		Vector3i(1, 1, 0): "forest",
	}
	for chunk: Vector3i in expected:
		if not ready.has(["biome", chunk]) or not _all(chunk, expected[chunk]):
			return false
	if repainted:
		if not dropped.has(["biome", Vector3i.ZERO]):
			_fail("a new painting arrived without dropping the chunks that read the old")
			return true
		print("verify_painted: a new painting dropped and regenerated the chunks that read it")
		quit(0)
		return true
	print("verify_painted: each chunk holds its pixel's category, the image's rows along +z")
	repainted = true
	ready.clear()
	if not world.give_painting("map", _painting([BLUE, BLUE, BLUE, GREEN])):
		_fail("the new painting was refused")
	return false

func _fail(message: String) -> void:
	printerr("verify_painted: " + message)
	quit(1)
