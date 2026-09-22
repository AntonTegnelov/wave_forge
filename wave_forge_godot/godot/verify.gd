## Drives the extension the way a game would and checks what it produced.
##
## Run through `../verify.sh`, which builds the extension, copies it in and starts Godot headless.
## A focus walks across a strip of chunks and back, which is the whole of what a game does: chunks
## appear ahead of it, chunks behind it are dropped, and the ones it returns to come back the same.
## The script exits non-zero with a message on any failure, so a shell can tell whether it passed.
extends SceneTree

## The world is this many chunks along x, three along y, one tall.
const CHUNKS_X := 8
const CHUNKS_Y := 3
## Cells along each axis of one chunk.
const CELLS := 8
## World units per cell, so one chunk is 16 units across.
const CELL_SIZE := 2.0
## How many chunks to keep around the focus, and how far beyond that to keep them.
const VIEW_RADIUS := 1
const EVICT_MARGIN := 1
## Tile indices, in the order `rules.ron` declares them.
const WATER := 0
const SAND := 1
const GRASS := 2
const FOREST := 3
## The focus walks to this chunk and back, one chunk at a time.
const WALK_TO := CHUNKS_X - 2

var world: Node
var updated := {}
var failed := {}
var evicted := {}
## The tiles of the chunk the walk starts in, to compare against when it comes back.
var first_tiles: PackedInt32Array
var seconds := 0.0
var frames := 0
var focus_x := 1
var walking_back := false
var done := false

func _initialize() -> void:
	world = ClassDB.instantiate("WaveForgeWorld")
	if world == null:
		_fail("the extension did not register WaveForgeWorld")
		return
	world.seed = 7
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = Vector3(CELL_SIZE, CELL_SIZE, CELL_SIZE)
	world.view_radius = VIEW_RADIUS
	world.halo = 1
	world.evict_margin = EVICT_MARGIN
	world.world_chunks = Vector3i(CHUNKS_X, CHUNKS_Y, 1)
	# What a game does with a prior: no water on the ground layer, and no forest against the edges
	# of a bounded world. The rule set only lets a material meet itself vertically, so a ground
	# layer without water is a world without water, which is easy to check. The layers above it are
	# left open with an empty array, which is how a scene says "anything goes here".
	var layers: Array[PackedInt32Array] = [PackedInt32Array([SAND, GRASS, FOREST]), PackedInt32Array()]
	world.set_layer_tiles(layers)
	for axis in 4:
		world.ban_tiles_on_face(axis, PackedInt32Array([FOREST]))
	world.chunk_updated.connect(_on_chunk_updated)
	world.chunk_failed.connect(_on_chunk_failed)
	world.chunk_evicted.connect(_on_chunk_evicted)
	world.generation_failed.connect(_on_generation_failed)
	root.add_child(world)

	var rules := FileAccess.get_file_as_string("res://rules.ron")
	if rules.is_empty():
		_fail("res://rules.ron is missing")
		return
	if not world.start(rules):
		_fail("the rule set was refused")
		return
	world.follow(_position_at(focus_x))
	print("verify: starting in chunk ", world.chunk_at(_position_at(focus_x)))

## A player who walks no faster than generation: the focus steps on once everything within its
## view radius is there, which is what a game would gate movement or streaming on.
func _process(delta: float) -> bool:
	if done:
		return true
	seconds += delta
	frames += 1
	if not world.is_generating():
		_fail("generation stopped")
		return true
	if seconds > 180.0:
		_fail("stuck at chunk %d after %d frames and %.0f s, %d generated" % [focus_x, frames, seconds, updated.size()])
		return true
	if not _view_complete():
		return false

	if focus_x == 1 and first_tiles.is_empty():
		first_tiles = world.tiles_at(Vector3i(1, 1, 0))
	if walking_back and focus_x == 1:
		_check()
		return true
	if focus_x == WALK_TO:
		walking_back = true
	focus_x += -1 if walking_back else 1
	world.follow(_position_at(focus_x))
	return false

## Whether every chunk of the world within the view radius of the focus has tiles.
func _view_complete() -> bool:
	for x in range(focus_x - VIEW_RADIUS, focus_x + VIEW_RADIUS + 1):
		if x < 0 or x >= CHUNKS_X:
			continue
		for y in CHUNKS_Y:
			if not _has_chunk(Vector3i(x, y, 0)):
				return false
	return true

func _position_at(chunk_x: int) -> Vector3:
	var chunk_units := CELLS * CELL_SIZE
	# Godot's xz plane is the lattice's xy: the strip runs along x, and the middle row is y = 1.
	return Vector3((float(chunk_x) + 0.5) * chunk_units, 0.0, 1.5 * chunk_units)

func _has_chunk(chunk: Vector3i) -> bool:
	return not world.tiles_at(chunk).is_empty()

func _on_chunk_updated(chunk: Vector3i) -> void:
	updated[chunk] = updated.get(chunk, 0) + 1

func _on_chunk_failed(chunk: Vector3i, status: String) -> void:
	failed[chunk] = status

func _on_chunk_evicted(chunk: Vector3i) -> void:
	evicted[chunk] = true

func _on_generation_failed(reason: String) -> void:
	_fail("generation failed: " + reason)

## Everything the extension promised a game, checked against what it can see from GDScript.
func _check() -> void:
	done = true
	var fps := frames / maxf(seconds, 0.001)
	print("verify: walked to chunk %d and back in %.2f s over %d frames (%.0f frames per second)" % [WALK_TO, seconds, frames, fps])
	print("verify: %d chunks generated, %d dropped, stats %s" % [updated.size(), evicted.size(), world.stats()])

	# The point of the worker thread: building the device, compiling kernels and dispatching all
	# happen off Godot's own thread, so the main loop keeps running while a world appears.
	if fps < 30.0:
		_fail("the main loop ran at %.0f frames per second while generating" % fps)
		return
	if not failed.is_empty():
		_fail("chunks could not be placed: %s" % failed)
		return
	if evicted.is_empty():
		_fail("nothing was dropped although the focus walked away from it")
		return
	for chunk: Vector3i in evicted:
		if _has_chunk(chunk) and not updated.has(chunk):
			_fail("chunk %s was reported as dropped but its tiles are still there" % chunk)
			return

	var generated: Array = world.generated_chunks()
	if generated.size() < (2 * VIEW_RADIUS + 1) * CHUNKS_Y:
		_fail("only %d chunks around the focus at the end: %s" % [generated.size(), generated])
		return

	var names := ["water", "sand", "grass", "forest"]
	var world_tiles := {}
	for chunk: Vector3i in generated:
		var tiles: PackedInt32Array = world.tiles_at(chunk)
		if tiles.size() != CELLS * CELLS * CELLS:
			_fail("chunk %s has %d tiles, expected %d" % [chunk, tiles.size(), CELLS * CELLS * CELLS])
			return
		for z in CELLS:
			for y in CELLS:
				for x in CELLS:
					var tile := tiles[(z * CELLS + y) * CELLS + x]
					if tile < 0 or tile >= names.size():
						_fail("chunk %s cell (%d, %d, %d) holds tile %d" % [chunk, x, y, z, tile])
						return
					# The lattice's y is the second axis of a chunk coordinate, as in Rust.
					world_tiles[Vector3i(chunk.x * CELLS + x, chunk.y * CELLS + y, z)] = tile

	# The rule set only lets a material meet itself vertically, so every column is one material.
	for cell: Vector3i in world_tiles:
		if cell.z != 0:
			continue
		var material: int = world_tiles[cell]
		for z in range(1, CELLS):
			var above: int = world_tiles[Vector3i(cell.x, cell.y, z)]
			if above != material:
				_fail("column (%d, %d) holds %s at z=0 and %s at z=%d" % [cell.x, cell.y, names[material], names[above], z])
				return

	# And horizontally, a material only meets itself or the next band along, which is the rule the
	# generated world has to obey across chunk seams as well as inside a chunk.
	for cell: Vector3i in world_tiles:
		var material: int = world_tiles[cell]
		for step: Vector3i in [Vector3i(1, 0, 0), Vector3i(0, 1, 0)]:
			var beside: Vector3i = cell + step
			if not world_tiles.has(beside):
				continue
			var other: int = world_tiles[beside]
			if absi(other - material) > 1:
				_fail("%s at %s sits beside %s" % [names[material], cell, names[other]])
				return

	# The prior the scene set: the ground layer had no water, so nothing does, and the world's
	# edges had no forest.
	for cell: Vector3i in world_tiles:
		var material: int = world_tiles[cell]
		if material == WATER:
			_fail("water at %s although the ground layer forbade it" % cell)
			return
		var on_edge: bool = cell.x == 0 or cell.x == CHUNKS_X * CELLS - 1 or cell.y == 0 or cell.y == CHUNKS_Y * CELLS - 1
		if on_edge and material == FOREST:
			_fail("forest at %s although the world's faces forbade it" % cell)
			return

	# A chunk the focus left and came back to is generated again from its coordinate alone, so it
	# has to hold what it held the first time.
	var again: PackedInt32Array = world.tiles_at(Vector3i(1, 1, 0))
	if first_tiles.is_empty():
		_fail("the starting chunk was never read")
		return
	if again != first_tiles:
		_fail("chunk (1, 1, 0) came back different after being dropped")
		return

	print("verify: %d cells, every column one material, every neighbour legal, the prior obeyed, and the chunk walked back to is unchanged" % world_tiles.size())
	quit(0)

func _fail(message: String) -> void:
	printerr("verify: " + message)
	quit(1)
