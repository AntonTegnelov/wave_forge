## Drives the extension the way a game would and checks what it produced.
##
## Run through `../verify.sh`, which builds the extension, copies it in and starts Godot headless.
## A focus runs across a strip of chunks and back in real time, which is the whole of what a game
## does: chunks appear ahead of it, chunks behind it are dropped, and the ones it returns to come back
## the same. It never waits for generation, so a generator that falls behind shows up as a chunk
## missing beside the player, and every frame is timed, so a stall on Godot's own thread shows up
## too. The script exits non-zero with a message on any failure, so a shell can tell whether it
## passed.
extends SceneTree

## The world is this many chunks along x, three along y, one tall.
const CHUNKS_X := 8
const CHUNKS_Y := 3
## Cells along each axis of one chunk.
const CELLS := 8
## World units per cell, so one chunk is 16 units across.
const CELL_SIZE := 2.0
## How many chunks to generate around the focus, and how far beyond that to keep them. A column two
## chunks out is asked for when the focus enters a chunk and is needed when it enters the next one,
## 16 units later.
const VIEW_RADIUS := 2
const EVICT_MARGIN := 1
## Every chunk this close to the focus's chunk has to have tiles on every frame of the walk.
const READY_RADIUS := 1
## Units per second: a running player, three times a walking pace of 1.4.
const PACE := 4.2
## The frame rate a game would cap its main loop at.
const FPS := 60
## How long Godot's own thread may spend processing a frame, including the extension and the signal
## handlers: half a frame at the 99th percentile, leaving the rest to rendering, and never a whole one.
const PROCESS_P99_MS := 8.0
const PROCESS_MAX_MS := 1000.0 / FPS
## Building the device, compiling kernels and generating the first view is loading, not play.
const LOAD_TIMEOUT_S := 180.0
## Tile indices, in the order `rules.ron` declares them.
const WATER := 0
const SAND := 1
const GRASS := 2
const FOREST := 3
## The focus starts in the middle of this chunk, runs to the middle of the other and back.
const WALK_FROM := 1
const WALK_TO := CHUNKS_X - 2

var world: Node
var updated := {}
var failed := {}
var evicted := {}
## The chunks that have tiles now, as the signals tell it.
var present := {}
## The tiles of the chunk the walk starts in, to compare against when it comes back.
var first_tiles: PackedInt32Array
var started_usec := 0
var walk_started_usec := -1
var last_frame_usec := 0
## Per frame of the walk: Godot's process time, and the time since the previous frame.
var process_ms := PackedFloat64Array()
var period_ms := PackedFloat64Array()
## Frames on which a chunk within the ready radius had no tiles, and the first of them.
var late_frames := 0
var first_late := ""
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
	Engine.max_fps = FPS
	started_usec = Time.get_ticks_usec()
	world.follow(_position_at(_start_x()))
	print("verify: starting in chunk ", world.chunk_at(_position_at(_start_x())))

## Loads until the chunks around the start are there, then runs the route by the clock, whatever
## generation is doing.
func _process(_delta: float) -> bool:
	if done:
		return true
	if not world.is_generating():
		_fail("generation stopped")
		return true
	var now := Time.get_ticks_usec()
	if walk_started_usec < 0:
		var loading := (now - started_usec) / 1e6
		if loading > LOAD_TIMEOUT_S:
			_fail("the first view was not there after %.0f s, %d generated" % [loading, updated.size()])
			return true
		if _ready_around(WALK_FROM):
			print("verify: loaded in %.2f s" % loading)
			first_tiles = world.tiles_at(Vector3i(WALK_FROM, 1, 0))
			walk_started_usec = now
			last_frame_usec = now
		return false

	# The time Godot's own thread spent on the previous frame: the extension draining its worker
	# and emitting signals, and this script's handlers.
	process_ms.append(Performance.get_monitor(Performance.TIME_PROCESS) * 1000.0)
	period_ms.append((now - last_frame_usec) / 1000.0)
	last_frame_usec = now
	var x := _x_at((now - walk_started_usec) / 1e6)
	if is_nan(x):
		_check()
		return true
	world.follow(_position_at(x))
	var chunk_x := int(floor(x / _chunk_units()))
	if not _ready_around(chunk_x):
		late_frames += 1
		if first_late.is_empty():
			first_late = "around chunk %d at %.2f s" % [chunk_x, (now - walk_started_usec) / 1e6]
	return false

## Where along x the focus is `t` seconds into the run, or NAN once it is back.
func _x_at(t: float) -> float:
	var one_way := (WALK_TO - WALK_FROM) * _chunk_units()
	var run := t * PACE
	if run > 2.0 * one_way:
		return NAN
	return _start_x() + (run if run <= one_way else 2.0 * one_way - run)

## Whether every chunk of the world within the ready radius of column `chunk_x` has tiles.
func _ready_around(chunk_x: int) -> bool:
	for x in range(chunk_x - READY_RADIUS, chunk_x + READY_RADIUS + 1):
		if x < 0 or x >= CHUNKS_X:
			continue
		for y in CHUNKS_Y:
			if not present.has(Vector3i(x, y, 0)):
				return false
	return true

func _chunk_units() -> float:
	return CELLS * CELL_SIZE

func _start_x() -> float:
	return (WALK_FROM + 0.5) * _chunk_units()

func _position_at(x: float) -> Vector3:
	# Godot's xz plane is the lattice's xy: the strip runs along x, and the middle row is y = 1.
	return Vector3(x, 0.0, 1.5 * _chunk_units())

## The value below which a share `q` of `values` lie.
func _quantile(values: PackedFloat64Array, q: float) -> float:
	var sorted := values.duplicate()
	sorted.sort()
	return sorted[int(round((sorted.size() - 1) * q))]

func _has_chunk(chunk: Vector3i) -> bool:
	return not world.tiles_at(chunk).is_empty()

func _on_chunk_updated(chunk: Vector3i) -> void:
	updated[chunk] = updated.get(chunk, 0) + 1
	present[chunk] = true

func _on_chunk_failed(chunk: Vector3i, status: String) -> void:
	failed[chunk] = status

func _on_chunk_evicted(chunk: Vector3i) -> void:
	evicted[chunk] = true
	present.erase(chunk)

func _on_generation_failed(reason: String) -> void:
	_fail("generation failed: " + reason)

## Everything the extension promised a game, checked against what it can see from GDScript.
func _check() -> void:
	done = true
	var seconds := (Time.get_ticks_usec() - walk_started_usec) / 1e6
	var slow_frames := 0
	for period in period_ms:
		if period > 1000.0 / FPS + 2.0:
			slow_frames += 1
	print("verify: ran to chunk %d and back at %.1f units/s in %.2f s over %d frames" % [WALK_TO, PACE, seconds, period_ms.size()])
	print("verify: process time p50 %.3f ms, p99 %.3f ms, max %.3f ms; frame period p50 %.2f ms, p99 %.2f ms, max %.2f ms, %d frames over %.1f ms" % [
		_quantile(process_ms, 0.5), _quantile(process_ms, 0.99), _quantile(process_ms, 1.0),
		_quantile(period_ms, 0.5), _quantile(period_ms, 0.99), _quantile(period_ms, 1.0),
		slow_frames, 1000.0 / FPS + 2.0])
	print("verify: %d chunks generated, %d dropped, %d late frames, stats %s" % [updated.size(), evicted.size(), late_frames, world.stats()])

	# The point of the worker thread: building the device, compiling kernels and dispatching all
	# happen off Godot's own thread, so the main loop keeps its frame time while a world appears.
	if _quantile(process_ms, 0.99) > PROCESS_P99_MS or _quantile(process_ms, 1.0) > PROCESS_MAX_MS:
		_fail("Godot's thread spent up to %.1f ms on a frame, %.1f ms at the 99th percentile" % [_quantile(process_ms, 1.0), _quantile(process_ms, 0.99)])
		return
	# And generation keeps ahead of a running player, which it cannot fake by making them wait.
	if late_frames > 0:
		_fail("%d frames had a chunk beside the focus without tiles, first %s" % [late_frames, first_late])
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
	if generated.size() < (2 * READY_RADIUS + 1) * CHUNKS_Y:
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
	var again: PackedInt32Array = world.tiles_at(Vector3i(WALK_FROM, 1, 0))
	if first_tiles.is_empty():
		_fail("the starting chunk was never read")
		return
	if again != first_tiles:
		_fail("chunk (%d, 1, 0) came back different after being dropped" % WALK_FROM)
		return

	print("verify: %d cells, every column one material, every neighbour legal, the prior obeyed, and the chunk walked back to is unchanged" % world_tiles.size())
	quit(0)

func _fail(message: String) -> void:
	printerr("verify: " + message)
	quit(1)
