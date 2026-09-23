## A toy history of the continent, run once before play. Rivers run downhill from the highlands to
## the sea, villages are founded beside them and grow for a century, roads join every village to its
## nearest neighbour, and some villages burn. What it made comes back as tables of facts, which the
## stages turn into carved rivers, levelled roads, towns and ruins.
##
## It reads the world map one column per chunk (8 cells) through the atlas, and gives positions in
## cells, the unit every table of the pack uses.
extends RefCounted

## How many map columns it reads each way from the centre.
const MAP := 44
const CELLS_PER_COLUMN := 8
const YEARS := 100

var rng := RandomNumberGenerator.new()
var land := PackedFloat32Array()
## The map columns a river runs through.
var wet := {}
var rivers: Array[Dictionary] = []
var villages: Array[Dictionary] = []
var roads: Array[Dictionary] = []

## Runs the history over `world`'s map, from `seed`, and returns its tables by name.
func run(world: Node, seed: int) -> Dictionary:
	rng.seed = seed
	land = world.atlas("land", Vector2i(-MAP, -MAP), Vector2i(2 * MAP, 2 * MAP))
	for i in 14:
		_trace_river(_random_highland())
	for year in YEARS:
		_found_village()
		for village in villages:
			village["population"] += rng.randi_range(0, 3)
	for village in villages:
		village["size"] = 2 if village["population"] > 150 else 1
		village["fate"] = "burned" if rng.randf() < 0.25 else "standing"
	_join_neighbours()
	return {"rivers": rivers, "roads": roads, "villages": villages}

## The height of a map column: above 0 is land.
func _height(column: Vector2i) -> float:
	if absi(column.x) >= MAP or absi(column.y) >= MAP:
		return -1.0
	return land[(column.y + MAP) * 2 * MAP + column.x + MAP]

## The middle of a map column, in cells.
func _centre(column: Vector2i) -> Vector2:
	return (Vector2(column) + Vector2(0.5, 0.5)) * CELLS_PER_COLUMN

func _random_highland() -> Vector2i:
	for attempt in 1000:
		var column := Vector2i(rng.randi_range(-MAP, MAP - 1), rng.randi_range(-MAP, MAP - 1))
		if _height(column) > 0.8:
			return column
	return Vector2i.ZERO

## Walks downhill from `at`, one column at a time, until the sea or a hollow it cannot leave.
func _trace_river(at: Vector2i) -> void:
	var width := 1.0
	while _height(at) > 0.0:
		wet[at] = true
		var lowest := at
		for step in [Vector2i(1, 0), Vector2i(-1, 0), Vector2i(0, 1), Vector2i(0, -1)]:
			if _height(at + step) < _height(lowest):
				lowest = at + step
		if lowest == at:
			return
		var from := _centre(at)
		var to := _centre(lowest)
		rivers.append({"id": rivers.size() + 1, "x0": from.x, "y0": from.y, "x1": to.x, "y1": to.y, "width": width})
		width = minf(width + 0.15, 3.0)
		at = lowest

## Founds a village beside a river, on low dry land, at least six columns from any other.
func _found_village() -> void:
	var banks := wet.keys()
	if banks.is_empty():
		return
	var bank: Vector2i = banks[rng.randi_range(0, banks.size() - 1)]
	var at: Vector2i = bank + [Vector2i(1, 0), Vector2i(-1, 0), Vector2i(0, 1), Vector2i(0, -1)][rng.randi_range(0, 3)]
	if wet.has(at) or _height(at) < 0.05 or _height(at) > 0.7:
		return
	for village in villages:
		if Vector2(village["x"], village["y"]).distance_to(_centre(at)) < 6 * CELLS_PER_COLUMN:
			return
	var centre := _centre(at)
	villages.append({"id": villages.size() + 1, "x": centre.x, "y": centre.y, "population": 20, "size": 1, "fate": "standing"})

## A road from every village to its nearest neighbour, once for each pair.
func _join_neighbours() -> void:
	var joined := {}
	for village in villages:
		var nearest: Dictionary = {}
		for other in villages:
			var closer: bool = nearest.is_empty() or _distance(village, other) < _distance(village, nearest)
			if other != village and closer:
				nearest = other
		if nearest.is_empty():
			continue
		var pair := [mini(village["id"], nearest["id"]), maxi(village["id"], nearest["id"])]
		if joined.has(pair):
			continue
		joined[pair] = true
		roads.append({"id": roads.size() + 1, "x0": village["x"], "y0": village["y"], "x1": nearest["x"], "y1": nearest["y"], "width": 1.5})

func _distance(a: Dictionary, b: Dictionary) -> float:
	return Vector2(a["x"], a["y"]).distance_to(Vector2(b["x"], b["y"]))
