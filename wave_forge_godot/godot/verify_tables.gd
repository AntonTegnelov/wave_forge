## Gives a pack tables of facts from GDScript and checks that the stages read them.
##
## Run by `../verify.sh` after `verify_stages.gd`. This is the seam a history the game simulates
## uses (story N11): rows are Dictionaries, a village's `wealth` field reads the focused village's
## population, houses are generated under every village, sharing out its population, every
## village has a site, and a road between the villages is levelled into the hills. The check gives villages, reads back both tables, focuses a village before
## asking for any chunk, and waits for its field and its site; then it gives the villages again with
## a new population, and only the stages that read them must be dropped and generated again. Rows
## that break the table's columns, or whose sites would crowd each other, are refused.
extends SceneTree

const CELLS := 8
const TIMEOUT_S := 30.0
const CENTRE := Vector3i.ZERO

var world: Node
var ready := {}
var dropped := {}
var started_usec := 0
## Which population the `wealth` field is waited for, or -1 once the check is done.
var expected := 400.0
var regiven := false

func _initialize() -> void:
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://facts.world.ron"
	world.targets = PackedStringArray(["wealth", "places", "paved"])
	world.seed = 5
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
	if not world.give_table("villages", [_village(3, 4, 400, "river"), _village(9, 40, 250.0, "hill")]):
		_fail("the villages were refused")
		return
	if not world.give_table("roads", [{"id": 1, "x0": 4, "y0": 4, "x1": 40, "y1": 4, "width": 1}]):
		_fail("the road was refused")
		return
	if not _check_tables() or not _check_refusals():
		return
	if not world.focus_row("villages", PackedInt64Array([3])):
		_fail("village 3 could not be focused")
		return
	world.follow(Vector3(4, 0, 4))
	started_usec = Time.get_ticks_usec()

## A village's row: it stands at `x` cells along the lattice's x, 4 cells in, on a site of one chunk.
func _village(id: int, x: float, population: float, culture: String) -> Dictionary:
	return {"id": id, "x": x, "y": 4, "size": 1, "population": population, "culture": culture}

## The villages read back as they were given, and each village's houses share out its population.
func _check_tables() -> bool:
	var villages: Array = world.table_rows("villages")
	if villages.size() != 2 or villages[0]["id"] != PackedInt64Array([3]) or villages[1]["culture"] != "hill":
		_fail("the villages read back as %s" % [villages])
		return false
	var houses: Array = world.table_rows("houses")
	var people := {}
	var count := {}
	for house: Dictionary in houses:
		var village: int = house["id"][0]
		people[village] = people.get(village, 0.0) + house["people"]
		count[village] = count.get(village, 0) + 1
	if count != {3: 4, 9: 2} or people != {3: 400.0, 9: 250.0}:
		_fail("the houses are %s, holding %s people" % [count, people])
		return false
	print("verify_tables: 2 villages read back; 6 houses share their populations exactly")
	return true

## Rows that break the table's columns are refused and change nothing.
func _check_refusals() -> bool:
	var missing := _village(1, 4, 5, "hill")
	missing.erase("culture")
	var negative := _village(1, 4, 5, "hill")
	negative["id"] = -1
	var vector := _village(1, 4, 5, "hill")
	vector["population"] = Vector2.ONE
	var refused := [
		[_village(1, 4, 5, "sea")],
		[missing],
		[negative],
		[vector],
		[_village(1, 4, 5, "hill"), _village(2, 12, 5, "hill")],
	]
	for rows: Array in refused:
		if world.give_table("villages", rows):
			_fail("the rows %s were taken" % [rows])
			return false
	if world.give_table("houses", []) or world.focus_row("villages", PackedInt64Array([42])):
		_fail("a generated table took rows, or a missing village was focused")
		return false
	if world.table_rows("villages").size() != 2:
		_fail("a refused give changed the villages")
		return false
	print("verify_tables: rows that break the columns or crowd each other's sites, a generated table and a missing row are refused")
	return true

func _process(_delta: float) -> bool:
	var waited := (Time.get_ticks_usec() - started_usec) / 1e6
	if waited > TIMEOUT_S:
		_fail("the wealth of %s at population %.0f had not arrived after %.0f s" % [CENTRE, expected, waited])
		return true
	for stage in ["wealth", "places", "paved"]:
		if not ready.has([stage, CENTRE]):
			return false
	var wealth: PackedFloat32Array = world.field_values("wealth", CENTRE)
	var hills: PackedFloat32Array = world.field_values("hills", CENTRE)
	if wealth.is_empty() or hills.is_empty() or absf(wealth[0] - hills[0] - expected) > 1e-3:
		return false
	for i in wealth.size():
		if absf(wealth[i] - hills[i] - expected) > 1e-3:
			_fail("column %d of wealth is %.3f over hills of %.3f; the population is %.0f" % [i, wealth[i], hills[i], expected])
			return true
	print("verify_tables: the wealth field reads the focused village's population, %.0f" % expected)
	var sites: Array = world.sites("places", CENTRE)
	if sites.size() != 1 or sites[0]["row"] != PackedInt64Array([3]) or sites[0]["min"] != Vector2i.ZERO:
		_fail("the sites of %s are %s; village 3's should stand there" % [CENTRE, sites])
		return true
	print("verify_tables: village 3 has its site, named by its row")
	if not _check_road():
		return true
	if regiven:
		if dropped.has(["hills", CENTRE]) or dropped.has(["paved", CENTRE]) or not dropped.has(["places", CENTRE]):
			_fail("new villages dropped %s; the stages that read them, and not the hills or the road, should go" % [dropped.keys()])
			return true
		print("verify_tables: new villages dropped and regenerated only the stages that read them")
		quit(0)
		return true
	regiven = true
	# The road reads no village, so only the stages that do are waited for again.
	ready.erase(["wealth", CENTRE])
	ready.erase(["places", CENTRE])
	expected = 1000.0
	if not world.give_table("villages", [_village(3, 4, 1000, "river"), _village(9, 40, 250, "hill")]):
		_fail("the new villages were refused")
		return true
	started_usec = Time.get_ticks_usec()
	return false

## The road runs along row 4 of the centre chunk: its curve is named by its row, and every column
## of that row takes the hills' height there, as does the row either side, within its radius.
func _check_road() -> bool:
	var curves: Array = world.curves("roads", CENTRE)
	if curves.size() != 1 or curves[0]["row"] != PackedInt64Array([1]):
		_fail("the curves of %s are %s; road 1 should pass there" % [CENTRE, curves])
		return false
	var paved: PackedFloat32Array = world.field_values("paved", CENTRE)
	var hills: PackedFloat32Array = world.field_values("hills", CENTRE)
	for x in range(4, CELLS):
		for y in [3, 4]:
			if paved[y * CELLS + x] != hills[4 * CELLS + x]:
				_fail("paved column (%d, %d) is %.3f; the road's centre there is %.3f" % [x, y, paved[y * CELLS + x], hills[4 * CELLS + x]])
				return false
	print("verify_tables: the road is named by its row and levels the hills across it")
	return true

func _fail(message: String) -> void:
	printerr("verify_tables: " + message)
	quit(1)
