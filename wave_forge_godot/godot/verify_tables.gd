## Gives a pack tables of facts from GDScript and checks that the stages read them.
##
## Run by `../verify.sh` after `verify_stages.gd`. This is the seam a history the game simulates
## uses (story N11): rows are Dictionaries, a village's `wealth` field reads the focused village's
## population, and houses are generated under every village, sharing out its population. The check
## gives villages, reads back both tables, focuses a village before asking for any chunk, and waits
## for its field; then it gives the villages again with a new population, and only the stage that
## reads it must be dropped and generated again. Rows that break the table's columns are refused.
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
	world.targets = PackedStringArray(["wealth"])
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
	if not world.give_table("villages", [
		{"id": 3, "population": 400, "culture": "river"},
		{"id": 9, "population": 250.0, "culture": "hill"},
	]):
		_fail("the villages were refused")
		return
	if not _check_tables() or not _check_refusals():
		return
	if not world.focus_row("villages", PackedInt64Array([3])):
		_fail("village 3 could not be focused")
		return
	world.follow(Vector3(4, 0, 4))
	started_usec = Time.get_ticks_usec()

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
	var refused := [
		[{"id": 1, "population": 5, "culture": "sea"}],
		[{"id": 1, "population": 5}],
		[{"id": -1, "population": 5, "culture": "hill"}],
		[{"id": 1, "population": Vector2.ONE, "culture": "hill"}],
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
	print("verify_tables: rows that break the columns, a generated table and a missing row are refused")
	return true

func _process(_delta: float) -> bool:
	var waited := (Time.get_ticks_usec() - started_usec) / 1e6
	if waited > TIMEOUT_S:
		_fail("the wealth of %s at population %.0f had not arrived after %.0f s" % [CENTRE, expected, waited])
		return true
	if not ready.has(["wealth", CENTRE]):
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
	if regiven:
		if dropped.has(["hills", CENTRE]):
			_fail("new villages dropped the hills, which read no table")
			return true
		print("verify_tables: new villages dropped and regenerated only the stage that reads them")
		quit(0)
		return true
	regiven = true
	ready.clear()
	expected = 1000.0
	if not world.give_table("villages", [
		{"id": 3, "population": 1000, "culture": "river"},
		{"id": 9, "population": 250, "culture": "hill"},
	]):
		_fail("the new villages were refused")
		return true
	started_usec = Time.get_ticks_usec()
	return false

func _fail(message: String) -> void:
	printerr("verify_tables: " + message)
	quit(1)
