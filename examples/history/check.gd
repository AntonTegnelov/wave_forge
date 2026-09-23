## Checks the example headless: what a person following the README should see, asserted.
##
## Run with `godot --headless --path examples/history --script check.gd` after prepare.sh; the
## Godot check (wave_forge_godot/verify.sh) runs it too. The history runs twice from one seed and
## must give the same tables, and has made rivers, villages both standing and burned, and roads.
## Given to the stages, every village gets a site named by its row with a town on it, ruins where it
## burned, and a road levels the ground it runs over. Then a second node, given only the history
## as saved to JSON and read back, must hold the same ground around a village as the first.
extends SceneTree

const HISTORY := preload("res://history.gd")
const CELLS := 8
const CELL_SIZE := 2.0
const SEED := 7
const TIMEOUT_S := 240.0

var first: Node
var second: Node
var history := {}
var ready := {}
var restored := {}
## The chunks of the first standing village and the first burned one.
var standing := Vector3i.ZERO
var burned := Vector3i.ZERO
var started_usec := 0
var phase := "towns"

func _initialize() -> void:
	first = _world()
	first.stage_ready.connect(func(stage: String, chunk: Vector3i) -> void: ready[[stage, chunk]] = true)
	if not first.start():
		_fail("the stages did not start")
		return
	var began := Time.get_ticks_usec()
	history = HISTORY.new().run(first, SEED)
	print("check: the history took %.0f ms: %d river stretches, %d villages, %d roads" % [
		(Time.get_ticks_usec() - began) / 1000.0, history["rivers"].size(), history["villages"].size(), history["roads"].size()])
	if JSON.stringify(HISTORY.new().run(first, SEED)) != JSON.stringify(history):
		_fail("one seed gave two histories")
		return
	var fates := {}
	for village: Dictionary in history["villages"]:
		fates[village["fate"]] = village
	if history["rivers"].size() < 10 or history["roads"].is_empty() or not fates.has("standing") or not fates.has("burned"):
		_fail("the history made too little: %s" % [fates.keys()])
		return
	standing = _chunk(fates["standing"])
	burned = _chunk(fates["burned"])
	for table: String in history:
		if not first.give_table(table, history[table]):
			_fail("the table %s was refused" % table)
			return
	first.follow(_position(standing))
	started_usec = Time.get_ticks_usec()

func _world() -> Node:
	var world: Node = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://continent.world.ron"
	world.rules_files = {"city": "res://city.ron", "ruins": "res://ruins.ron"}
	world.targets = PackedStringArray(["level", "towns", "carved", "paved"])
	world.seed = SEED
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = Vector3.ONE * CELL_SIZE
	world.view_radius = 1
	world.collider_radius = -1
	root.add_child(world)
	world.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	return world

func _chunk(village: Dictionary) -> Vector3i:
	return Vector3i(floori(village["x"] / CELLS), floori(village["y"] / CELLS), 0)

func _position(chunk: Vector3i) -> Vector3:
	return Vector3((chunk.x + 0.5) * CELLS * CELL_SIZE, 0, (chunk.y + 0.5) * CELLS * CELL_SIZE)

func _process(_delta: float) -> bool:
	var waited := (Time.get_ticks_usec() - started_usec) / 1e6
	if waited > TIMEOUT_S:
		_fail("%s had not arrived after %.0f s" % [phase, waited])
		return true
	if phase == "towns":
		return _check_town(standing, "standing") if ready.has(["towns", standing]) else false
	if phase == "ruins":
		return _check_town(burned, "burned") if ready.has(["towns", burned]) else false
	if phase == "restored":
		return _check_restored() if restored.has(["level", standing]) else false
	return false

## The village's chunk has a site named by its row and a town on it.
func _check_town(chunk: Vector3i, fate: String) -> bool:
	var sites: Array = first.sites("villages", chunk)
	var town: Dictionary = first.town("towns", chunk)
	if sites.is_empty() or not sites[0].has("row") or town.is_empty() or town["row"] != sites[0]["row"]:
		_fail("the %s village at %s has sites %s and town %s" % [fate, chunk, sites, town.keys()])
		return true
	print("check: the %s village at %s has its site and its town, after %.1f s" % [fate, chunk, (Time.get_ticks_usec() - started_usec) / 1e6])
	if fate == "standing":
		phase = "ruins"
		first.follow(_position(burned))
		return false
	_check_roads()
	_restore()
	return false

## Somewhere along a road the paved ground differs from the carved ground under it.
func _check_roads() -> void:
	for chunk in [standing, burned]:
		var paved: PackedFloat32Array = first.field_values("paved", chunk)
		var carved: PackedFloat32Array = first.field_values("carved", chunk)
		if paved != carved:
			print("check: a road levels the ground by the village at %s" % chunk)
			return
	_fail("no road changed the ground at either village")

## A second node, given only the saved history, holds the same ground as the first.
func _restore() -> void:
	var saved: Dictionary = JSON.parse_string(JSON.stringify(history))
	second = _world()
	second.stage_ready.connect(func(stage: String, chunk: Vector3i) -> void: restored[[stage, chunk]] = true)
	if not second.start():
		_fail("the second node did not start")
		return
	for table: String in saved:
		if not second.give_table(table, saved[table]):
			_fail("the saved table %s was refused" % table)
			return
	first.follow(_position(standing))
	second.follow(_position(standing))
	phase = "restored"
	started_usec = Time.get_ticks_usec()

func _check_restored() -> bool:
	if not ready.has(["level", standing]):
		return false
	if first.field_values("level", standing) != second.field_values("level", standing):
		_fail("the history restored from its save gives other ground at %s" % standing)
		return true
	print("check: the history restored from its save gives the same ground")
	quit(0)
	return true

func _fail(message: String) -> void:
	printerr("check: " + message)
	quit(1)
