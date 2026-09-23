## Fells a tree, cuts grass and raises the ground from GDScript, and checks what a save keeps.
##
## Run by `../verify.sh` after `verify_noise.gd`. A tree and a blade of grass the node placed are
## removed by the ids `point_sets` gave them, and the ground is raised under a position; the chunk
## comes back without them and with the ground higher. A second node given only the first one's
## `edits_log` holds the same. Then the first node's save, from `request_save` and the `saved`
## signal, is loaded into a third: the tree stays felled and the ground raised, but the grass,
## an ephemeral stage whose edits are never saved, grows back. Removing a point that is not there,
## raising a stage that is no field, and a log or a save that is not one are refused.
extends SceneTree

const CELLS := 8
const TIMEOUT_S := 30.0
const ORIGIN := Vector3i.ZERO

var first: Node
var second: Node
var ready := {}
var dropped := {}
var restored := {}
var loaded := {}
var third: Node
var cut := -1
var save_text := ""
var started_usec := 0
var phase := "arrive"
var felled := -1
var height := 0.0

func _initialize() -> void:
	first = _world()
	first.stage_ready.connect(func(stage: String, chunk: Vector3i) -> void: ready[[stage, chunk]] = true)
	first.stage_dropped.connect(func(stage: String, chunk: Vector3i) -> void: dropped[[stage, chunk]] = true)
	if not first.start():
		_fail("the stages did not start")
		return
	first.follow(Vector3(4, 0, 4))
	started_usec = Time.get_ticks_usec()

func _world() -> Node:
	var world: Node = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://edits.world.ron"
	world.targets = PackedStringArray(["ground", "trees", "grass"])
	world.seed = 5
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.view_radius = 1
	world.collider_radius = -1
	root.add_child(world)
	world.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	return world

func _ids(world: Node, stage := "trees") -> PackedInt64Array:
	var ids := PackedInt64Array()
	for set: Dictionary in world.point_sets(stage, ORIGIN):
		ids.append_array(set["ids"])
	return ids

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("the %s phase timed out" % phase)
		return true
	match phase:
		"arrive":
			if ready.has(["trees", ORIGIN]) and ready.has(["ground", ORIGIN]) and ready.has(["grass", ORIGIN]):
				_edit()
		"edited":
			if ready.has(["trees", ORIGIN]) and ready.has(["ground", ORIGIN]):
				return _check_edited()
		"restored":
			if restored.has(["trees", ORIGIN]) and restored.has(["ground", ORIGIN]):
				return _check_restored()
		"saving":
			if save_text != "":
				_load_save()
		"loaded":
			if loaded.has(["trees", ORIGIN]) and loaded.has(["ground", ORIGIN]) and loaded.has(["grass", ORIGIN]):
				return _check_loaded()
	return false

func _edit() -> void:
	felled = _ids(first)[0]
	cut = _ids(first, "grass")[0]
	height = first.field_values("ground", ORIGIN)[5 * CELLS + 5]
	if first.remove_point("trees", Vector3i(3, 3, 0), felled) or first.raise("trees", Vector3.ZERO, 1.0) or first.set_edits_log("nonsense"):
		_fail("a point that is not there, a stage that is no field or a log that is not one was taken")
		return
	ready.clear()
	if not first.remove_point("trees", ORIGIN, felled) or not first.remove_point("grass", ORIGIN, cut) or not first.raise("ground", Vector3(5.5, 0, 5.5), 7.0):
		_fail("the tree or the raise was refused")
		return
	phase = "edited"

func _check_edited() -> bool:
	if not dropped.has(["trees", ORIGIN]) or felled in _ids(first):
		_fail("the felled tree %d is still there" % felled)
		return true
	if first.field_values("ground", ORIGIN)[5 * CELLS + 5] != height + 7.0:
		_fail("the ground at (5, 5) is %s; it was %s before a raise of 7" % [first.field_values("ground", ORIGIN)[5 * CELLS + 5], height])
		return true
	print("verify_edits: the felled tree is gone and the ground raised by 7")
	second = _world()
	second.stage_ready.connect(func(stage: String, chunk: Vector3i) -> void: restored[[stage, chunk]] = true)
	if not second.start() or not second.set_edits_log(first.edits_log()):
		_fail("the second node did not take the first one's edits")
		return true
	second.follow(Vector3(4, 0, 4))
	phase = "restored"
	started_usec = Time.get_ticks_usec()
	return false

func _check_restored() -> bool:
	if _ids(second) != _ids(first) or second.field_values("ground", ORIGIN) != first.field_values("ground", ORIGIN):
		_fail("the edits restored from the log give another world")
		return true
	print("verify_edits: a node given only the edits log holds the same trees and ground")
	first.saved.connect(func(text: String) -> void: save_text = text)
	first.request_save()
	phase = "saving"
	started_usec = Time.get_ticks_usec()
	return false

func _load_save() -> void:
	third = _world()
	third.stage_ready.connect(func(stage: String, chunk: Vector3i) -> void: loaded[[stage, chunk]] = true)
	if not third.start() or third.load_save("nonsense") or not third.load_save(save_text):
		_fail("the third node took a save that is not one, or refused the first one's")
		return
	third.follow(Vector3(4, 0, 4))
	phase = "loaded"
	started_usec = Time.get_ticks_usec()

func _check_loaded() -> bool:
	if felled in _ids(third) or third.field_values("ground", ORIGIN) != first.field_values("ground", ORIGIN):
		_fail("the save lost the felled tree or the raised ground")
		return true
	if not cut in _ids(third, "grass"):
		_fail("the cut grass %d stayed cut; an ephemeral stage's edits are never saved" % cut)
		return true
	print("verify_edits: a save keeps the felled tree and the raised ground, and the cut grass grows back")
	quit(0)
	return true

func _fail(message: String) -> void:
	printerr("verify_edits: " + message)
	quit(1)
