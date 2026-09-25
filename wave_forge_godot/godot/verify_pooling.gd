## Pools the nodes of scenes that can reset themselves, and only those.
##
## Run by `../verify.sh` after `verify_scenes.gd`, on the same village. It runs the village twice.
## First its houses and streets are bound to a scene whose root script defines
## `_wave_forge_reset`, then to the same scene without it. Each run turns a promotion radius on and
## off five times, which frees the nodes of the pieces outside it and places them again in one
## frame, and then moves away and frees the node.
## - With the reset, every piece placed again reuses a node placed before, reset once per release
##   and out of the tree while it waited. No run creates more nodes than it had placed at once, and
##   freeing the node frees its pool.
## - Without it, no node is ever reused.
## Both runs print the time a node took to place again: the node's process is run once to place
## them, and once before that with nothing to place, and the difference is shared among them. It
## includes `instance_spawned` and this script's handler of it, the same in both runs.
extends SceneTree

const CELLS := 8
const RADIUS := 6
const CELL := Vector3(2, 1, 2)
const TIMEOUT_S := 60.0
const CYCLES := 5
const RESETTING := "extends Node3D\nvar opened := false\nvar resets := 0\nfunc _wave_forge_reset() -> void:\n\topened = false\n\tresets += 1\n"
const PLAIN := "extends Node3D\nvar opened := false\n"

var world: Node
var pooled := true
var ready := {}
var spawned := {}
var created := {}
var releases := {}
var started_usec := 0
var phase := "arrive"
var pieces := {}
var radius := 0
var cycle := 0
var per_node_ms := {"pooled": [], "fresh": []}

func _initialize() -> void:
	_start()

## Starts the village, its pieces bound to a scene with a reset when `pooled`, otherwise without.
func _start() -> void:
	ready = {}
	spawned = {}
	created = {}
	releases = {}
	pieces = {}
	cycle = 0
	phase = "arrive"
	var scene := _scene(RESETTING if pooled else PLAIN)
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://scenes.world.ron"
	world.targets = PackedStringArray(["village"])
	world.seed = 21
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = CELL
	world.view_radius = RADIUS
	world.collider_radius = -1
	world.placement_budget_ms = 1000.0
	world.scenes = {"house": scene, "street": scene}
	root.add_child(world)
	world.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	world.stage_ready.connect(func(stage: String, chunk: Vector3i) -> void: ready[[stage, chunk]] = true)
	world.instance_spawned.connect(_on_spawned)
	if not world.start():
		_fail("the stages did not start")
		return
	world.follow(Vector3(1, 0, 1))
	started_usec = Time.get_ticks_usec()

## A Node3D with a mesh child and the script `source`, packed, so it is placed as nodes.
func _scene(source: String) -> PackedScene:
	var script := GDScript.new()
	script.source_code = source
	if script.reload() != OK:
		_fail("the scene's script does not compile")
	var top := Node3D.new()
	top.set_script(script)
	var child := MeshInstance3D.new()
	child.mesh = BoxMesh.new()
	top.add_child(child)
	child.owner = top
	var scene := PackedScene.new()
	scene.pack(top)
	top.free()
	return scene

func _on_spawned(node: Node3D, chunk: Vector3i, id: int) -> void:
	var key := node.get_instance_id()
	if created.has(key):
		if not pooled:
			_fail("a node of a scene without a reset was reused")
			return
		if node.get("opened"):
			_fail("a node was reused without its reset")
			return
		if node.get("resets") != releases[key]:
			_fail("a node released %d times was reset %d times" % [releases[key], node.get("resets")])
			return
	created[key] = node
	releases[key] = releases.get(key, 0)
	node.set("opened", true)
	spawned[[chunk, id]] = node

func _area() -> Array[Vector3i]:
	var chunks: Array[Vector3i] = []
	for y in range(-RADIUS, RADIUS + 1):
		for x in range(-RADIUS, RADIUS + 1):
			chunks.append(Vector3i(x, y, 0))
	return chunks

func _outside() -> Array:
	var keys := []
	for key in pieces:
		if maxi(absi(key[0].x), absi(key[0].y)) > radius:
			keys.append(key)
	return keys

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("the %s phase timed out" % phase)
		return true
	var stats: Dictionary = world.stats()
	match phase:
		"arrive":
			for chunk in _area():
				if not ready.has(["village", chunk]):
					return false
			if stats["pending_signals"] > 0 or stats["pending_placements"] > 0:
				return false
			for key in spawned:
				pieces[key] = spawned[key].global_transform
			if pieces.size() < 20:
				_fail("only %d pieces to place" % pieces.size())
				return true
			var distances: Array[int] = []
			for key in pieces:
				distances.append(maxi(absi(key[0].x), absi(key[0].y)))
			distances.sort()
			radius = distances[distances.size() / 2]
			if _outside().is_empty() or _outside().size() == pieces.size():
				_fail("a promotion radius of %d puts %d of %d pieces outside" % [radius, _outside().size(), pieces.size()])
				return true
			return _promote()
		"promote":
			if stats["pending_placements"] > 0:
				return false
			return _check_promoted(stats)
		"restore":
			if stats["pending_placements"] > 0:
				return false
			return _check_restored(stats)
		"leave":
			if stats["placed_nodes"] > 0:
				return false
			var waiting: int = stats["pooled_nodes"]
			if waiting != (pieces.size() if pooled else 0):
				_fail("%d nodes wait in pools after leaving %d pieces" % [waiting, pieces.size()])
				return true
			var nodes := created.values()
			root.remove_child(world)
			world.free()
			for node in nodes:
				if is_instance_valid(node) and not node.is_queued_for_deletion():
					_fail("a node outlived the WaveForgeStages node")
					return true
			return _next_run()
	return false

func _promote() -> bool:
	for key in _outside():
		releases[spawned[key].get_instance_id()] += 1
	world.promotion_radius = radius
	phase = "promote"
	started_usec = Time.get_ticks_usec()
	return false

## The pieces outside the radius are no longer nodes: pooled ones wait out of the tree, reset.
func _check_promoted(stats: Dictionary) -> bool:
	var outside := _outside()
	for key in outside:
		var node = spawned[key]
		if pooled:
			if not is_instance_valid(node) or node.is_inside_tree():
				_fail("a pooled node of %s is not waiting out of the tree" % [key])
				return true
			if node.get("opened") or node.get("resets") != releases[node.get_instance_id()]:
				_fail("the node of %s was not reset as it was released" % [key])
				return true
		elif is_instance_valid(node) and not node.is_queued_for_deletion():
			_fail("the node of %s outlived its placement" % [key])
			return true
	if stats["pooled_nodes"] != (outside.size() if pooled else 0):
		_fail("%d nodes pooled for %d released" % [stats["pooled_nodes"], outside.size()])
		return true
	world.set_process(false)
	var idle := _time_process()
	world.promotion_radius = -1
	var placing := _time_process()
	world.set_process(true)
	per_node_ms["pooled" if pooled else "fresh"].append(maxf(0.0, (placing - idle) / 1000.0 / outside.size()))
	phase = "restore"
	started_usec = Time.get_ticks_usec()
	return false

## Every piece is a node again where its stage put it; a pooled run made no new ones.
func _check_restored(stats: Dictionary) -> bool:
	if stats["placed_nodes"] != pieces.size():
		_fail("%d nodes placed for %d pieces" % [stats["placed_nodes"], pieces.size()])
		return true
	for key in pieces:
		var node = spawned[key]
		if not is_instance_valid(node) or not node.is_inside_tree():
			_fail("the piece %s has no node" % [key])
			return true
		if not node.global_transform.is_equal_approx(pieces[key]):
			_fail("the node of %s stands at %s" % [key, node.global_transform])
			return true
	var expected := pieces.size() + (0 if pooled else _outside().size() * (cycle + 1))
	if created.size() != expected:
		_fail("%d nodes created for %d pieces over %d cycles" % [created.size(), pieces.size(), cycle + 1])
		return true
	if stats["pooled_nodes"] != 0:
		_fail("%d nodes left in pools with every piece placed" % stats["pooled_nodes"])
		return true
	cycle += 1
	if cycle < CYCLES:
		return _promote()
	world.follow(Vector3(4000, 0, 4000))
	phase = "leave"
	started_usec = Time.get_ticks_usec()
	return false

## How long one run of the node's process takes, in microseconds.
func _time_process() -> int:
	var started := Time.get_ticks_usec()
	world.notification(Node.NOTIFICATION_PROCESS)
	return Time.get_ticks_usec() - started

func _next_run() -> bool:
	var name := "pooled" if pooled else "fresh"
	var times: Array = per_node_ms[name]
	times.sort()
	print("verify_pooling: %s, %d pieces, %d placed again per cycle; per node placed again %.4f ms (median of %d, from %.4f to %.4f)" % [name, pieces.size(), _outside().size(), times[times.size() / 2], times.size(), times[0], times[-1]])
	if not pooled:
		print("verify_pooling: scenes with a reset are pooled and reset once per release, scenes without one are never reused, and pools are freed with the node")
		quit(0)
		return true
	pooled = false
	_start()
	return false

func _fail(message: String) -> void:
	printerr("verify_pooling: " + message)
	quit(1)
