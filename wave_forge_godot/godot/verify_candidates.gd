## Draws a Scatter stage's candidates coloured by what became of each.
##
## Run by `../verify.sh` after `verify_paint.gd`. With `candidates_stage` set to the islands
## preset's trees, every chunk around the player gets its candidates drawn, and the legend counts
## as many kept as there are trees, each in its colour, and candidates rejected by the tree's
## conditions: off the grass, or outside the woods. Hovering (N5): the candidate under a tree is a
## kept one with both conditions holding and no slope or water read, one off the grass reads 0 for
## its first condition, which fails, nothing lies far away, and the dock's panel shows what the
## node gives.
extends SceneTree

const CELLS := 8
const RADIUS := 2
const TIMEOUT_S := 30.0

var world: Node
var started_usec := 0

func _initialize() -> void:
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://islands.world.ron"
	world.targets = PackedStringArray(["height", "trees"])
	world.seed = 5
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.view_radius = RADIUS
	world.collider_radius = -1
	world.params = {"land": 0.7, "trees": 0.5}
	world.candidates_stage = "trees"
	root.add_child(world)
	world.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	if not world.start():
		_fail("the stages did not start")
		return
	world.follow(Vector3(1, 0, 1))
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("the candidates were not all drawn: %d chunks" % world.candidate_chunks().size())
		return true
	var drawn: Array = world.candidate_chunks()
	for y in range(-RADIUS, RADIUS + 1):
		for x in range(-RADIUS, RADIUS + 1):
			if not drawn.has(Vector3i(x, y, 0)):
				return false
	var trees := 0
	for chunk in drawn:
		for set: Dictionary in world.point_sets("trees", chunk):
			trees += set["ids"].size()
	var counts := {}
	for verdict: Dictionary in world.candidate_legend():
		counts[verdict["name"]] = verdict["count"]
		if not verdict["colour"] is Color:
			_fail("the verdict %s has no colour" % verdict["name"])
			return true
	if counts.get("kept", 0) != trees or trees == 0:
		_fail("%d kept candidates for %d trees: %s" % [counts.get("kept", 0), trees, counts])
		return true
	if counts.get("condition 0", 0) == 0 or counts.get("condition 1", 0) == 0:
		_fail("no candidate rejected off the grass or outside the woods: %s" % counts)
		return true
	print("verify_candidates: %d chunks drawn, the legend %s, the kept ones exactly the trees" % [drawn.size(), counts])
	var problem := _hovering(drawn)
	if not problem.is_empty():
		_fail(problem)
		return true
	print("verify_candidates: hovering a tree shows a kept candidate, both conditions holding; one off the grass reads 0 and fails its first; the panel shows it")
	quit(0)
	return true

## What is wrong with hovering over the drawn candidates, or nothing.
func _hovering(drawn: Array) -> String:
	# The first tree drawn, from its transform's origin.
	var buffer := PackedFloat32Array()
	for chunk in drawn:
		for set: Dictionary in world.point_sets("trees", chunk):
			if buffer.is_empty():
				buffer = set["transforms"]
	var tree := Vector3(buffer[3], buffer[7], buffer[11])
	# The dock asks along the ray under the mouse. This node has no ground_stage: the ray meets the
	# ground the candidates stand on, the Scatter stage's height.
	var looked: Dictionary = world.candidate_under(tree + Vector3(0.0, 40.0, 0.0), Vector3.DOWN)
	if looked.get("verdict") != "kept":
		return "looking down at the tree at %s: %s" % [tree, looked]
	var under: Dictionary = world.candidate_near(tree, 0.01)
	if under.get("verdict") != "kept" or under["conditions"].size() != 2:
		return "under the tree at %s: %s" % [tree, under]
	for condition: Dictionary in under["conditions"]:
		if not condition["holds"]:
			return "a kept candidate with a condition failing: %s" % under
	if under.has("slope") or under.has("water_depth"):
		return "a slope or water read by a stage with neither: %s" % under
	var off := {}
	for x in range(-RADIUS * CELLS, (RADIUS + 1) * CELLS):
		for z in range(-RADIUS * CELLS, (RADIUS + 1) * CELLS):
			var near: Dictionary = world.candidate_near(Vector3(x + 0.5, 0, z + 0.5), 0.75)
			if near.get("verdict") == "condition 0":
				off = near
	if off.is_empty() or off["conditions"][0]["holds"] or off["conditions"][0]["value"] != 0.0:
		return "a candidate off the grass: %s" % off
	if not world.candidate_near(Vector3(10000, 0, 10000), 1.0).is_empty():
		return "a candidate far from every chunk drawn"
	var panel: VBoxContainer = (load("res://addons/wave_forge/candidate_panel.gd") as GDScript).new()
	panel.show_candidate(off)
	var shown: String = panel.title.text + "\n" + panel.details.text
	panel.free()
	if not shown.contains("condition 0") or not shown.contains("condition 0: 0.000, fails"):
		return "the panel shows %s" % shown
	return ""

func _fail(message: String) -> void:
	printerr("verify_candidates: " + message)
	quit(1)
