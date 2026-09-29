## Draws a Scatter stage's candidates coloured by what became of each.
##
## Run by `../verify.sh` after `verify_paint.gd`. With `candidates_stage` set to the islands
## preset's trees, every chunk around the player gets its candidates drawn, and the legend counts
## as many kept as there are trees, each in its colour, and candidates rejected by the tree's
## conditions: off the grass, or outside the woods.
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
	quit(0)
	return true

func _fail(message: String) -> void:
	printerr("verify_candidates: " + message)
	quit(1)
