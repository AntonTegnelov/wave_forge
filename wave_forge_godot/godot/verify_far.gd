## Draws the near ground and the far ground beyond it and checks the far ground is drawn exactly
## where some near ground is missing.
##
## Run by `../verify.sh` after `verify_ground.gd`. The far ground check's pack (`far.world.ron`) has
## the same ground at full detail and at a coarse scale of 8; the near ground reaches 7 chunks from
## the focus, one less than the fields it is built from, so it covers the coarse chunk at the
## origin whole, which gets no far ground, and part of the ring around it, which does.
extends SceneTree

const CELLS := 8
const RADIUS := 8
const SCALE := 8
const FAR_RADIUS := 24
const TIMEOUT_S := 60.0

var world: Node
var started_usec := 0

func _initialize() -> void:
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://far.world.ron"
	world.targets = PackedStringArray(["height", "far"])
	var radii: Dictionary[StringName, int] = {&"far": FAR_RADIUS}
	world.target_radii = radii
	world.seed = 3
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.view_radius = RADIUS
	world.collider_radius = -1
	world.ground_stage = "height"
	world.far_ground_stage = "far"
	root.add_child(world)
	if not world.start():
		_fail("the stages did not start")
		return
	world.follow(Vector3(1, 0, 1))
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("the ground had not arrived: %d near grounds, %d far" % [world.ground_chunks().size(), world.far_ground_chunks().size()])
		return true
	var stats: Dictionary = world.stats()
	var side := 2 * (RADIUS - 1) + 1
	# Every far chunk within its radius whose fields have arrived around it: those of the coarse
	# chunks two from the edge of the radius.
	if stats["pending_grounds"] > 0 or stats["pending_far_grounds"] > 0 \
			or world.ground_chunks().size() < side * side or world.far_ground_chunks().size() < 12:
		return false
	var near := {}
	for chunk: Vector3i in world.ground_chunks():
		near[chunk] = true
	var far: Array[Vector3i] = world.far_ground_chunks()
	for chunk in far:
		if _covered(chunk, near):
			_fail("the coarse chunk %s, whose every chunk has near ground, has far ground" % chunk)
			return true
	for chunk in [Vector3i(1, 0, 0), Vector3i(-1, 0, 0), Vector3i(0, 1, 0), Vector3i(0, -1, 0)]:
		if not far.has(chunk):
			_fail("the coarse chunk %s, partly beyond the near ground, has no far ground" % chunk)
			return true
	print("verify_far: %d near grounds, %d far grounds, none on a coarse chunk the near ground covers" % [near.size(), far.size()])
	quit(0)
	return true

## Whether every chunk of the WFC lattice the coarse `chunk` covers has near ground.
func _covered(chunk: Vector3i, near: Dictionary) -> bool:
	for y in SCALE:
		for x in SCALE:
			if not near.has(Vector3i(chunk.x * SCALE + x, chunk.y * SCALE + y, 0)):
				return false
	return true

func _fail(message: String) -> void:
	printerr("verify_far: " + message)
	quit(1)
