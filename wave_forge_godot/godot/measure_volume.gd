## What a Volume stage costs while a player walks through it, for #71 and P1
## (docs/research/measurements.md). Headless it times the CPU's part: generating each chunk's
## volume on the stages' thread and building each surface, drawing and colliders included, on
## Godot's.
##
##     godot --headless --path . --script measure_volume.gd -- --speed 4.2 --seconds 20
##
## The pack is `caves.world.ron`: chunks of 16 by 16 columns and 64 levels, one cell a metre, seen
## 4 chunks around the player with colliders on the chunk around it. Once the first view is built
## the player walks along +x at `--speed` metres a second for `--seconds`, then one line of
## `key=value` fields is printed: the view's load time; each chunk's volume on the stages' thread,
## on average and at most; each surface's build on Godot's thread, overall and during the walk; the
## walk's frame intervals and the node's own time per frame, at the median, 99th percentile and
## most; and the slowest frame since the start with the surfaces and bodies it built.
extends SceneTree

const CELLS := 16
const RADIUS := 4
const LOAD_TIMEOUT_S := 300.0

var world: Node
var speed := 4.2
var seconds := 20.0
var phase := "load"
var walker := Vector3(8, 0, 8)
var started_usec := 0
var last_usec := 0
var frames := PackedFloat64Array()
## The node's own time on Godot's thread in each frame of the walk.
var node_frames := PackedFloat64Array()
var loaded := {}

func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	var i := 0
	while i + 1 < args.size():
		match args[i]:
			"--speed": speed = args[i + 1].to_float()
			"--seconds": seconds = args[i + 1].to_float()
			_:
				_fail("unknown option %s" % args[i])
				return
		i += 2
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://caves.world.ron"
	world.targets = PackedStringArray(["caves"])
	world.seed = 5
	world.chunk_cells = Vector3i(CELLS, CELLS, CELLS)
	world.cell_size = Vector3.ONE
	world.view_radius = RADIUS
	world.collider_radius = 1
	world.volume_stage = "caves"
	root.add_child(world)
	world.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	if not world.start():
		_fail("the stages did not start")
		return
	world.follow(walker)
	started_usec = Time.get_ticks_usec()

func _process(delta: float) -> bool:
	var now := Time.get_ticks_usec()
	match phase:
		"load":
			var stats: Dictionary = world.stats()
			if (now - started_usec) / 1e6 > LOAD_TIMEOUT_S:
				_fail("the view was not built in %d s" % LOAD_TIMEOUT_S)
				return true
			var inner := (2 * RADIUS - 1) * (2 * RADIUS - 1)
			if stats["pending_volumes"] == 0 and stats["pending_colliders"] == 0 and world.volume_chunks().size() >= inner:
				loaded = {"load_s": (now - started_usec) / 1e6, "surfaces": stats["volume_surfaces"], "surfaces_ms": stats["volume_surfaces_ms"]}
				phase = "walk"
				started_usec = now
				last_usec = now
		"walk":
			frames.append((now - last_usec) / 1000.0)
			node_frames.append(world.last_frame_ms())
			last_usec = now
			walker.x += speed * delta
			world.follow(walker)
			if (now - started_usec) / 1e6 >= seconds:
				_report(world.stats())
				quit(0)
				return true
	return false

func _report(stats: Dictionary) -> void:
	var sorted := frames.duplicate()
	sorted.sort()
	var node := node_frames.duplicate()
	node.sort()
	var caves: Dictionary = stats["stages"]["caves"]
	var walked: int = stats["volume_surfaces"] - loaded["surfaces"]
	var walked_ms: float = stats["volume_surfaces_ms"] - loaded["surfaces_ms"]
	print("measure_volume: load_s=%.2f chunks=%d volume_ms_per_chunk=%.3f volume_slowest_ms=%.3f surfaces=%d surface_ms_per_chunk=%.3f walk_surfaces=%d walk_surface_ms_per_chunk=%.3f frames=%d frame_p50_ms=%.2f frame_p99_ms=%.2f frame_max_ms=%.2f walk_node_p50_ms=%.3f walk_node_p99_ms=%.3f walk_node_max_ms=%.3f slowest_frame_ms=%.3f slowest_frame_grounds=%d slowest_frame_grounds_ms=%.3f slowest_frame_bodies=%d slowest_frame_bodies_ms=%.3f" % [
		loaded["load_s"], caves["products"], caves["ms"] / max(caves["products"], 1), caves["slowest_ms"],
		stats["volume_surfaces"], stats["volume_surfaces_ms"] / max(stats["volume_surfaces"], 1),
		walked, walked_ms / max(walked, 1),
		frames.size(), sorted[sorted.size() / 2], sorted[int(sorted.size() * 0.99)], sorted[sorted.size() - 1],
		node[node.size() / 2], node[int(node.size() * 0.99)], node[node.size() - 1],
		stats["slowest_frame_ms"], stats["slowest_frame_grounds"], stats["slowest_frame_grounds_ms"],
		stats["slowest_frame_bodies"], stats["slowest_frame_bodies_ms"]])

func _fail(message: String) -> void:
	printerr("measure_volume: " + message)
	quit(1)
