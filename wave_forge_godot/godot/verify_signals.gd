## Asks for many products at once with a slow handler on `stage_ready` and checks the node spreads
## the signals over frames by time, so a game's handlers cannot stall a frame, and that every
## product's signal still arrives.
##
## Run by `../verify.sh` after `verify_stages.gd`. The ground check's pack generates its height and
## surface over a wide view, hundreds of products in the first frames; each `stage_ready` handler
## here waits `HANDLER_USEC` before returning, as a game's handler that reads what arrived might.
## No frame spends more than the node's signal budget and its slowest handler emitting them, since
## the node looks at the clock between signals, and every product the stages report generated gets
## its signal.
extends SceneTree

const RADIUS := 6
const TARGETS := ["height", "surface"]
const HANDLER_USEC := 20
## The node's budget for emitting signals in one frame (`SIGNALS_BUDGET_MS` in stages_node.rs).
const SIGNALS_BUDGET_MS := 0.5
const TIMEOUT_S := 60.0

var world: Node
var received := 0
## The longest one handler took, in milliseconds: the operating system's sleep is coarser than asked.
var slowest_handler_ms := 0.0
var started_usec := 0
var settled := 0

func _initialize() -> void:
	world = ClassDB.instantiate("WaveForgeStages")
	world.pack_file = "res://ground.world.ron"
	world.targets = PackedStringArray(TARGETS)
	world.seed = 3
	world.view_radius = RADIUS
	world.collider_radius = -1
	world.stage_ready.connect(_on_stage_ready)
	root.add_child(world)
	if not world.start():
		_fail("the stages did not start")
		return
	world.follow(Vector3(1, 0, 1))
	started_usec = Time.get_ticks_usec()

func _on_stage_ready(_stage: String, _chunk: Vector3i) -> void:
	var start := Time.get_ticks_usec()
	received += 1
	OS.delay_usec(HANDLER_USEC)
	slowest_handler_ms = maxf(slowest_handler_ms, (Time.get_ticks_usec() - start) / 1000.0)

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("the products had not all arrived: %d signals" % received)
		return true
	var stats: Dictionary = world.stats()
	var products := 0
	for stage: String in TARGETS:
		products += stats["stages"][stage]["products"] if stats["stages"].has(stage) else 0
	# Every product generated and signalled, a while after the last arrived.
	if products == 0 or received < products:
		settled = 0
		return false
	settled += 1
	if settled < 10:
		return false
	var slowest: float = stats["slowest_frame_signals_ms"]
	print("verify_signals: %d products, each signalled; the slowest frame spent %.3f ms emitting %d signals, a handler at most %.3f ms" % [products, slowest, stats["slowest_frame_events"], slowest_handler_ms])
	if products <= 256:
		_fail("only %d products, too few to need more than one frame" % products)
		return true
	if received != products:
		_fail("%d signals for %d products" % [received, products])
		return true
	# The signals' own cost beside their handlers' is a few microseconds each.
	if slowest > SIGNALS_BUDGET_MS + slowest_handler_ms + 0.05:
		_fail("a frame spent %.3f ms emitting signals, over its budget of %.1f ms" % [slowest, SIGNALS_BUDGET_MS])
		return true
	quit(0)
	return true

func _fail(message: String) -> void:
	printerr("verify_signals: FAILED: %s" % message)
	quit(1)
