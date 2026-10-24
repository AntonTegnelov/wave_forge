## Asks for many products at once with a slow handler on `stage_ready` and checks the node spreads
## the signals over frames by time, so a game's handlers cannot stall a frame, and that every
## product's signal still arrives.
##
## Run by `../verify.sh` after `verify_stages.gd`. The ground check's pack generates its height and
## surface over a wide view, hundreds of products in the first frames; each `stage_ready` handler
## here waits `HANDLER_USEC` before returning, as a game's handler that reads what arrived might.
## The node looks at the clock between signals and starts another only while the frame is within
## its signal budget, so in no frame does a handler start later than the budget after the first;
## and every product the stages report generated gets its signal. The whole frame's emitting time is
## printed, not barred: a shared machine can preempt the thread during the last handler.
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
## The process frame of the last signal, when its first handler started, and the longest any frame's
## handlers went on starting after its first, in milliseconds.
var signal_frame := -1
var first_handler_usec := 0
var latest_start_ms := 0.0

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
	if Engine.get_process_frames() != signal_frame:
		signal_frame = Engine.get_process_frames()
		first_handler_usec = start
	latest_start_ms = maxf(latest_start_ms, (start - first_handler_usec) / 1000.0)
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
	print("verify_signals: %d products, each signalled; no handler started more than %.3f ms after its frame's first; the slowest frame spent %.3f ms emitting %d signals, a handler at most %.3f ms" % [products, latest_start_ms, slowest, stats["slowest_frame_events"], slowest_handler_ms])
	if products <= 256:
		_fail("only %d products, too few to need more than one frame" % products)
		return true
	if received != products:
		_fail("%d signals for %d products" % [received, products])
		return true
	# A signal's own cost before its handler starts is a few microseconds.
	if latest_start_ms > SIGNALS_BUDGET_MS + 0.05:
		_fail("a handler started %.3f ms after its frame's first, past the budget of %.1f ms" % [latest_start_ms, SIGNALS_BUDGET_MS])
		return true
	quit(0)
	return true

func _fail(message: String) -> void:
	printerr("verify_signals: FAILED: %s" % message)
	quit(1)
