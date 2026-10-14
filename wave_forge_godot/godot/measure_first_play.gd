## Time into a new world, for P3 (docs/product/user-stories.md): how long the small city preset
## takes from the moment its node enters the tree to a playable area around the player and to its
## first town, on a device of its own with its kernels compiled at run time. Run it twice with the
## same `--kernels` directory, first empty and then holding what the first run cached, as the
## desktop script does:
##     godot --headless --path . --script measure_first_play.gd -- --kernels <dir> --out results.txt
## It instances the addon's `presets/city.tscn` at its own settings, with a camera at the world's
## origin at eye height, and prints one line of `key=value` fields: whether the kernel directory
## held a cache when it started (`kernels=cached`) or not (`kernels=cold`), `playable_s`, the
## seconds until every chunk within one of the camera's has its ground and a collider, and
## `town_s`, the seconds until the first chunk of a town arrives; appended to `--out` as well when
## it is given.
extends SceneTree

const EYE_M := 1.7
const TIMEOUT_S := 300.0
## The chunks about the camera's own that a player needs to stand and walk on.
const AROUND := 1

var stages: Node
var kernels := ""
var out := ""
var cached := false
var started_usec := 0
var playable_s := -1.0
var town_s := -1.0

func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	var i := 0
	while i < args.size():
		if i + 1 >= args.size():
			_fail("%s needs a value" % args[i])
			return
		match args[i]:
			"--kernels": kernels = args[i + 1]
			"--out": out = args[i + 1]
			_:
				_fail("unknown option %s" % args[i])
				return
		i += 2
	if kernels.is_empty():
		_fail("--kernels names the directory the kernels are cached in")
		return
	cached = DirAccess.dir_exists_absolute(kernels) and not DirAccess.get_files_at(kernels).is_empty()
	var camera := Camera3D.new()
	camera.current = true
	root.add_child(camera)
	camera.look_at_from_position(Vector3(0.0, EYE_M, 0.0), Vector3(-1.0, EYE_M, -1.0))
	stages = (load("res://addons/wave_forge/presets/city.tscn") as PackedScene).instantiate()
	stages.kernel_cache = kernels
	stages.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	stages.stage_ready.connect(_on_ready)
	started_usec = Time.get_ticks_usec()
	root.add_child(stages)

func _on_ready(stage: String, chunk: Vector3i) -> void:
	if town_s < 0.0 and stage == "city" and not stages.town("city", chunk).is_empty():
		town_s = _seconds()

func _seconds() -> float:
	return (Time.get_ticks_usec() - started_usec) / 1e6

## Whether every chunk within `AROUND` of the camera's has its ground and a collider.
func _playable() -> bool:
	var grounds: Array = stages.ground_chunks()
	var colliders: Array = stages.collider_chunks()
	for y in range(-AROUND, AROUND + 1):
		for x in range(-AROUND, AROUND + 1):
			var chunk := Vector3i(x, y, 0)
			if not grounds.has(chunk) or not colliders.has(chunk):
				return false
	return true

func _process(_delta: float) -> bool:
	if started_usec == 0:
		return false
	if playable_s < 0.0 and _playable():
		playable_s = _seconds()
	if playable_s >= 0.0 and town_s >= 0.0:
		_report()
		quit(0)
		return true
	if _seconds() > TIMEOUT_S:
		_fail("after %d s the area was %s and the first town %s" % [TIMEOUT_S,
			"playable" if playable_s >= 0.0 else "not playable",
			"there" if town_s >= 0.0 else "not there"])
		return true
	return false

func _report() -> void:
	var line := "measure_first_play kernels=%s playable_s=%.2f town_s=%.2f" % [
		"cached" if cached else "cold", playable_s, town_s]
	print(line)
	if out != "":
		var file := FileAccess.open(out, FileAccess.READ_WRITE) if FileAccess.file_exists(out) else FileAccess.open(out, FileAccess.WRITE)
		if file == null:
			_fail("%s cannot be written" % out)
			return
		file.seek_end()
		file.store_line(line)

func _fail(message: String) -> void:
	printerr("measure_first_play: " + message)
	quit(1)
