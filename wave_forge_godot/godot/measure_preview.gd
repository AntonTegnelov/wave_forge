## How long the islands preset's 3 by 3 preview takes to show a changed parameter, for P6 on a
## desktop (docs/product/user-stories.md, docs/guides/desktop-measurements.md). Run with a display
## and the renderer to measure, or headless:
##     godot --path . --rendering-driver vulkan --script measure_preview.gd -- --out results.txt
## It instances the addon's `presets/islands.tscn` at a view of one chunk around the origin, its
## near stages alone, P6's 3 by 3 preview; with `--far` its far ground too, out to about 1 km as the
## preset has it, which is what the editor's preview of the preset generates. Once everything has
## generated it changes each of the preset's parameters five times, between
## two values in its range, through `update_params`, as the inspector's sliders do. A change is
## shown when every chunk it dropped has arrived again and no ground waits to be built. It prints
## one line per parameter of `key=value` fields: whether the far ground was there, the parameter, the
## chunks a change dropped, and the
## milliseconds until it was shown at the median and at most; appended to `--out` as well when it
## is given.
extends SceneTree

const Generated := preload("res://generated.gd")
const TIMEOUT_S := 300.0
## Frames everything has to stay generated for before the first change.
const SETTLE := 30
const CHANGES := 5
## Each parameter's two values, both in its range.
const VALUES := {"land": [0.45, 0.6], "roughness": [0.4, 0.7], "trees": [0.5, 0.8]}

var stages: Node
var out := ""
var far := false
var started_usec := 0
var settled := 0
var last := {}
## The parameter being changed, how many changes it has had, and when the last was made.
var param := ""
var changes := 0
var changed_usec := 0
## What the last change dropped, and what of that has arrived since.
var dropped := {}
var arrived := {}
var timings := {}

func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	var i := 0
	while i < args.size():
		if args[i] == "--far":
			far = true
			i += 1
			continue
		if i + 1 >= args.size():
			_fail("%s needs a value" % args[i])
			return
		match args[i]:
			"--out": out = args[i + 1]
			_:
				_fail("unknown option %s" % args[i])
				return
		i += 2
	stages = (load("res://addons/wave_forge/presets/islands.tscn") as PackedScene).instantiate()
	stages.follow_camera = false
	stages.view_radius = 1
	stages.collider_radius = -1
	stages.navigation_radius = -1
	if not far:
		# The targets with a radius of their own are the far ground's.
		var near := PackedStringArray()
		for target: String in stages.targets:
			if not stages.target_radii.has(target):
				near.append(target)
		stages.targets = near
		var none: Dictionary[StringName, int] = {}
		stages.target_radii = none
		stages.far_ground_stage = ""
		stages.far_ground_material_stage = ""
		stages.far_ground_cavity_stage = ""
	stages.generation_failed.connect(func(reason: String) -> void: _fail(reason))
	stages.stage_dropped.connect(func(stage: String, chunk: Vector3i) -> void:
		if changed_usec > 0:
			dropped[[stage, chunk]] = true)
	stages.stage_ready.connect(func(stage: String, chunk: Vector3i) -> void:
		if changed_usec > 0:
			arrived[[stage, chunk]] = true)
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	# Added on the first frame, as a scene's node is, so it starts on ready.
	if not stages.is_inside_tree():
		root.add_child(stages)
		stages.follow(Vector3.ZERO)
		return false
	var now := Time.get_ticks_usec()
	if (now - started_usec) / 1e6 > TIMEOUT_S:
		_fail("the preview had not settled after %d s, changing %s" % [TIMEOUT_S, param])
		return true
	if changed_usec == 0:
		var stats: Dictionary = stages.stats()
		if not Generated.all_generated(stages, stats, last) or stats["pending_grounds"] > 0:
			settled = 0
			return false
		settled += 1
		if settled < SETTLE:
			return false
		return _change()
	if dropped.is_empty() or stages.stats()["pending_grounds"] > 0:
		return false
	for key: Array in dropped:
		if not arrived.has(key):
			return false
	timings[param].append((now - changed_usec) / 1000.0)
	timings[param + " dropped"] = dropped.size()
	return _change()

## Makes the next change, or reports and quits after the last; returns whether the run is over.
func _change() -> bool:
	if param == "" or changes == CHANGES:
		var names := VALUES.keys()
		var next := names.find(param) + 1
		if next == names.size():
			_report()
			quit(0)
			return true
		param = names[next]
		changes = 0
		timings[param] = PackedFloat64Array()
	changes += 1
	dropped.clear()
	arrived.clear()
	changed_usec = Time.get_ticks_usec()
	var value: float = VALUES[param][changes % 2]
	if not stages.update_params({param: value}):
		_fail("%s = %s was refused" % [param, value])
		return true
	return false

func _report() -> void:
	var lines := PackedStringArray()
	for name: String in VALUES:
		var sorted: PackedFloat64Array = timings[name].duplicate()
		sorted.sort()
		lines.append("measure_preview far=%s param=%s changes=%d dropped=%d p50_ms=%.1f max_ms=%.1f" % [
			far, name, sorted.size(), timings[name + " dropped"], sorted[sorted.size() / 2], sorted[sorted.size() - 1]])
	for line in lines:
		print(line)
	if out != "":
		var file := FileAccess.open(out, FileAccess.READ_WRITE) if FileAccess.file_exists(out) else FileAccess.open(out, FileAccess.WRITE)
		if file == null:
			_fail("%s cannot be written" % out)
			return
		file.seek_end()
		for line in lines:
			file.store_line(line)

func _fail(message: String) -> void:
	printerr("measure_preview: " + message)
	quit(1)
