## The time a new world takes to make, for M2 on a desktop (docs/guides/desktop-measurements.md).
## Run headless: `godot --headless --path . --script measure.gd -- --out results.txt`. It runs the
## game's new-world flow (`main.tscn`) from nothing, in a folder of its own that it empties first,
## until the game plays the world, and prints one line of `key=value` fields (the seconds the
## history took, until the world was all there, the settlements, and the most memory the process
## held), appended to `--out` as well when it is given.
extends SceneTree

const FOLDER := "user://new_world_measure"
const PeakMemory := preload("res://peak_memory.gd")

var main: Node
var out := ""
var started_usec := 0
var history_s := -1.0

func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	if args.size() == 2 and args[0] == "--out":
		out = args[1]
	elif not args.is_empty():
		_fail("takes --out <file> or nothing, not %s" % " ".join(args))
		return
	_remove(ProjectSettings.globalize_path(FOLDER))
	main = (load("res://main.tscn") as PackedScene).instantiate()
	main.folder = FOLDER
	root.add_child(main)
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	var seconds := (Time.get_ticks_usec() - started_usec) / 1e6
	match main.phase:
		"generating":
			if history_s < 0.0:
				history_s = seconds
		"failed":
			_fail(main.status.text)
			return true
		"playing":
			var peak_mb := PeakMemory.peak_resident_mb()
			if peak_mb < 0.0:
				_fail("the system did not say how much memory the process held")
				return true
			var line := "measure_new_world history_s=%.1f seconds=%.1f settlements=%d peak_mb=%.0f" % [history_s, seconds, main.history.size(), peak_mb]
			print(line)
			if not out.is_empty():
				var file := FileAccess.open(out, FileAccess.READ_WRITE) if FileAccess.file_exists(out) else FileAccess.open(out, FileAccess.WRITE)
				if file == null:
					_fail("%s cannot be written" % out)
					return true
				file.seek_end()
				file.store_line(line)
			quit(0)
			return true
	return false

func _remove(path: String) -> void:
	var folder := DirAccess.open(path)
	if folder == null:
		return
	for name in folder.get_directories():
		_remove(path.path_join(name))
	for name in folder.get_files():
		folder.remove(name)
	DirAccess.remove_absolute(path)

func _fail(message: String) -> void:
	printerr("measure: " + message)
	quit(1)
