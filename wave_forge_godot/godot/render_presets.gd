## Renders a contact sheet of every preset the plugin ships, for judging its look (developer tool).
##
## Run through `../render_presets.sh`. For each preset, a tile per parameter and value: the
## parameter at its minimum, its default and its maximum, the others at their defaults, each a fresh
## node with the preset's settings, under the reference look (`reference_look.gd`) and seen from
## above at an angle once its view has settled. Rows are the parameters in the order of their
## names, columns minimum, default and maximum. Given `-- views`, it renders one sheet of every
## preset at its defaults from the reference views instead: a row per preset, columns eye level
## (1.7 m over the ground, looking toward the horizon), oblique (about 60 m away) and top-down, the
## sun to the side and behind, named for the renderer. Each sheet lands in Godot's user directory, and the script prints where and
## what each row is.
extends SceneTree

const Presets := preload("res://addons/wave_forge/presets.gd")
const ReferenceLook := preload("res://reference_look.gd")
const Generated := preload("res://generated.gd")
const TILE := Vector2i(640, 360)
const TIMEOUT_S := 120.0
## Frames to draw after a view settles, so what arrived last is on screen.
const SETTLE_FRAMES := 30
const AT := Vector3(1, 0, 1)
const VIEWS := ["eye", "oblique", "top"]

var tiles: Array = []
var tile := -1
var sheet: Image
var scene: Node3D
var stages: Node
var started_usec := 0
var settled_frames := 0
## Each target's products on the frame before, by name.
var last_products := {}
var camera: Camera3D

func _initialize() -> void:
	root.size = TILE
	if "views" in OS.get_cmdline_user_args():
		var paths := Presets.paths()
		var out := "views_%s" % RenderingServer.get_current_rendering_method()
		for row in paths.size():
			for column in VIEWS.size():
				tiles.append({"path": paths[row], "params": {}, "view": VIEWS[column], "row": row, "column": column, "rows": paths.size(), "last": row == paths.size() - 1 and column == VIEWS.size() - 1, "out": out})
		print("render_presets: %s, rows %s, columns %s" % [out, Array(paths).map(func(path: String) -> String: return path.get_file()), VIEWS])
		return
	for path in Presets.paths():
		var params := _params(path)
		var rows := params.size()
		for row in rows:
			var param: Dictionary = params[row]
			for column in 3:
				var value: float = [param["min"], param["default"], param["max"]][column]
				tiles.append({"path": path, "params": {param["name"]: value}, "view": "above", "row": row, "column": column, "rows": rows, "last": row == rows - 1 and column == 2, "out": path.get_file().get_basename()})
		print("render_presets: %s, rows %s, columns minimum, default and maximum" % [path.get_file(), params.map(func(param: Dictionary) -> String: return param["name"])])

func _process(_delta: float) -> bool:
	if stages == null:
		tile += 1
		if tile >= tiles.size():
			quit(0)
			return true
		_take(tiles[tile])
		return false
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		printerr("render_presets: tile %d did not settle" % tile)
		quit(1)
		return true
	# Above the ground at the followed point, however high it stands, and above the water over it.
	var ground: float = stages.ground_height(AT)
	if not is_nan(ground):
		var reach: float = stages.view_radius * stages.chunk_cells.x * stages.cell_size.x
		var water: Dictionary = stages.water()
		var surface := maxf(ground, water["level"] * stages.cell_size.y) if not water.is_empty() else ground
		var at := AT + Vector3(0, surface, 0)
		match tiles[tile]["view"]:
			"above":
				camera.look_at_from_position(at + Vector3(reach * 0.5, reach * 0.42, reach * 0.5), at)
			"eye":
				var eye := at + Vector3(0, 1.7, 0)
				camera.look_at_from_position(eye, eye + Vector3(1, -0.05, -1))
			"oblique":
				camera.look_at_from_position(at + Vector3(-30, 40, 30), at)
			"top":
				# High enough that the view's whole square fits the picture's height.
				camera.look_at_from_position(at + Vector3(0, reach * 2.0, 0), at, Vector3(0, 0, -1))
	var stats: Dictionary = stages.stats()
	var drawn: Array = stages.volume_chunks() if stages.ground_stage.is_empty() else stages.ground_chunks()
	# Every target stage has generated what it will: a stage that reads a slow one, trees on a
	# volume's top say, or a town still being solved, arrives after the ground is drawn.
	var generated := Generated.all_generated(stages, stats, last_products)
	if not generated or is_nan(ground) or drawn.is_empty() or stats["pending_grounds"] > 0 or stats["pending_volumes"] > 0 or stats["pending_placements"] > 0:
		settled_frames = 0
		return false
	settled_frames += 1
	if settled_frames < SETTLE_FRAMES:
		return false
	_capture(tiles[tile])
	scene.queue_free()
	stages = null
	return false

## A preset's parameters, read from a node given its settings.
func _params(path: String) -> Array:
	var probe: Node = ClassDB.instantiate("WaveForgeStages")
	var settings := Presets.settings(path)
	for property: String in settings:
		probe.set(property, settings[property])
	probe.start_on_ready = false
	root.add_child(probe)
	probe.start()
	var params: Array = probe.pack_params()
	probe.free()
	return params

func _take(spec: Dictionary) -> void:
	scene = Node3D.new()
	var sun := ReferenceLook.add_to(scene)
	camera = Camera3D.new()
	# Past the far ground, which reaches about 1 km.
	camera.far = 2000
	scene.add_child(camera)
	stages = ClassDB.instantiate("WaveForgeStages")
	var settings := Presets.settings(spec["path"])
	for property: String in settings:
		stages.set(property, settings[property])
	stages.params = spec["params"]
	stages.collider_radius = -1
	stages.navigation_radius = -1
	# Shadows reach the far corners of the view, seen from the highest camera.
	sun.directional_shadow_max_distance = 3.0 * stages.view_radius * stages.chunk_cells.x * stages.cell_size.x
	scene.add_child(stages)
	root.add_child(scene)
	stages.follow(AT)
	started_usec = Time.get_ticks_usec()
	settled_frames = 0
	last_products = {}

func _capture(spec: Dictionary) -> void:
	var picture := root.get_texture().get_image()
	picture.resize(TILE.x, TILE.y)
	if sheet == null:
		sheet = Image.create(TILE.x * 3, TILE.y * spec["rows"], false, picture.get_format())
	sheet.blit_rect(picture, Rect2i(Vector2i.ZERO, TILE), Vector2i(TILE.x * spec["column"], TILE.y * spec["row"]))
	if spec["last"]:
		DirAccess.make_dir_recursive_absolute("user://contact_sheets")
		var out := "user://contact_sheets/%s.png" % spec["out"]
		sheet.save_png(out)
		print("render_presets: wrote %s" % ProjectSettings.globalize_path(out))
		sheet = null
