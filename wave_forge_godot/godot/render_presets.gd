## Renders a contact sheet of every preset the plugin ships, for judging its look (developer tool).
##
## Run through `../render_presets.sh`. For each preset, a tile per parameter and value: the
## parameter at its minimum, its default and its maximum, the others at their defaults, each a fresh
## node with the preset's settings, lit and seen from above at an angle once its view has settled.
## Rows are the parameters in the order of their names, columns minimum, default and maximum. Each
## sheet lands in Godot's user directory, and the script prints where and what each row is.
extends SceneTree

const Presets := preload("res://addons/wave_forge/presets.gd")
const TILE := Vector2i(640, 360)
const TIMEOUT_S := 120.0
## Frames to draw after a view settles, so what arrived last is on screen.
const SETTLE_FRAMES := 30
const AT := Vector3(1, 0, 1)

var tiles: Array = []
var tile := -1
var sheet: Image
var scene: Node3D
var stages: Node
var started_usec := 0
var settled_frames := 0

func _initialize() -> void:
	root.size = TILE
	for path in Presets.paths():
		var params := _params(path)
		var rows := params.size()
		for row in rows:
			var param: Dictionary = params[row]
			for column in 3:
				var value: float = [param["min"], param["default"], param["max"]][column]
				tiles.append({"path": path, "params": {param["name"]: value}, "row": row, "column": column, "rows": rows, "last": row == rows - 1 and column == 2})
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
	var stats: Dictionary = stages.stats()
	if stages.ground_chunks().is_empty() or stats["pending_grounds"] > 0 or stats["pending_placements"] > 0:
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
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-55, 35, 0)
	sun.shadow_enabled = true
	scene.add_child(sun)
	var environment := WorldEnvironment.new()
	environment.environment = Environment.new()
	environment.environment.background_mode = Environment.BG_COLOR
	environment.environment.background_color = Color(0.62, 0.76, 0.9)
	environment.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.environment.ambient_light_color = Color(0.55, 0.58, 0.62)
	scene.add_child(environment)
	var camera := Camera3D.new()
	camera.far = 1000
	scene.add_child(camera)
	stages = ClassDB.instantiate("WaveForgeStages")
	var settings := Presets.settings(spec["path"])
	for property: String in settings:
		stages.set(property, settings[property])
	stages.params = spec["params"]
	stages.collider_radius = -1
	stages.navigation_radius = -1
	scene.add_child(stages)
	root.add_child(scene)
	var reach: float = stages.view_radius * stages.chunk_cells.x * stages.cell_size.x
	camera.look_at_from_position(AT + Vector3(reach * 0.5, reach * 0.42, reach * 0.5), AT)
	stages.follow(AT)
	started_usec = Time.get_ticks_usec()
	settled_frames = 0

func _capture(spec: Dictionary) -> void:
	var picture := root.get_texture().get_image()
	picture.resize(TILE.x, TILE.y)
	if sheet == null:
		sheet = Image.create(TILE.x * 3, TILE.y * spec["rows"], false, picture.get_format())
	sheet.blit_rect(picture, Rect2i(Vector2i.ZERO, TILE), Vector2i(TILE.x * spec["column"], TILE.y * spec["row"]))
	if spec["last"]:
		DirAccess.make_dir_recursive_absolute("user://contact_sheets")
		var out := "user://contact_sheets/%s.png" % String(spec["path"]).get_file().get_basename()
		sheet.save_png(out)
		print("render_presets: wrote %s" % ProjectSettings.globalize_path(out))
		sheet = null
