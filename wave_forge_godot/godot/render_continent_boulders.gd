## Draws the continent's boulders where its Talus stage's debris placed them, from eye level, and
## saves the picture (developer tool).
##
## Run with a display, after baking the continent with `measure_world_run.gd`:
## `godot --path . --script render_continent_boulders.gd -- --directory <dir> --at <x>,<z>`.
## `continent.tscn`'s node plays the directory around `--at`, a column of the lattice where the
## debris lies deep, its boulders drawn with the plugin's nature kit's rocks and the kinds around
## them with the kit's trees, bushes and ferns and the plugin's props (the continent binds no scenes
## of its own, so it would draw placeholders), loaded before the node starts. Once the view has
## settled it finds the boulder with the most others within 15 m in the 9 by 9 chunks around and
## follows it, so the near ground stands all round it. Once that has settled the camera stands 1.7 m
## over the lowest ground 14 m from it, or 20 or 28 m where no spot that near will do, with nothing
## of the ground in between and nothing placed within 4 m, looking up at it against the slope its
## debris came from under the reference look (`reference_look.gd`). The script fails if no spot
## will do. The picture lands in Godot's user directory, and the script prints where, with how many
## boulders stand within 15 m.
extends SceneTree

const ReferenceLook := preload("res://reference_look.gd")
const TIMEOUT_S := 300.0
const EYE_M := 1.7
## How far from the boulders the camera may stand, nearest first.
const BACK_M: Array[float] = [14.0, 20.0, 28.0]
const CLUSTER_M := 15.0
const NATURE := "res://addons/wave_forge/nature/%s_%d.tscn"
## The continent's kinds drawn with the nature kit's kinds, and how many variants each has there.
const KIT := {
	"rock": [["boulder", "stone_pile", "tide_rock"], 3],
	"pine": [["pine", "spruce", "fir", "larch", "cypress", "juniper"], 5],
	"broadleaf": [["oak", "beech", "birch", "hazel", "willow", "dwarf_willow", "acacia", "baobab",
		"bottle_tree", "cloud_tree", "jungle_giant", "mangrove_tree"], 5],
	"bush": [["berry_bush", "bramble", "heather_bush", "thornbush", "sagebrush", "saltbush",
		"creosote", "bog_myrtle"], 2],
	"fern": [["fern_clump", "tree_fern"], 3],
}
## The continent's kinds drawn with the plugin's own props.
const PROPS := {"cactus": "res://addons/wave_forge/cactus.tscn", "palm": "res://addons/wave_forge/palm.tscn",
	"date_palm": "res://addons/wave_forge/palm.tscn"}

var stages: Node
var camera: Camera3D
var at := Vector3.ZERO
var started_usec := 0
var settled := 0
## The boulders around `at` once the view has settled, in Godot's world; empty before.
var boulders := PackedVector3Array()
## The densest boulder once found and followed, which the camera frames once the view about it has
## settled.
var cluster := Vector3.ZERO
var framed := false

func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	var directory := ""
	var given_at := false
	var i := 0
	while i < args.size():
		if i + 1 >= args.size():
			_fail("%s needs a value" % args[i])
			return
		match args[i]:
			"--directory": directory = args[i + 1]
			"--at":
				var parts := args[i + 1].split(",")
				if parts.size() != 2 or not parts[0].is_valid_float() or not parts[1].is_valid_float():
					_fail("--at takes a column as <x>,<z>, not %s" % args[i + 1])
					return
				at = Vector3(parts[0].to_float(), 0.0, parts[1].to_float())
				given_at = true
			_:
				_fail("unknown option %s" % args[i])
				return
		i += 2
	if not given_at:
		_fail("--at is needed: the column to look at")
		return
	if not DirAccess.dir_exists_absolute(directory):
		_fail("--directory names no directory: %s" % directory)
		return
	var scene: Node = (load("res://continent.tscn") as PackedScene).instantiate()
	stages = scene.get_node("Continent")
	stages.play_directory = directory
	stages.collider_radius = -1
	stages.follow_camera = false
	var scenes := {}
	for kind: String in KIT:
		var variants: Array = []
		for variant in KIT[kind][1]:
			variants.append(load(NATURE % [kind, variant + 1]))
		for continent_kind: String in KIT[kind][0]:
			scenes[continent_kind] = variants
	for kind: String in PROPS:
		scenes[kind] = load(PROPS[kind])
	stages.scenes = scenes
	root.add_child(scene)
	camera = Camera3D.new()
	camera.current = true
	camera.fov = 70.0
	camera.far = 2500.0
	camera.position = at + Vector3(0.0, 200.0, 0.0)
	root.add_child(camera)
	ReferenceLook.add_to(root)
	if not stages.start():
		_fail("the continent did not start")
		return
	stages.follow(at)
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		_fail("the view did not settle: %s" % stages.stats())
		return true
	if is_nan(stages.ground_height(at)):
		return false
	var stats: Dictionary = stages.stats()
	if stats["pending_grounds"] > 0 or stats["pending_placements"] > 0:
		settled = 0
		return false
	settled += 1
	if settled < 60:
		return false
	if boulders.is_empty():
		boulders = _boulders()
		if boulders.is_empty():
			_fail("no boulders in the 9 by 9 chunks around %s" % at)
			return true
		cluster = _densest()
		at = cluster
		stages.follow(cluster)
		settled = 0
		return false
	if not framed:
		if not _place_camera(cluster):
			_fail("no spot within %s m of the boulders at %s stands in sight of them, clear of anything placed" % [BACK_M, cluster])
			return true
		framed = true
		settled = 0
		return false
	var path := "user://continent_boulders.png"
	root.get_viewport().get_texture().get_image().save_png(path)
	print("render_continent_boulders: saved %s, %d boulders within %.0f m of %s" % [ProjectSettings.globalize_path(path), _near(at), CLUSTER_M, at])
	quit(0)
	return true

## Stands the camera at eye height as near `cluster` as `BACK_M` allows, on the lowest ground there,
## so the slope behind the boulders shows, looking at it; and moves `at` there. Returns false, with
## the camera unmoved, where no spot will do.
func _place_camera(cluster: Vector3) -> bool:
	at = cluster
	var target := cluster + Vector3(0.0, 1.0, 0.0)
	for back: float in BACK_M:
		var from := Vector3.INF
		var lowest := INF
		for step in 16:
			var angle := TAU * step / 16.0
			var there := cluster + Vector3(cos(angle), 0.0, sin(angle)) * back
			var height: float = stages.ground_height(there)
			if is_nan(height) or height >= lowest:
				continue
			var eye := Vector3(there.x, height + EYE_M, there.z)
			if not _in_sight(eye, target) or _crowded(eye):
				continue
			lowest = height
			from = eye
		if from != Vector3.INF:
			camera.look_at_from_position(from, target)
			return true
	return false

## Whether the ground stands below the line from `eye` to `target` all the way.
func _in_sight(eye: Vector3, target: Vector3) -> bool:
	for step in range(1, 20):
		var point := eye.lerp(target, step / 20.0)
		var ground: float = stages.ground_height(point)
		if not is_nan(ground) and ground > point.y:
			return false
	return true

## Whether anything the node has placed, of any kind, stands within 4 m of `eye` on the ground plane.
func _crowded(eye: Vector3) -> bool:
	var chunk_side: float = stages.chunk_cells.x * stages.cell_size.x
	var centre := Vector2i(floori(eye.x / chunk_side), floori(eye.z / chunk_side))
	for stage: String in stages.targets:
		for y in range(-1, 2):
			for x in range(-1, 2):
				for set: Dictionary in stages.point_sets(stage, Vector3i(centre.x + x, centre.y + y, 0)):
					var transforms: PackedFloat32Array = set["transforms"]
					for k in range(0, transforms.size(), 12):
						if Vector2(transforms[k + 3] - eye.x, transforms[k + 11] - eye.z).length() < 4.0:
							return true
	return false

## The boulder with the most others within `CLUSTER_M`.
func _densest() -> Vector3:
	var best := boulders[0]
	var most := -1
	for point in boulders:
		var count := _near(point)
		if count > most:
			most = count
			best = point
	return best

## How many boulders lie within `CLUSTER_M` of `centre` on the ground plane.
func _near(centre: Vector3) -> int:
	var count := 0
	for point in boulders:
		if Vector2(point.x - centre.x, point.z - centre.z).length() <= CLUSTER_M:
			count += 1
	return count

## The boulders the node has placed in the 9 by 9 chunks around `at`.
func _boulders() -> PackedVector3Array:
	var found := PackedVector3Array()
	var chunk_side: float = stages.chunk_cells.x * stages.cell_size.x
	var centre := Vector2i(floori(at.x / chunk_side), floori(at.z / chunk_side))
	for y in range(-4, 5):
		for x in range(-4, 5):
			for set: Dictionary in stages.point_sets("boulders", Vector3i(centre.x + x, centre.y + y, 0)):
				var transforms: PackedFloat32Array = set["transforms"]
				for k in range(0, transforms.size(), 12):
					found.push_back(Vector3(transforms[k + 3], transforms[k + 7], transforms[k + 11]))
	return found

func _fail(message: String) -> void:
	printerr("render_continent_boulders: " + message)
	quit(1)
