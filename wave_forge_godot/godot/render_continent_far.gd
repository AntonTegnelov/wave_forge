## Draws the continent's far ground from the lowlands toward a mountain range 1 to 2 km away and
## saves the picture (developer tool).
##
## Run through `../render_ground.sh`. `continent.tscn` is instanced with only its ground, biomes and
## far ground asked for, the far ground out to 128 chunks, about 2 km. The camera stands 40 m over
## the lowland east of the central range and looks west across it under the reference look
## (`reference_look.gd`), so the range's silhouette stands against the sky. The picture lands in
## Godot's user directory, and the script prints where, with how many chunks each coarse stage
## generated per second of its own time (P2).
extends SceneTree

const ReferenceLook := preload("res://reference_look.gd")
const FAR_RADIUS := 128
const TIMEOUT_S := 300.0
## The camera's column and the column it looks at, in cells of the lattice.
const FROM := Vector2(1640, 970)
const TOWARD := Vector2(1050, 970)
const COARSE := ["coarse_ridges", "coarse_uplift", "coarse_eroded", "coarse_height", "far_height", "far_biome"]

var scene: Node
var world: Node
var camera: Camera3D
var started_usec := 0
var waited_frames := 0

func _initialize() -> void:
	scene = (load("res://continent.tscn") as PackedScene).instantiate()
	world = scene.get_node("Continent")
	world.targets = PackedStringArray(["ground", "biome", "far_height", "far_biome"])
	var radii: Dictionary[StringName, int] = {&"far_height": FAR_RADIUS, &"far_biome": FAR_RADIUS}
	world.target_radii = radii
	world.view_radius = 2
	world.collider_radius = -1
	world.volume_stage = ""
	root.add_child(scene)
	camera = Camera3D.new()
	camera.far = 4000
	camera.fov = 50
	root.add_child(camera)
	ReferenceLook.add_to(root)
	if not world.start():
		printerr("render_continent_far: the continent did not start")
		quit(1)
		return
	var cell: Vector3 = world.cell_size
	var from := Vector3(FROM.x * cell.x, 0, FROM.y * cell.z)
	var toward := Vector3(TOWARD.x * cell.x, 0, TOWARD.y * cell.z)
	from.y = world.sample("height", from) * cell.y + 40.0
	toward.y = world.sample("far_height", toward) * cell.y
	world.follow(from)
	camera.look_at_from_position(from, toward)
	started_usec = Time.get_ticks_usec()

func _process(_delta: float) -> bool:
	if (Time.get_ticks_usec() - started_usec) / 1e6 > TIMEOUT_S:
		printerr("render_continent_far: the far ground had not arrived")
		quit(1)
		return true
	var stats: Dictionary = world.stats()
	if stats["pending_far_grounds"] > 0 or world.far_ground_chunks().size() < 64:
		return false
	# A few frames for the renderer to draw what just arrived.
	waited_frames += 1
	if waited_frames < 20:
		return false
	var path := "user://continent_far.png"
	root.get_viewport().get_texture().get_image().save_png(path)
	var lines := PackedStringArray()
	var stages: Dictionary = stats["stages"]
	for stage: String in COARSE:
		if not stages.has(stage):
			continue
		var cost: Dictionary = stages[stage]
		lines.append("%s %d chunks in %.1f ms, %.0f a second" % [stage, cost["products"], cost["ms"], cost["products"] / max(cost["ms"], 0.001) * 1000.0])
	print("render_continent_far: saved %s, %d far grounds; %s" % [ProjectSettings.globalize_path(path), world.far_ground_chunks().size(), "; ".join(lines)])
	quit(0)
	return true
