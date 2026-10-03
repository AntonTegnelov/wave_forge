## Writes the plugin's built-in props, the scenes the presets bind to their Scatter kinds: a tree
## (`addons/wave_forge/tree.tres` and `tree.tscn`), a trunk and a cone of a crown, and a cactus
## (`cactus.tres` and `cactus.tscn`), a column with two arms. Each is one mesh coloured by its
## vertices, its base at the origin so it stands on the point a Scatter stage gives it, with no
## script, so the node draws it as a MultiMesh. Run from `wave_forge_godot/godot`:
## `godot --headless --path . --script tools/make_props.gd`.
extends SceneTree

const TRUNK := Color(0.42, 0.29, 0.18)
const CROWN := Color(0.20, 0.45, 0.18)
const CACTUS := Color(0.31, 0.49, 0.23)
const SIDES := 8

func _initialize() -> void:
	# Each part: from, to, the radius at each end, and its colour.
	_write("Tree", "tree", [
		[Vector3(0, 0, 0), Vector3(0, 1.0, 0), 0.16, 0.12, TRUNK],
		[Vector3(0, 0.8, 0), Vector3(0, 3.4, 0), 1.0, 0.0, CROWN],
	])
	_write("Cactus", "cactus", [
		[Vector3(0, 0, 0), Vector3(0, 2.4, 0), 0.28, 0.24, CACTUS],
		[Vector3(0, 1.0, 0), Vector3(0.6, 1.0, 0), 0.15, 0.15, CACTUS],
		[Vector3(0.6, 0.9, 0), Vector3(0.6, 1.8, 0), 0.15, 0.12, CACTUS],
		[Vector3(0, 1.4, 0), Vector3(0, 1.4, -0.5), 0.14, 0.14, CACTUS],
		[Vector3(0, 1.3, -0.5), Vector3(0, 2.0, -0.5), 0.14, 0.11, CACTUS],
	])
	quit()

## Saves the mesh of `parts` as `addons/wave_forge/<file>.tres` and a scene of it, its root named
## `node`, as `<file>.tscn`.
func _write(node: String, file: String, parts: Array) -> void:
	var tool := SurfaceTool.new()
	tool.begin(Mesh.PRIMITIVE_TRIANGLES)
	for part: Array in parts:
		_cone(tool, part[0], part[1], part[2], part[3], part[4])
	tool.generate_normals()
	var material := StandardMaterial3D.new()
	material.vertex_color_use_as_albedo = true
	material.vertex_color_is_srgb = true
	material.roughness = 0.9
	tool.set_material(material)
	var path := "res://addons/wave_forge/%s" % file
	ResourceSaver.save(tool.commit(), path + ".tres")
	var instance := MeshInstance3D.new()
	instance.name = node
	instance.mesh = load(path + ".tres")
	var scene := PackedScene.new()
	scene.pack(instance)
	ResourceSaver.save(scene, path + ".tscn")
	instance.free()
	print("make_props: wrote addons/wave_forge/%s.tres and %s.tscn" % [file, file])

## A cone or frustum from `from` to `to`, of radius `bottom` at `from` and `top` at `to`, with a cap
## on each end that has a radius, in `colour`.
func _cone(tool: SurfaceTool, from: Vector3, to: Vector3, bottom: float, top: float, colour: Color) -> void:
	var axis := (to - from).normalized()
	# Across the axis: x and z for an upright part. `v` is `u` crossed with the axis for every
	# part, so the sides wind the same way whichever way a part points.
	var u := Vector3.RIGHT if absf(axis.y) > 0.9 else Vector3.UP.cross(axis).normalized()
	var v := u.cross(axis)
	tool.set_color(colour)
	for side in SIDES:
		var a := TAU * side / SIDES
		var b := TAU * (side + 1) / SIDES
		var lower := [from + u * (cos(a) * bottom) + v * (sin(a) * bottom), from + u * (cos(b) * bottom) + v * (sin(b) * bottom)]
		var upper := [to + u * (cos(a) * top) + v * (sin(a) * top), to + u * (cos(b) * top) + v * (sin(b) * top)]
		# Clockwise seen from outside, which Godot draws as the front.
		for vertex in [lower[0], lower[1], upper[0], lower[1], upper[1], upper[0]]:
			tool.add_vertex(vertex)
		for vertex in [from, lower[1], lower[0]]:
			tool.add_vertex(vertex)
		if top > 0.0:
			for vertex in [to, upper[0], upper[1]]:
				tool.add_vertex(vertex)
