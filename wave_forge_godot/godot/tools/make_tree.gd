## Writes the plugin's built-in tree (`addons/wave_forge/tree.tres` and `tree.tscn`), the scene the
## presets bind to `tree`: a trunk and a cone of a crown in one mesh, coloured by its vertices, its
## base at the origin so it stands on the point a Scatter stage gives it. One mesh and no script, so
## the node draws it as a MultiMesh. Run from `wave_forge_godot/godot`:
## `godot --headless --path . --script tools/make_tree.gd`.
extends SceneTree

const TRUNK := Color(0.42, 0.29, 0.18)
const CROWN := Color(0.20, 0.45, 0.18)
const SIDES := 8

func _initialize() -> void:
	var tool := SurfaceTool.new()
	tool.begin(Mesh.PRIMITIVE_TRIANGLES)
	_cone(tool, 0.0, 1.0, 0.16, 0.12, TRUNK)
	_cone(tool, 0.8, 3.4, 1.0, 0.0, CROWN)
	tool.generate_normals()
	var material := StandardMaterial3D.new()
	material.vertex_color_use_as_albedo = true
	material.vertex_color_is_srgb = true
	material.roughness = 0.9
	tool.set_material(material)
	var mesh := tool.commit()
	ResourceSaver.save(mesh, "res://addons/wave_forge/tree.tres")
	var instance := MeshInstance3D.new()
	instance.name = "Tree"
	instance.mesh = load("res://addons/wave_forge/tree.tres")
	var scene := PackedScene.new()
	scene.pack(instance)
	ResourceSaver.save(scene, "res://addons/wave_forge/tree.tscn")
	instance.free()
	print("make_tree: wrote addons/wave_forge/tree.tres and tree.tscn")
	quit()

## A cone or frustum from `low` to `high` along y, of radius `bottom` there and `top` at the top,
## with a cap on each end that has a radius, in `colour`.
func _cone(tool: SurfaceTool, low: float, high: float, bottom: float, top: float, colour: Color) -> void:
	tool.set_color(colour)
	for side in SIDES:
		var a := TAU * side / SIDES
		var b := TAU * (side + 1) / SIDES
		var lower := [Vector3(cos(a) * bottom, low, sin(a) * bottom), Vector3(cos(b) * bottom, low, sin(b) * bottom)]
		var upper := [Vector3(cos(a) * top, high, sin(a) * top), Vector3(cos(b) * top, high, sin(b) * top)]
		# Clockwise seen from outside, which Godot draws as the front.
		for vertex in [lower[0], lower[1], upper[0], lower[1], upper[1], upper[0]]:
			tool.add_vertex(vertex)
		for vertex in [Vector3(0, low, 0), lower[1], lower[0]]:
			tool.add_vertex(vertex)
		if top > 0.0:
			for vertex in [Vector3(0, high, 0), upper[0], upper[1]]:
				tool.add_vertex(vertex)
