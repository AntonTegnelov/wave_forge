## Writes the plugin's nature kit, the scenes the presets bind to their trees, undergrowth and rocks
## (docs/reference/godot.md, "Scenes"), from a curated subset of Quaternius's Stylized Nature
## MegaKit (CC0, its licence kept beside it as `LICENSE_Quaternius.txt`). Each model becomes one
## mesh (`addons/wave_forge/nature/<name>.res`) and a scene of it (`<name>.tscn`) whose root is a
## lone MeshInstance3D without a script, so the node draws it as a MultiMesh, its base at the
## origin. Plants are drawn with the reference vegetation shader, so they sway in the wind and vary
## by their look; rocks with a plain material. Each texture is shrunk to `TEXTURE` texels a side and
## kept, losslessly compressed, in a resource of its own (`nature/textures/<name>.res`) that every
## mesh using it shares, so no file needs Godot's importer. The bushes take the broadleaf trees' green
## leaves, as the kit's own previews show them, rather than the twisted tree's autumn leaves their
## files name. Each material's colour is scaled so its texture's mean, where it is not cut out, is
## as bright as the measured material of its class (docs/research/terrain-look.md, "Measured
## albedo"): leaves as tree foliage, bark as bark, rock as rock; its hue stays the kit's.
##
## Run from `wave_forge_godot/godot` with the Standard version of the kit unpacked (its `glTF` and
## `Textures` folders and `License_Standard.txt`):
## `godot --headless --path . --script tools/make_nature_kit.gd -- <the unpacked kit>`.
extends SceneTree

const OUT := "res://addons/wave_forge/nature"
const VEGETATION := "res://addons/wave_forge/shaders/vegetation.gdshader"
const TEXTURE := 512
## The models kept: the file's name, the scene's, and whether it is a plant, which sways.
const MODELS := [
	["CommonTree_1", "broadleaf_1", true], ["CommonTree_2", "broadleaf_2", true],
	["CommonTree_3", "broadleaf_3", true], ["CommonTree_4", "broadleaf_4", true],
	["CommonTree_5", "broadleaf_5", true],
	["Pine_1", "pine_1", true], ["Pine_2", "pine_2", true], ["Pine_3", "pine_3", true],
	["Pine_4", "pine_4", true], ["Pine_5", "pine_5", true],
	["Bush_Common", "bush_1", true], ["Bush_Common_Flowers", "bush_2", true],
	["Fern_1", "fern_1", true], ["Plant_1", "fern_2", true], ["Plant_7", "fern_3", true],
	["Rock_Medium_1", "rock_1", false], ["Rock_Medium_2", "rock_2", false],
	["Rock_Medium_3", "rock_3", false],
]

## The texture a material takes in place of the one its file names.
const SWAPS := {"Leaves_TwistedTree": "Leaves_NormalTree"}
## The linear luminance each class of material is scaled to: the measured tree foliage 01113 and
## bark 00350 of the Spectral Materials Database, and its Rock 6, which the presets' rock is.
const FOLIAGE := 0.138
const BARK := 0.149
const ROCK := 0.229

## Each texture written, by its name, so meshes share it, and its mean linear luminance where it is
## not cut out.
var textures := {}
var means := {}

func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	if args.size() != 1:
		_fail("give the unpacked kit's folder after --")
		return
	var kit: String = args[0]
	# Outside the editor a compressed texture drops its data once made, and would save empty.
	PortableCompressedTexture2D.set_keep_all_compressed_buffers(true)
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(OUT + "/textures"))
	for model: Array in MODELS:
		if not _write(kit, model[0], model[1], model[2]):
			return
	var licence := FileAccess.get_file_as_string(kit.path_join("License_Standard.txt"))
	if licence.is_empty():
		_fail("the kit's License_Standard.txt is missing")
		return
	var file := FileAccess.open(OUT + "/LICENSE_Quaternius.txt", FileAccess.WRITE)
	file.store_string(licence)
	file.close()
	print("make_nature_kit: wrote %d models and %d textures to %s" % [MODELS.size(), textures.size(), OUT])
	quit()

## Writes the model of `source`, the kit's glTF file of that name, as the mesh and scene `name`.
func _write(kit: String, source: String, name: String, plant: bool) -> bool:
	var document := GLTFDocument.new()
	var state := GLTFState.new()
	if document.append_from_file(kit.path_join("glTF").path_join(source + ".gltf"), state) != OK:
		return _fail("%s could not be read" % source)
	var root := document.generate_scene(state)
	# Each material's texture file, from the glTF itself: a headless run keeps no texture's image.
	var files := {}
	var json: Dictionary = state.json
	for material: Dictionary in json.get("materials", []):
		var texture: Dictionary = material.get("pbrMetallicRoughness", {}).get("baseColorTexture", {})
		if texture.has("index"):
			var image: int = json["textures"][int(texture["index"])]["source"]
			files[material["name"]] = kit.path_join("glTF").path_join(json["images"][image]["uri"])
	# A swapped texture's file, by the kit's own name for it.
	for swapped: String in SWAPS:
		if files.has(swapped):
			files[swapped] = kit.path_join("glTF").path_join(SWAPS[swapped] + "_C.png")
	# Every surface's arrays, its place in the model baked in, with its material, and the model's
	# height, at which a plant sways fully.
	var surfaces := []
	var height := 0.0
	for node in root.find_children("*", "MeshInstance3D", true, false):
		var instance := node as MeshInstance3D
		var place := _place(root, instance)
		for surface in instance.mesh.get_surface_count():
			var arrays := instance.mesh.surface_get_arrays(surface)
			var vertices: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
			var normals: PackedVector3Array = arrays[Mesh.ARRAY_NORMAL]
			for i in vertices.size():
				vertices[i] = place * vertices[i]
				height = maxf(height, vertices[i].y)
			for i in normals.size():
				normals[i] = (place.basis * normals[i]).normalized()
			arrays[Mesh.ARRAY_VERTEX] = vertices
			arrays[Mesh.ARRAY_NORMAL] = normals
			# Tangents no longer follow the normals; the materials use no normal map.
			arrays[Mesh.ARRAY_TANGENT] = null
			surfaces.append([arrays, instance.mesh.surface_get_material(surface)])
	var mesh := ArrayMesh.new()
	for surface: Array in surfaces:
		var material := _material(surface[1], files, plant, height)
		if material == null:
			return _fail("%s has a surface without a texture" % source)
		mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, surface[0])
		mesh.surface_set_material(mesh.get_surface_count() - 1, material)
	root.free()
	if mesh.get_surface_count() == 0:
		return _fail("%s holds no mesh" % source)
	var mesh_path := OUT.path_join(name + ".res")
	if ResourceSaver.save(mesh, mesh_path, ResourceSaver.FLAG_COMPRESS) != OK:
		return _fail("%s could not be saved" % mesh_path)
	var instance := MeshInstance3D.new()
	instance.name = name.to_pascal_case()
	instance.mesh = load(mesh_path)
	var scene := PackedScene.new()
	scene.pack(instance)
	instance.free()
	if ResourceSaver.save(scene, OUT.path_join(name + ".tscn")) != OK:
		return _fail("%s's scene could not be saved" % name)
	return true

## Where `node` stands relative to `root`, which is out of the tree.
func _place(root: Node, node: Node3D) -> Transform3D:
	var place := node.transform
	var parent := node.get_parent()
	while parent != root and parent is Node3D:
		place = (parent as Node3D).transform * place
		parent = parent.get_parent()
	return place

## The material for a surface that the kit draws with `source`, whose texture file `files` gives by
## the material's name: the vegetation shader for a plant, swaying fully at `height` with its
## texture shrunk and cut out where the kit cuts it, or a plain material for a rock. Null if
## `source` has no texture.
func _material(source: Material, files: Dictionary, plant: bool, height: float) -> Material:
	var standard := source as StandardMaterial3D
	if standard == null or not files.has(standard.resource_name):
		return null
	var name: String = SWAPS.get(standard.resource_name, standard.resource_name)
	var texture := _texture(files[standard.resource_name], name)
	if texture == null:
		return null
	var bark := standard.resource_name.begins_with("Bark")
	var target := ROCK if not plant else (BARK if bark else FOLIAGE)
	var scale: Color = Color.WHITE * (target / float(means[name]))
	scale.a = 1.0
	if not plant:
		var rock := StandardMaterial3D.new()
		rock.albedo_texture = texture
		rock.albedo_color = scale.linear_to_srgb()
		rock.roughness = 0.9
		return rock
	var material := ShaderMaterial.new()
	material.shader = load(VEGETATION)
	material.set_shader_parameter("albedo", scale.linear_to_srgb())
	material.set_shader_parameter("albedo_texture", texture)
	var cut := standard.transparency == BaseMaterial3D.TRANSPARENCY_ALPHA_SCISSOR
	material.set_shader_parameter("alpha_scissor", standard.alpha_scissor_threshold if cut else 0.0)
	material.set_shader_parameter("bend_height", maxf(height, 1.0))
	# Bark neither flutters nor dries; leaves do both.
	material.set_shader_parameter("flutter", 0.0 if bark else 0.03)
	material.set_shader_parameter("dryness", 0.0 if bark else 0.6)
	return material

## The kit's texture of `name` from the file `file`, shrunk and compressed into a resource of its
## own, written once. Null if the file cannot be read.
func _texture(file: String, name: String) -> Texture2D:
	if textures.has(name):
		return textures[name]
	var image := Image.load_from_file(file)
	if image == null or image.is_empty():
		return null
	if image.get_width() > TEXTURE:
		image.resize(TEXTURE, TEXTURE * image.get_height() / image.get_width(), Image.INTERPOLATE_LANCZOS)
	means[name] = _mean(image)
	image.generate_mipmaps()
	var texture := PortableCompressedTexture2D.new()
	texture.create_from_image(image, PortableCompressedTexture2D.COMPRESSION_MODE_LOSSLESS)
	var path := OUT.path_join("textures").path_join(name.to_snake_case() + ".res")
	ResourceSaver.save(texture, path)
	textures[name] = load(path)
	return textures[name]

## The mean linear luminance of `image`'s texels that a cut-out keeps, those of alpha above a half.
func _mean(image: Image) -> float:
	var sum := 0.0
	var count := 0
	for y in range(0, image.get_height(), 2):
		for x in range(0, image.get_width(), 2):
			var texel := image.get_pixel(x, y)
			if texel.a > 0.5:
				var linear := texel.srgb_to_linear()
				sum += 0.2126 * linear.r + 0.7152 * linear.g + 0.0722 * linear.b
				count += 1
	return sum / maxi(count, 1)

func _fail(message: String) -> bool:
	printerr("make_nature_kit: " + message)
	quit(1)
	return false
