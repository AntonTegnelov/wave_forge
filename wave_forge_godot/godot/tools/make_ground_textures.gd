## Writes the plugin's ground textures, the texture sets the reference ground shader can draw a
## material with instead of its colour (docs/reference/godot.md, "Ground and colliders"), from
## eight of ambientCG's CC0 materials (https://ambientcg.com, its licence kept beside them as
## `LICENSE_ambientCG.txt`). Each set becomes two textures of `TEXTURE` texels a side:
## `<index>_<name>_albedo_height.res`, its colour with its height in alpha, and
## `<index>_<name>_normal_roughness.res`, its OpenGL normal with its roughness in alpha, each a
## PortableCompressedTexture2D compressed lossily and without mipmaps, so no file needs Godot's
## importer and the set stays small. The node stacks them into texture arrays by index and makes
## their mipmaps as it does.
##
## Run from `wave_forge_godot/godot` with each material's 1K-JPG download unpacked into one folder
## (its `_Color`, `_Displacement`, `_NormalGL` and `_Roughness` maps):
## `godot --headless --path . --script tools/make_ground_textures.gd -- <the folder>`.
extends SceneTree

const OUT := "res://addons/wave_forge/ground_textures"
const TEXTURE := 1024
## How much the lossy compression keeps, 0 to 1.
const QUALITY := 0.7
## The sets kept, in the order of their index: ambientCG's asset id and the set's name.
const SETS := [
	["Grass004", "grass"], ["Ground106", "dirt"], ["Gravel022", "gravel"], ["Ground080", "sand"],
	["Rock060", "rock"], ["Rock051", "cliff"], ["Snow010A", "snow"], ["Ground071", "mud"],
]

func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	if args.size() != 1:
		printerr("make_ground_textures: give the folder of unpacked downloads after --")
		quit(1)
		return
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(OUT))
	# Headless, Godot keeps no compressed buffer unless asked, and saves an empty texture.
	PortableCompressedTexture2D.set_keep_all_compressed_buffers(true)
	for index in SETS.size():
		var id: String = SETS[index][0]
		var name: String = SETS[index][1]
		var map := func(kind: String) -> Image:
			var image := Image.load_from_file(args[0].path_join("%s_1K-JPG_%s.jpg" % [id, kind]))
			if image == null:
				return null
			image.convert(Image.FORMAT_RGBA8)
			image.resize(TEXTURE, TEXTURE, Image.INTERPOLATE_LANCZOS)
			return image
		var colour: Image = map.call("Color")
		var height: Image = map.call("Displacement")
		var normal: Image = map.call("NormalGL")
		var roughness: Image = map.call("Roughness")
		if colour == null or height == null or normal == null or roughness == null:
			printerr("make_ground_textures: %s is missing a map in %s" % [id, args[0]])
			quit(1)
			return
		for y in TEXTURE:
			for x in TEXTURE:
				var c := colour.get_pixel(x, y)
				colour.set_pixel(x, y, Color(c.r, c.g, c.b, height.get_pixel(x, y).r))
				var n := normal.get_pixel(x, y)
				normal.set_pixel(x, y, Color(n.r, n.g, n.b, roughness.get_pixel(x, y).r))
		for pair in [[colour, "albedo_height"], [normal, "normal_roughness"]]:
			var image: Image = pair[0]
			var texture := PortableCompressedTexture2D.new()
			texture.create_from_image(image, PortableCompressedTexture2D.COMPRESSION_MODE_LOSSY, false, QUALITY)
			var path := OUT.path_join("%d_%s_%s.res" % [index, name, pair[1]])
			if ResourceSaver.save(texture, path) != OK:
				printerr("make_ground_textures: %s could not be written" % path)
				quit(1)
				return
		print("make_ground_textures: %d %s from %s" % [index, name, id])
	quit(0)
