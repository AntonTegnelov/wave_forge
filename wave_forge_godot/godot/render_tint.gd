## Draws three plants through the reference vegetation shader and checks each takes the colour
## its custom data gives (developer tool).
##
## Run through `../render_ground.sh`, on Forward+, whose lighting is linear, so a colour's ratios
## survive it; Compatibility lights the same colours brighter and not in proportion. A MultiMesh of three boxes carries custom data as the stages
## node packs a point's: no sway, then the random and the first appearance value. The first has a
## random of 0, the second of 1, the third a middling random and an appearance of 1. Lit by a white
## ambient light alone and seen from straight above, each box's top is its colour times one factor,
## which the first gives: the second is brighter and warmer by the shader's variation, and the third
## is gone toward the dry colour by `dryness`, both as the shader's defaults have them.
extends SceneTree

## The shader's defaults: its albedo, dry colour, variation and dryness.
const ALBEDO := Color(0.3, 0.5, 0.2)
const DRY := Color(0.365, 0.322, 0.212)
const VARIATION := 0.15
const DRYNESS := 0.6
## Each box's random and appearance.
const LOOKS := [Vector2(0.0, 0.0), Vector2(1.0, 0.0), Vector2(0.5, 1.0)]

var frames := 0

func _initialize() -> void:
	var stages: Node = ClassDB.instantiate("WaveForgeStages")
	var shader := Shader.new()
	shader.code = stages.vegetation_shader_code()
	stages.free()
	var material := ShaderMaterial.new()
	material.shader = shader
	var box := BoxMesh.new()
	box.size = Vector3(2, 2, 2)
	box.material = material
	var multimesh := MultiMesh.new()
	multimesh.transform_format = MultiMesh.TRANSFORM_3D
	multimesh.use_colors = true
	multimesh.use_custom_data = true
	multimesh.mesh = box
	multimesh.instance_count = LOOKS.size()
	for i in LOOKS.size():
		multimesh.set_instance_transform(i, Transform3D(Basis.IDENTITY, Vector3(i * 3.0 - 3.0, 0, 0)))
		multimesh.set_instance_color(i, Color.WHITE)
		multimesh.set_instance_custom_data(i, Color(0.0, 1.0, LOOKS[i].x, LOOKS[i].y))
	var plants := MultiMeshInstance3D.new()
	plants.multimesh = multimesh
	root.add_child(plants)
	var environment := Environment.new()
	environment.background_mode = Environment.BG_COLOR
	environment.background_color = Color.BLACK
	environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.ambient_light_color = Color.WHITE
	environment.reflected_light_source = Environment.REFLECTION_SOURCE_DISABLED
	environment.tonemap_mode = Environment.TONE_MAPPER_LINEAR
	var world_environment := WorldEnvironment.new()
	world_environment.environment = environment
	root.add_child(world_environment)
	var camera := Camera3D.new()
	camera.projection = Camera3D.PROJECTION_ORTHOGONAL
	camera.size = 9.0
	root.add_child(camera)
	camera.look_at_from_position(Vector3(0, 10, 0), Vector3.ZERO, Vector3(0, 0, -1))
	RenderingServer.global_shader_parameter_set("wave_forge_wind", Vector4(1, 0, 0, 3))

## The linear colour at the centre of box `index`'s top: the camera sees 9 units from top to bottom.
func _top(picture: Image, index: int) -> Color:
	var pixels_per_unit := picture.get_height() / 9.0
	var x := int(picture.get_width() / 2.0 + (index * 3.0 - 3.0) * pixels_per_unit)
	return picture.get_pixel(x, picture.get_height() / 2).srgb_to_linear()

func _process(_delta: float) -> bool:
	frames += 1
	if frames < 30:
		return false
	var picture := root.get_viewport().get_texture().get_image()
	var tops := [_top(picture, 0), _top(picture, 1), _top(picture, 2)]
	var albedo := ALBEDO.srgb_to_linear()
	var dry := DRY.srgb_to_linear()
	# The light's one factor, from the first box: its random of 0 darkens and cools it.
	var shift := -VARIATION
	var light: float = tops[0].g / (albedo.g * (1.0 + shift))
	if light < 0.2:
		printerr("render_tint: the boxes are unlit, %s" % [tops])
		quit(1)
		return true
	var expected := [
		_tinted(albedo, -VARIATION, light),
		_tinted(albedo, VARIATION, light),
		_tinted(albedo.lerp(dry, DRYNESS), 0.0, light),
	]
	for i in tops.size():
		var got: Color = tops[i]
		var want: Color = expected[i]
		for channel in 3:
			if absf(got[channel] - want[channel]) > 0.04 * want[channel] + 0.002:
				printerr("render_tint: box %d is %s, not %s" % [i, got, want])
				quit(1)
				return true
	print("render_tint: a random of 0 and of 1 and a dry plant take %s, %s and %s, as the shader's defaults give" % tops)
	quit(0)
	return true

## `colour` varied by `shift`, brighter and warmer or darker and cooler, under `light`.
func _tinted(colour: Color, shift: float, light: float) -> Color:
	return Color(colour.r * (1.0 + shift) * (1.0 + 0.3 * shift), colour.g * (1.0 + shift), colour.b * (1.0 + shift) * (1.0 - 0.3 * shift)) * light
