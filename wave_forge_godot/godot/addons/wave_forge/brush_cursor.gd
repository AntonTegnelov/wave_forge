@tool
extends Decal
## The brush's cursor in the editor (docs/reference/godot.md, "Editor"): a ring projected onto the
## ground under the mouse, as wide as the brush reaches, so the artist sees what a stroke there
## would change. Decals draw on every renderer from Godot 4.8, and on Forward+ and Mobile before it.

## The ring's colour.
const COLOUR := Color(1.0, 0.85, 0.3)
## How far above and below the point under the mouse the ring reaches, in brush radii, so it lies
## on slopes as steep as 45 degrees across its whole width.
const DEPTH_PER_RADIUS := 2.0

func _init() -> void:
	var gradient := Gradient.new()
	gradient.offsets = PackedFloat32Array([0.0, 0.86, 0.9, 0.97, 1.0])
	gradient.colors = PackedColorArray([
		Color(COLOUR, 0.08), Color(COLOUR, 0.08), Color(COLOUR, 0.9), Color(COLOUR, 0.9), Color(COLOUR, 0.0),
	])
	var ring := GradientTexture2D.new()
	ring.gradient = gradient
	ring.fill = GradientTexture2D.FILL_RADIAL
	ring.fill_from = Vector2(0.5, 0.5)
	ring.fill_to = Vector2(1.0, 0.5)
	texture_albedo = ring
	visible = false

## Shows the ring centred on `ground`, a point in world space, for a brush of `radius` cells on a
## world whose cells are `cell` wide; hides it when `ground` is null, the mouse being off the ground.
func show_at(ground: Variant, radius: float, cell: Vector3) -> void:
	visible = ground != null
	if ground == null:
		return
	global_position = ground
	var reach := radius * cell.x
	size = Vector3(2.0 * reach, 2.0 * DEPTH_PER_RADIUS * reach, 2.0 * reach)
