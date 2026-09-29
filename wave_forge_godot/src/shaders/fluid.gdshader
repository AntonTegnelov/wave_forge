// Wave Forge's reference fluid shader (docs/reference/godot.md, "Volume"): each vertex of a
// fluid's surface carries its fluid's colour, whose alpha is how opaque it is, and in its first UV
// how brightly it glows, so water shows the rock under it and lava lights the cave. Seen from both
// sides, from over a pool and from under its surface. Works on every renderer, Compatibility
// included.
shader_type spatial;
render_mode cull_disabled;

uniform float roughness = 0.1;

void fragment() {
	ALBEDO = COLOR.rgb;
	ALPHA = COLOR.a;
	EMISSION = COLOR.rgb * UV.x;
	ROUGHNESS = roughness;
}
