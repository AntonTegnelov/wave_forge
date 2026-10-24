// Wave Forge's reference water shader (docs/reference/godot.md, "Water"): lakes and rivers drawn by
// how deep they stand over the ground, which the node hands in per vertex in UV2's x
// (`wave_forge::WaterMesh::depths`). Shallow water shows the ground through it in a light colour,
// deep water turns dark and opaque, and at the shore it fades out to nothing where it dips under
// the bank, so the bank shows no edge. Needs no depth texture; works on every renderer,
// Compatibility included.
shader_type spatial;
render_mode blend_mix, depth_draw_opaque, cull_back;

// The water's colour where it is shallow and where it is deep.
uniform vec3 shallow_colour : source_color = vec3(0.30, 0.42, 0.38);
uniform vec3 deep_colour : source_color = vec3(0.03, 0.10, 0.13);
// The depth, in world units, over which the water turns from shallow to deep and from clear to
// opaque: about two thirds of the way at this depth.
uniform float clarity : hint_range(0.01, 100.0, 0.01) = 1.0;
// How opaque the water is at its shallowest and its deepest.
uniform vec2 opacity = vec2(0.6, 0.95);
// The depth, in world units, over which the water fades in from the shore.
uniform float shore : hint_range(0.01, 10.0, 0.01) = 0.25;
uniform float roughness : hint_range(0.0, 1.0) = 0.06;

varying float depth;

void vertex() {
	depth = UV2.x;
}

void fragment() {
	// The ranges stop the inspector at a positive value, and a script setting less is held there:
	// at zero both divide by zero.
	float deep = 1.0 - exp(-max(depth, 0.0) / max(clarity, 0.01));
	ALBEDO = mix(shallow_colour, deep_colour, deep);
	ALPHA = mix(opacity.x, opacity.y, deep) * smoothstep(0.0, max(shore, 0.01), depth);
	ROUGHNESS = roughness;
}
