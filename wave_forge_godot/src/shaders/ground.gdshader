// Wave Forge's reference ground shader (docs/reference/godot.md, "Ground"): each chunk's ground
// carries its vertices' materials as an id texture, one texel per vertex, and every fragment
// blends the colours of the four vertices around it, so materials meet in a smooth band rather
// than along triangle edges. Works on every renderer, Compatibility included.
shader_type spatial;

// A chunk's material id per ground vertex, in the red channel as id / 255.
uniform sampler2D wave_forge_materials : filter_nearest, repeat_disable;
// One cell's width along x and z in world units: the ground's vertices stand at the centres of
// the cells, (i + 0.5) cells from the chunk's corner.
uniform vec2 wave_forge_cell = vec2(1.0);
// A colour per material id, in texel id of a 256 by 1 texture.
uniform sampler2D wave_forge_palette : filter_nearest, repeat_disable, source_color;
uniform float roughness = 0.9;

varying vec2 column;

void vertex() {
	column = VERTEX.xz / wave_forge_cell - 0.5;
}

vec3 colour_at(ivec2 vertex) {
	int id = int(texelFetch(wave_forge_materials, vertex, 0).r * 255.0 + 0.5);
	return texelFetch(wave_forge_palette, ivec2(id, 0), 0).rgb;
}

void fragment() {
	ivec2 size = textureSize(wave_forge_materials, 0);
	vec2 at = clamp(column, vec2(0.0), vec2(size - 1));
	ivec2 low = ivec2(floor(at));
	ivec2 high = min(low + 1, size - 1);
	vec2 t = at - vec2(low);
	vec3 near_row = mix(colour_at(low), colour_at(ivec2(high.x, low.y)), t.x);
	vec3 far_row = mix(colour_at(ivec2(low.x, high.y)), colour_at(high), t.x);
	ALBEDO = mix(near_row, far_row, t.y);
	ROUGHNESS = roughness;
}
