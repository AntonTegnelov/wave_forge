// Wave Forge's reference ground shader (docs/reference/godot.md, "Ground"): each chunk's ground
// carries its vertices' materials as an id texture, one texel per vertex, and every fragment draws
// the materials of the four vertices around it. Where they differ, each is weighted by how near
// its vertices are plus a noise of its own, and the highest wins with a short blend, so materials
// meet along a broken line rather than the grid's. Steep ground turns to rock by its own slope,
// its colour muted and banded in strata; a broad noise varies the colour across the land, and a
// fine one bumps the ground and grains its colour near the camera. Asset free; works on every
// renderer, Compatibility included. Channels the node hands in darken hollows and wet ground, and
// each material's colour may run along a ramp of its own by height, slope, cavity and wetness.
// `wave_forge_bevy/src/shaders/ground.wesl` is the same shader for Bevy.
shader_type spatial;

// A chunk's material id per ground vertex, in the red channel as id / 255.
uniform sampler2D wave_forge_materials : filter_nearest, repeat_disable;
// One cell's width along x and z in world units: the ground's vertices stand at the centres of
// the cells, (i + 0.5) cells from the chunk's corner.
uniform vec2 wave_forge_cell = vec2(1.0);
// A colour per material id, in texel id of a 256 by 1 texture.
uniform sampler2D wave_forge_palette : filter_nearest, repeat_disable, source_color;
// Per material id, in texel id of a 256 by 2 texture: the colour, linear, its ramp runs toward in
// row 0, and in row 1 how far the ground's height (0 to 1 over `ramp_heights`), slope, cavity and
// wetness each move its colour along that ramp, from -1 to 1. Black, no ramp, where the node hands
// in none.
uniform sampler2D wave_forge_ramps : filter_nearest, repeat_disable, hint_default_black;
// The heights, in world units, over which the ramps' height runs from 0 to 1.
uniform vec2 ramp_heights = vec2(0.0, 1.0);
// Per ground vertex, 0 to 1: a cavity in red, which darkens the ground, a wetness in green, which
// darkens and smooths it, and a cover in blue, which tints it toward its grass. Black, no effect,
// where the node hands in none.
uniform sampler2D wave_forge_channels : filter_nearest, repeat_disable, hint_default_black;
uniform float roughness = 0.9;
// How far a full cavity darkens the ground.
uniform float cavity_darkening : hint_range(0.0, 1.0) = 0.6;
// How far full wetness darkens the ground, and the roughness it leaves.
uniform float wet_darkening : hint_range(0.0, 1.0) = 0.45;
uniform float wet_roughness : hint_range(0.0, 1.0) = 0.35;
// Where the grass's blades shrink away, as the grass shader measures it (in chunks from the followed
// chunk's centre along the farther axis, with a chunk's width and that centre in world units):
// covered ground tints in over it, so where the blades stand the ground is as it is.
uniform vec2 wave_forge_fade = vec2(1e9, 2e9);
uniform vec2 wave_forge_span = vec2(1.0);
uniform vec2 wave_forge_focus = vec2(0.0);
// The colour fully covered ground tints toward beyond the grass, the grass's own seen from afar
// (measured grass, docs/research/terrain-look.md, "Measured albedo"), and how far.
uniform vec3 cover_colour : source_color = vec3(0.376, 0.384, 0.231);
uniform float cover_tint : hint_range(0.0, 1.0) = 0.5;
// Rock on steep ground is the ground's own colour muted and darkened, or with this colour's alpha
// above zero, this colour instead, by that much; and the slopes (one minus the normal's height) at
// which rock starts and has taken over.
uniform vec4 rock_colour : source_color = vec4(0.42, 0.40, 0.38, 0.0);
uniform vec2 rock_slope = vec2(0.3, 0.5);
// How strongly the rock is banded in strata.
uniform float strata : hint_range(0.0, 1.0) = 0.6;
// How deep the gullies that run down steep ground along its fall line read, in colour and in its
// normal: finer than a column, so a wall too narrow for the Erode stage still shows them.
uniform float gullies : hint_range(0.0, 1.0) = 0.5;
// How far noise breaks the line where materials meet.
uniform float edge_noise : hint_range(0.0, 1.0) = 0.5;
// How much a broad noise varies the colour across the land.
uniform float macro_variation : hint_range(0.0, 1.0) = 0.5;
// How much fine grain and detail normal the ground has near the camera.
uniform float detail : hint_range(0.0, 1.0) = 1.0;

// The slopes (one minus the normal's height) at which gullies start and are cut fully, and how many
// gullies a world unit of the ground plane holds.
const vec2 GULLY_SLOPE = vec2(0.08, 0.2);
const float GULLY_SCALE = 0.7;

varying vec2 column;
varying vec3 world;
varying vec3 world_normal;

void vertex() {
	column = VERTEX.xz / wave_forge_cell - 0.5;
	world = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	world_normal = (MODEL_MATRIX * vec4(NORMAL, 0.0)).xyz;
}

// A hash without trigonometry, cheaper than sin and the same on every GPU.
float hash(vec2 p) {
	vec3 p3 = fract(vec3(p.xyx) * 0.1031);
	p3 += dot(p3, p3.yzx + 33.33);
	return fract((p3.x + p3.y) * p3.z);
}

// Value noise in 0..1 with a smooth step between the lattice's points.
float value(vec2 p) {
	vec2 i = floor(p);
	vec2 f = p - i;
	vec2 u = f * f * (3.0 - 2.0 * f);
	float a = hash(i);
	float b = hash(i + vec2(1.0, 0.0));
	float c = hash(i + vec2(0.0, 1.0));
	float d = hash(i + vec2(1.0, 1.0));
	return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Three octaves of value noise, in 0..1, each turned against the last so no feature lines up with
// the axes.
float fbm(vec2 p) {
	mat2 turn = mat2(vec2(1.6, 1.2), vec2(-1.2, 1.6));
	vec2 q = turn * p;
	return (value(p) * 4.0 + value(q) * 2.0 + value(turn * q)) / 7.0;
}

// Value noise and its slope, (value, d/dx, d/dy).
vec3 value_with_slope(vec2 p) {
	vec2 i = floor(p);
	vec2 f = p - i;
	vec2 u = f * f * (3.0 - 2.0 * f);
	vec2 du = 6.0 * f * (1.0 - f);
	float a = hash(i);
	float b = hash(i + vec2(1.0, 0.0));
	float c = hash(i + vec2(0.0, 1.0));
	float d = hash(i + vec2(1.0, 1.0));
	float k = a - b - c + d;
	return vec3(a + (b - a) * u.x + (c - a) * u.y + k * u.x * u.y, du * (vec2(b - a, c - a) + k * u.yx));
}

// `fbm` and its slope, (value, d/dx, d/dy): each turned octave's slope turned back.
vec3 fbm_with_slope(vec2 p) {
	mat2 turn = mat2(vec2(1.6, 1.2), vec2(-1.2, 1.6));
	mat2 back = transpose(turn);
	vec2 q = turn * p;
	vec3 first = value_with_slope(p);
	vec3 second = value_with_slope(q);
	vec3 third = value_with_slope(turn * q);
	float v = first.x * 4.0 + second.x * 2.0 + third.x;
	vec2 slope = first.yz * 4.0 + back * (second.yz * 2.0 + back * third.yz);
	return vec3(v, slope) / 7.0;
}

int id_at(ivec2 vertex) {
	return int(texelFetch(wave_forge_materials, vertex, 0).r * 255.0 + 0.5);
}

// Material `id`'s colour, moved along its ramp by `drivers`: the ground's height, slope, cavity and
// wetness here.
vec3 colour_of(int id, vec4 drivers) {
	vec3 colour = texelFetch(wave_forge_palette, ivec2(id, 0), 0).rgb;
	ivec2 last = textureSize(wave_forge_ramps, 0) - 1;
	vec3 toward = texelFetch(wave_forge_ramps, min(ivec2(id, 0), last), 0).rgb;
	vec4 by = texelFetch(wave_forge_ramps, min(ivec2(id, 1), last), 0);
	return mix(colour, toward, clamp(dot(by, drivers), 0.0, 1.0));
}

void fragment() {
	ivec2 size = textureSize(wave_forge_materials, 0);
	vec2 at = clamp(column, vec2(0.0), vec2(size - 1));
	ivec2 low = ivec2(floor(at));
	ivec2 high = min(low + 1, size - 1);
	vec2 t = at - vec2(low);
	ivec2 corners[4] = ivec2[4](low, ivec2(high.x, low.y), ivec2(low.x, high.y), high);
	float near_weights[4] = float[4]((1.0 - t.x) * (1.0 - t.y), t.x * (1.0 - t.y), (1.0 - t.x) * t.y, t.x * t.y);
	vec2 ground = world.xz;
	// Rock by the ground's own slope, its edge broken by noise.
	vec3 surface = normalize(world_normal);
	float steepness = 1.0 - surface.y + edge_noise * 0.15 * (value(ground / 2.0) - 0.5);
	float rock = smoothstep(rock_slope.x, rock_slope.y, steepness);
	// The materials come per column of x and z, so a sharp border would run down a wall as a
	// streak: on rock the border is soft and unbroken.
	float border_noise = edge_noise * (1.0 - rock);
	float border_width = mix(0.1, 1.0, rock);
	// Each material's weight is the sum of its corners', so its edge follows the smooth line between
	// the vertices; raised by a noise of the material's own, the highest wins, blending over a band
	// of the border's width. A material on several corners is counted once.
	// The channels, blended between the vertices as the materials are.
	ivec2 channel_size = textureSize(wave_forge_channels, 0);
	vec3 channels = vec3(0.0);
	for (int k = 0; k < 4; k++) {
		channels += texelFetch(wave_forge_channels, min(corners[k], channel_size - 1), 0).rgb * near_weights[k];
	}
	float cavity = clamp(channels.r, 0.0, 1.0);
	float wet = clamp(channels.g, 0.0, 1.0);
	float cover = clamp(channels.b, 0.0, 1.0);
	// What moves each material's colour along its ramp.
	float height = clamp((world.y - ramp_heights.x) / max(ramp_heights.y - ramp_heights.x, 1e-6), 0.0, 1.0);
	vec4 drivers = vec4(height, steepness, cavity, wet);
	int ids[4];
	for (int k = 0; k < 4; k++) {
		ids[k] = id_at(corners[k]);
	}
	vec3 albedo = colour_of(ids[0], drivers);
	// Most fragments have one material on all four corners and need no border.
	if (ids[1] != ids[0] || ids[2] != ids[0] || ids[3] != ids[0]) {
		float heights[4];
		float shares[4];
		float highest = -2.0;
		for (int k = 0; k < 4; k++) {
			float weight = 0.0;
			float same = 0.0;
			for (int j = 0; j < 4; j++) {
				if (ids[j] == ids[k]) {
					weight += near_weights[j];
					same += 1.0;
				}
			}
			heights[k] = weight + border_noise * (fbm(ground / 2.0 + float(ids[k]) * 17.31) - 0.5) * 2.0;
			shares[k] = 1.0 / same;
			highest = max(highest, heights[k]);
		}
		albedo = vec3(0.0);
		float total = 0.0;
		for (int k = 0; k < 4; k++) {
			float weight = max(heights[k] - highest + border_width, 0.0) * shares[k];
			albedo += colour_of(ids[k], drivers) * weight;
			total += weight;
		}
		albedo /= total;
	}

	float near = 1.0 - smoothstep(30.0, 120.0, length(world - CAMERA_POSITION_WORLD));
	float grain = detail * near;
	// The fine noise bumps the surface by its slope, fading with distance.
	vec3 fine = fbm_with_slope(ground * 1.7);
	vec2 slope = fine.yz * 1.7;
	vec3 normal = normalize(surface + vec3(-slope.x, 0.0, -slope.y) * 0.25 * grain);
	// The rock is the ground's colour muted, banded in warped strata.
	if (rock > 0.0) {
		float band = 0.5 + 0.5 * sin(world.y * 1.8 + fbm(ground / 12.0) * 2.0);
		vec3 muted = mix(albedo, vec3(dot(albedo, vec3(0.299, 0.587, 0.114))), 0.3) * 0.8;
		vec3 rock_base = mix(muted, rock_colour.rgb, rock_colour.a);
		albedo = mix(albedo, rock_base * mix(1.0, 0.8 + 0.4 * band, strata), rock);
	}
	// Gullies on steep ground: a ridged noise over the ground plane, which on a slope steep enough
	// changes little down the fall line and much along the slope, so its ridges run down it as
	// grooves; each groove is darker and the normal leans into it by the noise's own slope. They
	// come in from the slope of a badland's or a wall's flank, gentler than the rock's.
	float cut = gullies * smoothstep(GULLY_SLOPE.x, GULLY_SLOPE.y, steepness);
	if (cut > 0.0) {
		vec3 gully = value_with_slope(ground * GULLY_SCALE);
		float side = 2.0 * gully.x - 1.0;
		float groove = 1.0 - abs(side);
		vec2 lean = -sign(side) * 2.0 * gully.yz * GULLY_SCALE;
		albedo *= 1.0 - 0.3 * cut * smoothstep(0.6, 1.0, groove);
		normal = normalize(normal - vec3(lean.x, 0.0, lean.y) * 0.3 * cut);
	}
	albedo *= mix(1.0, 0.8 + 0.4 * fbm(ground / 48.0), macro_variation);
	vec2 from_focus = abs(world.xz - wave_forge_focus) / wave_forge_span;
	float beyond_grass = smoothstep(wave_forge_fade.x, wave_forge_fade.y, max(from_focus.x, from_focus.y));
	albedo = mix(albedo, cover_colour, cover * cover_tint * beyond_grass);
	albedo *= (1.0 - cavity_darkening * cavity) * (1.0 - wet_darkening * wet);
	albedo *= mix(1.0, 0.86 + 0.28 * fine.x, grain);

	ALBEDO = albedo;
	ROUGHNESS = mix(roughness, wet_roughness, wet);
	NORMAL = normalize((VIEW_MATRIX * vec4(normal, 0.0)).xyz);
}
