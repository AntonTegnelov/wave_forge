// Wave Forge's reference ground shader (docs/reference/godot.md, "Ground"): each chunk's ground
// carries its vertices' materials as an id texture, one texel per vertex, and every fragment draws
// the materials of the four vertices around it. Where they differ, each is weighted by how near
// its vertices are plus a noise of its own, and the highest wins with a short blend, so materials
// meet along a broken line rather than the grid's. A material may be drawn with a texture set
// instead of its colour and relief, seen from above and, on steep ground, from the sides, its tiles
// broken up by a second sample. Each material has a micro-height, its relief's (pebbles, cracks or
// tufts) or its texture's height, and where materials meet the one standing higher wins; near the
// camera a relief bumps the ground and grains its colour. Steep ground turns to rock by its own
// slope, its colour muted and banded in strata, unless a texture set draws it; a broad noise varies
// the colour across the land. Asset free without texture sets; works on every
// renderer, Compatibility included. Channels the node hands in darken hollows and wet ground, and
// each material's colour may run along a ramp of its own by height, slope, cavity and wetness.
// `wave_forge_bevy/src/shaders/ground.wesl` is the same shader for Bevy.
shader_type spatial;

// A chunk's material id per ground vertex, in the red channel as id / 255.
uniform sampler2D wave_forge_materials : filter_nearest, repeat_disable;
// One cell's width along x and z in world units: the ground's vertices stand at the centres of
// the cells, (i + 0.5) cells from the chunk's corner.
uniform vec2 wave_forge_cell = vec2(1.0);
// Where the first vertex stands from the mesh's origin, in cells of `wave_forge_cell`: the centre
// of the first cell for a near ground, less for the far ground, whose squares are coarse columns
// with their first vertex at a fine cell's centre.
uniform vec2 wave_forge_first = vec2(0.5);
// How many fine cells wide one of this mesh's cells is: 1 for a near ground, a coarse column's
// width for the far ground, so the noise that breaks the borders between materials is as coarse as
// the cells whose borders it breaks.
uniform float wave_forge_coarseness = 1.0;
// A colour per material id, in texel id of a 256 by 1 texture.
uniform sampler2D wave_forge_palette : filter_nearest, repeat_disable, source_color;
// Per material id, in texel id of a 256 by 2 texture: the colour, linear, its ramp runs toward in
// row 0, and in row 1 how far the ground's height (0 to 1 over `ramp_heights`), slope, cavity and
// wetness each move its colour along that ramp, from -1 to 1. Black, no ramp, where the node hands
// in none.
uniform sampler2D wave_forge_ramps : filter_nearest, repeat_disable, hint_default_black;
// Per material id, in texel id of a 256 by 1 texture: how much of it is pebbles, cracks and tufts,
// and how wide one of its features is in world units (`wave_forge::GroundRelief`). Black, a flat
// ground, where the node hands in none.
uniform sampler2D wave_forge_relief : filter_nearest, repeat_disable, hint_default_black;
// How far the materials' micro-heights move the line where they meet.
uniform float relief_blend : hint_range(0.0, 1.0) = 0.6;
// Texture sets a material can be drawn with instead of its colour: layers of two arrays, the
// colour with the height in alpha and the OpenGL normal with the roughness in alpha. Per material
// id, in texel id of a 256 by 1 texture, its set's layer plus one, as id / 255 in red; 0, and black
// where the node hands in none, draws the material in its colour.
uniform sampler2DArray wave_forge_albedo_height : source_color, filter_linear_mipmap, repeat_enable;
uniform sampler2DArray wave_forge_normal_roughness : filter_linear_mipmap, repeat_enable;
uniform sampler2D wave_forge_texture_layers : filter_nearest, repeat_disable, hint_default_black;
// How wide one tile of a texture set is, in world units, and how far its normal leans the ground's.
uniform float texture_size = 3.0;
uniform float texture_normal : hint_range(0.0, 2.0) = 1.0;
// How far a texture set's own roughness moves the ground's from `roughness`: photographed sets
// tend to read glossier than ground does in sunlight.
uniform float texture_roughness : hint_range(0.0, 1.0) = 0.25;
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
// How much the micro-heights bump the ground and grain its colour near the camera.
uniform float detail : hint_range(0.0, 1.0) = 1.0;

// The slopes (one minus the normal's height) at which gullies start and are cut fully, and how many
// gullies a world unit of the ground plane holds.
const vec2 GULLY_SLOPE = vec2(0.08, 0.2);
const float GULLY_SCALE = 0.7;
// How far a micro-height of full depth leans the normal, per unit of its slope in feature widths.
const float RELIEF_BUMP = 0.25;
// How wide a crack is, in feature widths, and how far its sides lean the normal against a plate's.
const float CRACK_WIDTH = 0.06;
const float CRACK_DEPTH = 0.15;

varying vec2 column;
varying vec3 world;
varying vec3 world_normal;

void vertex() {
	column = VERTEX.xz / wave_forge_cell - wave_forge_first;
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

// Round stones in about half the cells of a grid over `p`, in feature widths, each at a hashed
// place and of a hashed size up to a third of a feature: (height 0 to 1, its slope along x and z).
vec3 pebbles(vec2 p) {
	vec2 cell = floor(p);
	vec3 stone = vec3(0.0);
	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			vec2 c = cell + vec2(float(x), float(y));
			float size = hash(c + 7.7);
			if (size < 0.45) {
				continue;
			}
			float radius = 0.15 + 0.35 * (size - 0.45);
			vec2 d = (p - (c + 0.25 + 0.5 * vec2(hash(c), hash(c + 19.19)))) / radius;
			float h = 1.0 - dot(d, d);
			if (h > stone.x) {
				stone = vec3(h, -2.0 * d / radius);
			}
		}
	}
	return stone;
}

// Plates parted by narrow cracks: the cells of a cellular noise over `p`, in feature widths, a
// plate standing at 1 and a crack falling to 0 where its two nearest features are about as near,
// shallow, so it reads as a line: (height 0 to 1, its slope along x and z).
vec3 cracks(vec2 p) {
	vec2 cell = floor(p);
	float first = 1e9;
	float second = 1e9;
	vec2 to_first = vec2(0.0);
	vec2 to_second = vec2(0.0);
	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			vec2 c = cell + vec2(float(x), float(y));
			vec2 d = p - (c + vec2(hash(c + 3.7), hash(c + 41.3)));
			float squared = dot(d, d);
			if (squared < first) {
				second = first;
				to_second = to_first;
				first = squared;
				to_first = d;
			} else if (squared < second) {
				second = squared;
				to_second = d;
			}
		}
	}
	float near = sqrt(first);
	float far = sqrt(second);
	float x = clamp((far - near) / CRACK_WIDTH, 0.0, 1.0);
	vec2 across = to_second / max(far, 1e-4) - to_first / max(near, 1e-4);
	return vec3(x * x * (3.0 - 2.0 * x), CRACK_DEPTH * 6.0 * x * (1.0 - x) / CRACK_WIDTH * across);
}

// Material `id`'s micro-height at `ground`, world units along x and z: (height 0 to 1, its slope
// along x and z in feature widths, its depth 0 to 1).
vec4 relief_of(int id, vec2 ground) {
	vec4 shape = texelFetch(wave_forge_relief, ivec2(id, 0), 0);
	float total = shape.x + shape.y + shape.z;
	if (total <= 0.0 || shape.w <= 0.0) {
		return vec4(0.5, 0.0, 0.0, 0.0);
	}
	vec2 p = ground / shape.w;
	vec3 h = vec3(0.0);
	if (shape.x > 0.0) {
		h += shape.x * pebbles(p);
	}
	if (shape.y > 0.0) {
		h += shape.y * cracks(p + 31.7);
	}
	if (shape.z > 0.0) {
		h += shape.z * fbm_with_slope(p + 57.3);
	}
	return vec4(h / total, min(total, 1.0));
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

// What a material looks like at a fragment: its colour, its micro-height (its height 0 to 1 and its
// slope along x and z, scaled by its depth), how deep that height is, 0 to 1, its roughness, and 1
// for a texture set or 0 for its colour and relief.
struct Look {
	vec3 albedo;
	vec3 micro;
	float depth;
	float roughness;
	float textured;
};

// A texture set's layer `layer` at `ground`, seen along `axis`: two samples, the second turned and
// scaled, blended by a broad noise, so the tiles do not repeat; (colour and height, normal and
// roughness).
void sample_set(float layer, vec2 ground, out vec4 colour, out vec4 normal) {
	vec2 uv = ground / texture_size;
	mat2 turn = mat2(vec2(0.8, 0.6), vec2(-0.6, 0.8));
	vec2 other = turn * uv * 0.73 + 0.37;
	float blend = smoothstep(0.3, 0.7, value(ground / (texture_size * 2.3)));
	colour = mix(texture(wave_forge_albedo_height, vec3(uv, layer)), texture(wave_forge_albedo_height, vec3(other, layer)), blend);
	normal = mix(texture(wave_forge_normal_roughness, vec3(uv, layer)), texture(wave_forge_normal_roughness, vec3(other, layer)), blend);
}

// Material `id`'s look at `at`, `ground` along the ground plane, on a surface of normal `surface`:
// its texture set's, seen from above and, on steep ground, from the sides as well, or its colour
// and, with `relief`, its relief's.
Look look_of(int id, vec3 at, vec2 ground, vec3 surface, vec4 drivers, bool relief) {
	Look look;
	float layer = floor(texelFetch(wave_forge_texture_layers, ivec2(id, 0), 0).r * 255.0 + 0.5) - 1.0;
	if (layer < 0.0) {
		vec4 r = relief ? relief_of(id, ground) : vec4(0.5, 0.0, 0.0, 0.0);
		look.albedo = colour_of(id, drivers);
		look.micro = vec3(r.x, r.yz * r.w);
		look.depth = r.w;
		look.roughness = roughness;
		look.textured = 0.0;
		return look;
	}
	vec3 weights = pow(abs(surface), vec3(4.0));
	weights /= weights.x + weights.y + weights.z;
	vec4 colour;
	vec4 normal;
	sample_set(layer, ground, colour, normal);
	// The normal's lean along the ground's x and z, from above.
	vec2 lean = (normal.rg * 2.0 - 1.0) * vec2(1.0, -1.0);
	if (weights.y < 0.98) {
		vec4 side_x;
		vec4 side_x_normal;
		vec4 side_z;
		vec4 side_z_normal;
		sample_set(layer, vec2(at.z, -at.y), side_x, side_x_normal);
		sample_set(layer, vec2(at.x, -at.y), side_z, side_z_normal);
		colour = colour * weights.y + side_x * weights.x + side_z * weights.z;
		normal = normal * weights.y + side_x_normal * weights.x + side_z_normal * weights.z;
		lean *= weights.y;
	}
	look.albedo = colour.rgb;
	look.micro = vec3(colour.a, -lean * texture_normal);
	look.depth = 1.0;
	look.roughness = mix(roughness, normal.a, texture_roughness);
	look.textured = 1.0;
	return look;
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
	float near = 1.0 - smoothstep(30.0, 120.0, length(world - CAMERA_POSITION_WORLD));
	float grain = detail * near;
	// What the ground looks like here: one material's look, or where materials meet, theirs blended
	// by how high each stands.
	Look here = look_of(ids[0], world, ground, surface, drivers, grain > 0.0);
	// Most fragments have one material on all four corners and need no border.
	if (ids[1] != ids[0] || ids[2] != ids[0] || ids[3] != ids[0]) {
		float heights[4];
		float shares[4];
		Look looks[4];
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
			// A material on several corners has its look worked out once.
			int first = k;
			for (int j = k - 1; j >= 0; j--) {
				if (ids[j] == ids[k]) {
					first = j;
				}
			}
			looks[k] = first == k ? look_of(ids[k], world, ground, surface, drivers, true) : looks[first];
			// The material standing higher, by its micro-height or its texture's height, wins: sand
			// settles between pebbles, rock plates stand out of their cracks.
			heights[k] = weight + border_noise * (fbm(ground / (2.0 * wave_forge_coarseness) + float(ids[k]) * 17.31) - 0.5) * 2.0
				+ (1.0 - rock) * relief_blend * looks[k].depth * (looks[k].micro.x - 0.5);
			shares[k] = 1.0 / same;
			highest = max(highest, heights[k]);
		}
		Look blended;
		blended.albedo = vec3(0.0);
		blended.micro = vec3(0.0);
		blended.roughness = 0.0;
		blended.textured = 0.0;
		float total = 0.0;
		for (int k = 0; k < 4; k++) {
			float weight = max(heights[k] - highest + border_width, 0.0) * shares[k];
			blended.albedo += looks[k].albedo * weight;
			blended.micro += looks[k].micro * weight;
			blended.roughness += looks[k].roughness * weight;
			blended.textured += looks[k].textured * weight;
			total += weight;
		}
		here.albedo = blended.albedo / total;
		here.micro = blended.micro / total;
		here.roughness = blended.roughness / total;
		here.textured = blended.textured / total;
	}
	vec3 albedo = here.albedo;
	vec3 micro = here.micro;
	// The micro-height bumps the surface by its slope, fading with distance; a texture's normal
	// does at every distance, its mipmaps fading it.
	vec3 normal = normalize(surface + vec3(-micro.y, 0.0, -micro.z) * mix(RELIEF_BUMP * grain, 1.0, here.textured));
	// The rock is the ground's colour muted, banded in warped strata; a texture set draws its own.
	rock *= 1.0 - here.textured;
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
	albedo *= mix(1.0, 0.86 + 0.28 * micro.x, grain * (1.0 - here.textured));

	ALBEDO = albedo;
	ROUGHNESS = mix(here.roughness, wet_roughness, wet);
	NORMAL = normalize((VIEW_MATRIX * vec4(normal, 0.0)).xyz);
}
