// Wave Forge's reference ground shader (docs/reference/godot.md, "Ground"): each chunk's ground
// carries its vertices' materials as an id texture, one texel per vertex, and every fragment draws
// the materials of the four vertices around it. Where they differ, each is weighted by how near
// its vertices are plus a noise of its own, and the highest wins with a short blend, so materials
// meet along a broken line rather than the grid's. Steep ground turns to rock by its own slope,
// its colour muted and banded in strata; a broad noise varies the colour across the land, and a fine one and a detail normal give
// the ground grain near the camera. Asset free; works on every renderer, Compatibility included.
// `wave_forge_bevy/src/shaders/ground.wesl` is the same shader for Bevy.
shader_type spatial;

// A chunk's material id per ground vertex, in the red channel as id / 255.
uniform sampler2D wave_forge_materials : filter_nearest, repeat_disable;
// One cell's width along x and z in world units: the ground's vertices stand at the centres of
// the cells, (i + 0.5) cells from the chunk's corner.
uniform vec2 wave_forge_cell = vec2(1.0);
// A colour per material id, in texel id of a 256 by 1 texture.
uniform sampler2D wave_forge_palette : filter_nearest, repeat_disable, source_color;
uniform float roughness = 0.9;
// Rock on steep ground is the ground's own colour muted and darkened, or with this colour's alpha
// above zero, this colour instead, by that much; and the slopes (one minus the normal's height) at
// which rock starts and has taken over.
uniform vec4 rock_colour : source_color = vec4(0.42, 0.40, 0.38, 0.0);
uniform vec2 rock_slope = vec2(0.3, 0.5);
// How strongly the rock is banded in strata.
uniform float strata : hint_range(0.0, 1.0) = 0.6;
// How far noise breaks the line where materials meet.
uniform float edge_noise : hint_range(0.0, 1.0) = 0.5;
// How much a broad noise varies the colour across the land.
uniform float macro_variation : hint_range(0.0, 1.0) = 0.5;
// How much fine grain and detail normal the ground has near the camera.
uniform float detail : hint_range(0.0, 1.0) = 1.0;

varying vec2 column;
varying vec3 world;
varying vec3 world_normal;

void vertex() {
	column = VERTEX.xz / wave_forge_cell - 0.5;
	world = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	world_normal = (MODEL_MATRIX * vec4(NORMAL, 0.0)).xyz;
}

float hash(vec2 p) {
	return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
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

int id_at(ivec2 vertex) {
	return int(texelFetch(wave_forge_materials, vertex, 0).r * 255.0 + 0.5);
}

vec3 colour_of(int id) {
	return texelFetch(wave_forge_palette, ivec2(id, 0), 0).rgb;
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
	int ids[4];
	for (int k = 0; k < 4; k++) {
		ids[k] = id_at(corners[k]);
	}
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
	vec3 albedo = vec3(0.0);
	float total = 0.0;
	for (int k = 0; k < 4; k++) {
		float weight = max(heights[k] - highest + border_width, 0.0) * shares[k];
		albedo += colour_of(ids[k]) * weight;
		total += weight;
	}
	albedo /= total;

	float near = 1.0 - smoothstep(30.0, 120.0, length(world - CAMERA_POSITION_WORLD));
	vec3 normal = surface;
	// The detail normal: the slope of a fine noise, fading with distance.
	if (detail > 0.0) {
		float step_m = 0.15;
		float here = fbm(ground * 1.7);
		vec2 slope = vec2(fbm((ground + vec2(step_m, 0.0)) * 1.7) - here, fbm((ground + vec2(0.0, step_m)) * 1.7) - here) / step_m;
		normal = normalize(normal + vec3(-slope.x, 0.0, -slope.y) * 0.25 * detail * near);
	}
	// The rock is the ground's colour muted, banded in warped strata.
	float band = 0.5 + 0.5 * sin(world.y * 1.8 + fbm(ground / 12.0) * 2.0);
	vec3 muted = mix(albedo, vec3(dot(albedo, vec3(0.299, 0.587, 0.114))), 0.3) * 0.8;
	vec3 rock_base = mix(muted, rock_colour.rgb, rock_colour.a);
	albedo = mix(albedo, rock_base * mix(1.0, 0.8 + 0.4 * band, strata), rock);
	albedo *= mix(1.0, 0.8 + 0.4 * fbm(ground / 48.0), macro_variation);
	albedo *= mix(1.0, 0.86 + 0.28 * fbm(ground * 1.3), detail * near);

	ALBEDO = albedo;
	ROUGHNESS = roughness;
	NORMAL = normalize((VIEW_MATRIX * vec4(normal, 0.0)).xyz);
}
