// Wave Forge's reference grass shader (docs/reference/godot.md, "Grass"): every chunk draws one
// MultiMesh of blades with no data of their own. Blade INSTANCE_ID belongs to a column, stands at a
// hashed place in it, and shows only where a hash is below the column's cover, so the cover map
// alone decides where grass grows. It stands on the ground's heights and sways in the global wind.
// Works on every renderer, Compatibility included.
shader_type spatial;
render_mode cull_disabled;

// The ground's height at each of its vertices, in world units, one texel each.
uniform sampler2D wave_forge_heights : filter_nearest, repeat_disable;
// The cover of each of the chunk's columns, from 0 (bare) to 1 (every blade shown).
uniform sampler2D wave_forge_cover : filter_nearest, repeat_disable;
// One cell's width along x and z in world units.
uniform vec2 wave_forge_cell = vec2(1.0);
// Blades per column.
uniform int wave_forge_per_column = 8;
// The chunk, so neighbouring chunks do not repeat each other's blades.
uniform ivec2 wave_forge_chunk;
uniform float blade_height = 0.6;
uniform float blade_width = 0.1;
uniform vec3 base_colour : source_color = vec3(0.16, 0.32, 0.09);
uniform vec3 tip_colour : source_color = vec3(0.55, 0.74, 0.3);
// The wind: a direction along x and z, a strength in world units at a blade's tip, and a speed.
global uniform vec4 wave_forge_wind;

varying float along;

uint hash(uint x) {
	x ^= x >> 16u;
	x *= 0x7feb352du;
	x ^= x >> 15u;
	x *= 0x846ca68bu;
	x ^= x >> 16u;
	return x;
}

float unit(uint x) {
	return float(hash(x) & 0xFFFFFFu) / 16777216.0;
}

float height_at(ivec2 vertex) {
	return texelFetch(wave_forge_heights, vertex, 0).r;
}

void vertex() {
	ivec2 columns = textureSize(wave_forge_cover, 0);
	int column = INSTANCE_ID / wave_forge_per_column;
	ivec2 at = ivec2(column % columns.x, column / columns.x);
	uint seed = uint(INSTANCE_ID) * 747796405u ^ uint(wave_forge_chunk.x) * 2891336453u
		^ uint(wave_forge_chunk.y) * 1181783497u;
	vec2 place = vec2(at) + vec2(unit(seed), unit(seed ^ 0x9E3779B9u));
	float cover = texelFetch(wave_forge_cover, at, 0).r;
	bool shown = column < columns.x * columns.y && unit(seed ^ 0x85EBCA6Bu) < cover;
	// The ground's vertices stand at the columns' centres.
	ivec2 last = textureSize(wave_forge_heights, 0) - 1;
	vec2 grid = clamp(place - 0.5, vec2(0.0), vec2(last));
	ivec2 low = ivec2(floor(grid));
	ivec2 high = min(low + 1, last);
	vec2 t = grid - vec2(low);
	float ground = mix(
		mix(height_at(low), height_at(ivec2(high.x, low.y)), t.x),
		mix(height_at(ivec2(low.x, high.y)), height_at(high), t.x),
		t.y);
	float angle = unit(seed ^ 0x27D4EB2Fu) * TAU;
	vec2 facing = vec2(cos(angle), sin(angle));
	along = VERTEX.y;
	float tall = blade_height * (0.7 + 0.6 * unit(seed ^ 0x165667B1u));
	vec3 blade = vec3(facing.x * VERTEX.x * blade_width, VERTEX.y * tall, facing.y * VERTEX.x * blade_width);
	float phase = unit(seed ^ 0xD3A2646Cu) * TAU;
	blade.xz += wave_forge_wind.xy * wave_forge_wind.z * along * along
		* (0.6 + 0.4 * sin(TIME * wave_forge_wind.w + phase));
	VERTEX = shown ? vec3(place.x * wave_forge_cell.x, ground, place.y * wave_forge_cell.y) + blade : vec3(0.0);
	NORMAL = vec3(0.0, 1.0, 0.0);
}

void fragment() {
	// A blade is seen from both sides; both are lit as its front.
	if (!FRONT_FACING) {
		NORMAL = -NORMAL;
	}
	ALBEDO = mix(base_colour, tip_colour, along);
	ROUGHNESS = 0.8;
}
