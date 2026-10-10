// Wave Forge's reference grass shader (docs/reference/godot.md, "Grass"): every chunk draws one
// MultiMesh of blades with no data of their own. Blade INSTANCE_ID belongs to a column, stands at a
// hashed place in it, and shows only where a hash is below the column's cover, so the cover map
// alone decides where grass grows. It stands on the ground's heights and sways in the global wind.
// Blades grow in clumps, the cells of a cellular noise, each clump of one height, leaning one way
// and of one colour between healthy and dry; a blade's base takes the ground's colour under it, its
// normal is rounded across it, and a blade seen edge-on widens. Farther away fewer blades show, each
// wider, so the meadow keeps its cover at less cost. Works on every renderer, Compatibility
// included.
shader_type spatial;
render_mode cull_disabled;

// The ground's height at each of its vertices, in world units, one texel each.
uniform sampler2D wave_forge_heights : filter_nearest, repeat_disable;
// The cover of each of the chunk's columns, from 0 (bare) to 1 (every blade shown).
uniform sampler2D wave_forge_cover : filter_nearest, repeat_disable;
// One cell's width along x and z in world units.
uniform vec2 wave_forge_cell = vec2(1.0);
// Blades per column in this chunk's MultiMesh, and in a full one: a chunk far enough away draws
// fewer, the first of the full set's, since those farther down a column's ranks would not show.
uniform int wave_forge_per_column = 8;
uniform int wave_forge_blades = 8;
// The chunk, so neighbouring chunks do not repeat each other's blades.
uniform ivec2 wave_forge_chunk;
// The chunks from the followed chunk's centre, along the farther axis, from which blades shrink and
// at which they are gone, so the grass fades out before its edge rather than stopping at it; and a
// chunk's width along x and z and that centre on the ground plane, in world units. Measured so, the
// fade follows the square of chunks that have grass, wherever the camera is.
uniform vec2 wave_forge_fade = vec2(1e9, 2e9);
uniform vec2 wave_forge_span = vec2(1.0);
uniform vec2 wave_forge_focus = vec2(0.0);
uniform float blade_height = 0.6;
uniform float blade_width = 0.07;
// How wide a clump is, in world units, how far its blades lean its way at the tip, and how much of
// the clumps are dry, from none to all.
uniform float clump_size = 1.5;
uniform float clump_lean = 0.15;
uniform float dry_clumps : hint_range(0.0, 1.0) = 0.2;
// The colour a dry clump's blades take, a measured dry grass (docs/research/terrain-look.md,
// "Measured albedo").
uniform vec3 dry_colour : source_color = vec3(0.365, 0.322, 0.212);
// The ground's material per vertex and its palette, as the ground shader takes them, and how far a
// blade's base takes the ground's colour. The node hands them in when the ground has materials.
uniform sampler2D wave_forge_materials : filter_nearest, repeat_disable;
uniform sampler2D wave_forge_palette : filter_nearest, repeat_disable, source_color;
uniform bool wave_forge_ground_colours = false;
uniform float ground_tint : hint_range(0.0, 1.0) = 0.6;
// From how far and to how far from the camera, in world units, the blades thin out, and the share
// of them still standing at the far end; those left widen to keep the cover.
uniform vec2 thin_distance = vec2(4.0, 14.0);
uniform float thin_keep : hint_range(0.05, 1.0) = 0.12;
// Measured grass, dark at the base and light at the tip (docs/research/terrain-look.md, "Measured
// albedo").
uniform vec3 base_colour : source_color = vec3(0.31, 0.306, 0.157);
uniform vec3 tip_colour : source_color = vec3(0.435, 0.435, 0.267);
// The wind: a direction along x and z, a strength in world units at a blade's tip, and a speed.
global uniform vec4 wave_forge_wind;

varying float along;
varying vec3 ground_colour;
varying float dry;

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

// The clump `point` lies in: a cell of a cellular noise over the ground plane, whose nearest
// feature point names it.
uint clump_of(vec2 point) {
	vec2 cell = floor(point);
	float nearest = 1e9;
	uint id = 0u;
	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			vec2 c = cell + vec2(float(x), float(y));
			uint key = uint(int(c.x)) * 73856093u ^ uint(int(c.y)) * 19349663u;
			vec2 feature = c + vec2(unit(key), unit(key ^ 0x68E31DA4u));
			float d = dot(point - feature, point - feature);
			if (d < nearest) {
				nearest = d;
				id = key;
			}
		}
	}
	return id;
}

void vertex() {
	ivec2 columns = textureSize(wave_forge_cover, 0);
	int column = INSTANCE_ID / wave_forge_per_column;
	// The blade's rank in its column: a blade is the same in a full MultiMesh and a smaller one.
	int rank = INSTANCE_ID % wave_forge_per_column;
	ivec2 at = ivec2(column % columns.x, column / columns.x);
	uint seed = (uint(column) * 4096u + uint(rank)) * 747796405u ^ uint(wave_forge_chunk.x) * 2891336453u
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
	along = VERTEX.y;
	// Which edge of the blade the vertex is on, -0.5 to 0.5, its tip in the middle.
	float side = VERTEX.x;
	vec3 root = (MODEL_MATRIX * vec4(place.x * wave_forge_cell.x, ground, place.y * wave_forge_cell.y, 1.0)).xyz;
	// The clump: its height, the way it leans and whether it is dry.
	uint clump = clump_of(root.xz / clump_size);
	float clump_tall = 0.65 + 0.7 * unit(clump ^ 0x3C6EF372u);
	float clump_angle = unit(clump ^ 0xA54FF53Au) * TAU;
	dry = smoothstep(1.0 - dry_clumps, 1.0 - dry_clumps + 0.15, unit(clump ^ 0x510E527Fu));
	// A blade faces about its clump's way, turned a little by its own hash.
	float angle = clump_angle + (unit(seed ^ 0x27D4EB2Fu) - 0.5) * 1.6;
	vec2 facing = vec2(cos(angle), sin(angle));
	vec2 from_focus = abs(root.xz - wave_forge_focus) / wave_forge_span;
	float faded = 1.0 - smoothstep(wave_forge_fade.x, wave_forge_fade.y, max(from_focus.x, from_focus.y));
	// Farther away fewer blades show, a column's first by rank, each wider by as much as their
	// number fell.
	float distance = length(root - CAMERA_POSITION_WORLD);
	float keep = mix(1.0, thin_keep, smoothstep(thin_distance.x, thin_distance.y, distance));
	float ranked = (float(rank) + unit(seed ^ 0x9B05688Cu)) / float(wave_forge_blades);
	shown = shown && faded > 0.0 && ranked < keep;
	float tall = blade_height * clump_tall * (0.85 + 0.3 * unit(seed ^ 0x165667B1u)) * faded;
	// A blade seen edge-on widens, so it still covers what it would face-on.
	vec3 face = vec3(-facing.y, 0.0, facing.x);
	vec3 view = normalize(root - CAMERA_POSITION_WORLD);
	float edge_on = 1.0 - abs(dot(view.xz, face.xz)) / max(length(view.xz), 1e-4);
	float wide = blade_width * inversesqrt(keep) * (1.0 + 0.5 * edge_on);
	vec3 blade = vec3(facing.x * VERTEX.x * wide, VERTEX.y * tall, facing.y * VERTEX.x * wide);
	float phase = unit(seed ^ 0xD3A2646Cu) * TAU;
	blade.xz += vec2(cos(clump_angle), sin(clump_angle)) * clump_lean * tall * along * along;
	blade.xz += wave_forge_wind.xy * wave_forge_wind.z * along * along
		* (0.6 + 0.4 * sin(TIME * wave_forge_wind.w + phase));
	VERTEX = shown ? vec3(place.x * wave_forge_cell.x, ground, place.y * wave_forge_cell.y) + blade : vec3(0.0);
	// Mostly the sky's, as a meadow is lit, rounded across the blade, leaning out at its edges, so
	// both of its sides take the light alike.
	vec3 across = vec3(facing.x, 0.0, facing.y);
	NORMAL = normalize(vec3(0.0, 1.0, 0.0) + face * 0.3 + across * side * 0.8);
	// The ground's colour under the blade's root, from its nearest vertex.
	ground_colour = base_colour;
	if (wave_forge_ground_colours) {
		ivec2 vertex = clamp(ivec2(round(place - 0.5)), ivec2(0), textureSize(wave_forge_materials, 0) - 1);
		int id = int(texelFetch(wave_forge_materials, vertex, 0).r * 255.0 + 0.5);
		ground_colour = mix(base_colour, texelFetch(wave_forge_palette, ivec2(id, 0), 0).rgb, ground_tint);
	}
}

void fragment() {
	// A blade is seen from both sides; both are lit as its front.
	if (!FRONT_FACING) {
		NORMAL = -NORMAL;
	}
	ALBEDO = mix(ground_colour, mix(tip_colour, dry_colour, dry), along);
	ROUGHNESS = 0.8;
	// A blade's thin cards would catch the sky's sheen as a broad surface does; grass takes little.
	SPECULAR = 0.2;
}
