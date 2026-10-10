// Wave Forge's reference vegetation shader (docs/reference/godot.md, "Wind"): trees and bushes
// placed as MultiMeshes bend in the global wind, after GPU Gems 3, chapter 16. The main bending
// leans the whole plant downwind, more the higher up a vertex is and the less stiff the plant;
// the detail bending flutters the outer parts along their normals. Each instance's phase and
// stiffness come from the MultiMesh's custom data, which the stages node fills per point, so no
// two plants move alike; so does its look, a random and its appearance values, by which its colour
// varies and dries. Works on every renderer, Compatibility included.
shader_type spatial;

// The height, in the mesh's own units, at which a plant bends fully.
uniform float bend_height = 4.0;
// How far the outer parts flutter, in the mesh's own units at full wind.
uniform float flutter = 0.03;
uniform vec3 albedo : source_color = vec3(0.3, 0.5, 0.2);
uniform float roughness = 0.9;
// How far each plant's random varies its colour, brighter and warmer or darker and cooler.
uniform float variation : hint_range(0.0, 1.0) = 0.15;
// The colour a plant goes toward as its first appearance value, a dryness say, goes to 1, and how
// far it goes: a measured dry grass by default (docs/research/terrain-look.md, "Measured albedo").
uniform vec3 dry_albedo : source_color = vec3(0.365, 0.322, 0.212);
uniform float dryness : hint_range(0.0, 1.0) = 0.6;

varying float random;
varying float dry;
// The wind: a direction along x and z, a strength in world units at a blade's tip, and a speed.
global uniform vec4 wave_forge_wind;

void vertex() {
	// Custom data: x the phase as a fraction of a turn, y the stiffness, z the plant's random and
	// w its first appearance value.
	random = INSTANCE_CUSTOM.z;
	dry = clamp(INSTANCE_CUSTOM.w, 0.0, 1.0);
	float phase = INSTANCE_CUSTOM.x * TAU;
	float stiffness = max(INSTANCE_CUSTOM.y, 0.05);
	float strength = wave_forge_wind.z / stiffness;
	// The wind's direction in the mesh's own axes, which the instance's turn and lean change.
	vec3 across = inverse(mat3(MODEL_MATRIX)) * vec3(wave_forge_wind.x, 0.0, wave_forge_wind.y);
	float up = max(VERTEX.y, 0.0) / bend_height;
	float gust = 0.7 + 0.3 * sin(TIME * wave_forge_wind.w + phase);
	float reach = length(VERTEX);
	VERTEX.xz += across.xz * up * up * strength * gust;
	// Bending keeps a vertex's distance from the root, so the plant leans rather than stretches.
	VERTEX = normalize(VERTEX) * reach;
	float wave = sin(TIME * wave_forge_wind.w * 3.0 + phase + dot(VERTEX, vec3(2.3, 1.7, 3.1)));
	VERTEX += NORMAL * wave * flutter * up * strength;
}

void fragment() {
	float shift = variation * (random - 0.5) * 2.0;
	vec3 colour = mix(albedo, dry_albedo, dry * dryness);
	ALBEDO = colour * (1.0 + shift) * vec3(1.0 + 0.3 * shift, 1.0, 1.0 - 0.3 * shift);
	ROUGHNESS = roughness;
}
