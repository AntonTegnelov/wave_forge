// Wave Forge's reference vegetation shader for Bevy (docs/reference/bevy.md, "Wind"): a plant's
// mesh bends in the wind, after GPU Gems 3, chapter 16. The main bending leans the whole plant
// downwind, more the higher up a vertex is and the less stiff the plant; the detail bending flutters
// the outer parts out from the plant's axis. The chapter flutters along the normals, but Bevy's
// prepass has no normals unless a normal prepass runs, and both passes must move a vertex alike. A plant's phase is hashed from where it stands and its
// stiffness is its scale, so no two plants move alike and a larger one bends less. The same vertex
// shader serves the main pass and the prepass, so depth, shadows and motion vectors follow the
// swaying plant.
import bevy_pbr::render::{mesh_functions, view_transformations::position_world_to_clip};

@if(PREPASS_PIPELINE)
import bevy_pbr::prepass::io::{Vertex, VertexOutput, decompress_vertex};
@else
import bevy_pbr::render::forward_io::{Vertex, VertexOutput, decompress_vertex};

@if(PREPASS_PIPELINE)
import bevy_render::globals::Globals;
@else
import bevy_pbr::render::mesh_view_bindings::globals;

// The prepass binds the frame's globals at a binding of its own.
@if(PREPASS_PIPELINE)
@group(0) @binding(1) var<uniform> globals: Globals;

struct VegetationSettings {
    // The wind: a direction along x and z, a strength in world units, and a speed.
    wind: vec4<f32>,
    // The height, in the mesh's own units, at which a plant bends fully, then how far its outer
    // parts flutter, in the mesh's own units at full wind.
    bend: vec4<f32>,
};

@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<uniform> vegetation: VegetationSettings;

fn hash(value: u32) -> u32 {
    var x = value;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

// Where a vertex at `local` is at `time`, in the mesh's own units, on a plant placed by
// `world_from_local` with a phase of `phase` turns.
fn bent(local: vec3<f32>, world_from_local: mat4x4<f32>, phase: f32, time: f32) -> vec3<f32> {
    let stiffness = max(length(world_from_local[1].xyz), 0.05);
    let strength = vegetation.wind.z / stiffness;
    // The wind's direction in the mesh's own axes, which the plant's turn and lean change. For a
    // plant scaled evenly by s, the inverse of its turn and scale is their transpose over s squared.
    let model = mat3x3<f32>(world_from_local[0].xyz, world_from_local[1].xyz, world_from_local[2].xyz);
    let across = transpose(model) * vec3<f32>(vegetation.wind.x, 0.0, vegetation.wind.y) / (stiffness * stiffness);
    let up = max(local.y, 0.0) / vegetation.bend.x;
    let gust = 0.7 + 0.3 * sin(time * vegetation.wind.w + phase);
    let reach = length(local);
    var position = local;
    position.x += across.x * up * up * strength * gust;
    position.z += across.z * up * up * strength * gust;
    // Bending keeps a vertex's distance from the root, so the plant leans rather than stretches.
    if reach > 0.0 {
        position = normalize(position) * reach;
    }
    let wave = sin(time * vegetation.wind.w * 3.0 + phase + dot(position, vec3<f32>(2.3, 1.7, 3.1)));
    let out = vec3<f32>(position.x, 0.0, position.z);
    if dot(out, out) == 0.0 {
        return position;
    }
    return position + normalize(out) * wave * vegetation.bend.y * up * strength;
}

@vertex
fn vertex(vertex_in: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let in = decompress_vertex(vertex_in, vertex_in.instance_index);
    let world_from_local = mesh_functions::get_world_from_local(vertex_in.instance_index);
    let origin = bitcast<vec3<u32>>(world_from_local[3].xyz);
    let phase = f32(hash(origin.x ^ hash(origin.y ^ hash(origin.z))) & 0xFFFFu) / 65536.0 * 6.2831853;
    let local = bent(in.position, world_from_local, phase, globals.time);
    out.world_position = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(local, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);
@if(PREPASS_PIPELINE && UNCLIPPED_DEPTH_ORTHO_EMULATION) {
    out.unclipped_depth = out.position.z;
    out.position.z = min(out.position.z, 1.0);
}
@if(VERTEX_UVS_A)
    out.uv = in.uv;
@if(VERTEX_UVS_B)
    out.uv_b = in.uv_b;
@if(VERTEX_NORMALS && (!PREPASS_PIPELINE || NORMAL_PREPASS_OR_DEFERRED_PREPASS))
    out.world_normal = mesh_functions::mesh_normal_local_to_world(in.normal, vertex_in.instance_index);
@if(VERTEX_TANGENTS && (!PREPASS_PIPELINE || NORMAL_PREPASS_OR_DEFERRED_PREPASS))
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(world_from_local, in.tangent, vertex_in.instance_index);
@if(VERTEX_COLORS)
    out.color = in.color;
@if(PREPASS_PIPELINE && MOTION_VECTOR_PREPASS) {
    let previous_from_local = mesh_functions::get_previous_world_from_local(vertex_in.instance_index);
    let previous = bent(in.position, previous_from_local, phase, globals.time - globals.delta_time);
    out.previous_world_position = mesh_functions::mesh_position_local_to_world(previous_from_local, vec4<f32>(previous, 1.0));
}
@if(VERTEX_OUTPUT_INSTANCE_INDEX)
    out.instance_index = vertex_in.instance_index;
@if(VISIBILITY_RANGE_DITHER)
    out.visibility_range_dither = mesh_functions::get_visibility_range_dither_level(vertex_in.instance_index, world_from_local[3]);
    return out;
}
