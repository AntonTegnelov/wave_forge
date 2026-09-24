// Wave Forge's reference grass shader for Bevy (docs/reference/bevy.md, "Grass"): every chunk draws
// one shared mesh of blades whose vertices carry nothing but their blade's index. Blade b belongs to
// a column, stands at a hashed place in it, and shows only where a hash is below
// the column's cover, so the cover map alone decides where grass grows. It stands on the ground's
// heights and sways in the wind.
import bevy_pbr::render::{
    mesh_functions,
    mesh_view_bindings::globals,
    forward_io::{Vertex, VertexOutput, FragmentOutput, decompress_vertex},
    view_transformations::position_world_to_clip,
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT,
};

struct GrassSettings {
    // A cell's width along x and z, then a blade's height and width.
    cell_and_blade: vec4<f32>,
    // The wind: a direction along x and z, a strength at a blade's tip, and a speed.
    wind: vec4<f32>,
    // The chunk along x and z, then blades per column.
    chunk_and_count: vec4<i32>,
    base_colour: vec4<f32>,
    tip_colour: vec4<f32>,
};

@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<uniform> grass: GrassSettings;
// The ground's height at each of its vertices, in world units, one texel each.
@group(constants::MATERIAL_BIND_GROUP) @binding(101) var heights: texture_2d<f32>;
// The cover of each of the chunk's columns, from 0 (bare) to 1 (every blade shown).
@group(constants::MATERIAL_BIND_GROUP) @binding(102) var cover: texture_2d<f32>;

fn hash(value: u32) -> u32 {
    var x = value;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

fn unit(x: u32) -> f32 {
    return f32(hash(x) & 0xFFFFFFu) / 16777216.0;
}

fn height_at(vertex: vec2<i32>) -> f32 {
    return textureLoad(heights, vertex, 0).r;
}

@vertex
fn vertex(vertex_in: Vertex) -> VertexOutput {
    var out: VertexOutput;
    // Bevy may store a mesh's attributes compressed; its own decompression reads them.
    let in = decompress_vertex(vertex_in, vertex_in.instance_index);
    // The blade's index is in the second UV channel's x.
    let blade = i32(in.uv_b.x + 0.5);
    let columns = vec2<i32>(textureDimensions(cover));
    let column = blade / grass.chunk_and_count.z;
    let at = vec2<i32>(column % columns.x, column / columns.x);
    let seed = (u32(blade) * 747796405u) ^ (u32(grass.chunk_and_count.x) * 2891336453u)
        ^ (u32(grass.chunk_and_count.y) * 1181783497u);
    let place = vec2<f32>(at) + vec2<f32>(unit(seed), unit(seed ^ 0x9E3779B9u));
    let covered = textureLoad(cover, at, 0).r;
    let shown = column < columns.x * columns.y && unit(seed ^ 0x85EBCA6Bu) < covered;
    // The ground's vertices stand at the columns' centres.
    let last = vec2<i32>(textureDimensions(heights)) - 1;
    let grid = clamp(place - 0.5, vec2<f32>(0.0), vec2<f32>(last));
    let low = vec2<i32>(floor(grid));
    let high = min(low + 1, last);
    let t = grid - vec2<f32>(low);
    let ground = mix(
        mix(height_at(low), height_at(vec2<i32>(high.x, low.y)), t.x),
        mix(height_at(vec2<i32>(low.x, high.y)), height_at(high), t.x),
        t.y);
    let angle = unit(seed ^ 0x27D4EB2Fu) * 6.2831853;
    let facing = vec2<f32>(cos(angle), sin(angle));
    let along = in.position.y;
    let tall = grass.cell_and_blade.z * (0.7 + 0.6 * unit(seed ^ 0x165667B1u));
    let width = grass.cell_and_blade.w;
    var blade_offset = vec3<f32>(facing.x * in.position.x * width, along * tall, facing.y * in.position.x * width);
    let phase = unit(seed ^ 0xD3A2646Cu) * 6.2831853;
    blade_offset.x += grass.wind.x * grass.wind.z * along * along * (0.6 + 0.4 * sin(globals.time * grass.wind.w + phase));
    blade_offset.z += grass.wind.y * grass.wind.z * along * along * (0.6 + 0.4 * sin(globals.time * grass.wind.w + phase));
    var local = vec3<f32>(0.0);
    if shown {
        local = vec3<f32>(place.x * grass.cell_and_blade.x, ground, place.y * grass.cell_and_blade.y) + blade_offset;
    }
    let world_from_local = mesh_functions::get_world_from_local(vertex_in.instance_index);
    out.world_position = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(local, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);
    out.world_normal = vec3<f32>(0.0, 1.0, 0.0);
    out.uv = vec2<f32>(0.0, along);
    // The fragment finds its material through the instance.
    @if(VERTEX_OUTPUT_INSTANCE_INDEX)
    out.instance_index = vertex_in.instance_index;
    return out;
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = mix(grass.base_colour, grass.tip_colour, in.uv.y);
    var out: FragmentOutput;
    if (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u {
        out.color = apply_pbr_lighting(pbr_input);
    } else {
        out.color = pbr_input.material.base_color;
    }
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
