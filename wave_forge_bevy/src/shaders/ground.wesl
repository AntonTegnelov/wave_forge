// Wave Forge's reference ground shader for Bevy (docs/reference/bevy.md, "Ground materials"): each
// chunk's ground carries its vertices' materials as an id texture, one texel per vertex, and every
// fragment blends the colours of the four vertices around it through a palette, so materials meet
// in a smooth band rather than along triangle edges. Bevy's own lighting does the rest.
import bevy_pbr::render::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT,
};

// The chunk's corner on the ground plane (x, z) and a cell's width along x and z.
@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<uniform> grid: vec4<f32>;
// The chunk's material id per ground vertex, in the red channel as id / 255.
@group(constants::MATERIAL_BIND_GROUP) @binding(101) var materials: texture_2d<f32>;
// A colour per material id, in texel id of a 256 by 1 texture.
@group(constants::MATERIAL_BIND_GROUP) @binding(102) var palette: texture_2d<f32>;

fn colour_at(vertex: vec2<i32>) -> vec4<f32> {
    let id = i32(textureLoad(materials, vertex, 0).r * 255.0 + 0.5);
    return textureLoad(palette, vec2<i32>(id, 0), 0);
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    let size = vec2<i32>(textureDimensions(materials));
    // The ground's vertices stand at the columns' centres.
    let column = (in.world_position.xz - grid.xy) / grid.zw - 0.5;
    let at = clamp(column, vec2<f32>(0.0), vec2<f32>(size - 1));
    let low = vec2<i32>(floor(at));
    let high = min(low + 1, size - 1);
    let t = at - vec2<f32>(low);
    let near_row = mix(colour_at(low), colour_at(vec2<i32>(high.x, low.y)), t.x);
    let far_row = mix(colour_at(vec2<i32>(low.x, high.y)), colour_at(high), t.x);
    pbr_input.material.base_color = mix(near_row, far_row, t.y);
    var out: FragmentOutput;
    if (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u {
        out.color = apply_pbr_lighting(pbr_input);
    } else {
        out.color = pbr_input.material.base_color;
    }
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
