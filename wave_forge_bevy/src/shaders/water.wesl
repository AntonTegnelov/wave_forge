// Wave Forge's reference water shader for Bevy (docs/reference/bevy.md, "Water"): lakes and rivers
// drawn by how deep they stand over the ground, which `water_mesh` hands in per vertex in the first
// UV's x (`wave_forge::WaterMesh::depths`). Shallow water shows the ground through it in a light
// colour, deep water turns dark and opaque, and at the shore it fades out to nothing where it dips
// under the bank. Bevy's own lighting does the rest.
// `wave_forge_godot/godot/addons/wave_forge/shaders/water.gdshader` is the same shader for Godot.
import bevy_pbr::render::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT,
};

// What `WaterLook` in materials.rs holds.
struct WaterLook {
    shallow_colour: vec4<f32>,
    deep_colour: vec4<f32>,
    opacity: vec2<f32>,
    clarity: f32,
    shore: f32,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<uniform> look: WaterLook;

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    let depth = in.uv.x;
    let deep = 1.0 - exp(-max(depth, 0.0) / look.clarity);
    pbr_input.material.base_color = vec4<f32>(
        mix(look.shallow_colour.rgb, look.deep_colour.rgb, deep),
        mix(look.opacity.x, look.opacity.y, deep) * smoothstep(0.0, look.shore, depth),
    );
    var out: FragmentOutput;
    if (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u {
        out.color = apply_pbr_lighting(pbr_input);
    } else {
        out.color = pbr_input.material.base_color;
    }
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
