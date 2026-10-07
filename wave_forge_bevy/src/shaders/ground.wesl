// Wave Forge's reference ground shader for Bevy (docs/reference/bevy.md, "Ground materials"): each
// chunk's ground carries its vertices' materials as an id texture, one texel per vertex, and every
// fragment draws the materials of the four vertices around it. Where they differ, each is weighted
// by how near its vertices are plus a noise of its own, and the highest wins with a short blend, so
// materials meet along a broken line rather than the grid's. Steep ground turns to rock by its own
// slope, its colour muted and banded in strata; a broad noise varies the colour across the land,
// and a fine one bumps the ground and grains its colour near the camera. Bevy's own lighting does
// the rest. `wave_forge_godot/godot/addons/wave_forge/shaders/ground.gdshader` is the same
// shader for Godot.
import bevy_pbr::render::{
    mesh_view_bindings::view,
    forward_io::{VertexOutput, FragmentOutput},
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT,
};

// What `GroundLook` in materials.rs holds.
struct GroundLook {
    rock_colour: vec4<f32>,
    rock_slope: vec2<f32>,
    strata: f32,
    edge_noise: f32,
    macro_variation: f32,
    detail: f32,
}

// The chunk's corner on the ground plane (x, z) and a cell's width along x and z.
@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<uniform> grid: vec4<f32>;
// The chunk's material id per ground vertex, in the red channel as id / 255.
@group(constants::MATERIAL_BIND_GROUP) @binding(101) var materials: texture_2d<f32>;
// A colour per material id, in texel id of a 256 by 1 texture.
@group(constants::MATERIAL_BIND_GROUP) @binding(102) var palette: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(103) var<uniform> look: GroundLook;

// A hash without trigonometry, cheaper than sin and the same on every GPU.
fn hash(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Value noise in 0..1 with a smooth step between the lattice's points.
fn value(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = p - i;
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash(i);
    let b = hash(i + vec2<f32>(1.0, 0.0));
    let c = hash(i + vec2<f32>(0.0, 1.0));
    let d = hash(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Three octaves of value noise, in 0..1, each turned against the last so no feature lines up with
// the axes.
fn fbm(p: vec2<f32>) -> f32 {
    let turn = mat2x2<f32>(vec2<f32>(1.6, 1.2), vec2<f32>(-1.2, 1.6));
    let q = turn * p;
    return (value(p) * 4.0 + value(q) * 2.0 + value(turn * q)) / 7.0;
}

fn id_at(vertex: vec2<i32>) -> i32 {
    return i32(textureLoad(materials, vertex, 0).r * 255.0 + 0.5);
}

fn colour_of(id: i32) -> vec3<f32> {
    return textureLoad(palette, vec2<i32>(id, 0), 0).rgb;
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
    let corners = array<vec2<i32>, 4>(low, vec2<i32>(high.x, low.y), vec2<i32>(low.x, high.y), high);
    let near_weights = array<f32, 4>((1.0 - t.x) * (1.0 - t.y), t.x * (1.0 - t.y), (1.0 - t.x) * t.y, t.x * t.y);
    let world = in.world_position.xyz;
    let ground = world.xz;
    // Rock by the ground's own slope, its edge broken by noise.
    let surface = normalize(in.world_normal);
    let steepness = 1.0 - surface.y + look.edge_noise * 0.15 * (value(ground / 2.0) - 0.5);
    let rock = smoothstep(look.rock_slope.x, look.rock_slope.y, steepness);
    // The materials come per column of x and z, so a sharp border would run down a wall as a
    // streak: on rock the border is soft and unbroken.
    let border_noise = look.edge_noise * (1.0 - rock);
    let border_width = mix(0.1, 1.0, rock);
    // Each material's weight is the sum of its corners', so its edge follows the smooth line between
    // the vertices; raised by a noise of the material's own, the highest wins, blending over a band
    // of the border's width. A material on several corners is counted once.
    var ids: array<i32, 4>;
    for (var k = 0; k < 4; k++) {
        ids[k] = id_at(corners[k]);
    }
    var albedo = colour_of(ids[0]);
    // Most fragments have one material on all four corners and need no border.
    if ids[1] != ids[0] || ids[2] != ids[0] || ids[3] != ids[0] {
        var heights: array<f32, 4>;
        var shares: array<f32, 4>;
        var highest = -2.0;
        for (var k = 0; k < 4; k++) {
            var weight = 0.0;
            var same = 0.0;
            for (var j = 0; j < 4; j++) {
                if ids[j] == ids[k] {
                    weight += near_weights[j];
                    same += 1.0;
                }
            }
            heights[k] = weight + border_noise * (fbm(ground / 2.0 + f32(ids[k]) * 17.31) - 0.5) * 2.0;
            shares[k] = 1.0 / same;
            highest = max(highest, heights[k]);
        }
        albedo = vec3<f32>(0.0);
        var total = 0.0;
        for (var k = 0; k < 4; k++) {
            let weight = max(heights[k] - highest + border_width, 0.0) * shares[k];
            albedo += colour_of(ids[k]) * weight;
            total += weight;
        }
        albedo /= total;
    }

    let near = 1.0 - smoothstep(30.0, 120.0, length(world - view.world_position));
    let grain = look.detail * near;
    // The fine noise bumps the surface by its slope on screen (Mikkelsen's surface gradient), fading
    // with distance. Screen derivatives need every fragment of a quad, so this runs unbranched.
    let fine = fbm(ground * 1.7);
    let dpdx_world = dpdx(world);
    let dpdy_world = dpdy(world);
    let r1 = cross(dpdy_world, pbr_input.N);
    let r2 = cross(pbr_input.N, dpdx_world);
    let det = dot(dpdx_world, r1);
    let bump = (dpdx(fine) * r1 + dpdy(fine) * r2) * 0.25 * grain;
    pbr_input.N = normalize(abs(det) * pbr_input.N - sign(det) * bump);
    // The rock is the ground's colour muted, banded in warped strata.
    if rock > 0.0 {
        let band = 0.5 + 0.5 * sin(world.y * 1.8 + fbm(ground / 12.0) * 2.0);
        let muted = mix(albedo, vec3<f32>(dot(albedo, vec3<f32>(0.299, 0.587, 0.114))), 0.3) * 0.8;
        let rock_base = mix(muted, look.rock_colour.rgb, look.rock_colour.a);
        albedo = mix(albedo, rock_base * mix(1.0, 0.8 + 0.4 * band, look.strata), rock);
    }
    albedo *= mix(1.0, 0.8 + 0.4 * fbm(ground / 48.0), look.macro_variation);
    albedo *= mix(1.0, 0.86 + 0.28 * fine, grain);

    pbr_input.material.base_color = vec4<f32>(albedo, 1.0);
    var out: FragmentOutput;
    if (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u {
        out.color = apply_pbr_lighting(pbr_input);
    } else {
        out.color = pbr_input.material.base_color;
    }
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
