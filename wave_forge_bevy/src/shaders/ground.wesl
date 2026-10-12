// Wave Forge's reference ground shader for Bevy (docs/reference/bevy.md, "Ground materials"): each
// chunk's ground carries its vertices' materials as an id texture, one texel per vertex, and every
// fragment draws the materials of the four vertices around it. Where they differ, each is weighted
// by how near its vertices are plus a noise of its own, and the highest wins with a short blend, so
// materials meet along a broken line rather than the grid's. A material may be drawn with a texture
// set instead of its colour and relief, seen from above and, on steep ground, from the sides, its
// tiles broken up by a second sample. Each material has a micro-height, its relief's (pebbles,
// cracks or tufts) or its texture's height, and where materials meet the one standing higher wins;
// near the camera a relief bumps the ground and grains its colour. Steep ground turns to rock by
// its own slope, its colour muted and banded in strata, unless a texture set draws it; a broad
// noise varies the colour across the land. Channels the stages hand
// in darken hollows and wet ground, and each material's colour may run along a ramp of its own by
// height, slope, cavity and wetness. Bevy's own lighting does the rest.
// `wave_forge_godot/godot/addons/wave_forge/shaders/ground.gdshader` is the same shader for Godot.
import bevy_pbr::render::{
    mesh_view_bindings::view,
    forward_io::{VertexOutput, FragmentOutput},
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT,
};

// What `GrassFade` in materials.rs holds, as the grass shader reads it.
struct GrassFade {
    band: vec2<f32>,
    span: vec2<f32>,
    focus: vec2<f32>,
}

// What `GroundLook` in materials.rs holds.
struct GroundLook {
    rock_colour: vec4<f32>,
    rock_slope: vec2<f32>,
    strata: f32,
    edge_noise: f32,
    macro_variation: f32,
    detail: f32,
    cavity_darkening: f32,
    wet_darkening: f32,
    wet_roughness: f32,
    cover_tint: f32,
    cover_colour: vec4<f32>,
    gullies: f32,
    relief_blend: f32,
    ramp_heights: vec2<f32>,
    texture_size: f32,
    texture_normal: f32,
    texture_roughness: f32,
    // A struct within a uniform starts on 16 bytes.
    @align(16) cover_fade: GrassFade,
}

// The slopes (one minus the normal's height) at which gullies start and are cut fully, and how many
// gullies a world unit of the ground plane holds.
const GULLY_SLOPE: vec2<f32> = vec2<f32>(0.08, 0.2);
const GULLY_SCALE: f32 = 0.7;
// How far a micro-height of full depth leans the normal, per unit of its slope in feature widths.
const RELIEF_BUMP: f32 = 0.25;
// How wide a crack is, in feature widths, and how far its sides lean the normal against a plate's.
const CRACK_WIDTH: f32 = 0.06;
const CRACK_DEPTH: f32 = 0.15;

// The chunk's corner on the ground plane (x, z) and a cell's width along x and z.
@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<uniform> grid: vec4<f32>;
// The chunk's material id per ground vertex, in the red channel as id / 255.
@group(constants::MATERIAL_BIND_GROUP) @binding(101) var materials: texture_2d<f32>;
// A colour per material id, in texel id of a 256 by 1 texture.
@group(constants::MATERIAL_BIND_GROUP) @binding(102) var palette: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(103) var<uniform> look: GroundLook;
// Per ground vertex, 0 to 1: a cavity in red, which darkens the ground, a wetness in green, which
// darkens and smooths it, and a cover in blue, which tints it toward the grass's colour. A single
// texel of zeros for a ground without channels.
@group(constants::MATERIAL_BIND_GROUP) @binding(104) var channels: texture_2d<f32>;
// Per material id, in texel id of a 256 by 2 texture: the colour, linear, its ramp runs toward in
// row 0, and in row 1 how far the ground's height (0 to 1 over `look.ramp_heights`), slope, cavity
// and wetness each move its colour along that ramp, from -1 to 1. A single texel of zeros, no ramp,
// for a ground without ramps.
@group(constants::MATERIAL_BIND_GROUP) @binding(105) var ramps: texture_2d<f32>;
// Per material id, in texel id of a 256 by 1 texture: how much of it is pebbles, cracks and tufts,
// and how wide one of its features is in world units (`wave_forge::GroundRelief`).
@group(constants::MATERIAL_BIND_GROUP) @binding(106) var relief: texture_2d<f32>;
// Where the first vertex stands from the grid's corner, in cells (x, y), and how many fine cells
// wide one cell is (z): (0.5, 0.5, 1) for a near ground, less and more for the far ground, whose
// cells are coarse columns with their first vertex at a fine cell's centre, so the noise breaking
// its borders is as coarse as its cells.
@group(constants::MATERIAL_BIND_GROUP) @binding(107) var<uniform> vertex_layout: vec4<f32>;
// Per material id, in texel id of a 256 by 1 texture, the layer plus one of the texture set it is
// drawn with, as id / 255 in red; 0, as a single texel of zeros gives every material, draws it in
// its colour.
@group(constants::MATERIAL_BIND_GROUP) @binding(108) var texture_layers: texture_2d<f32>;
// Texture sets a material can be drawn with instead of its colour: layers of two arrays, the colour
// with the height in alpha and the OpenGL normal with the roughness in alpha. A textured material's
// palette colour is its set's mean, and its ramp tints the set by how far it moves that colour.
@group(constants::MATERIAL_BIND_GROUP) @binding(109) var albedo_height: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(110) var albedo_height_sampler: sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(111) var normal_roughness: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(112) var normal_roughness_sampler: sampler;

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

// Value noise and its slope, (value, d/dx, d/dy).
fn value_with_slope(p: vec2<f32>) -> vec3<f32> {
    let i = floor(p);
    let f = p - i;
    let u = f * f * (3.0 - 2.0 * f);
    let du = 6.0 * f * (1.0 - f);
    let a = hash(i);
    let b = hash(i + vec2<f32>(1.0, 0.0));
    let c = hash(i + vec2<f32>(0.0, 1.0));
    let d = hash(i + vec2<f32>(1.0, 1.0));
    let k = a - b - c + d;
    return vec3<f32>(a + (b - a) * u.x + (c - a) * u.y + k * u.x * u.y, du * (vec2<f32>(b - a, c - a) + k * u.yx));
}

// `fbm` and its slope, (value, d/dx, d/dy): each turned octave's slope turned back.
fn fbm_with_slope(p: vec2<f32>) -> vec3<f32> {
    let turn = mat2x2<f32>(vec2<f32>(1.6, 1.2), vec2<f32>(-1.2, 1.6));
    let back = transpose(turn);
    let q = turn * p;
    let first = value_with_slope(p);
    let second = value_with_slope(q);
    let third = value_with_slope(turn * q);
    let v = first.x * 4.0 + second.x * 2.0 + third.x;
    let slope = first.yz * 4.0 + back * (second.yz * 2.0 + back * third.yz);
    return vec3<f32>(v, slope) / 7.0;
}

// Round stones in about half the cells of a grid over `p`, in feature widths, each at a hashed
// place and of a hashed size up to a third of a feature: (height 0 to 1, its slope along x and z).
fn pebbles(p: vec2<f32>) -> vec3<f32> {
    let cell = floor(p);
    var stone = vec3<f32>(0.0);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let c = cell + vec2<f32>(f32(x), f32(y));
            let size = hash(c + 7.7);
            if size < 0.45 {
                continue;
            }
            let radius = 0.15 + 0.35 * (size - 0.45);
            let d = (p - (c + 0.25 + 0.5 * vec2<f32>(hash(c), hash(c + 19.19)))) / radius;
            let h = 1.0 - dot(d, d);
            if h > stone.x {
                stone = vec3<f32>(h, -2.0 * d / radius);
            }
        }
    }
    return stone;
}

// Plates parted by narrow cracks: the cells of a cellular noise over `p`, in feature widths, a
// plate standing at 1 and a crack falling to 0 where its two nearest features are about as near,
// shallow, so it reads as a line: (height 0 to 1, its slope along x and z).
fn cracks(p: vec2<f32>) -> vec3<f32> {
    let cell = floor(p);
    var first = 1e9;
    var second = 1e9;
    var to_first = vec2<f32>(0.0);
    var to_second = vec2<f32>(0.0);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let c = cell + vec2<f32>(f32(x), f32(y));
            let d = p - (c + vec2<f32>(hash(c + 3.7), hash(c + 41.3)));
            let squared = dot(d, d);
            if squared < first {
                second = first;
                to_second = to_first;
                first = squared;
                to_first = d;
            } else if squared < second {
                second = squared;
                to_second = d;
            }
        }
    }
    let near = sqrt(first);
    let far = sqrt(second);
    let x = clamp((far - near) / CRACK_WIDTH, 0.0, 1.0);
    let across = to_second / max(far, 1e-4) - to_first / max(near, 1e-4);
    return vec3<f32>(x * x * (3.0 - 2.0 * x), CRACK_DEPTH * 6.0 * x * (1.0 - x) / CRACK_WIDTH * across);
}

// Material `id`'s micro-height at `ground`, world units along x and z: (height 0 to 1, its slope
// along x and z in feature widths, its depth 0 to 1).
fn relief_of(id: i32, ground: vec2<f32>) -> vec4<f32> {
    let last = i32(textureDimensions(relief).x) - 1;
    let shape = textureLoad(relief, vec2<i32>(min(id, last), 0), 0);
    let total = shape.x + shape.y + shape.z;
    if total <= 0.0 || shape.w <= 0.0 {
        return vec4<f32>(0.5, 0.0, 0.0, 0.0);
    }
    let p = ground / shape.w;
    var h = vec3<f32>(0.0);
    if shape.x > 0.0 {
        h += shape.x * pebbles(p);
    }
    if shape.y > 0.0 {
        h += shape.y * cracks(p + 31.7);
    }
    if shape.z > 0.0 {
        h += shape.z * fbm_with_slope(p + 57.3);
    }
    return vec4<f32>(h / total, min(total, 1.0));
}

fn id_at(vertex: vec2<i32>) -> i32 {
    return i32(textureLoad(materials, vertex, 0).r * 255.0 + 0.5);
}

// Material `id`'s colour, moved along its ramp by `drivers`: the ground's height, slope, cavity and
// wetness here.
fn colour_of(id: i32, drivers: vec4<f32>) -> vec3<f32> {
    let colour = textureLoad(palette, vec2<i32>(id, 0), 0).rgb;
    let last = vec2<i32>(textureDimensions(ramps)) - 1;
    let toward = textureLoad(ramps, min(vec2<i32>(id, 0), last), 0).rgb;
    let by = textureLoad(ramps, min(vec2<i32>(id, 1), last), 0);
    return mix(colour, toward, clamp(dot(by, drivers), 0.0, 1.0));
}

// What a material looks like at a fragment: its colour, its micro-height (its height 0 to 1 and its
// slope along x and z, scaled by its depth), how deep that height is, 0 to 1, its perceptual
// roughness, and 1 for a texture set or 0 for its colour and relief.
struct Look {
    albedo: vec3<f32>,
    micro: vec3<f32>,
    depth: f32,
    roughness: f32,
    textured: f32,
}

// A texture set's colour and height, and its normal and roughness, at one place.
struct SetSample {
    colour: vec4<f32>,
    normal: vec4<f32>,
}

// A texture set's layer `layer` at `ground`, whose screen derivatives are `dx` and `dy`: two
// samples, the second turned and scaled, blended by a broad noise, so the tiles do not repeat. The
// derivatives are explicit since the layer, read per material, leaves uniform control flow.
fn sample_set(layer: i32, ground: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>) -> SetSample {
    let uv = ground / look.texture_size;
    let turn = mat2x2<f32>(vec2<f32>(0.8, 0.6), vec2<f32>(-0.6, 0.8));
    let other = turn * uv * 0.73 + 0.37;
    let uv_dx = dx / look.texture_size;
    let uv_dy = dy / look.texture_size;
    let other_dx = turn * uv_dx * 0.73;
    let other_dy = turn * uv_dy * 0.73;
    let blend = smoothstep(0.3, 0.7, value(ground / (look.texture_size * 2.3)));
    var sample: SetSample;
    sample.colour = mix(
        textureSampleGrad(albedo_height, albedo_height_sampler, uv, layer, uv_dx, uv_dy),
        textureSampleGrad(albedo_height, albedo_height_sampler, other, layer, other_dx, other_dy),
        blend,
    );
    sample.normal = mix(
        textureSampleGrad(normal_roughness, normal_roughness_sampler, uv, layer, uv_dx, uv_dy),
        textureSampleGrad(normal_roughness, normal_roughness_sampler, other, layer, other_dx, other_dy),
        blend,
    );
    return sample;
}

// Material `id`'s look at `at`, whose screen derivatives are `dx` and `dy`, on a surface of normal
// `surface` and a base roughness of `roughness`: its texture set's, seen from above and, on steep
// ground, from the sides as well, or its colour and, with `relief`, its relief's.
fn look_of(id: i32, at: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>, surface: vec3<f32>, drivers: vec4<f32>, roughness: f32, relief: bool) -> Look {
    var out: Look;
    let last = i32(textureDimensions(texture_layers).x) - 1;
    let layer = i32(textureLoad(texture_layers, vec2<i32>(min(id, last), 0), 0).r * 255.0 + 0.5) - 1;
    if layer < 0 {
        var r = vec4<f32>(0.5, 0.0, 0.0, 0.0);
        if relief {
            r = relief_of(id, at.xz);
        }
        out.albedo = colour_of(id, drivers);
        out.micro = vec3<f32>(r.x, r.yz * r.w);
        out.depth = r.w;
        out.roughness = roughness;
        out.textured = 0.0;
        return out;
    }
    var weights = pow(abs(surface), vec3<f32>(4.0));
    weights /= weights.x + weights.y + weights.z;
    let top = sample_set(layer, at.xz, dx.xz, dy.xz);
    var colour = top.colour;
    var normal = top.normal;
    // The normal's lean along the ground's x and z, from above.
    var lean = (normal.rg * 2.0 - 1.0) * vec2<f32>(1.0, -1.0);
    if weights.y < 0.98 {
        let side_x = sample_set(layer, vec2<f32>(at.z, -at.y), vec2<f32>(dx.z, -dx.y), vec2<f32>(dy.z, -dy.y));
        let side_z = sample_set(layer, vec2<f32>(at.x, -at.y), vec2<f32>(dx.x, -dx.y), vec2<f32>(dy.x, -dy.y));
        colour = colour * weights.y + side_x.colour * weights.x + side_z.colour * weights.z;
        normal = normal * weights.y + side_x.normal * weights.x + side_z.normal * weights.z;
        lean *= weights.y;
    }
    // Its ramp moves the set's colour as it would the set's mean, its palette colour.
    let mean = textureLoad(palette, vec2<i32>(id, 0), 0).rgb;
    out.albedo = colour.rgb * colour_of(id, drivers) / max(mean, vec3<f32>(1e-3));
    out.micro = vec3<f32>(colour.a, -lean * look.texture_normal);
    out.depth = 1.0;
    out.roughness = mix(roughness, normal.a, look.texture_roughness);
    out.textured = 1.0;
    return out;
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    let size = vec2<i32>(textureDimensions(materials));
    // The ground's vertices stand at the columns' centres.
    let column = (in.world_position.xz - grid.xy) / grid.zw - vertex_layout.xy;
    let at = clamp(column, vec2<f32>(0.0), vec2<f32>(size - 1));
    let low = vec2<i32>(floor(at));
    let high = min(low + 1, size - 1);
    let t = at - vec2<f32>(low);
    let corners = array<vec2<i32>, 4>(low, vec2<i32>(high.x, low.y), vec2<i32>(low.x, high.y), high);
    let near_weights = array<f32, 4>((1.0 - t.x) * (1.0 - t.y), t.x * (1.0 - t.y), (1.0 - t.x) * t.y, t.x * t.y);
    let world = in.world_position.xyz;
    let ground = world.xz;
    // The texture sets' derivatives, taken while control flow is uniform.
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
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
    // The channels, blended between the vertices as the materials are.
    let channel_size = vec2<i32>(textureDimensions(channels));
    var near_channels = vec3<f32>(0.0);
    for (var k = 0; k < 4; k++) {
        near_channels += textureLoad(channels, min(corners[k], channel_size - 1), 0).rgb * near_weights[k];
    }
    let cavity = clamp(near_channels.x, 0.0, 1.0);
    let wet = clamp(near_channels.y, 0.0, 1.0);
    let cover = clamp(near_channels.z, 0.0, 1.0);
    // What moves each material's colour along its ramp.
    let height = clamp((world.y - look.ramp_heights.x) / max(look.ramp_heights.y - look.ramp_heights.x, 1e-6), 0.0, 1.0);
    let drivers = vec4<f32>(height, steepness, cavity, wet);
    var ids: array<i32, 4>;
    for (var k = 0; k < 4; k++) {
        ids[k] = id_at(corners[k]);
    }
    let near = 1.0 - smoothstep(30.0, 120.0, length(world - view.world_position));
    let grain = look.detail * near;
    let base_roughness = pbr_input.material.perceptual_roughness;
    // What the ground looks like here: one material's look, or where materials meet, theirs blended
    // by how high each stands.
    var here = look_of(ids[0], world, world_dx, world_dy, surface, drivers, base_roughness, grain > 0.0);
    // Most fragments have one material on all four corners and need no border.
    if ids[1] != ids[0] || ids[2] != ids[0] || ids[3] != ids[0] {
        var heights: array<f32, 4>;
        var shares: array<f32, 4>;
        var looks: array<Look, 4>;
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
            // A material on several corners has its look worked out once.
            var first = k;
            for (var j = k - 1; j >= 0; j--) {
                if ids[j] == ids[k] {
                    first = j;
                }
            }
            if first == k {
                looks[k] = look_of(ids[k], world, world_dx, world_dy, surface, drivers, base_roughness, true);
            } else {
                looks[k] = looks[first];
            }
            // The material standing higher, by its micro-height or its texture's height, wins: sand
            // settles between pebbles, rock plates stand out of their cracks.
            heights[k] = weight + border_noise * (fbm(ground / (2.0 * vertex_layout.z) + f32(ids[k]) * 17.31) - 0.5) * 2.0
                + (1.0 - rock) * look.relief_blend * looks[k].depth * (looks[k].micro.x - 0.5);
            shares[k] = 1.0 / same;
            highest = max(highest, heights[k]);
        }
        var blended = Look(vec3<f32>(0.0), vec3<f32>(0.0), 0.0, 0.0, 0.0);
        var total = 0.0;
        for (var k = 0; k < 4; k++) {
            let weight = max(heights[k] - highest + border_width, 0.0) * shares[k];
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
    var albedo = here.albedo;
    let micro = here.micro;
    // The micro-height bumps the surface by its slope, fading with distance; a texture's normal
    // does at every distance, its mipmaps fading it.
    pbr_input.N = normalize(pbr_input.N + vec3<f32>(-micro.y, 0.0, -micro.z) * mix(RELIEF_BUMP * grain, 1.0, here.textured));
    // The rock is the ground's colour muted, banded in warped strata; a texture set draws its own.
    let rock_here = rock * (1.0 - here.textured);
    if rock_here > 0.0 {
        let band = 0.5 + 0.5 * sin(world.y * 1.8 + fbm(ground / 12.0) * 2.0);
        let muted = mix(albedo, vec3<f32>(dot(albedo, vec3<f32>(0.299, 0.587, 0.114))), 0.3) * 0.8;
        let rock_base = mix(muted, look.rock_colour.rgb, look.rock_colour.a);
        albedo = mix(albedo, rock_base * mix(1.0, 0.8 + 0.4 * band, look.strata), rock_here);
    }
    // Gullies on steep ground: a ridged noise over the ground plane, which on a slope steep enough
    // changes little down the fall line and much along the slope, so its ridges run down it as
    // grooves; each groove is darker and the normal leans into it by the noise's own slope. They
    // come in from the slope of a badland's or a wall's flank, gentler than the rock's.
    let cut = look.gullies * smoothstep(GULLY_SLOPE.x, GULLY_SLOPE.y, steepness);
    if cut > 0.0 {
        let gully = value_with_slope(ground * GULLY_SCALE);
        let side = 2.0 * gully.x - 1.0;
        let groove = 1.0 - abs(side);
        let lean = -sign(side) * 2.0 * gully.yz * GULLY_SCALE;
        albedo *= 1.0 - 0.3 * cut * smoothstep(0.6, 1.0, groove);
        pbr_input.N = normalize(pbr_input.N - vec3<f32>(lean.x, 0.0, lean.y) * 0.3 * cut);
    }
    albedo *= mix(1.0, 0.8 + 0.4 * fbm(ground / 48.0), look.macro_variation);
    // Covered ground tints in as the grass's blades shrink away, so where they stand it is as it is.
    let from_focus = abs(world.xz - look.cover_fade.focus) / look.cover_fade.span;
    let beyond_grass = smoothstep(look.cover_fade.band.x, look.cover_fade.band.y, max(from_focus.x, from_focus.y));
    albedo = mix(albedo, look.cover_colour.rgb, cover * look.cover_tint * beyond_grass);
    albedo *= (1.0 - look.cavity_darkening * cavity) * (1.0 - look.wet_darkening * wet);
    pbr_input.material.perceptual_roughness = mix(here.roughness, look.wet_roughness, wet);
    albedo *= mix(1.0, 0.86 + 0.28 * micro.x, grain * (1.0 - here.textured));

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
