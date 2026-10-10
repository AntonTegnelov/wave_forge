// Wave Forge's reference grass shader for Bevy (docs/reference/bevy.md, "Grass"): every chunk draws
// one shared mesh of blades whose vertices carry nothing but their blade's index. Blade b belongs to
// a column, stands at a hashed place in it, and shows only where a hash is below the column's cover,
// so the cover map alone decides where grass grows. It stands on the ground's heights and sways in
// the wind. Blades grow in clumps, the cells of a cellular noise, each clump of one height, leaning
// one way and of one colour between healthy and dry; a blade's base takes the ground's colour under
// it, its normal is rounded across it, and a blade seen edge-on widens. Farther away fewer blades
// show, a column's first by rank, each wider. The Godot grass shader is the same.
import bevy_pbr::render::{
    mesh_functions,
    mesh_view_bindings::{globals, view},
    forward_io::{Vertex, VertexOutput, FragmentOutput, decompress_vertex},
    view_transformations::position_world_to_clip,
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT,
};

// What `GrassFade` in materials.rs holds: the chunks from `focus`, along the farther axis, over
// which blades shrink away, a chunk's width along x and z, and the centre of the chunk the grass is
// given around, so the fade follows the square of chunks that have grass wherever the camera is.
struct GrassFade {
    band: vec2<f32>,
    span: vec2<f32>,
    focus: vec2<f32>,
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
    dry_colour: vec4<f32>,
    // A clump's width, its blades' lean, the share of clumps that are dry, how far a base takes the
    // ground's colour.
    clump: vec4<f32>,
    // From and to how far blades thin, and the share left at the far end.
    thinning: vec4<f32>,
    fade: GrassFade,
};

@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<uniform> grass: GrassSettings;
// The ground's height at each of its vertices, in world units, one texel each.
@group(constants::MATERIAL_BIND_GROUP) @binding(101) var heights: texture_2d<f32>;
// The cover of each of the chunk's columns, from 0 (bare) to 1 (every blade shown).
@group(constants::MATERIAL_BIND_GROUP) @binding(102) var cover: texture_2d<f32>;
// The ground's material per vertex, id / 255 in red, and its palette.
@group(constants::MATERIAL_BIND_GROUP) @binding(103) var materials: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(104) var palette: texture_2d<f32>;


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

// The clump `point` lies in: a cell of a cellular noise over the ground plane, whose nearest feature
// point names it.
fn clump_of(point: vec2<f32>) -> u32 {
    let cell = floor(point);
    var nearest = 1e9;
    var id = 0u;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let c = cell + vec2<f32>(f32(x), f32(y));
            let key = (u32(i32(c.x)) * 73856093u) ^ (u32(i32(c.y)) * 19349663u);
            let feature = c + vec2<f32>(unit(key), unit(key ^ 0x68E31DA4u));
            let d = dot(point - feature, point - feature);
            if d < nearest {
                nearest = d;
                id = key;
            }
        }
    }
    return id;
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
    // The blade's rank in its column: a blade is the same in a full mesh and a smaller one.
    let rank = blade % grass.chunk_and_count.z;
    let at = vec2<i32>(column % columns.x, column / columns.x);
    let seed = ((u32(column) * 4096u + u32(rank)) * 747796405u) ^ (u32(grass.chunk_and_count.x) * 2891336453u)
        ^ (u32(grass.chunk_and_count.y) * 1181783497u);
    let place = vec2<f32>(at) + vec2<f32>(unit(seed), unit(seed ^ 0x9E3779B9u));
    let covered = textureLoad(cover, at, 0).r;
    let world_from_local = mesh_functions::get_world_from_local(vertex_in.instance_index);
    let root = mesh_functions::mesh_position_local_to_world(
        world_from_local,
        vec4<f32>(place.x * grass.cell_and_blade.x, 0.0, place.y * grass.cell_and_blade.y, 1.0));
    let from_focus = abs(root.xz - grass.fade.focus) / grass.fade.span;
    let faded = 1.0 - smoothstep(grass.fade.band.x, grass.fade.band.y, max(from_focus.x, from_focus.y));
    // Farther away fewer blades show, a column's first by rank, each wider by as much as their number
    // fell.
    let distance = length(root.xyz - view.world_position);
    let keep = mix(1.0, grass.thinning.z, smoothstep(grass.thinning.x, grass.thinning.y, distance));
    let ranked = (f32(rank) + unit(seed ^ 0x9B05688Cu)) / f32(grass.chunk_and_count.w);
    let shown = column < columns.x * columns.y && unit(seed ^ 0x85EBCA6Bu) < covered && faded > 0.0
        && ranked < keep;
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
    // The clump: its height, the way it leans and whether it is dry.
    let clump = clump_of(root.xz / grass.clump.x);
    let clump_tall = 0.65 + 0.7 * unit(clump ^ 0x3C6EF372u);
    let clump_angle = unit(clump ^ 0xA54FF53Au) * 6.2831853;
    let dry = smoothstep(1.0 - grass.clump.z, 1.0 - grass.clump.z + 0.15, unit(clump ^ 0x510E527Fu));
    // A blade faces about its clump's way, turned a little by its own hash.
    let angle = clump_angle + (unit(seed ^ 0x27D4EB2Fu) - 0.5) * 1.6;
    let facing = vec2<f32>(cos(angle), sin(angle));
    let along = in.position.y;
    let side = in.position.x;
    let tall = grass.cell_and_blade.z * clump_tall * (0.85 + 0.3 * unit(seed ^ 0x165667B1u)) * faded;
    // A blade seen edge-on widens, so it still covers what it would face-on.
    let face = vec3<f32>(-facing.y, 0.0, facing.x);
    let to_root = root.xyz - view.world_position;
    let edge_on = 1.0 - abs(dot(to_root.xz, face.xz)) / max(length(to_root.xz), 1e-4);
    let width = grass.cell_and_blade.w * inverseSqrt(keep) * (1.0 + 0.5 * edge_on);
    var blade_offset = vec3<f32>(facing.x * side * width, along * tall, facing.y * side * width);
    blade_offset.x += cos(clump_angle) * grass.clump.y * tall * along * along;
    blade_offset.z += sin(clump_angle) * grass.clump.y * tall * along * along;
    let phase = unit(seed ^ 0xD3A2646Cu) * 6.2831853;
    blade_offset.x += grass.wind.x * grass.wind.z * along * along * (0.6 + 0.4 * sin(globals.time * grass.wind.w + phase));
    blade_offset.z += grass.wind.y * grass.wind.z * along * along * (0.6 + 0.4 * sin(globals.time * grass.wind.w + phase));
    var local = vec3<f32>(0.0);
    if shown {
        local = vec3<f32>(place.x * grass.cell_and_blade.x, ground, place.y * grass.cell_and_blade.y) + blade_offset;
    }
    out.world_position = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(local, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);
    // Mostly the sky's, as a meadow is lit, rounded across the blade, leaning out at its edges.
    let across = vec3<f32>(facing.x, 0.0, facing.y);
    out.world_normal = normalize(vec3<f32>(0.0, 1.0, 0.0) + face * 0.3 + across * side * 0.8);
    // The clump's dryness and the height along the blade, and the blade's place in its chunk, from
    // which the fragment finds the ground's colour under its root.
    out.uv = vec2<f32>(dry, along);
    out.uv_b = place;
    // The fragment finds its material through the instance.
    @if(VERTEX_OUTPUT_INSTANCE_INDEX)
    out.instance_index = vertex_in.instance_index;
    return out;
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    // The ground's colour under the blade's root, from its nearest vertex.
    let vertex_at = clamp(vec2<i32>(round(in.uv_b - 0.5)), vec2<i32>(0), vec2<i32>(textureDimensions(materials)) - 1);
    let id = i32(textureLoad(materials, vertex_at, 0).r * 255.0 + 0.5);
    let palette_last = i32(textureDimensions(palette).x) - 1;
    let under = textureLoad(palette, vec2<i32>(min(id, palette_last), 0), 0).rgb;
    let ground_colour = mix(grass.base_colour.rgb, under, grass.clump.w);
    let tip = mix(grass.tip_colour.rgb, grass.dry_colour.rgb, in.uv.x);
    pbr_input.material.base_color = vec4<f32>(mix(ground_colour, tip, in.uv.y), 1.0);
    var out: FragmentOutput;
    if (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u {
        out.color = apply_pbr_lighting(pbr_input);
    } else {
        out.color = pbr_input.material.base_color;
    }
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
