// WGSL Shader for Wave Function Collapse constraint propagation (Test version)
//
// This is a simplified version for test purposes only.

// Struct defining shader parameters
struct Params {
    grid_width: u32,
    grid_height: u32,
    grid_depth: u32,
    num_tiles: u32,
    num_axes: u32,
    worklist_size: u32,
    boundary_mode: u32, // 0: Clamped, 1: Periodic
    _padding1: u32, // padding to align to 16 bytes
};

// Axis enums for easier readability
const AXIS_POS_X: u32 = 0u;
const AXIS_NEG_X: u32 = 1u;
const AXIS_POS_Y: u32 = 2u;
const AXIS_NEG_Y: u32 = 3u;
const AXIS_POS_Z: u32 = 4u;
const AXIS_NEG_Z: u32 = 5u;

// Numeric constants - use named constants instead of numeric identifiers
const ONE: u32 = 1u;
const TWO: u32 = 2u;

// Uniform buffer with parameters
@group(0) @binding(0) var<uniform> params: Params;

// Storage buffers - simplified for test purposes
@group(0) @binding(1) var<storage, read_write> grid_possibilities: array<u32>;

@compute @workgroup_size(64)
fn main_propagate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Empty test function
    let x = global_id.x;
    if (x >= params.grid_width) {
        return;
    }
} 