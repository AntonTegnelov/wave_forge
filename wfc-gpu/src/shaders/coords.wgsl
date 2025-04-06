// coords.wgsl - Coordinate and neighbor calculation for Wave Function Collapse
//
// This module contains functions for working with grid coordinates, calculating
// neighbor positions, and handling different boundary conditions.

// --- Constants (Moved from utils.wgsl) ---
const AXIS_POS_X: u32 = 0u;
const AXIS_NEG_X: u32 = 1u;
const AXIS_POS_Y: u32 = 2u;
const AXIS_NEG_Y: u32 = 3u;
const AXIS_POS_Z: u32 = 4u;
const AXIS_NEG_Z: u32 = 5u;

// --- Structures (Params struct might live elsewhere eventually, e.g., shared defs) ---
// Common struct defining shader parameters (Moved from utils.wgsl, may need refinement)
// Consider moving to a shared definitions file if used by many components independently.
struct Params {
    // Grid dimensions
    grid_width: u32,
    grid_height: u32,
    grid_depth: u32,

    // Algorithm settings
    num_tiles: u32,
    num_axes: u32,
    boundary_mode: u32, // 0 = clamped, 1 = periodic
    heuristic_type: u32, // Entropy calculation specific?
    tie_breaking: u32, // Entropy calculation specific?

    // Propagation parameters
    max_propagation_steps: u32, // Propagation specific?
    contradiction_check_frequency: u32, // Propagation specific?

    // Worklist parameters
    worklist_size: u32, // Worklist specific?

    // For SoA structure
    grid_element_count: u32,

    // Padding to ensure 16 byte alignment
    _padding: u32,
};


// --- Coordinate Conversion --- 

// Helper function to get 1D flat index from 3D coords (Moved from utils.wgsl)
fn grid_index(x: u32, y: u32, z: u32, grid_width: u32, grid_height: u32) -> u32 {
    return z * grid_width * grid_height + y * grid_width + x;
}

// Helper function to get 3D coords from 1D flat index
fn get_coords_from_index(index: u32, grid_width: u32, grid_height: u32) -> vec3<u32> {
    let z = index / (grid_width * grid_height);
    let temp = index % (grid_width * grid_height);
    let y = temp / grid_width;
    let x = temp % grid_width;
    return vec3<u32>(x, y, z);
}

// --- Boundary Handling --- 

// Helper to calculate wrapped coordinate for Periodic boundary mode (Moved from utils.wgsl)
fn wrap_coord(coord: i32, max_dim: u32) -> u32 {
    if (max_dim == 0u) { return 0u; } // Avoid modulo by zero
    // Efficient modulo for potentially negative numbers
    let m = coord % i32(max_dim);
    if (m < 0) {
        return u32(m + i32(max_dim));
    } else {
        return u32(m);
    }
}

// Check if the given coordinates are within the grid bounds
fn is_in_bounds(coords: vec3<i32>, params: Params) -> bool {
    return coords.x >= 0 && coords.x < i32(params.grid_width) &&
           coords.y >= 0 && coords.y < i32(params.grid_height) &&
           coords.z >= 0 && coords.z < i32(params.grid_depth);
}

// Apply boundary conditions to the coordinates
// Returns the adjusted coordinates and a bool indicating if they're valid
fn apply_boundary_conditions(coords: vec3<i32>, params: Params) -> vec4<u32> {
    var result = vec4<u32>(0u, 0u, 0u, 0u); // x, y, z, is_valid

    if (params.boundary_mode == 0u) { // Clamped mode
        if (is_in_bounds(coords, params)) {
            result.x = u32(coords.x);
            result.y = u32(coords.y);
            result.z = u32(coords.z);
            result.w = 1u; // Valid
        } else {
            result.w = 0u; // Invalid - out of bounds
        }
    } else { // Periodic mode
        result.x = wrap_coord(coords.x, params.grid_width);
        result.y = wrap_coord(coords.y, params.grid_height);
        result.z = wrap_coord(coords.z, params.grid_depth);
        result.w = 1u; // Always valid in periodic mode
    }

    return result;
}

// --- Neighbor Calculation --- 

// Calculate the raw coordinates of a neighbor cell in a given direction
fn get_neighbor_coords(x: u32, y: u32, z: u32, axis_idx: u32, params: Params) -> vec3<i32> {
    var nx: i32 = i32(x);
    var ny: i32 = i32(y);
    var nz: i32 = i32(z);

    // Convert axis index to offset
    switch (axis_idx) {
        case AXIS_POS_X: { nx = nx + 1; }
        case AXIS_NEG_X: { nx = nx - 1; }
        case AXIS_POS_Y: { ny = ny + 1; }
        case AXIS_NEG_Y: { ny = ny - 1; }
        case AXIS_POS_Z: { nz = nz + 1; }
        case AXIS_NEG_Z: { nz = nz - 1; }
        default: { /* Invalid axis, return original coords */ }
    }

    return vec3<i32>(nx, ny, nz);
}

// Get the 1D index and validity of a neighbor in a given direction, handling boundaries.
fn get_neighbor_index(x: u32, y: u32, z: u32, axis_idx: u32, params: Params) -> vec2<u32> {
    let neighbor_coords_i32 = get_neighbor_coords(x, y, z, axis_idx, params);
    let adjusted_coords = apply_boundary_conditions(neighbor_coords_i32, params);

    if (adjusted_coords.w == 1u) {
        let neighbor_idx = grid_index(adjusted_coords.x, adjusted_coords.y, adjusted_coords.z,
                                     params.grid_width, params.grid_height);
        return vec2<u32>(neighbor_idx, 1u); // Return (index, valid)
    } else {
        return vec2<u32>(0xffffffffu, 0u); // Return (invalid_index, invalid_flag)
    }
} 