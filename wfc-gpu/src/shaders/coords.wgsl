// coords.wgsl - Coordinate and neighbor calculation for Wave Function Collapse
//
// This module contains functions for working with grid coordinates, calculating
// neighbor positions, and handling different boundary conditions.

// Calculate the coordinates of a neighbor cell in a given direction
fn get_neighbor_coords(x: u32, y: u32, z: u32, axis_idx: u32, params: Params) -> vec3<i32> {
    var nx: i32 = i32(x);
    var ny: i32 = i32(y);
    var nz: i32 = i32(z);
    
    // Convert axis index to offset
    switch (axis_idx) {
        case AXIS_POS_X: {
            nx = i32(x) + 1;
        }
        case AXIS_NEG_X: {
            nx = i32(x) - 1;
        }
        case AXIS_POS_Y: {
            ny = i32(y) + 1;
        }
        case AXIS_NEG_Y: {
            ny = i32(y) - 1;
        }
        case AXIS_POS_Z: {
            nz = i32(z) + 1;
        }
        case AXIS_NEG_Z: {
            nz = i32(z) - 1;
        }
        default: {
            // Invalid axis, return original coords
        }
    }
    
    return vec3<i32>(nx, ny, nz);
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
        // Check if the coordinates are in bounds
        if (is_in_bounds(coords, params)) {
            result.x = u32(coords.x);
            result.y = u32(coords.y);
            result.z = u32(coords.z);
            result.w = 1u; // Valid
        } else {
            result.w = 0u; // Invalid - out of bounds
        }
    } else { // Periodic mode
        // Wrap coordinates
        result.x = wrap_coord(coords.x, params.grid_width);
        result.y = wrap_coord(coords.y, params.grid_height);
        result.z = wrap_coord(coords.z, params.grid_depth);
        result.w = 1u; // Always valid in periodic mode
    }
    
    return result;
}

// Get the index and validity of a neighbor in a given direction
fn get_neighbor_index(x: u32, y: u32, z: u32, axis_idx: u32, params: Params) -> vec2<u32> {
    // Calculate raw neighbor coordinates
    let neighbor_coords = get_neighbor_coords(x, y, z, axis_idx, params);
    
    // Apply boundary conditions
    let adjusted_coords = apply_boundary_conditions(neighbor_coords, params);
    
    // If valid, calculate the 1D index
    if (adjusted_coords.w == 1u) {
        let neighbor_idx = grid_index(adjusted_coords.x, adjusted_coords.y, adjusted_coords.z, 
                                     params.grid_width, params.grid_height);
        return vec2<u32>(neighbor_idx, 1u); // Return (index, valid)
    } else {
        return vec2<u32>(0u, 0u); // Invalid neighbor
    }
} 