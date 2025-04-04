// WGSL Shader for Wave Function Collapse constraint propagation (Fallback version)
//
// This is a simplified version of the propagation shader that doesn't use atomics.
// Instead of atomically updating possibility masks and building a worklist, this version:
// 1. Only applies a single pass of propagation
// 2. Directly updates possibility masks without atomics (less safe for parallel updates)
// 3. Does not generate a new worklist for the next propagation step
//
// This version is compatible with hardware that doesn't support atomics at the cost 
// of reduced functionality and possibly requiring multiple shader dispatches from the host.

// Example bindings (adjust based on actual buffer structures)
@group(0) @binding(0) var<storage, read_write> grid_possibilities: array<u32>; // Non-atomic version
@group(0) @binding(1) var<storage, read> adjacency_rules: array<u32>; // Flattened rules
@group(0) @binding(2) var<storage, read> worklist: array<u32>; // Coordinates or indices of updated cells
@group(0) @binding(4) var<uniform> params: Params;
// No atomic buffers for worklist or contradiction flags in this fallback version

// Specialization constant for number of u32s per cell
override NUM_TILES_U32: u32 = 1u; // Default value, MUST be overridden by pipeline

// Specialization constant for workgroup size (X dimension)
const WORKGROUP_SIZE_X: u32 = 64u; // Hardcoded size

// Uniforms for grid dimensions, num_tiles etc.
struct Params {
    grid_width: u32,
    grid_height: u32,
    grid_depth: u32,
    num_tiles: u32,
    num_axes: u32,
    worklist_size: u32,
    boundary_mode: u32, // 0: Clamped, 1: Periodic
    _padding1: u32,
};

// Constants for axes (match CPU version)
const AXIS_POS_X: u32 = 0u;
const AXIS_NEG_X: u32 = 1u;
const AXIS_POS_Y: u32 = 2u;
const AXIS_NEG_Y: u32 = 3u;
const AXIS_POS_Z: u32 = 4u;
const AXIS_NEG_Z: u32 = 5u;

// Helper function to get 1D index from 3D coords
fn grid_index(x: u32, y: u32, z: u32) -> u32 {
    // Assumes packed u32s for possibilities are handled by multiplying by num_tiles_u32 later
    return z * params.grid_width * params.grid_height + y * params.grid_width + x;
}

// Helper to calculate wrapped coordinate for Periodic boundary mode
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

// Helper function to check if a specific bit is set in a u32 array mask
fn is_tile_possible(tile_index: u32, mask: ptr<function, array<u32, 4>>) -> bool {
    let u32_index = tile_index / 32u;
    let bit_index = tile_index % 32u;
    // Critical safety check - use specialization constant
    if (u32_index >= NUM_TILES_U32 || u32_index >= 4u) { return false; } // Keep fixed size check for now
    return ((*mask)[u32_index] & (1u << bit_index)) != 0u;
}

// Helper function to set a specific bit in a u32 array mask
fn set_tile_possible(tile_index: u32, mask: ptr<function, array<u32, 4>>) {
    let u32_index = tile_index / 32u;
    let bit_index = tile_index % 32u;
    // Critical safety check - use specialization constant
    if (u32_index >= NUM_TILES_U32 || u32_index >= 4u) { return; } // Keep fixed size check for now
    (*mask)[u32_index] = (*mask)[u32_index] | (1u << bit_index);
}

// Helper function to check adjacency rule
// Assumes rules are packed tightly: rule[axis][tile1][tile2]
fn check_rule(tile1: u32, tile2: u32, axis: u32) -> bool {
    // Bounds check for tile indices
    if (tile1 >= params.num_tiles || tile2 >= params.num_tiles || axis >= params.num_axes) {
        return false; // Out of bounds - rule doesn't exist
    }

    // Index: axis * num_tiles * num_tiles + tile1 * num_tiles + tile2
    let rule_idx = axis * params.num_tiles * params.num_tiles + tile1 * params.num_tiles + tile2;

    // Determine the index within the u32 array and the bit position
    let u32_idx = rule_idx / 32u;
    let bit_idx = rule_idx % 32u;

    // Bounds check for adjacency_rules array 
    if (u32_idx >= (params.num_axes * params.num_tiles * params.num_tiles + 31u) / 32u) {
        return false; // Out of bounds access
    }

    // Check the specific bit
    return (adjacency_rules[u32_idx] & (1u << bit_idx)) != 0u;
}

@compute @workgroup_size(64) // Use hardcoded size
fn main_propagate(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    // Use flat 1D indexing - much more reliable
    let thread_idx = global_id.x;
    
    // Bounds check for worklist
    if (thread_idx >= params.worklist_size) {
        return;
    }
    
    let coords_packed = worklist[thread_idx];
    let z = coords_packed / (params.grid_width * params.grid_height);
    let temp_coord = coords_packed % (params.grid_width * params.grid_height);
    let y = temp_coord / params.grid_width;
    let x = temp_coord % params.grid_width;
    let current_cell_idx_1d = grid_index(x, y, z); // Pre-calculate 1D index

    // SAFETY CHECK: Only process if NUM_TILES_U32 <= 4 to avoid out of bounds
    if (NUM_TILES_U32 > 4u) {
        // No atomic flag in fallback version
        return;
    }

    var current_possibilities: array<u32, 4>; // Example for up to 128 tiles
    // Initialize to 0
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        current_possibilities[i] = 0u;
    }

    // Calculate number of cells for SoA indexing
    let num_cells = params.grid_width * params.grid_height * params.grid_depth;

    // Only load as many as we need and are within bounds
    for (var i: u32 = 0u; i < NUM_TILES_U32; i = i + 1u) {
        // SoA indexing - current_cell_idx selects the cell, i selects the chunk
        let soa_idx = i * num_cells + current_cell_idx_1d;
        if (soa_idx < NUM_TILES_U32 * num_cells) {
            // Non-atomic read in fallback version
            current_possibilities[i] = grid_possibilities[soa_idx];
        }
    }
    
    // --- Check if the cell has already been collapsed to a single possibility ---
    // If it's already collapsed, can skip most of the work
    
    // Just process each axis direction (typically 6 for 3D grid)
    for (var axis_idx: u32 = 0u; axis_idx < params.num_axes; axis_idx = axis_idx + 1u) {
        // Calculate neighbor coordinates based on axis
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
                // Invalid axis, skip
                continue;
            }
        }
        
        // Check boundary conditions
        var neighbor_in_bounds = true;
        
        if (params.boundary_mode == 0u) { // Clamped mode
            if (nx < 0 || nx >= i32(params.grid_width) ||
                ny < 0 || ny >= i32(params.grid_height) ||
                nz < 0 || nz >= i32(params.grid_depth)) {
                neighbor_in_bounds = false;
            }
        } else { // Periodic mode
            // Wrap coordinates
            nx = i32(wrap_coord(nx, params.grid_width));
            ny = i32(wrap_coord(ny, params.grid_height));
            nz = i32(wrap_coord(nz, params.grid_depth));
        }
        
        if (!neighbor_in_bounds) {
            continue; // Skip this direction
        }
        
        let neighbor_idx_1d = grid_index(u32(nx), u32(ny), u32(nz));
        
        // Get the current possibility state for the neighbor cell
        var neighbor_possibilities: array<u32, 4>; // Up to 128 tiles
        
        // Initialize to 0
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            neighbor_possibilities[i] = 0u;
        }
        
        // Load the possibility state for the neighbor
        for (var i: u32 = 0u; i < NUM_TILES_U32; i = i + 1u) {
            let soa_idx = i * num_cells + neighbor_idx_1d;
            if (soa_idx < NUM_TILES_U32 * num_cells) {
                // Non-atomic read in fallback version
                neighbor_possibilities[i] = grid_possibilities[soa_idx];
            }
        }
        
        // --- Calculate constraints from current cell to neighbor ---
        
        // This is the mask of allowed possibilities in the neighbor
        // based on the current cell's constraints
        var allowed_neighbor_mask: array<u32, 4>; // Up to 128 tiles
        
        // Initialize to 0
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            allowed_neighbor_mask[i] = 0u;
        }
        
        // Calculate the opposite axis for checking constraints
        var opposite_axis = axis_idx;
        if (axis_idx % 2u == 0u) {
            opposite_axis = axis_idx + 1u;
        } else {
            opposite_axis = axis_idx - 1u;
        }
        
        // Determine which tiles are allowed in the neighbor based on
        // the current cell's possibilities and the adjacency rules
        
        // Iterate over all possible tiles (tile2) for the *neighbor* cell
        for (var tile2_idx: u32 = 0u; tile2_idx < params.num_tiles; tile2_idx = tile2_idx + 1u) {
            var tile2_is_supported: bool = false;
            // Check if *any* currently possible tile (tile1) in the *current* cell
            // supports tile2 in the neighbor cell along the relevant axis.
            for (var tile1_idx: u32 = 0u; tile1_idx < params.num_tiles; tile1_idx = tile1_idx + 1u) {
                // Is tile1 possible in the current cell?
                if (is_tile_possible(tile1_idx, &current_possibilities)) {
                    // Does tile1 support tile2 in the neighbor's direction?
                    // Note: check_rule(tile_in_current, tile_in_neighbor, axis_from_current_to_neighbor)
                    if (check_rule(tile1_idx, tile2_idx, axis_idx)) { 
                        tile2_is_supported = true;
                        break; // Found support, no need to check other tile1 for this tile2
                    }
                }
            } // end loop tile1_idx (current cell possibilities)

            // If tile2 is supported by *at least one* possible tile in the current cell,
            // mark it as potentially allowed in the neighbor.
            if (tile2_is_supported) {
                set_tile_possible(tile2_idx, &allowed_neighbor_mask);
            }
        } // end loop tile2_idx (neighbor cell possibilities)

        // --- Update Neighbor's Possibilities Without Atomics ---
        // Apply the constraints by ANDing the current neighbor possibilities with the allowed mask
        var neighbor_mask_changed: bool = false;
        var neighbor_mask_is_zero: bool = true; // Assume contradiction until proven otherwise
        
        for (var i: u32 = 0u; i < NUM_TILES_U32; i = i + 1u) {
            let old_neighbor_mask = neighbor_possibilities[i];
            let new_neighbor_mask = old_neighbor_mask & allowed_neighbor_mask[i];
            
            // Check if the mask changed
            if (old_neighbor_mask != new_neighbor_mask) {
                neighbor_mask_changed = true;
                
                // Update directly without atomics in fallback version
                let soa_idx = i * num_cells + neighbor_idx_1d;
                if (soa_idx < NUM_TILES_U32 * num_cells) {
                    grid_possibilities[soa_idx] = new_neighbor_mask;
                }
            }
            
            // Check for contradictions - if any chunk has bits set, this is not a contradiction
            if (new_neighbor_mask != 0u) {
                neighbor_mask_is_zero = false;
            }
        }
        
        // No atomic flag updates for contradictions in fallback version
        // Host code will need to check for contradictions after the shader runs
    }
} 