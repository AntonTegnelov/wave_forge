// WGSL Shader for Wave Function Collapse constraint propagation
//
// This compute shader handles the propagation of constraints through the grid
// after a cell's possibilities have been restricted or collapsed. It ensures that
// all neighboring cells maintain consistent possibility states based on the 
// adjacency rules.
//
// CRITICAL SAFETY FEATURES:
// 1. Uses 1D workgroup layout (64,1,1) for simpler thread indexing
// 2. Enforces strict bounds checking on all array accesses
// 3. Limits maximum tiles to 128 (4 u32s) to prevent buffer overflows
// 4. Contains output worklist size limits to prevent infinite propagation loops
// 5. Detects and reports contradictions early
//
// The shader processes each cell in the input worklist, updates all valid 
// neighbors according to adjacency rules, and adds any changed neighbors
// to the output worklist for further processing if needed.

// Placeholder for WGSL shader code

/*
struct UpdateInfo {
    coords: vec3<u32>,
    // Add other necessary info, potentially the collapsed tile ID
};

@group(0) @binding(0) var<storage, read_write> grid_possibilities: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> adjacency_rules: array<u32>; // Flattened/indexed rules
@group(0) @binding(2) var<uniform> grid_dims: vec3<u32>;
@group(0) @binding(3) var<storage, read> updates_to_process: array<UpdateInfo>; // List of cells that changed
@group(0) @binding(4) var<storage, read_write> contradiction_flag: atomic<u32>; // Flag for contradictions

const MAX_NEIGHBORS: u32 = 6;

fn get_linear_index(coords: vec3<u32>) -> u32 {
    return coords.z * grid_dims.x * grid_dims.y + coords.y * grid_dims.x + coords.x;
}

@compute @workgroup_size(64, 1, 1) // Example: process updates in parallel
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let update_idx = global_id.x;
    // Check bounds for updates_to_process array

    let update = updates_to_process[update_idx];
    let current_coords = update.coords;
    let current_cell_idx = get_linear_index(current_coords);

    // TODO: Implement propagation logic in WGSL
    // For the updated cell (current_coords):
    // 1. Get its current possibility mask (after potential collapse)
    // 2. Iterate through its neighbors (e.g., 6 directions)
    // 3. For each neighbor:
    //    a. Determine the valid possibilities for the neighbor based on the
    //       current cell's mask and the adjacency rules.
    //    b. Get the neighbor's current possibility mask.
    //    c. Calculate the intersection (bitwise AND) of the neighbor's mask
    //       and the valid possibilities derived from the current cell.
    //    d. Atomically update the neighbor's mask in grid_possibilities using
    //       atomicAnd. Check if the mask changed.
    //    e. If the neighbor's mask becomes zero, set the contradiction_flag (atomicMax?)
    //    f. If the neighbor's mask changed, potentially add the neighbor to a
    //       *new* list of updates for the *next* propagation step (more complex).
}
*/

// TODO: Define appropriate data structures for grid, rules, worklist

// Example bindings (adjust based on actual buffer structures)
@group(0) @binding(0) var<storage, read_write> grid_possibilities: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> adjacency_rules: array<u32>; // Flattened rules
@group(0) @binding(2) var<storage, read> worklist: array<u32>; // Coordinates or indices of updated cells
@group(0) @binding(3) var<storage, read_write> output_worklist: array<atomic<u32>>; // For new cells that need propagation
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read_write> output_worklist_count: atomic<u32>; // Atomic counter for output_worklist size
@group(0) @binding(6) var<storage, read_write> contradiction_flag: atomic<u32>; // Global flag for contradictions
@group(0) @binding(7) var<storage, read_write> contradiction_location: atomic<u32>; // Global index of first contradiction (initialized to u32::MAX)

// Uniforms for grid dimensions, num_tiles etc.
struct Params {
    grid_width: u32,
    grid_height: u32,
    grid_depth: u32,
    num_tiles: u32,
    num_tiles_u32: u32, // num_tiles rounded up for u32 array indexing
    num_axes: u32,
    worklist_size: u32,
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

// Helper function to check if a specific bit is set in a u32 array mask
fn is_tile_possible(tile_index: u32, mask: ptr<function, array<u32, 4>>) -> bool {
    let u32_index = tile_index / 32u;
    let bit_index = tile_index % 32u;
    // Critical safety check - must ensure we don't access beyond array bounds
    if (u32_index >= params.num_tiles_u32 || u32_index >= 4u) { return false; }
    return ((*mask)[u32_index] & (1u << bit_index)) != 0u;
}

// Helper function to set a specific bit in a u32 array mask
fn set_tile_possible(tile_index: u32, mask: ptr<function, array<u32, 4>>) {
    let u32_index = tile_index / 32u;
    let bit_index = tile_index % 32u;
    // Critical safety check - must ensure we don't access beyond array bounds
    if (u32_index >= params.num_tiles_u32 || u32_index >= 4u) { return; }
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

@compute
@workgroup_size(64, 1, 1) // Switch to simple 1D layout - much less error prone
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
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

    // SAFETY CHECK: Only process if params.num_tiles_u32 <= 4 to avoid out of bounds
    if (params.num_tiles_u32 > 4u) {
        atomicMax(&contradiction_flag, 1u); // Mark as contradiction
        // Atomically store the location if it hasn't been stored yet (atomicMin with u32::MAX as initial)
        atomicMin(&contradiction_location, current_cell_idx_1d); 
        return;
    }

    var current_possibilities: array<u32, 4>; // Example for up to 128 tiles
    // Initialize to 0
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        current_possibilities[i] = 0u;
    }

    // Only load as many as we need and are within bounds
    for (var i: u32 = 0u; i < params.num_tiles_u32 && i < 4u; i = i + 1u) {
       current_possibilities[i] = atomicLoad(&grid_possibilities[current_cell_idx_1d * params.num_tiles_u32 + i]);
    }

    // --- Iterate through Neighbors ---
    for (var axis_idx: u32 = 0u; axis_idx < 6u; axis_idx = axis_idx + 1u) {
        // Determine offset and axes using a switch based on axis_idx
        var neighbor_offset: vec3<i32>;
        var current_axis: u32;
        var neighbor_axis: u32;

        switch axis_idx {
            case 0u: { // +X
                neighbor_offset = vec3<i32>(1, 0, 0);
                current_axis = 0u;
                neighbor_axis = 1u;
            }
            case 1u: { // -X
                neighbor_offset = vec3<i32>(-1, 0, 0);
                current_axis = 1u;
                neighbor_axis = 0u;
            }
            case 2u: { // +Y
                neighbor_offset = vec3<i32>(0, 1, 0);
                current_axis = 2u;
                neighbor_axis = 3u;
            }
            case 3u: { // -Y
                neighbor_offset = vec3<i32>(0, -1, 0);
                current_axis = 3u;
                neighbor_axis = 2u;
            }
            case 4u: { // +Z
                neighbor_offset = vec3<i32>(0, 0, 1);
                current_axis = 4u;
                neighbor_axis = 5u;
            }
            case 5u: { // -Z
                neighbor_offset = vec3<i32>(0, 0, -1);
                current_axis = 5u;
                neighbor_axis = 4u;
            }
            default: { // Should not happen
                // Optionally handle error or continue
                continue;
            }
        }

        let nx = i32(x) + neighbor_offset.x;
        let ny = i32(y) + neighbor_offset.y;
        let nz = i32(z) + neighbor_offset.z;

        // --- Bounds Check ---
        if (nx >= 0 && nx < i32(params.grid_width) &&
            ny >= 0 && ny < i32(params.grid_height) &&
            nz >= 0 && nz < i32(params.grid_depth)) {

            let unx = u32(nx);
            let uny = u32(ny);
            let unz = u32(nz);
            let neighbor_idx_1d = grid_index(unx, uny, unz);

            // --- Calculate Allowed Neighbor Tiles ---
            var allowed_neighbor_mask: array<u32, 4>; // Max 128 tiles example
            // Initialize mask to all zeros
            for (var i: u32 = 0u; i < 4u; i=i+1u) {
                allowed_neighbor_mask[i] = 0u;
            }

            // Iterate over all possible tiles (tile1) for the *current* cell
            for (var tile1_idx: u32 = 0u; tile1_idx < params.num_tiles; tile1_idx = tile1_idx + 1u) {
                // Check if tile1 is actually possible in the current cell
                if (is_tile_possible(tile1_idx, &current_possibilities)) {
                    // If tile1 is possible, iterate over all possible tiles (tile2) for the *neighbor*
                    for (var tile2_idx: u32 = 0u; tile2_idx < params.num_tiles; tile2_idx = tile2_idx + 1u) {
                        // Check the rule: Is tile2 allowed next to tile1 along this axis?
                        if (check_rule(tile1_idx, tile2_idx, current_axis)) {
                            // If the rule allows it, mark tile2 as possible for the neighbor
                            set_tile_possible(tile2_idx, &allowed_neighbor_mask);
                        }
                    }
                }
            }

            // --- Perform Atomic Update on Neighbor ---
            var original_neighbor_mask: array<u32, 4>;
            var changed = false;
            var became_empty = true;

            for (var i: u32 = 0u; i < params.num_tiles_u32 && i < 4u; i = i + 1u) {
                let neighbor_atomic_ptr = &grid_possibilities[neighbor_idx_1d * params.num_tiles_u32 + i];
                // Perform atomic AND
                let original_val = atomicAnd(neighbor_atomic_ptr, allowed_neighbor_mask[i]);
                original_neighbor_mask[i] = original_val;

                // Check if the mask actually changed
                if ((original_val & allowed_neighbor_mask[i]) != original_val) {
                    changed = true;
                }

                // Check if the updated mask part is non-zero
                if ((original_val & allowed_neighbor_mask[i]) != 0u) {
                    became_empty = false;
                }
            }

            // --- Check for Contradiction --- 
            if (became_empty) {
                // Set global contradiction flag
                atomicMax(&contradiction_flag, 1u);
                // Atomically store the location of the CONTRADICTION (neighbor cell)
                // Only store if no other location has been stored yet (atomicMin vs u32::MAX)
                atomicMin(&contradiction_location, neighbor_idx_1d); 

                // Important: Even if a contradiction is found, we might need to continue
                // processing other neighbors or updates depending on the desired propagation behavior.
                // For now, we continue processing other neighbors of the current cell.
            }

            // --- Add to Output Worklist if Changed (Optional/Complex) ---
            if (changed) {
                // Atomically increment the output worklist counter and get the index
                let output_index = atomicAdd(&output_worklist_count, 1u);
                
                // CRITICAL: Check for worklist overflow - we must have a maximum size limit
                // This prevents infinite iterations and potential hangs
                let max_worklist_size = params.grid_width * params.grid_height * params.grid_depth;
                
                if (output_index < max_worklist_size) {
                    // Store the packed coordinates of the neighbor
                    let neighbor_coords_packed = unz * params.grid_width * params.grid_height + uny * params.grid_width + unx;
                    // Use atomicStore; although not strictly necessary if only one thread writes per index, it's safer.
                    atomicStore(&output_worklist[output_index], neighbor_coords_packed);
                } else {
                    // If worklist overflows, mark as contradiction to halt processing
                    atomicMax(&contradiction_flag, 1u);
                }
            }
        } // End bounds check
    } // End neighbor loop
} 