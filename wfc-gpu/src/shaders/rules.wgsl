// rules.wgsl - Adjacency rules and possibility mask handling for WFC
//
// This module contains functions for working with adjacency rules, possibility masks,
// and checking tile compatibility between adjacent cells.

// --- Constants (Consider moving NUM_TILES_U32 if purely configuration) ---
// Number of u32s needed per cell based on total number of tiles (Moved from utils.wgsl)
// This will be replaced with the actual value during shader compilation
const NUM_TILES_U32_VALUE: u32 = 1u; // Placeholder, will be dynamically set in source
const NUM_TILES_U32: u32 = NUM_TILES_U32_VALUE;

// --- Types (Moved from utils.wgsl) ---
// Possibility mask array type using the NUM_TILES_U32 constant
// Represents the set of possible tiles for a cell.
alias PossibilityMask = array<u32, NUM_TILES_U32_VALUE>;

// --- Possibility Mask Helpers (Moved from utils.wgsl) ---

// Helper function to check if a specific bit (tile) in the mask is set.
// Assumes NUM_TILES_U32_VALUE is handled correctly during compilation.
fn is_tile_possible(tile_index: u32, mask: PossibilityMask) -> bool {
    let u32_index = tile_index / 32u;
    let bit_index = tile_index % 32u;
    // Bounds check might be needed if NUM_TILES_U32_VALUE > 1
    if (u32_index >= NUM_TILES_U32_VALUE) { return false; }
    return (mask[u32_index] & (1u << bit_index)) != 0u;
}

// Helper function to set a specific bit (tile) in a u32 array mask.
// This version takes a pointer to allow modification.
fn set_tile_possible(tile_index: u32, mask_ptr: ptr<function, PossibilityMask>) {
    let u32_index = tile_index / 32u;
    let bit_index = tile_index % 32u;
    // Bounds check might be needed if NUM_TILES_U32_VALUE > 1
    if (u32_index < NUM_TILES_U32_VALUE) {
         (*mask_ptr)[u32_index] = (*mask_ptr)[u32_index] | (1u << bit_index);
    }
}

// Helper function to clear a specific bit (tile) in a u32 array mask.
fn clear_tile_possible(tile_index: u32, mask_ptr: ptr<function, PossibilityMask>) {
    let u32_index = tile_index / 32u;
    let bit_index = tile_index % 32u;
    if (u32_index < NUM_TILES_U32_VALUE) {
         (*mask_ptr)[u32_index] = (*mask_ptr)[u32_index] & !(1u << bit_index);
    }
}

// --- Adjacency Rule Checking --- 

// Helper function to check adjacency rule bit.
// Assumes rules are packed tightly: rule[axis][tile1][tile2]
// Requires access to params (for num_tiles, num_axes) and the adjacency_rules buffer.
fn check_rule(tile1: u32, tile2: u32, axis: u32, params: Params, adjacency_rules: array<u32>) -> bool {
    // Bounds check for tile indices
    if (tile1 >= params.num_tiles || tile2 >= params.num_tiles || axis >= params.num_axes) {
        return false; // Out of bounds - rule doesn't exist
    }

    // Index: axis * num_tiles * num_tiles + tile1 * num_tiles + tile2
    let rule_idx = axis * params.num_tiles * params.num_tiles + tile1 * params.num_tiles + tile2;

    // Determine the index within the u32 array and the bit position
    let u32_idx = rule_idx / 32u;
    let bit_idx = rule_idx % 32u;

    // Calculate the expected size of the rules buffer in u32s
    let rules_buffer_size = (params.num_axes * params.num_tiles * params.num_tiles + 31u) / 32u;

    // Bounds check for adjacency_rules array
    if (u32_idx >= rules_buffer_size) {
        // Log error or handle gracefully? For now, return false.
        // This indicates a potential mismatch between params and buffer size.
        return false; // Out of bounds access
    }

    // Check the specific bit
    return (adjacency_rules[u32_idx] & (1u << bit_idx)) != 0u;
}

// Helper function to get rule weight.
// For non-weighted rules (the default), this returns 1.0.
// Requires access to params, adjacency_rules, and rule_weights buffers.
fn get_rule_weight(tile1: u32, tile2: u32, axis: u32, params: Params,
                   adjacency_rules: array<u32>, rule_weights: array<u32>) -> f32 {
    // First check if the rule exists at all
    if (!check_rule(tile1, tile2, axis, params, adjacency_rules)) {
        return 0.0;
    }

    // Index into the weights array (if it exists)
    let rule_idx = axis * params.num_tiles * params.num_tiles + tile1 * params.num_tiles + tile2;

    // Check if this rule has an entry in the weights buffer
    // TODO: Replace linear search with a more efficient lookup (e.g., hash map on CPU, binary search if sorted)
    for (var i = 0u; i < arrayLength(&rule_weights); i += 2u) {
        // Each weight entry consists of two u32s: (rule_idx, weight_bits)
        if (rule_weights[i] == rule_idx) {
            // Found it, convert the bits back to float
            return bitcast<f32>(rule_weights[i + 1u]);
        }
    }

    // Default weight for valid rules with no explicit weight entry
    return 1.0;
}

// Computes the mask of tiles allowed in a neighboring cell, based on the
// possibilities of the current cell and the adjacency rules along the given axis.
// Requires access to params and adjacency_rules.
fn compute_allowed_neighbor_mask(current_possibilities: PossibilityMask,
                                axis_idx: u32,
                                params: Params,
                                adjacency_rules: array<u32>) -> PossibilityMask {
    var allowed_neighbor_mask: PossibilityMask;
    // Initialize all mask elements to 0
    for (var i = 0u; i < NUM_TILES_U32_VALUE; i = i + 1u) {
         allowed_neighbor_mask[i] = 0u;
    }

    // Iterate through each possible tile in the *current* cell
    for (var current_tile: u32 = 0u; current_tile < params.num_tiles; current_tile = current_tile + 1u) {
        if (!is_tile_possible(current_tile, current_possibilities)) {
            continue; // Skip if this tile isn't possible in the current cell
        }

        // For the current possible tile, find all compatible *neighbor* tiles
        for (var neighbor_tile: u32 = 0u; neighbor_tile < params.num_tiles; neighbor_tile = neighbor_tile + 1u) {
            // Check the rule: Is neighbor_tile allowed next to current_tile along axis_idx?
            if (check_rule(current_tile, neighbor_tile, axis_idx, params, adjacency_rules)) {
                // If yes, mark neighbor_tile as possible in the result mask
                set_tile_possible(neighbor_tile, &allowed_neighbor_mask);
            }
        }
    }

    return allowed_neighbor_mask;
} 