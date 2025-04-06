// rules.wgsl - Adjacency rules handling for Wave Function Collapse
//
// This module contains functions for working with adjacency rules, including
// checking if specific tile combinations are allowed across different axes.

// Helper function to check adjacency rule
// Assumes rules are packed tightly: rule[axis][tile1][tile2]
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

    // Bounds check for adjacency_rules array 
    if (u32_idx >= (params.num_axes * params.num_tiles * params.num_tiles + 31u) / 32u) {
        return false; // Out of bounds access
    }

    // Check the specific bit
    return (adjacency_rules[u32_idx] & (1u << bit_idx)) != 0u;
}

// Helper function to get rule weight
// For non-weighted rules (the default), this returns 1.0
// For weighted rules, this returns a value between 0.0 and 1.0
fn get_rule_weight(tile1: u32, tile2: u32, axis: u32, params: Params, 
                   adjacency_rules: array<u32>, rule_weights: array<u32>) -> f32 {
    // First check if the rule exists at all
    if (!check_rule(tile1, tile2, axis, params, adjacency_rules)) {
        return 0.0;
    }
    
    // Index into the weights array (if it exists)
    // Since most rules have weight 1.0, we only store the ones that differ
    let rule_idx = axis * params.num_tiles * params.num_tiles + tile1 * params.num_tiles + tile2;
    
    // Check if this rule has an entry in the weights buffer
    // For now, we'll use a simple linear search approach
    // Future optimization: Use a proper mapping structure
    for (var i = 0u; i < arrayLength(&rule_weights); i += 2u) {
        // Each weight entry consists of two u32s:
        // - rule_idx: The packed rule index
        // - weight_bits: The f32 weight encoded as bits
        
        if (rule_weights[i] == rule_idx) {
            // Found it, get the weight
            // Convert the bits back to float
            return bitcast<f32>(rule_weights[i + 1u]);
        }
    }
    
    // Default weight for valid rules with no explicit weight
    return 1.0;
}

// Computes what tiles are allowed in a neighboring cell based on the current cell's possibilities
fn compute_allowed_neighbor_mask(current_possibilities: ptr<function, PossibilityMask>, 
                                axis_idx: u32, 
                                params: Params,
                                adjacency_rules: array<u32>) -> PossibilityMask {
    // This mask represents the set of tiles allowed in the neighbor
    var allowed_neighbor_mask: PossibilityMask;
    
    // Initialize to 0 - use only index 0 for simplicity
    allowed_neighbor_mask[0] = 0u;
    
    // For each possible tile in the current cell
    for (var current_tile: u32 = 0u; current_tile < params.num_tiles; current_tile = current_tile + 1u) {
        // Skip if this tile isn't possible in the current cell
        if (!is_tile_possible(current_tile, current_possibilities)) {
            continue;
        }
        
        // For each potential tile in the neighbor cell
        for (var neighbor_tile: u32 = 0u; neighbor_tile < params.num_tiles; neighbor_tile = neighbor_tile + 1u) {
            // Check if this neighbor tile is allowed according to the rule
            if (check_rule(current_tile, neighbor_tile, axis_idx, params, adjacency_rules)) {
                // If allowed, mark this tile as possible in the allowed_neighbor_mask
                set_tile_possible(neighbor_tile, &allowed_neighbor_mask);
            }
        }
    }
    
    return allowed_neighbor_mask;
} 