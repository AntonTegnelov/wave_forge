// wfc-gpu/src/shaders/components/entropy_calculation.wgsl
// Contains functions related to calculating cell entropy.

// Include common utilities - Assume this will be handled by the assembly process
// #include "../utils.wgsl"

// Binding declarations (will be managed by the pipeline setup)
// These are placeholders to show what the component needs.
// @group(0) @binding(0) var<uniform> params: Params;
// @group(0) @binding(1) var<storage, read> grid_possibilities: array<u32>; // Or atomic<u32>
// @group(0) @binding(2) var<storage, read_write> grid_entropy: array<f32>;
// @group(0) @binding(3) var<storage, read_write> min_entropy_info: array<atomic<u32>>; // Or non-atomic

// Placeholder constant - will be replaced during compilation/assembly
const NUM_TILES: u32 = 32u; // Example value

// Placeholder type - may vary based on NUM_TILES
alias PossibilityMask = array<u32, 1>; // Assume NUM_TILES <= 32 for simplicity here

// Function to count set bits (from entropy_fallback.wgsl)
// Kept simple for broader compatibility initially.
// Assumes utils.wgsl might provide an optimized version later.
fn count_set_bits(n: u32) -> u32 {
    var count = 0u;
    var num = n;
    while (num > 0u) {
        num = num & (num - 1u); // Clear the least significant set bit
        count = count + 1u;
    }
    return count;
}

// Helper to check if a tile is possible (adapted, assuming utils.wgsl might refine)
fn has_tile_possibility(mask: PossibilityMask, tile_index: u32) -> bool {
    let u32_index = tile_index / 32u;
    let bit_index = tile_index % 32u;
    if (u32_index >= arrayLength(&mask)) { return false; } // Basic bounds check
    return (mask[u32_index] & (1u << bit_index)) != 0u;
}

// Applies the configured tie-breaking strategy (from entropy_modular.wgsl)
// Depends on params.tie_breaking
fn apply_tie_breaking(entropy: f32, possibilities: PossibilityMask, cell_idx: u32, params: Params) -> f32 {
    // If tie-breaking is disabled, return the raw entropy
    if (params.tie_breaking == 0u) {
        return entropy;
    }

    let noise_scale = 0.00001;
    var tie_breaking_value = 0.0;

    // Simple hash for deterministic tie-breaking (case 1 and default)
    var pattern_hash = possibilities[0]; // Assumes single u32 mask

    switch (params.tie_breaking) {
        case 1u: { // Deterministic
            tie_breaking_value = f32(pattern_hash % 1024u) / 1024.0;
        }
        case 2u: { // Random pattern (example adaptation)
             pattern_hash = pattern_hash ^ (pattern_hash >> 13u);
             pattern_hash = pattern_hash ^ (pattern_hash << 17u);
             tie_breaking_value = f32(pattern_hash % 1024u) / 1024.0;
        }
        case 3u: { // Position-based
            // Simplified position calculation assuming flat index maps well
            // A more robust coord conversion might be needed depending on context
             let pos_weight = sin(f32(cell_idx % params.grid_width) * 0.1) *
                             cos(f32((cell_idx / params.grid_width) % params.grid_height) * 0.1) +
                             sin(f32(cell_idx / (params.grid_width * params.grid_height)) * 0.1);
            tie_breaking_value = (pos_weight + 1.0) * 0.5; // Normalize to 0-1 range
        }
        default: { // Fallback to deterministic
            tie_breaking_value = f32(pattern_hash % 1024u) / 1024.0;
        }
    }

    return entropy + noise_scale * tie_breaking_value;
}


// Calculates the Shannon entropy for a given set of possibilities (from entropy_modular.wgsl)
fn calculate_shannon_entropy(possibilities: PossibilityMask, cell_idx: u32, params: Params) -> f32 {
    // Count the number of possible tiles
    let possible_count = count_set_bits(possibilities[0]); // Use simple count

    // If fully collapsed (1 possibility) or impossible (0 possibilities), return special values
    if (possible_count <= 1u) {
        return f32(possible_count) - 1.0; // -1.0 for impossible, 0.0 for collapsed
    }

    // Base entropy calculation: log2(count)
    var entropy = log2(f32(possible_count));

    // Apply the configured tie-breaking strategy
    return apply_tie_breaking(entropy, possibilities, cell_idx, params);
}


// Calculates entropy based on simple possibility count (heuristic_type 0u)
fn calculate_count_entropy(possibilities: PossibilityMask, cell_idx: u32, params: Params) -> f32 {
    let count = count_set_bits(possibilities[0]);
    var entropy: f32;
    if (count <= 1u) {
        entropy = f32(count) - 1.0; // -1.0 for impossible, 0.0 for collapsed
    } else {
         // Count heuristic: n-1 (count minus 1)
         entropy = f32(count - 1u);
         // Apply tie-breaking using the count-based entropy value
         entropy = apply_tie_breaking(entropy, possibilities, cell_idx, params);
    }
    return entropy;
}


// Calculates weighted entropy (from entropy_modular.wgsl, simplified weights)
// Assumes weights are provided or accessible via bindings later.
// For now, uses uniform weights as a placeholder.
fn calculate_weighted_entropy(possibilities: PossibilityMask, cell_idx: u32, params: Params) -> f32 {
    let possible_count = count_set_bits(possibilities[0]);

    if (possible_count <= 1u) {
        return f32(possible_count) - 1.0;
    }

    // Placeholder: Use uniform weights for now.
    // A real implementation would need access to a weights buffer.
    var weights: array<f32, NUM_TILES>;
    for (var i = 0u; i < NUM_TILES; i++) {
        weights[i] = 1.0;
    }

    var sum_weights = 0.0;
    var weighted_log_sum = 0.0;
    var active_count = 0u;

    // Collect weights and calculate sum
    for (var tile = 0u; tile < NUM_TILES; tile++) {
        if (has_tile_possibility(possibilities, tile)) {
            let w = weights[tile]; // Using placeholder uniform weight
            sum_weights += w;
            active_count += 1u;
        }
    }

    // Calculate entropy part: sum(p * log2(p)) where p = w / sum_weights
    for (var tile = 0u; tile < NUM_TILES; tile++) {
        if (has_tile_possibility(possibilities, tile)) {
            let w = weights[tile];
            let normalized_w = w / sum_weights;
            if (normalized_w > 0.00001) { // Avoid log(0)
                weighted_log_sum += normalized_w * log2(normalized_w);
            }
        }
    }

    let entropy = -weighted_log_sum; // Shannon entropy formula

    return apply_tie_breaking(entropy, possibilities, cell_idx, params);
}

// --- Main Entry Point Function (Example Structure) ---
// The actual entry point (@compute) will likely be in the assembled shader variant.
// This function shows how the components might be used.
fn calculate_cell_entropy(cell_idx: u32, grid_possibilities: array<u32>, params: Params) -> f32 {
     // Load current cell's possibilities (simplified loading)
    var possibilities: PossibilityMask;
    possibilities[0] = grid_possibilities[cell_idx]; // Assumes flat layout

    var entropy = 0.0;

    switch (params.heuristic_type) {
        case 0u: { // Simple count - 1
            entropy = calculate_count_entropy(possibilities, cell_idx, params);
        }
        case 1u: { // Shannon entropy
            entropy = calculate_shannon_entropy(possibilities, cell_idx, params);
        }
        case 2u: { // Weighted entropy (using placeholder weights)
            entropy = calculate_weighted_entropy(possibilities, cell_idx, params);
        }
        default: { // Fallback to Shannon
            entropy = calculate_shannon_entropy(possibilities, cell_idx, params);
        }
    }
    return entropy;
}


// --- Minimum Entropy Tracking (Example, requires atomics binding) ---
// This part needs the atomic min_entropy_info buffer.
fn update_min_entropy_atomic(cell_idx: u32, entropy: f32, min_entropy_info: array<atomic<u32>>) {
    // Skip already collapsed or impossible cells
    if (entropy <= 0.0) {
        return;
    }

    // Convert to sortable integer representation for atomicMin
    // Use bitcast, potentially invert based on ordering needs (e.g., 1.0/entropy for min)
    let entropy_bits = bitcast<u32>(1.0 / entropy); // Example: Find min positive entropy

    // Basic atomicMin update
    atomicMin(&min_entropy_info[0], entropy_bits);

    // Storing the index associated with the min requires careful synchronization
    // A common pattern is to store {entropy_bits, cell_index} packed into a u64,
    // or use a compare-and-swap loop if only u32 atomics are available.
    // For simplicity, this example only updates the minimum value.
    // A more robust implementation would handle index updates correctly.

     // Example of storing index if min was updated (needs CAS or careful logic)
    // let old_min = atomicMin(&min_entropy_info[0], entropy_bits);
    // if (old_min > entropy_bits) { // Check if we actually set a new minimum
    //    atomicStore(&min_entropy_info[1], cell_idx); // Store the index (potential race condition without CAS)
    // }
} 