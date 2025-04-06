// entropy_modular.wgsl - Modular Wave Function Collapse entropy calculation
//
// This compute shader calculates the Shannon entropy of each cell in the grid
// based on the current possibility space. It finds cells with minimum entropy
// that are prime candidates for collapsing in the next step of the WFC algorithm.
//
// This version uses a modular approach, importing common utility functions
// for better maintainability.

// Include common utilities
#include "utils.wgsl"

// --- Bindings ---

// Uniform buffer with parameters
@group(0) @binding(0) var<uniform> params: Params;

// Storage buffers for grid data - Structure of Array (SoA) layout
// Each u32 contains up to 32 bits for tile possibilities
@group(0) @binding(1) var<storage, read> grid_possibilities: array<u32>;

// Output storage buffer for entropy values
@group(0) @binding(2) var<storage, read_write> grid_entropy: array<f32>;

// Atomic output for min entropy tracking
@group(0) @binding(3) var<storage, read_write> min_entropy_info: array<atomic<u32>>;

// --- Helper functions ---

// Calculates the Shannon entropy for a given set of possibilities
fn calculate_shannon_entropy(possibilities: PossibilityMask) -> f32 {
    // Count the number of possible tiles
    let possible_count = count_bits(possibilities[0]);
    
    // If fully collapsed (1 possibility) or impossible (0 possibilities), return special values
    if (possible_count <= 1u) {
        return f32(possible_count) - 1.0; // -1.0 for impossible, 0.0 for collapsed
    }
    
    // Calculate Shannon entropy: -sum(p * log(p))
    // For uniform distribution with n possibilities: -log(1/n) = log(n)
    // Add small epsilon to avoid precision issues
    return log2(f32(possible_count)) + 0.0001 * f32(possible_count);
}

// Calculates weighted entropy based on tile weights
fn calculate_weighted_entropy(possibilities: PossibilityMask, weights: array<f32, NUM_TILES>) -> f32 {
    // Count the number of possible tiles
    let possible_count = count_bits(possibilities[0]);
    
    // If fully collapsed (1 possibility) or impossible (0 possibilities), return special values
    if (possible_count <= 1u) {
        return f32(possible_count) - 1.0; // -1.0 for impossible, 0.0 for collapsed
    }
    
    // Calculate weight sum and entropy
    var sum_weights = 0.0;
    var sum_weighted_logw = 0.0;
    
    // Loop through all possible tiles in the current cell
    for (var tile = 0u; tile < NUM_TILES; tile++) {
        if (has_tile_possibility(possibilities, tile)) {
            let w = weights[tile];
            sum_weights += w;
            sum_weighted_logw += w * log2(w);
        }
    }
    
    // Calculate final weighted entropy
    // H = log(sum(w)) - sum(w*log(w))/sum(w)
    let entropy = log2(sum_weights) - sum_weighted_logw / sum_weights;
    
    // Add small noise for randomness in case of ties
    return entropy + 0.0001 * f32(possible_count);
}

// Gets the minimum entropy value and updates atomic tracking value
fn update_min_entropy(cell_idx: u32, entropy: f32) {
    // Skip already collapsed or impossible cells
    if (entropy <= 0.0) {
        return;
    }
    
    // Convert to sortable integer representation
    // Invert to make smaller entropy = larger integer for min finding
    let entropy_bits = bitcast<u32>(1.0 / entropy);
    
    // Pack the entropy and cell index into a single u64 represented as two u32s
    // First 32 bits: entropy as sortable bits
    // Second 32 bits: cell index
    atomicMin(&min_entropy_info[0], entropy_bits);
    
    // Update cell index if this is the new minimum
    let current_min = atomicLoad(&min_entropy_info[0]);
    if (current_min == entropy_bits) {
        atomicStore(&min_entropy_info[1], cell_idx);
    }
}

// --- Main compute shader entry point ---

@compute @workgroup_size(64) // Use hardcoded size
fn main_entropy(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    // Use flat 1D indexing
    let thread_idx = global_id.x;
    
    // Calculate grid dimensions
    let num_cells = params.grid_width * params.grid_height * params.grid_depth;
    
    // Bounds check
    if (thread_idx >= num_cells) {
        return;
    }
    
    // Load current cell's possibilities
    var possibilities: PossibilityMask;
    possibilities[0] = grid_possibilities[thread_idx];
    
    // Calculate entropy based on the heuristic type
    var entropy = 0.0;
    
    switch (params.heuristic_type) {
        case 0u: { // Simple count
            let count = count_bits(possibilities[0]);
            entropy = f32(count);
            if (count <= 1u) {
                entropy = f32(count) - 1.0; // -1.0 for impossible, 0.0 for collapsed
            }
        }
        case 1u: { // Shannon entropy
            entropy = calculate_shannon_entropy(possibilities);
        }
        case 2u: { // Weighted entropy - actually use uniform weights for now
            var weights: array<f32, NUM_TILES>;
            for (var i = 0u; i < NUM_TILES; i++) {
                weights[i] = 1.0;
            }
            entropy = calculate_weighted_entropy(possibilities, weights);
        }
        default: { // Fallback to Shannon
            entropy = calculate_shannon_entropy(possibilities);
        }
    }
    
    // Write entropy to output buffer
    grid_entropy[thread_idx] = entropy;
    
    // Update minimum entropy info if this cell has valid entropy
    update_min_entropy(thread_idx, entropy);
} 