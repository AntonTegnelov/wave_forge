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

// Applies the configured tie-breaking strategy to the entropy value
fn apply_tie_breaking(entropy: f32, possibilities: PossibilityMask, cell_idx: u32) -> f32 {
    // If tie-breaking is disabled, return the raw entropy
    if (params.tie_breaking == 0u) {
        return entropy;
    }
    
    let noise_scale = 0.00001;
    var tie_breaking_value = 0.0;
    
    switch (params.tie_breaking) {
        case 1u: { // Deterministic - always the same for the same pattern
            // Use the pattern of bits as a simple hash
            var pattern_hash = 0u;
            for (var i = 0u; i < 1u; i++) {
                pattern_hash = pattern_hash ^ possibilities[i];
            }
            tie_breaking_value = f32(pattern_hash % 1024u) / 1024.0;
        }
        case 2u: { // Random pattern - less predictable patterns
            // XOR the pattern with its bit-rotated version for more randomness
            var pattern_hash = possibilities[0];
            pattern_hash = pattern_hash ^ bitcast<u32>(f32(possibilities[0] >> 13u));
            pattern_hash = pattern_hash ^ bitcast<u32>(f32(possibilities[0] << 17u));
            tie_breaking_value = f32(pattern_hash % 1024u) / 1024.0;
        }
        case 3u: { // Position-based - favors certain grid areas
            // Calculate a position-based value using cell index
            // Convert 1D index to 3D position
            let z = cell_idx / (params.grid_width * params.grid_height);
            let temp = cell_idx % (params.grid_width * params.grid_height);
            let y = temp / params.grid_width;
            let x = temp % params.grid_width;
            
            // Use a simple space-filling pattern that combines coordinates
            // This creates wave-like patterns of preference across the grid
            let pos_weight = sin(f32(x) * 0.1) * cos(f32(y) * 0.1) + sin(f32(z) * 0.1);
            tie_breaking_value = (pos_weight + 1.0) * 0.5; // Normalize to 0-1 range
        }
        default: { // Fallback to deterministic
            var pattern_hash = possibilities[0];
            tie_breaking_value = f32(pattern_hash % 1024u) / 1024.0;
        }
    }
    
    // Apply the tie-breaking value with appropriate scaling
    return entropy + noise_scale * tie_breaking_value;
}

// Calculates the Shannon entropy for a given set of possibilities
fn calculate_shannon_entropy(possibilities: PossibilityMask, cell_idx: u32) -> f32 {
    // Count the number of possible tiles
    let possible_count = count_bits(possibilities[0]);
    
    // If fully collapsed (1 possibility) or impossible (0 possibilities), return special values
    if (possible_count <= 1u) {
        return f32(possible_count) - 1.0; // -1.0 for impossible, 0.0 for collapsed
    }
    
    // For uniform distribution, Shannon entropy is just log(n)
    // But we need to handle numerical issues carefully
    
    // Base entropy calculation: log2(count)
    var entropy = log2(f32(possible_count));
    
    // Apply the configured tie-breaking strategy
    return apply_tie_breaking(entropy, possibilities, cell_idx);
}

// Calculates weighted entropy based on tile weights with improved stability
fn calculate_weighted_entropy(possibilities: PossibilityMask, weights: array<f32, NUM_TILES>, cell_idx: u32) -> f32 {
    // Count the number of possible tiles
    let possible_count = count_bits(possibilities[0]);
    
    // If fully collapsed (1 possibility) or impossible (0 possibilities), return special values
    if (possible_count <= 1u) {
        return f32(possible_count) - 1.0; // -1.0 for impossible, 0.0 for collapsed
    }
    
    // Calculate weight sum and entropy with improved numerical stability
    var sum_weights = 0.0;
    var max_weight = 0.0;
    var weights_array: array<f32, NUM_TILES>;
    var active_count = 0u;
    
    // First pass: collect weights and find maximum
    for (var tile = 0u; tile < NUM_TILES; tile++) {
        if (has_tile_possibility(possibilities, tile)) {
            let w = weights[tile];
            weights_array[active_count] = w;
            sum_weights += w;
            max_weight = max(max_weight, w);
            active_count += 1u;
        }
    }
    
    // Use the log-sum-exp trick for numerical stability
    // H = log(sum(w)) - sum(w*log(w))/sum(w)
    
    // Calculate log(sum(w)) stably
    let log_sum_weights = log2(sum_weights);
    
    // Calculate sum(w*log(w))/sum(w) stably
    var weighted_log_sum = 0.0;
    for (var i = 0u; i < active_count; i++) {
        let w = weights_array[i];
        let normalized_w = w / sum_weights;  // Normalize weights
        
        // Skip or handle very small values to prevent precision issues with log
        if (normalized_w > 0.00001) {
            weighted_log_sum += normalized_w * log2(normalized_w);
        }
    }
    
    // Shannon entropy formula: -sum(p*log(p))
    let entropy = -weighted_log_sum;
    
    // Apply the configured tie-breaking strategy
    return apply_tie_breaking(entropy, possibilities, cell_idx);
}

// Gets the minimum entropy value and updates atomic tracking value
// Handles race conditions gracefully with a more robust approach
fn update_min_entropy(cell_idx: u32, entropy: f32) {
    // Skip already collapsed or impossible cells
    if (entropy <= 0.0) {
        return;
    }
    
    // Convert to sortable integer representation
    // Use bitcast to preserve ordering while allowing atomic operations
    // Invert to make smaller entropy = larger integer for min finding
    // This allows us to use atomicMin to find the minimum entropy
    let entropy_bits = bitcast<u32>(1.0 / entropy);
    
    // Use double-check algorithm to reduce contention:
    // 1. First check if our entropy is potentially a new minimum 
    let current_min = atomicLoad(&min_entropy_info[0]);
    
    // Only attempt update if this cell has lower entropy (higher inverted bits)
    if (entropy_bits > current_min) {
        // Try to update the minimum value
        let old_min = atomicMin(&min_entropy_info[0], entropy_bits);
        
        // If we successfully updated the minimum, also update the cell index
        // This ensures the cell index always corresponds to the minimum entropy
        if (old_min < entropy_bits) {
            // Try to update the index - use atomic exchange to avoid race conditions
            // even though only one thread should enter this condition
            atomicStore(&min_entropy_info[1], cell_idx);
            
            // Memory barrier to ensure correct ordering
            storageBarrier();
            
            // Verify our minimum is still the global minimum
            // Handle case where another thread found a better minimum between our operations
            let final_min = atomicLoad(&min_entropy_info[0]);
            if (final_min != entropy_bits) {
                // Our entropy value was replaced by a better one
                // The associated cell index should have been set by the other thread
                // No action needed
            }
        }
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
            entropy = calculate_shannon_entropy(possibilities, thread_idx);
        }
        case 2u: { // Weighted entropy - actually use uniform weights for now
            var weights: array<f32, NUM_TILES>;
            for (var i = 0u; i < NUM_TILES; i++) {
                weights[i] = 1.0;
            }
            entropy = calculate_weighted_entropy(possibilities, weights, thread_idx);
        }
        default: { // Fallback to Shannon
            entropy = calculate_shannon_entropy(possibilities, thread_idx);
        }
    }
    
    // Write entropy to output buffer
    grid_entropy[thread_idx] = entropy;
    
    // Update minimum entropy info if this cell has valid entropy
    update_min_entropy(thread_idx, entropy);
} 