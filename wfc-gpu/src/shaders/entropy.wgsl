// WGSL Shader for Wave Function Collapse entropy calculation
//
// This compute shader calculates the entropy (uncertainty) for each cell in the grid
// based on the number of remaining possibilities. Cells with higher entropy have
// more possibilities and are less constrained.
//
// CRITICAL SAFETY FEATURES:
// 1. Uses 1D workgroup layout (64) for simpler thread indexing
// 2. Enforces strict bounds checking on grid indices
// 3. Uses grid dimensions from uniform buffer to prevent out-of-bounds access
// 4. Simple count-based entropy calculation to avoid NaN/infinity issues
//
// The entropy values are used in the WFC algorithm to determine which cell
// to collapse next (typically the one with the lowest non-zero entropy).

// Placeholder for WGSL shader code
// This file might eventually use include_str! or a build script
// to embed the actual shader code.

/*
@group(0) @binding(0) var<storage, read_write> grid_possibilities: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> adjacency_rules: array<u32>; // Example layout
@group(0) @binding(2) var<uniform> grid_dims: vec3<u32>;
@group(0) @binding(3) var<uniform> num_tiles: u32;

@compute @workgroup_size(8, 8, 1) // Example workgroup size
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;

    if (x >= grid_dims.x || y >= grid_dims.y || z >= grid_dims.z) {
        return;
    }

    // TODO: Implement entropy calculation logic in WGSL
    // - Read possibilities for the current cell (x, y, z)
    // - Calculate Shannon entropy (or similar metric)
    // - Write result to an output entropy buffer (needs another binding)
}
*/

// Binding for the input possibility grid (read-only)
// Assumes possibilities are packed, e.g., one u32 per cell if <= 32 tiles
@group(0) @binding(0) var<storage, read> possibilities: array<u32>;

// Binding for the output entropy grid (write-only)
@group(0) @binding(1) var<storage, read_write> entropy: array<f32>;

// Binding for minimum entropy info [f32 bits, index] (atomic access required)
// Using array<atomic<u32>, 2> to store f32 bits and index atomically.
@group(0) @binding(3) var<storage, read_write> min_entropy_info: array<atomic<u32>, 2>;

// Specialization constant for number of u32s per cell
override NUM_TILES_U32: u32 = 1u; // Default value, MUST be overridden by pipeline

// Params struct containing grid dimensions
struct Params {
    grid_width: u32,
    grid_height: u32,
    grid_depth: u32,
    num_tiles: u32,
    // num_tiles_u32: u32, // Removed - Now specialization constant
    num_axes: u32,
    worklist_size: u32,
};
@group(0) @binding(2) var<uniform> params: Params;

// TODO: Consider passing grid dimensions (width, height, depth) via uniform buffer
// For now, assume global_invocation_id maps directly to flat grid index

// Function to atomically update the minimum entropy info
// Only updates if the new entropy is lower than the current minimum.
fn atomicMinF32Index(index: u32, entropy_value: f32) {
    let entropy_bits = bitcast<u32>(entropy_value);

    // Loop to ensure atomic update succeeds
    loop {
        // Atomically load the current minimum entropy bits and index
        let current_min_bits = atomicLoad(&min_entropy_info[0]);
        let current_min_entropy = bitcast<f32>(current_min_bits);

        // Check if the new entropy is lower, or if it's the same but the index is lower (for tie-breaking)
        // Also handle the initial state (f32::MAX)
        if (entropy_value < current_min_entropy || current_min_entropy >= 3.402823e+38) { // Check against ~f32::MAX
            // Attempt to atomically swap the value
            // atomicCompareExchangeWeak returns a vec2<u32> where [0] is old value, [1] is success flag (1 if success)
            let compare_result = atomicCompareExchangeWeak(&min_entropy_info[0], current_min_bits, entropy_bits);

            // If the swap was successful (meaning no other thread updated it in the meantime)
            if (compare_result.exchanged) {
                // Atomically store the new index (no compare needed, we won the race for entropy)
                atomicStore(&min_entropy_info[1], index);
                break; // Exit loop
            }
            // If compare failed, another thread updated, loop again to retry
        } else {
            // New entropy is not lower, no need to update
            break;
        }
    }
}

// Function to count set bits in a u32
fn count_set_bits(n: u32) -> u32 {
    var count = 0u;
    var num = n;
    while (num > 0u) {
        num = num & (num - 1u); // Clear the least significant set bit
        count = count + 1u;
    }
    return count;
}

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    // Get 1D index from thread ID
    let index = global_id.x;
    
    // Get total grid size from params
    let grid_size = params.grid_width * params.grid_height * params.grid_depth;
    
    // Bounds check to prevent out-of-bounds access
    if (index >= grid_size) {
        return;
    }

    // Possibilities are read from a separate, non-atomic buffer (`possibilities`)
    // which should already be in the correct SoA layout if prepared correctly on CPU.
    // Calculate number of cells for SoA indexing
    let num_cells = params.grid_width * params.grid_height * params.grid_depth;

    var total_possibility_mask: array<u32, 4>; // Max 128 tiles example
    var possibilities_count: u32 = 0u;

    // Read the relevant chunks for this cell using SoA indexing
    // SAFETY: Check NUM_TILES_U32 <= 4 to prevent out-of-bounds access
    if (NUM_TILES_U32 <= 4u) { 
        for (var i: u32 = 0u; i < NUM_TILES_U32; i = i + 1u) {
            // SoA Index: chunk_idx * num_cells + cell_idx
            let chunk = possibilities[i * num_cells + index];
            total_possibility_mask[i] = chunk;
            possibilities_count = possibilities_count + count_set_bits(chunk);
        }
    } else {
        // Handle error case: too many tiles for current static array size
        // Mark entropy as 0 or a special value, or potentially flag an error if possible
        possibilities_count = 0u; // Indicate error state or collapsed/contradiction
    }

    var calculated_entropy = 0.0;
    if (possibilities_count == 1u) {
        // Collapsed cell, entropy is 0 (or effectively infinite/ignored)
        calculated_entropy = 0.0;
    } else if (possibilities_count > 1u) {
        // Implement proper Shannon entropy (with uniform probability assumption):
        // H = log2(N), where N is possibilities_count
        calculated_entropy = log2(f32(possibilities_count));

        // Add a small amount of noise to break ties consistently
        // Hash the index to get a pseudo-random offset
        var hash = index;
        hash = hash ^ (hash >> 16u);
        hash = hash * 0x45d9f3b5u;
        hash = hash ^ (hash >> 16u);
        hash = hash * 0x45d9f3b5u;
        hash = hash ^ (hash >> 16u);
        let noise = f32(hash) * (1.0 / 4294967296.0); // Map hash to [0, 1)
        let noise_scale = 0.0001; // Small scale factor for noise
        calculated_entropy = calculated_entropy + noise * noise_scale;

        // Atomically update the global minimum entropy if this cell's entropy is lower
        // Only consider cells that are not yet collapsed (possibilities_count > 1)
        atomicMinF32Index(index, calculated_entropy);

    } else { // possibilities_count == 0u
        // Contradiction! Entropy is effectively infinite (use a large negative number or specific flag if needed)
        // For simplicity, setting entropy to 0 here, but contradiction handling should occur elsewhere.
        calculated_entropy = 0.0;
        // Optionally, signal contradiction via another atomic buffer if needed.
    }

    entropy[index] = calculated_entropy;
} 