// WGSL Shader for Wave Function Collapse entropy calculation (Fallback version)
//
// This is a simplified version of the entropy calculation shader that does not
// use atomics. It calculates entropy values but doesn't find the minimum entropy
// cell directly on the GPU. Instead, finding the minimum entropy cell is performed
// on the CPU after reading back the entropy buffer.
//
// This provides better compatibility with hardware that doesn't support atomics,
// at the cost of additional CPU-GPU synchronization.

// Binding for the input possibility grid (read-only)
// Assumes possibilities are packed, e.g., one u32 per cell if <= 32 tiles
@group(0) @binding(0) var<storage, read> possibilities: array<u32>;

// Binding for the output entropy grid (write-only)
@group(0) @binding(1) var<storage, read_write> entropy: array<f32>;

// No atomic buffer for minimum entropy in this fallback version

// Specialization constant for number of u32s per cell
override NUM_TILES_U32: u32 = 1u; // Default value, MUST be overridden by pipeline

// Specialization constant for workgroup size (X dimension)
override WORKGROUP_SIZE_X: u32 = 64u; // Default value, can be overridden by pipeline creation

// Params struct containing grid dimensions
struct Params {
    grid_width: u32,
    grid_height: u32,
    grid_depth: u32,
    num_tiles: u32,
    num_axes: u32,
    worklist_size: u32,
};
@group(0) @binding(2) var<uniform> params: Params;

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

@compute @workgroup_size(64) // Use hardcoded size
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
        possibilities_count = 0u; // Indicate error state or collapsed/contradiction
    }

    var calculated_entropy = 0.0;
    if (possibilities_count == 1u) {
        // Collapsed cell, entropy is 0
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

        // NO ATOMIC MINIMUM CALCULATION IN FALLBACK VERSION
        // The minimum entropy cell will be found on the CPU
    } else { // possibilities_count == 0u
        // Contradiction! 
        calculated_entropy = 0.0;
    }

    entropy[index] = calculated_entropy;
} 