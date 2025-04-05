// Entropy calculation shader for Wave Function Collapse
// This shader calculates the Shannon entropy of each cell in the grid
// and finds the cell with the minimum positive entropy.

// Bind group 0: Grid state
@group(0) @binding(0) var<storage, read> grid_possibilities: array<u32>;
@group(0) @binding(1) var<storage, read> rules: array<u32>;

// Bind group 1: Output
@group(1) @binding(0) var<storage, read_write> entropy_grid: array<f32>;
@group(1) @binding(1) var<storage, read_write> min_entropy_info: array<u32>;

// Bind group 2: Parameters
@group(2) @binding(0) var<uniform> params: array<u32, 8>;
// params[0] = width
// params[1] = height
// params[2] = depth
// params[3] = padding
// params[4] = entropy_heuristic (0=Shannon, 1=Count, 2=CountSimple, 3=WeightedCount)
// params[5-7] = padding

const WORKGROUP_SIZE = 8;

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Extract grid dimensions from params
    let width = params[0];
    let height = params[1];
    let depth = params[2];
    let heuristic_type = params[4];
    
    // Skip if outside grid bounds
    if (global_id.x >= width || global_id.y >= height || global_id.z >= depth) {
        return;
    }
    
    // Get flat index in grid
    let flat_idx = (global_id.z * height + global_id.y) * width + global_id.x;
    
    // Get number of tile types from rules buffer
    let num_tiles = rules[0];
    
    // Calculate number of u32s per cell
    let u32s_per_cell = (num_tiles + 31u) / 32u;
    
    // Calculate entropy for this cell
    let cell_start = flat_idx * u32s_per_cell;
    
    // Count possibilities in this cell
    var possibilities = 0u;
    for (var i = 0u; i < u32s_per_cell; i = i + 1u) {
        let bits = grid_possibilities[cell_start + i];
        possibilities = possibilities + count_ones(bits);
    }
    
    var entropy: f32;
    
    // Calculate entropy based on heuristic type
    if (possibilities <= 1u) {
        // Zero or one possibility (collapsed or contradiction)
        entropy = -1.0;
    } else {
        switch (heuristic_type) {
            case 0u: {
                // Shannon entropy: log2(n)
                entropy = log2(f32(possibilities));
            }
            case 1u: {
                // Count heuristic: n-1 (count minus 1)
                entropy = f32(possibilities - 1u);
            }
            case 2u: {
                // CountSimple: n/total (normalized count)
                entropy = f32(possibilities) / f32(num_tiles);
            }
            case 3u: {
                // WeightedCount: fall back to count for now
                entropy = f32(possibilities - 1u);
            }
            default: {
                // Default to Shannon entropy
                entropy = log2(f32(possibilities));
            }
        }
    }
    
    // Write entropy to output grid
    entropy_grid[flat_idx] = entropy;
    
    // Update minimum entropy if this is positive and less than current minimum
    if (entropy > 0.0) {
        // Atomic_min takes u32, but we need to compare floats
        // Convert float to u32 in a way that preserves ordering
        let entropy_bits = bitcast<u32>(entropy);
        
        // Use atomic min to update the min entropy
        let old_min = atomicMin(&min_entropy_info[0], entropy_bits);
        
        // If we just set a new minimum, also update the cell index
        if (entropy_bits < old_min) {
            min_entropy_info[1] = flat_idx;
        }
    }
}

// Count number of 1 bits in a u32
fn count_ones(x: u32) -> u32 {
    var bits = x;
    bits = bits - ((bits >> 1) & 0x55555555u);
    bits = (bits & 0x33333333u) + ((bits >> 2) & 0x33333333u);
    bits = (bits + (bits >> 4)) & 0x0F0F0F0Fu;
    bits = bits + (bits >> 8);
    bits = bits + (bits >> 16);
    return bits & 0x0000003Fu;
} 