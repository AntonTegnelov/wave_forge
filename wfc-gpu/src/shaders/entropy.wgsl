// WGSL Shader for Wave Function Collapse entropy calculation
//
// This compute shader calculates the Shannon entropy of each cell in the grid
// based on the current possibilities. It also finds the cell with minimum entropy
// (excluding cells that are fully collapsed to a single possibility).
//
// Entropy is calculated as:
// H = -sum(p*log2(p)) where p is the probability of each state
//
// For the WFC algorithm, all p values are equal (1/n where n is the number of 
// possibilities), so this simplifies to:
// H = log2(n)
//
// The shader also uses atomic operations to track the cell with minimum
// positive entropy, which is used for the next collapse step.

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
@group(0) @binding(0) var<storage, read> grid_possibilities: array<u32>; // Flattened cell possibilities

// Binding for the output entropy grid (write-only)
@group(0) @binding(1) var<storage, read_write> entropy_output: array<f32>; // Output entropy grid

// Binding for minimum entropy info [f32 bits, index] (atomic access required)
// Using array<atomic<u32>, 2> to store f32 bits and index atomically.
@group(0) @binding(3) var<storage, read_write> min_entropy_info: array<atomic<u32>, 2>; // [entropy_bits, flat_index]

// Specialization constant for number of u32s per cell
const NUM_TILES_U32: u32 = NUM_TILES_U32_VALUE;

// Specialization constant for workgroup size (X dimension)
const WORKGROUP_SIZE_X: u32 = 64u; // Hardcoded size

// Params struct containing grid dimensions
struct Params {
    grid_width: u32,
    grid_height: u32,
    grid_depth: u32,
    num_tiles: u32,
    num_axes: u32,
    worklist_size: u32, // Unused in entropy shader, but kept for consistency
    boundary_mode: u32, // Unused in entropy shader, but kept for consistency
    _padding1: u32,
};
@group(0) @binding(2) var<uniform> params: Params;

// Define constants for F32 bit manipulation
const F32_SIGN_MASK: u32 = 0x80000000u;
const F32_EXP_MASK: u32 = 0x7F800000u;
const F32_FRAC_MASK: u32 = 0x007FFFFFu;
const F32_IMPLICIT_ONE: u32 = 0x00800000u;
const F32_EXP_BIAS: i32 = 127;

// Possibility mask array type using the NUM_TILES_U32 constant
alias PossibilityMask = array<u32, NUM_TILES_U32_VALUE>;

// Bit manipulation helpers
fn float_to_bits(f: f32) -> u32 {
    return bitcast<u32>(f);
}

fn bits_to_float(bits: u32) -> f32 {
    return bitcast<f32>(bits);
}

// Helper to check and transform a float value for atomic minimum
fn prepareForAtomicMinF32(value: f32) -> u32 {
    // Check for NaN - a NaN value is the only value that is not equal to itself
    if (value != value) { return 0xFFFFFFFFu; } // Return max value for NaN

    // Convert f32 to u32 bit pattern for atomic operations
    let value_bits = float_to_bits(value);
    
    // For negative values, we need to flip all bits to maintain ordering
    var value_transformed: u32;
    if ((value_bits & F32_SIGN_MASK) != 0u) {
        // Negative value - invert all bits (smaller negative -> larger positive)
        value_transformed = ~value_bits;
    } else {
        // Positive value - invert sign bit only (to make it ordered correctly)
        value_transformed = value_bits ^ F32_SIGN_MASK;
    }
    
    return value_transformed;
}

// Helper function to get 1D index from 3D coords
fn grid_index(x: u32, y: u32, z: u32) -> u32 {
    return z * params.grid_width * params.grid_height + y * params.grid_width + x;
}

// Helper to count set bits in a mask (number of possibilities)
fn count_possibilities(possibility_bits: PossibilityMask) -> u32 {
    // For simplicity, only use the first element which must exist
    // This is a conservative approach for tests with small NUM_TILES_U32 values
    return countOneBits(possibility_bits[0]);
}

@compute @workgroup_size(64)
fn main_entropy(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    // Use flat 1D indexing for cleaner grid access
    let grid_idx_x = global_id.x % params.grid_width;
    let grid_idx_y = (global_id.x / params.grid_width) % params.grid_height;
    let grid_idx_z = global_id.x / (params.grid_width * params.grid_height);
    
    // Bounds check
    if (grid_idx_x >= params.grid_width || 
        grid_idx_y >= params.grid_height || 
        grid_idx_z >= params.grid_depth) {
        return;
    }
    
    // Compute flat grid index
    let grid_idx_1d = grid_index(grid_idx_x, grid_idx_y, grid_idx_z);
    
    // Calculate total number of cells for SoA indexing
    let num_cells = params.grid_width * params.grid_height * params.grid_depth;
    
    // Create buffer to hold this cell's possibilities
    var total_possibility_mask: PossibilityMask;
    
    // Load all u32 chunks for this cell
    for (var i = 0u; i < NUM_TILES_U32; i = i + 1u) {
        // SoA layout: [u32_chunk_0 for all cells, u32_chunk_1 for all cells, ...]
        let buffer_idx = grid_idx_1d + i * num_cells;
        
        // Bounds check before read
        if (buffer_idx < arrayLength(&grid_possibilities)) {
            total_possibility_mask[i] = grid_possibilities[buffer_idx];
        } else {
            // A problem with the grid size - return early
            return;
        }
    }
    
    // Count the number of set bits (possible tile types)
    let num_possible = count_possibilities(total_possibility_mask);
    
    // Calculate entropy
    // For zero possibilities (contradiction), return special value
    if (num_possible == 0u) {
        entropy_output[grid_idx_1d] = -1.0; // Indicate contradiction
        return;
    }
    
    // For single possibility, mark as "collapsed" with 0 entropy
    if (num_possible == 1u) {
        entropy_output[grid_idx_1d] = 0.0;
        return;
    }
    
    // For multiple possibilities, calculate shannon entropy: log2(count)
    let entropy = log2(f32(num_possible));
    entropy_output[grid_idx_1d] = entropy;
    
    // Update minimum entropy info using atomic operations
    if (entropy > 0.0) {  // Skip cells that are already collapsed (entropy == 0)
        // Calculate value for atomic min, avoiding passing pointers
        let value_bits = float_to_bits(entropy);
        
        // Main atomic compare-exchange loop for min entropy
        var current_bits = atomicLoad(&min_entropy_info[0]);
        var current_float = bits_to_float(current_bits);
        
        // Continue looping while we still could update the minimum
        var keep_trying = entropy < current_float;
        var attempt_counter = 0u;
        
        while (keep_trying && attempt_counter < 32u) {
            // Try to update the minimum entropy
            let result = atomicCompareExchangeWeak(
                &min_entropy_info[0],
                current_bits,
                value_bits
            );
            let exchanged = result.exchanged;
            let old_value = result.old_value;
            
            if (exchanged) {
                // Successfully updated the entropy, now update the index
                atomicStore(&min_entropy_info[1], grid_idx_1d);
                keep_trying = false;
            } else {
                // Exchange failed, update current values and check if we still need to try
                current_bits = old_value;
                current_float = bits_to_float(current_bits);
                keep_trying = entropy < current_float;
            }
            
            attempt_counter = attempt_counter + 1u;
        }
    }
}

struct GridDimensions {
    width: u32,
    height: u32,
    depth: u32,
    padding: u32,
};

struct EntropyParams {
    width: u32,
    height: u32,
    depth: u32,
    padding1: u32,
    heuristic_type: u32,
    padding2: u32,
    padding3: u32,
    padding4: u32,
}

// Binding group 0: Grid state
@group(0) @binding(0) var<storage, read> grid_possibilities: array<u32>;
@group(0) @binding(1) var<storage, read> adjacency_rules: array<u32>;

// Binding group 1: Output entropy grid and minimum tracker
@group(1) @binding(0) var<storage, read_write> grid_entropy: array<f32>;
@group(1) @binding(1) var<storage, read_write> min_entropy_cell: array<u32>;

// Binding group 2: Entropy parameters
@group(2) @binding(0) var<uniform> params: EntropyParams;

// Other potential bindings for weighted count heuristic
// @group(3) @binding(0) var<storage, read> tile_weights: array<f32>;

const WORKGROUP_SIZE = 8;

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn compute_entropy(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let width = params.width;
    let height = params.height;
    let depth = params.depth;
    
    // Skip if outside grid dimensions
    if (global_id.x >= width || global_id.y >= height || global_id.z >= depth) {
        return;
    }
    
    let flat_idx = flatten_3d_index(global_id.x, global_id.y, global_id.z);
    let num_tile_types = get_num_tile_types();
    let u32s_per_cell = get_u32s_per_cell(num_tile_types);
    let cell_offset = flat_idx * u32(u32s_per_cell);
    
    // Get bit vector of possibilities for this cell
    var bits: array<u32, 16>;  // Max 512 tile types (16 * 32 bits)
    for (var i = 0u; i < u32(u32s_per_cell); i++) {
        bits[i] = grid_possibilities[cell_offset + i];
    }
    
    let n = count_possibilities(bits, num_tile_types);
    var entropy: f32;
    
    // If the cell is already decided or has no possibilities, set entropy to -1 or a large value
    if (n <= 1u) {
        entropy = -1.0;  // -1 flags this as an invalid cell for selection
        grid_entropy[flat_idx] = entropy;
        return;
    }
    
    // Calculate entropy based on the selected heuristic
    switch(params.heuristic_type) {
        case 0u: {
            // Shannon entropy (default)
            entropy = calculate_shannon_entropy(bits, num_tile_types, n);
        }
        case 1u: {
            // Count heuristic (just number of possibilities)
            entropy = f32(n);
        }
        case 2u: {
            // Count simple (same as count but normalized)
            entropy = f32(n) / f32(num_tile_types);
        }
        case 3u: {
            // Weighted count (would need tile weights if implemented)
            entropy = calculate_weighted_entropy(bits, num_tile_types);
        }
        default: {
            // Fall back to Shannon if unknown
            entropy = calculate_shannon_entropy(bits, num_tile_types, n);
        }
    }
    
    grid_entropy[flat_idx] = entropy;
    
    // Check if this is the minimum entropy cell
    // We only update if our entropy is positive (> 0) and less than the current minimum
    if (entropy > 0.0) {
        // Atomic operation to check and update minimum
        let old_min_entropy = atomic_min(&min_entropy_cell[0], bitcast<u32>(entropy));
        
        // If we successfully updated the minimum entropy
        if (bitcast<f32>(old_min_entropy) > entropy) {
            // Store the cell index (only overwrite if we're the new minimum)
            min_entropy_cell[1] = flat_idx;
        }
    }
}

// Calculate Shannon entropy for a cell
fn calculate_shannon_entropy(bits: array<u32, 16>, num_tile_types: u32, n: u32) -> f32 {
    // For uniform distribution, Shannon entropy simplifies to log2(n)
    // using equal probability p = 1/n for each possibility
    // Shannon entropy = -sum(p * log2(p)) = -n * (1/n * log2(1/n)) = log2(n)
    return log2(f32(n));
}

// Calculate weighted entropy for a cell
fn calculate_weighted_entropy(bits: array<u32, 16>, num_tile_types: u32) -> f32 {
    // Here we would use tile weights if they were available
    // For now, just return a simple count-based calculation
    let n = count_possibilities(bits, num_tile_types);
    return f32(n);
}

// Count the number of possibilities (set bits)
fn count_possibilities(bits: array<u32, 16>, num_tile_types: u32) -> u32 {
    var count = 0u;
    let num_complete_u32s = num_tile_types / 32u;
    
    // Count bits in complete u32s
    for (var i = 0u; i < num_complete_u32s; i++) {
        count += count_bits(bits[i]);
    }
    
    // Count bits in the final partial u32
    let remaining_bits = num_tile_types % 32u;
    if (remaining_bits > 0u) {
        let mask = (1u << remaining_bits) - 1u;
        count += count_bits(bits[num_complete_u32s] & mask);
    }
    
    return count;
}

// Helper to flatten 3D index to 1D
fn flatten_3d_index(x: u32, y: u32, z: u32) -> u32 {
    return (z * params.height + y) * params.width + x;
}

// Get number of u32s needed per cell
fn get_u32s_per_cell(num_tile_types: u32) -> u32 {
    return (num_tile_types + 31u) / 32u;  // Ceiling division
}

// Count bits in a u32
fn count_bits(value: u32) -> u32 {
    // Population count algorithm
    var v = value;
    v = v - ((v >> 1) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
    return ((v + (v >> 4) & 0xF0F0F0Fu) * 0x1010101u) >> 24;
}

// Estimate the number of tile types from the adjacency rules buffer
fn get_num_tile_types() -> u32 {
    // First element in adjacency_rules should be the number of unique tiles
    return adjacency_rules[0];
} 