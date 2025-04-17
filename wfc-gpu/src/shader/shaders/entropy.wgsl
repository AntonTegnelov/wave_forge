// Entropy calculation shader for Wave Function Collapse
// This shader calculates the entropy of each cell in the grid
// and finds the cell with the minimum positive entropy using workgroup shared memory optimization.

// Bind group 0: Grid state & Parameters
struct Params {
    grid_dims: vec3<u32>,
    heuristic_type: u32,
    num_tiles: u32,
    u32s_per_cell: u32,
    // Add padding or other params if needed
};
@group(0) @binding(0) var<storage, read> grid_possibilities: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

// Bind group 1: Output & Global Minimum
@group(1) @binding(0) var<storage, read_write> entropy_grid: array<f32>; // For debug or other uses
@group(1) @binding(1) var<storage, read_write> min_entropy_info: array<atomic<u32>, 2>; // [0] = min_entropy_bits, [1] = flat_index

// Shared memory for workgroup reduction
struct MinInfo {
    entropy_bits: u32,
    flat_index: u32,
};
var<workgroup> local_min_info: array<MinInfo, WORKGROUP_SIZE * WORKGROUP_SIZE>; // Size based on 2D workgroup

const WORKGROUP_SIZE = 8u;
const MAX_FLOAT_BITS = 0x7F7FFFFFu; // ~f32.max as u32 bits

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1u)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    // Extract grid dimensions
    let width = params.grid_dims.x;
    let height = params.grid_dims.y;
    let depth = params.grid_dims.z;
    let heuristic_type = params.heuristic_type;
    let num_tiles = params.num_tiles;
    let u32s_per_cell = params.u32s_per_cell;

    var current_min_entropy_bits = MAX_FLOAT_BITS;
    var current_min_index = 0xFFFFFFFFu; // u32::MAX

    // --- Calculate Entropy for this invocation's cell --- 
    var entropy = -1.0f;
    if (global_id.x < width && global_id.y < height && global_id.z < depth) {
        let flat_idx = (global_id.z * height + global_id.y) * width + global_id.x;
        let cell_start = flat_idx * u32s_per_cell;

        // Count possibilities
        var possibilities = 0u;
        for (var i = 0u; i < u32s_per_cell; i = i + 1u) {
            possibilities += count_ones(grid_possibilities[cell_start + i]);
        }

        // Calculate entropy based on heuristic
        if (possibilities > 1u) {
             switch (heuristic_type) {
                 case 0u: { entropy = log2(f32(possibilities)); }
                 case 1u: { entropy = f32(possibilities - 1u); }
                 case 2u: { entropy = f32(possibilities) / f32(num_tiles); }
                 case 3u: { entropy = f32(possibilities - 1u); } // WeightedCount fallback
                 default: { entropy = log2(f32(possibilities)); }
             }
        }

        // Write to output grid (optional, can be removed if only min is needed)
        entropy_grid[flat_idx] = entropy;

        // Prepare for reduction: only consider positive entropy
        if (entropy > 0.0f) {
            current_min_entropy_bits = bitcast<u32>(entropy);
            current_min_index = flat_idx;
        }
    }

    // --- Workgroup Reduction --- 
    // Initialize shared memory for this invocation
    local_min_info[local_index].entropy_bits = current_min_entropy_bits;
    local_min_info[local_index].flat_index = current_min_index;

    // Synchronize all threads within the workgroup
    workgroupBarrier();

    // Perform reduction in shared memory
    // Assumes workgroup size is power of 2 (e.g., 64 for 8x8)
    let wg_num_threads = WORKGROUP_SIZE * WORKGROUP_SIZE; // Total threads in 2D workgroup
    for (var stride = wg_num_threads / 2u; stride > 0u; stride >>= 1u) {
        if (local_index < stride) {
            let other_index = local_index + stride;
            let other_info = local_min_info[other_index];
            let current_info = local_min_info[local_index];

            // If other thread has smaller entropy, update current thread's min
            if (other_info.entropy_bits < current_info.entropy_bits) {
                local_min_info[local_index] = other_info;
            }
            // Tie-breaking: if entropies are equal, choose lower index
            else if (other_info.entropy_bits == current_info.entropy_bits && other_info.flat_index < current_info.flat_index) {
                 local_min_info[local_index] = other_info;
            }
        }
        // Synchronize after each reduction step
        workgroupBarrier();
    }

    // --- Atomic Update to Global Minimum --- 
    // Only the first thread in the workgroup performs the atomic update
    if (local_index == 0u) {
        let local_best = local_min_info[0];
        
        // Only attempt atomic update if the workgroup found a valid minimum
        if (local_best.entropy_bits < MAX_FLOAT_BITS) {
            // Loop to ensure atomic update succeeds (CAS-like approach)
            // Read current global minimum
            var current_global_min_bits = atomicLoad(&min_entropy_info[0]);

            while (local_best.entropy_bits < current_global_min_bits) {
                // Attempt to atomically swap if our local minimum is still better
                let old_val = atomicCompareExchangeWeak(
                    &min_entropy_info[0], 
                    current_global_min_bits, // Expected current value
                    local_best.entropy_bits // New value if swap occurs
                );

                // Check if the swap succeeded (old_val.exchanged == true)
                // If swap succeeded, update the index atomically as well
                if (old_val.exchanged) {
                     // Atomically store the index corresponding to the new minimum entropy
                     // Note: This isn't strictly race-free if another workgroup updates
                     // min_entropy_info[0] between our CAS and this store, but it's usually good enough.
                     // A true atomic 64-bit CAS for both values would be better if available/needed.
                     atomicStore(&min_entropy_info[1], local_best.flat_index);
                    break; // Exit loop after successful update
                }
                 else {
                    // Swap failed, means another thread updated the global minimum.
                    // Reload the global minimum and retry the comparison.
                    current_global_min_bits = old_val.old_value;
                }
            }
             // Tie-breaking at the global level (optional, but good practice)
             // If entropies are equal, prefer the lower index
             if (local_best.entropy_bits == current_global_min_bits) {
                 // Atomically update the index only if our index is lower
                 var current_global_index = atomicLoad(&min_entropy_info[1]);
                 while (local_best.flat_index < current_global_index) {
                      let old_idx_val = atomicCompareExchangeWeak(
                          &min_entropy_info[1],
                          current_global_index,
                          local_best.flat_index
                      );
                      if (old_idx_val.exchanged) {
                          break;
                      }
                      else {
                           current_global_index = old_idx_val.old_value;
                      }
                 }
             }
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