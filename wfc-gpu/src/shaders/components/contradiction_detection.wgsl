// wfc-gpu/src/shaders/components/contradiction_detection.wgsl
// Contains functions related to detecting and handling contradictions.

// Placeholder bindings:
// @group(0) @binding(7) var<storage, read_write> contradiction_flag: atomic<u32>; // Or non-atomic
// @group(0) @binding(8) var<storage, read_write> contradiction_location: atomic<u32>; // Or non-atomic
// @group(0) @binding(9) var<storage, read_write> pass_statistics: array<atomic<u32>>; // Or non-atomic

// Signals that a contradiction has been detected.
// Uses atomic operations for thread-safe signaling.
fn signal_contradiction_atomic(cell_idx: u32,
                              contradiction_flag: atomic<u32>,
                              contradiction_location: atomic<u32>,
                              pass_statistics: array<atomic<u32>>) {
    // Set the global contradiction flag
    atomicStore(&contradiction_flag, 1u);

    // Attempt to store the location of the first contradiction found
    // Use atomicMin to record the lowest index where contradiction occurred.
    // Initialize the location to a max value if needed.
    atomicMin(&contradiction_location, cell_idx);

    // Increment contradiction count in statistics
    atomicAdd(&pass_statistics[2], 1u);
}

// Non-atomic version for fallback implementations.
// NOTE: Potential race conditions if used concurrently.
fn signal_contradiction_fallback(cell_idx: u32,
                                contradiction_flag_ptr: ptr<storage, u32, read_write>,
                                contradiction_location_ptr: ptr<storage, u32, read_write>,
                                pass_statistics_ptr: ptr<storage, array<u32>, read_write>) {
    *contradiction_flag_ptr = 1u;

    // Update location non-atomically (might not be the *first* one detected)
    if (*contradiction_location_ptr == 0xffffffffu || cell_idx < *contradiction_location_ptr) {
         *contradiction_location_ptr = cell_idx;
    }

    // Update stats non-atomically
    (*pass_statistics_ptr)[2] = (*pass_statistics_ptr)[2] + 1u;
}

// Checks if a contradiction has been signaled globally.
// Used for early termination in propagation loops.
fn check_contradiction_atomic(contradiction_flag: atomic<u32>) -> bool {
    return atomicLoad(&contradiction_flag) == 1u;
}

// Non-atomic version for fallback.
fn check_contradiction_fallback(contradiction_flag_ptr: ptr<storage, u32, read>) -> bool {
     return *contradiction_flag_ptr == 1u;
}

// Example usage within a propagation step (part of update_neighbor_possibilities)
// This shows where contradiction detection logic typically fits.
fn update_neighbor_and_detect_contradiction_atomic(
    neighbor_idx: u32,
    allowed_mask: u32, // Simplified: single u32 mask
    grid_possibilities: array<atomic<u32>>, // The main grid
    contradiction_flag: atomic<u32>,
    contradiction_location: atomic<u32>,
    pass_statistics: array<atomic<u32>>
    // ... other params ...
) -> bool { // Returns true if a change was made

    // Placeholder: Assume SoA layout and calculate index
    let soa_idx = neighbor_idx; // Simplified index

    // --- Core Logic --- 
    // Atomically update the neighbor's mask by ANDing with the allowed mask.
    // Use atomicAnd or a compare-and-swap loop.
    let old_neighbor_mask = atomicAnd(&grid_possibilities[soa_idx], allowed_mask);

    // Calculate the mask *after* the update (since atomicAnd returns the *old* value)
    let new_neighbor_mask = old_neighbor_mask & allowed_mask;

    // Check if any change occurred
    let changed = (new_neighbor_mask != old_neighbor_mask);

    // Check for contradiction: if the new mask is zero AND the old mask was non-zero
    if (new_neighbor_mask == 0u && old_neighbor_mask != 0u) {
        signal_contradiction_atomic(neighbor_idx, contradiction_flag, contradiction_location, pass_statistics);
    }

    // Update statistics if change occurred (e.g., possibilities removed)
    if (changed) {
         // Count bits removed (requires bit count function)
        // let removed_count = count_bits(old_neighbor_mask) - count_bits(new_neighbor_mask);
        // atomicAdd(&pass_statistics[1], removed_count);
    }

    return changed;
} 