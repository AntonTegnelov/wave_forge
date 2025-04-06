// wfc-gpu/src/shaders/components/worklist_management.wgsl
// Contains functions related to managing the propagation worklist.

// Assumes access to params uniform and worklist/output_worklist buffers.
// Placeholder bindings:
// @group(0) @binding(0) var<uniform> params: Params;
// @group(0) @binding(5) var<storage, read_write> output_worklist: array<atomic<u32>>;
// @group(0) @binding(6) var<storage, read_write> output_worklist_count: atomic<u32>;
// @group(0) @binding(9) var<storage, read_write> pass_statistics: array<atomic<u32>>;

// Adds a cell to the output worklist for future processing (from propagate_modular.wgsl)
// Requires atomic bindings for output_worklist_count and output_worklist.
fn add_to_worklist_atomic(cell_idx: u32, output_worklist_count: atomic<u32>, output_worklist: array<atomic<u32>>, pass_statistics: array<atomic<u32>>) {
    // Get current output worklist index atomically
    let worklist_idx = atomicAdd(&output_worklist_count, 1u);

    // Bounds check for worklist
    if (worklist_idx < arrayLength(&output_worklist)) {
        // Add cell to worklist atomically
        // output_worklist[worklist_idx] = cell_idx; // Non-atomic assignment if buffer isn't atomic
        atomicStore(&output_worklist[worklist_idx], cell_idx); // Assuming atomic storage buffer

        // Update statistics for current pass (atomically)
        atomicAdd(&pass_statistics[0], 1u); // Total cells added
    } else {
        // Worklist overflow - signal for host to resize buffers (atomically)
        atomicStore(&pass_statistics[3], 1u); // Overflow flag
        // Decrement count since we couldn't add - important to keep count accurate
        atomicSub(&output_worklist_count, 1u);
    }
}

// Non-atomic version for fallback implementations
// Assumes output_worklist_count and output_worklist are non-atomic storage buffers.
// NOTE: This is inherently unsafe if multiple threads write concurrently.
// It's suitable for single-threaded processing or specific fallback scenarios.
fn add_to_worklist_fallback(cell_idx: u32, output_worklist_count_ptr: ptr<storage, u32, read_write>, output_worklist_ptr: ptr<storage, array<u32>, read_write>, pass_statistics_ptr: ptr<storage, array<u32>, read_write>) {
    // Increment count non-atomically (potential race condition!)
    let worklist_idx = *output_worklist_count_ptr;
    *output_worklist_count_ptr = worklist_idx + 1u;

    let worklist_len = arrayLength(&(*output_worklist_ptr));
    if (worklist_idx < worklist_len) {
        (*output_worklist_ptr)[worklist_idx] = cell_idx;
        // Update stats non-atomically (potential race condition!)
        (*pass_statistics_ptr)[0] = (*pass_statistics_ptr)[0] + 1u;
    } else {
        // Overflow
        (*pass_statistics_ptr)[3] = 1u;
         // Revert count increment
        *output_worklist_count_ptr = worklist_idx;
    }
}


// Resets pass statistics (typically done once per dispatch)
// Requires atomic bindings for pass_statistics.
fn reset_pass_statistics_atomic(thread_idx: u32, pass_statistics: array<atomic<u32>>) {
     // Reset statistics for first thread only
    if (thread_idx == 0u) {
        atomicStore(&pass_statistics[0], 0u); // Reset cells added
        atomicStore(&pass_statistics[1], 0u); // Reset possibilities removed
        atomicStore(&pass_statistics[2], 0u); // Reset contradiction count
        atomicStore(&pass_statistics[3], 0u); // Reset overflow flag
    }
    // Ensure all threads in the workgroup see the reset before proceeding
    workgroupBarrier();
}

// Non-atomic version for fallback.
fn reset_pass_statistics_fallback(thread_idx: u32, pass_statistics_ptr: ptr<storage, array<u32>, read_write>) {
     if (thread_idx == 0u) {
        (*pass_statistics_ptr)[0] = 0u;
        (*pass_statistics_ptr)[1] = 0u;
        (*pass_statistics_ptr)[2] = 0u;
        (*pass_statistics_ptr)[3] = 0u;
    }
    // workgroupBarrier(); // Still needed if run in a workgroup context
} 