// Worklist Management Component
// Manages the worklist of cells that need to be processed during propagation

// Import required utilities
@import "utils.wgsl"

// Worklist management functions
fn add_to_worklist(cell_index: u32) -> bool {
    // Try to add the cell to the worklist if it's not already there
    let was_in_worklist = atomic_load(&worklist_flags[cell_index]);
    if was_in_worklist == 0u {
        // Atomically set the flag and add to worklist
        atomic_store(&worklist_flags[cell_index], 1u);
        let worklist_pos = atomic_add(&worklist_count, 1u);
        if worklist_pos < max_worklist_size {
            worklist[worklist_pos] = cell_index;
            return true;
        }
    }
    return false;
}

fn clear_worklist_flag(cell_index: u32) {
    atomic_store(&worklist_flags[cell_index], 0u);
}

// Main worklist management workgroup entry point
@compute @workgroup_size(256)
fn process_worklist(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let worklist_index = global_id.x;
    if worklist_index >= worklist_count {
        return;
    }

    let cell_index = worklist[worklist_index];
    clear_worklist_flag(cell_index);
} 