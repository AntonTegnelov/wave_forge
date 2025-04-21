// Contradiction Detection Component
// Detects contradictions in the wave function collapse grid

// Import required utilities
@import "utils.wgsl"

// Contradiction detection function
fn detect_contradiction(cell_possibilities: u32) -> bool {
    // A cell has a contradiction if it has no possible values (all bits are 0)
    return cell_possibilities == 0u;
}

// Main contradiction detection workgroup entry point
@compute @workgroup_size(256)
fn check_contradictions(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let cell_index = global_id.x + global_id.y * grid_size.x + global_id.z * grid_size.x * grid_size.y;
    if cell_index >= total_cells {
        return;
    }

    let possibilities = grid[cell_index];
    if detect_contradiction(possibilities) {
        // Set the contradiction flag for this cell
        atomic_store(&contradiction_flags[cell_index], 1u);
    }
} 