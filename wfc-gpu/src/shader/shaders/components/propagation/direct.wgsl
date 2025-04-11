// Direct propagation component for Wave Function Collapse
// This component handles the standard direct propagation algorithm
// for constraint propagation between adjacent cells

// Propagate constraints from a cell to its neighbors
// This function propagates based on adjacency rules
fn propagate_constraints(
    cell_idx: u32,              // The index of the cell being processed
    grid_width: u32,            // Width of the grid
    grid_height: u32,           // Height of the grid
    grid_depth: u32,            // Depth of the grid
    u32s_per_cell: u32,         // Number of u32s per cell
    possibilities: array<u32>,  // Grid possibilities array
    adjacency_rules: array<u32> // Adjacency rules defining allowed neighbors
) -> bool {
    // This is a placeholder for the core propagation algorithm
    // In a real implementation, this would:
    // 1. Check if this cell has been collapsed to a single possibility
    // 2. Loop through all adjacent cells
    // 3. Apply constraints based on adjacency rules
    // 4. Return whether any changes were made

    let x = cell_idx % grid_width;
    let y = (cell_idx / grid_width) % grid_height;
    let z = cell_idx / (grid_width * grid_height);
    
    // Get the starting indices for this cell's data
    let cell_start = cell_idx * u32s_per_cell;
    
    // Check if this cell has been collapsed to a single possibility
    var single_possibility_idx = -1;
    var has_single_possibility = false;
    
    // Look for a single set bit across all u32s
    for (var i = 0u; i < u32s_per_cell; i = i + 1u) {
        let cell_value = possibilities[cell_start + i];
        
        // If value is a power of 2, it has exactly one bit set
        if (cell_value != 0u && (cell_value & (cell_value - 1u)) == 0u) {
            // This u32 has exactly one bit set, make sure others are 0
            var all_others_zero = true;
            for (var j = 0u; j < u32s_per_cell; j = j + 1u) {
                if (j != i && possibilities[cell_start + j] != 0u) {
                    all_others_zero = false;
                    break;
                }
            }
            
            if (all_others_zero) {
                // Find the index of the set bit (0-31)
                let bit_pos = first_bit_position(cell_value);
                single_possibility_idx = i32(i * 32u + bit_pos);
                has_single_possibility = true;
                break;
            }
        }
    }
    
    // Only propagate constraints if this cell has a single possibility
    if (!has_single_possibility) {
        return false;
    }
    
    var changes_made = false;
    
    // Check all 6 directions for 3D (+-X, +-Y, +-Z)
    // Loop through all 6 adjacent cells (in 3D)
    let directions = array<vec3<i32>, 6>(
        vec3<i32>(-1, 0, 0),  // -X
        vec3<i32>(1, 0, 0),   // +X
        vec3<i32>(0, -1, 0),  // -Y
        vec3<i32>(0, 1, 0),   // +Y
        vec3<i32>(0, 0, -1),  // -Z
        vec3<i32>(0, 0, 1)    // +Z
    );
    
    // Loop through all 6 directions
    for (var dir_idx = 0u; dir_idx < 6u; dir_idx++) {
        let dir = directions[dir_idx];
        let nx = i32(x) + dir.x;
        let ny = i32(y) + dir.y;
        let nz = i32(z) + dir.z;
        
        // Skip if outside grid bounds
        if (nx < 0 || nx >= i32(grid_width) || 
            ny < 0 || ny >= i32(grid_height) || 
            nz < 0 || nz >= i32(grid_depth)) {
            continue;
        }
        
        let neighbor_idx = u32(nz) * grid_width * grid_height + u32(ny) * grid_width + u32(nx);
        let neighbor_start = neighbor_idx * u32s_per_cell;
        
        // Apply constraint from this cell to neighbor
        // In a real implementation, this would use adjacency_rules to determine which
        // possibilities should be removed from the neighbor based on this cell's state
        // For demonstration, just showing how it would be structured
        
        // Example: Look up allowed neighbors in adjacency rules
        // let rule_idx = (single_possibility_idx * 6) + dir_idx;
        // let allowed_neighbors = adjacency_rules[rule_idx];
        
        // For each u32 in the neighbor cell, apply constraints
        for (var i = 0u; i < u32s_per_cell; i++) {
            // Get current possibilities
            let current = possibilities[neighbor_start + i];
            
            // Apply constraint (in a real implementation)
            // let new_value = current & allowed_neighbors;
            let new_value = current; // Placeholder
            
            // If changed, update and mark changes
            if (new_value != current) {
                // Would update possibilities[neighbor_start + i] = new_value;
                changes_made = true;
            }
        }
    }
    
    return changes_made;
}

// Helper function: Find position of first set bit
fn first_bit_position(value: u32) -> u32 {
    if (value == 0u) {
        return 0u;
    }
    
    var temp = value;
    var pos = 0u;
    
    // Binary search for the first set bit
    if ((temp & 0xFFFFu) == 0u) { pos += 16u; temp >>= 16u; }
    if ((temp & 0xFFu) == 0u) { pos += 8u; temp >>= 8u; }
    if ((temp & 0xFu) == 0u) { pos += 4u; temp >>= 4u; }
    if ((temp & 0x3u) == 0u) { pos += 2u; temp >>= 2u; }
    if ((temp & 0x1u) == 0u) { pos += 1u; }
    
    return pos;
} 