// Count-based entropy calculation component for Wave Function Collapse
// This component handles the specific calculation of count-based entropy

// Calculate standard count-based entropy (possibilities - 1)
// This approach emphasizes cells with fewer possibilities
fn calculate_count_entropy(possibility_count: u32) -> f32 {
    // Only calculate for cells with more than 1 possibility (uncollapsed)
    if (possibility_count <= 1u) {
        return -1.0; // Marker for already collapsed or invalid cells
    }
    
    // Count-based entropy is simply the count of possibilities minus 1
    // Subtracting 1 ensures collapsed cells (with count=1) get entropy=0
    return f32(possibility_count - 1u);
}

// Calculate normalized count-based entropy (possibilities / total_tiles)
// This approach normalizes by the total possible tiles
fn calculate_normalized_count_entropy(possibility_count: u32, num_tiles: u32) -> f32 {
    // Only calculate for cells with more than 1 possibility (uncollapsed)
    if (possibility_count <= 1u) {
        return -1.0; // Marker for already collapsed or invalid cells
    }
    
    // Normalized count divides by the total number of possible tiles
    return f32(possibility_count) / f32(num_tiles);
}

// Apply count-based entropy calculation to a cell's possibility state
fn apply_count_entropy(cell_start: u32, u32s_per_cell: u32, grid_possibilities: array<u32>) -> f32 {
    // Count the total number of allowed tiles (possibilities)
    var possibility_count = 0u;
    
    // Loop through all u32s for this cell and count set bits
    for (var i = 0u; i < u32s_per_cell; i = i + 1u) {
        let value = grid_possibilities[cell_start + i];
        possibility_count += count_ones(value);
    }
    
    // Return the count-based entropy calculation
    return calculate_count_entropy(possibility_count);
}

// Apply normalized count-based entropy calculation to a cell's possibility state
fn apply_normalized_count_entropy(cell_start: u32, u32s_per_cell: u32, grid_possibilities: array<u32>, num_tiles: u32) -> f32 {
    // Count the total number of allowed tiles (possibilities)
    var possibility_count = 0u;
    
    // Loop through all u32s for this cell and count set bits
    for (var i = 0u; i < u32s_per_cell; i = i + 1u) {
        let value = grid_possibilities[cell_start + i];
        possibility_count += count_ones(value);
    }
    
    // Return the normalized count-based entropy calculation
    return calculate_normalized_count_entropy(possibility_count, num_tiles);
} 