// Shannon entropy calculation component for Wave Function Collapse
// This component handles the specific calculation of Shannon information entropy

// Calculate Shannon entropy for a given cell
// Takes the count of possibilities and calculates log2(possibilities)
fn calculate_shannon_entropy(possibility_count: u32) -> f32 {
    // Shannon entropy is just log2 of the number of possible states
    // Only calculate for cells with more than 1 possibility (uncollapsed)
    if (possibility_count <= 1u) {
        return -1.0; // Marker for already collapsed or invalid cells
    }
    
    // Information entropy calculation: log2(count)
    return log2(f32(possibility_count));
}

// Apply Shannon entropy calculation to a cell's possibility state
// This function can be used from the main entropy calculation shader
fn apply_shannon_entropy(cell_start: u32, u32s_per_cell: u32, grid_possibilities: array<u32>) -> f32 {
    // Count the total number of allowed tiles (possibilities)
    var possibility_count = 0u;
    
    // Loop through all u32s for this cell and count set bits
    for (var i = 0u; i < u32s_per_cell; i = i + 1u) {
        let value = grid_possibilities[cell_start + i];
        possibility_count += count_ones(value);
    }
    
    // Return the Shannon entropy calculation
    return calculate_shannon_entropy(possibility_count);
} 