// wfc-gpu/src/shaders/components/cell_collapse.wgsl
// Contains functions related to collapsing a cell to a single possibility.

// Placeholder bindings:
// @group(0) @binding(0) var<uniform> params: Params;
// @group(0) @binding(1) var<storage, read_write> grid_possibilities: array<atomic<u32>>; // Or non-atomic
// @group(0) @binding(4) var<storage, read> worklist: array<u32>; // Often used to find the cell to collapse
// @group(0) @binding(5) var<storage, read_write> output_worklist: array<atomic<u32>>;
// @group(0) @binding(6) var<storage, read_write> output_worklist_count: atomic<u32>;

// Placeholder type
alias PossibilityMask = array<u32, 1>;

// Selects a single tile to keep based on possibilities and weights/randomness.
// This is a simplified version. A real implementation would use weights and
// potentially a random number generator (passed via buffer or params).
fn select_tile_to_keep(possibilities: PossibilityMask, cell_idx: u32, /* rng_state: ptr<...> */) -> u32 {
    // Find the index of the first set bit (lowest tile index possible)
    let mask = possibilities[0];
    if (mask == 0u) { return 0u; } // Should not happen if called correctly

    // Find lowest set bit using trailingZeros (or equivalent logic)
    // Note: WGSL might not have trailingZeros directly, emulate if needed.
    // Emulation example:
    var lowest_bit_index = 0u;
    var temp_mask = mask;
    while ((temp_mask & 1u) == 0u) {
        temp_mask = temp_mask >> 1u;
        lowest_bit_index += 1u;
        if (lowest_bit_index >= 32u) { break; } // Safety break
    }

    // For now, just return the index of the lowest possible tile.
    // TODO: Implement weighted random selection based on tile weights and RNG.
    return lowest_bit_index;
}

// Collapses a cell to a single chosen tile.
// Updates the grid and potentially adds the cell to the output worklist.
// Returns true if the collapse was successful (cell wasn't already collapsed/contradicted).
fn collapse_cell_atomic(cell_idx: u32,
                       grid_possibilities: array<atomic<u32>>,
                       output_worklist_count: atomic<u32>,
                       output_worklist: array<atomic<u32>>,
                       pass_statistics: array<atomic<u32>>,
                       params: Params
                       /* rng_state: ptr<...> */) -> bool {

    let num_cells = params.grid_width * params.grid_height * params.grid_depth;
    let soa_idx = 0u * num_cells + cell_idx; // Assuming NUM_TILES_U32 = 1

    // Load current possibilities atomically
    let current_mask = atomicLoad(&grid_possibilities[soa_idx]);

    // Count possibilities
    var count = 0u;
    var temp = current_mask;
    while (temp > 0u) {
        temp = temp & (temp - 1u);
        count += 1u;
    }

    // Don't collapse if already collapsed or in contradiction
    if (count <= 1u) {
        return false;
    }

    // Select the tile to keep
    var possibilities_struct : PossibilityMask;
    possibilities_struct[0] = current_mask;
    let chosen_tile_index = select_tile_to_keep(possibilities_struct, cell_idx /*, rng_state */);
    let new_mask = 1u << chosen_tile_index;

    // Atomically update the cell's possibilities
    // Use atomicCompareExchangeWeak for efficiency if possible, or atomicStore.
    // atomicStore is simpler but might overwrite concurrent changes (less likely in collapse).
    let old_mask = atomicExchange(&grid_possibilities[soa_idx], new_mask);

    // Check if we actually changed the state (e.g., wasn't already collapsed to this tile)
    if (old_mask != new_mask) {
        // Add the collapsed cell to the worklist for propagation
        add_to_worklist_atomic(cell_idx, output_worklist_count, output_worklist, pass_statistics);
        return true;
    } else {
        return false; // No change made
    }
}

// Non-atomic version for fallback
fn collapse_cell_fallback(cell_idx: u32,
                         grid_possibilities_ptr: ptr<storage, array<u32>, read_write>,
                         output_worklist_count_ptr: ptr<storage, u32, read_write>,
                         output_worklist_ptr: ptr<storage, array<u32>, read_write>,
                         pass_statistics_ptr: ptr<storage, array<u32>, read_write>,
                         params: Params
                         /* rng_state: ptr<...> */) -> bool {

    let current_mask = (*grid_possibilities_ptr)[cell_idx]; // Assumes flat layout

    var count = 0u;
    var temp = current_mask;
    while (temp > 0u) {
        temp = temp & (temp - 1u);
        count += 1u;
    }

    if (count <= 1u) {
        return false;
    }

    var possibilities_struct : PossibilityMask;
    possibilities_struct[0] = current_mask;
    let chosen_tile_index = select_tile_to_keep(possibilities_struct, cell_idx /*, rng_state */);
    let new_mask = 1u << chosen_tile_index;

    // Non-atomically update (potential race condition!)
    if ((*grid_possibilities_ptr)[cell_idx] != new_mask) {
        (*grid_possibilities_ptr)[cell_idx] = new_mask;
        add_to_worklist_fallback(cell_idx, output_worklist_count_ptr, output_worklist_ptr, pass_statistics_ptr);
        return true;
    } else {
        return false;
    }
} 