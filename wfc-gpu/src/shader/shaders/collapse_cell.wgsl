// wfc-gpu/src/shader/shaders/collapse_cell.wgsl
// Shader to collapse a single cell to a specific tile ID.

struct Params {
    grid_dims: vec3<u32>,
    heuristic_type: u32, // Not used here, but part of standard Params
    num_tiles: u32,
    u32s_per_cell: u32,
    boundary_mode: u32, // Not used here
    tie_breaking: u32, // Not used here
    max_propagation_steps: u32, // Not used here
    contradiction_check_frequency: u32, // Not used here
    worklist_size: u32, // Not used here
    grid_element_count: u32, // Not used here
    _padding: u32, // Ensure correct struct size/alignment
};

struct CollapseInfo {
    coord_x: u32,
    coord_y: u32,
    coord_z: u32,
    chosen_tile_id: u32,
};

// Bind group 0: Grid possibilities (read/write), Params (read), CollapseInfo (read)
@group(0) @binding(0) var<storage, read_write> grid_possibilities: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<uniform> collapse_info: CollapseInfo;

@compute @workgroup_size(1, 1, 1) // Only need a single invocation
fn main() {
    let width = params.grid_dims.x;
    let height = params.grid_dims.y;
    let u32s_per_cell = params.u32s_per_cell;

    let x = collapse_info.coord_x;
    let y = collapse_info.coord_y;
    let z = collapse_info.coord_z;
    let tile_id = collapse_info.chosen_tile_id;

    // Calculate the flat index for the cell
    let flat_idx = (z * height + y) * width + x;
    let cell_start = flat_idx * u32s_per_cell;

    // Calculate the specific u32 chunk and bit index for the chosen tile
    let u32_idx = tile_id / 32u;
    let bit_idx = tile_id % 32u;

    // Ensure we don't write out of bounds for the cell's data
    if (u32_idx >= u32s_per_cell) {
        // This shouldn't happen if tile_id is valid
        return;
    }

    // Create the new possibility mask (only the chosen tile is possible)
    let new_mask = 1u << bit_idx;

    // Update the grid possibilities buffer
    // Set the target chunk to the new mask
    grid_possibilities[cell_start + u32_idx] = new_mask;

    // Zero out all other chunks for this cell
    for (var i = 0u; i < u32s_per_cell; i = i + 1u) {
        if (i != u32_idx) {
            grid_possibilities[cell_start + i] = 0u;
        }
    }
} 