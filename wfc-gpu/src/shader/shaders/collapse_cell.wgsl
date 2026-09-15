// wfc-gpu/src/shader/shaders/collapse_cell.wgsl
// Shader to collapse a single cell to a specific tile ID.

// Mirrors GpuParamsUniform in buffers/mod.rs field for field
struct Params {
    grid_width: u32,
    grid_height: u32,
    grid_depth: u32,
    num_tiles: u32,
    num_axes: u32,
    boundary_mode: u32, // 0: Clamped, 1: Periodic
    heuristic_type: u32,
    tie_breaking: u32,
    max_propagation_steps: u32,
    contradiction_check_frequency: u32,
    worklist_size: u32, // Number of cells in the input worklist for this pass
    grid_element_count: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
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
    let width = params.grid_width;
    let height = params.grid_height;
    let u32s_per_cell = (params.num_tiles + 31u) / 32u;

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