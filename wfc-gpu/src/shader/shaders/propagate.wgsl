// WGSL Shader for Wave Function Collapse constraint propagation
//
// This compute shader handles the propagation of constraints through the grid
// after a cell's possibilities have been restricted or collapsed. It ensures that
// all neighboring cells maintain consistent possibility states based on the 
// adjacency rules.
//
// CRITICAL SAFETY FEATURES:
// 1. Uses 1D workgroup layout (64,1,1) for simpler thread indexing
// 2. Enforces strict bounds checking on all array accesses
// 3. Contains output worklist size limits to prevent infinite propagation loops
// 4. Detects and reports contradictions early
//
// Memory access optimizations:
// 1. Pre-loads and caches cell possibilities to reduce atomic operations
// 2. Uses local variables for intermediate calculations to reduce memory traffic
// 3. Batches atomic operations when possible for more efficient GPU utilization
// 4. Caches rule check results to avoid redundant calculations
//
// The shader processes each cell in the input worklist, updates all valid 
// neighbors according to adjacency rules, and adds any changed neighbors
// to the output worklist for further processing if needed.

// Struct defining shader parameters
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

// Axis enums for easier readability
const AXIS_POS_X: u32 = 0u;
const AXIS_NEG_X: u32 = 1u;
const AXIS_POS_Y: u32 = 2u;
const AXIS_NEG_Y: u32 = 3u;
const AXIS_POS_Z: u32 = 4u;
const AXIS_NEG_Z: u32 = 5u;

// Bind group 0: Parameters and buffers
@group(0) @binding(0) var<storage, read_write> grid_possibilities: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> adjacency_rules: array<u32>;
@group(0) @binding(2) var<storage, read> rule_weights: array<u32>;
@group(0) @binding(3) var<storage, read> worklist: array<u32>;
@group(0) @binding(4) var<storage, read_write> output_worklist: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read_write> worklist_count: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> contradiction_flag: atomic<u32>;
@group(0) @binding(8) var<storage, read_write> contradiction_location: atomic<u32>;
@group(0) @binding(9) var<storage, read_write> pass_statistics: array<atomic<u32>>;

// Possibilities are stored as `words_per_cell()` u32 words per cell, 32 tiles per word, in cell
// order (`cell * words + word`), the same layout the host uploads and the entropy shader reads.
// Function-local arrays in WGSL need a constant size, so masks are sized for MAX_WORDS and every
// loop only runs over the words the current rule set uses. Keep MAX_WORDS in sync with
// MAX_WORDS_PER_CELL in shader/pipeline.rs; the host rejects rule sets that do not fit.
const MAX_WORDS: u32 = 8u;
alias PossibilityMask = array<u32, 8>;

fn words_per_cell() -> u32 {
    return (params.num_tiles + 31u) / 32u;
}

// Entry point
@compute @workgroup_size(64)
fn propagate_constraints(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    // Get global thread ID
    let global_id = workgroup_id.x * 64u + local_id.x;
    
    // Check if this thread should process a cell from the worklist
    // worklist_count is the output counter for this pass; the input size comes from params
    if (global_id >= params.worklist_size || global_id >= arrayLength(&worklist)) {
        return; // No more cells to process
    }
    
    // Get the cell index from the worklist
    let cell_idx = worklist[global_id];
    
    // Load cell's current possibilities
    var current_possibilities = load_cell_possibilities(cell_idx);
    
    // Get cell's 3D coordinates
    let z = cell_idx / (params.grid_width * params.grid_height);
    let y = (cell_idx % (params.grid_width * params.grid_height)) / params.grid_width;
    let x = cell_idx % params.grid_width;
    
    // Process each neighbor
    for (var axis = 0u; axis < params.num_axes; axis = axis + 1u) {
        // Compute allowed neighbor mask for this axis
        let allowed_neighbor_mask = compute_allowed_neighbor_mask(&current_possibilities, axis);
        
        // Calculate neighbor coordinates based on axis
        var nx = x;
        var ny = y;
        var nz = z;
        
        switch (axis) {
            case AXIS_POS_X: { nx = x + 1u; }
            case AXIS_NEG_X: { nx = x - 1u; }
            case AXIS_POS_Y: { ny = y + 1u; }
            case AXIS_NEG_Y: { ny = y - 1u; }
            case AXIS_POS_Z: { nz = z + 1u; }
            case AXIS_NEG_Z: { nz = z - 1u; }
            default: { break; }
        }
        
        // Handle boundary conditions
        if (params.boundary_mode == 0u) { // Clamped
            if (nx >= params.grid_width || ny >= params.grid_height || nz >= params.grid_depth) {
                continue; // Skip out-of-bounds neighbors
            }
        } else { // Periodic
            nx = wrap_coord(i32(nx), params.grid_width);
            ny = wrap_coord(i32(ny), params.grid_height);
            nz = wrap_coord(i32(nz), params.grid_depth);
        }
        
        // Calculate neighbor's 1D index
        let neighbor_idx = grid_index(nx, ny, nz);
        
        // Update neighbor's possibilities and add to worklist if changed
        let changed = update_neighbor(neighbor_idx, allowed_neighbor_mask);
        
        // Add to next worklist if any changes were made (one atomic operation)
        if (changed) {
            let worklist_idx = atomicAdd(&worklist_count[0], 1u);
            // Bounds check for worklist
            if (worklist_idx < arrayLength(&output_worklist)) {
                atomicStore(&output_worklist[worklist_idx], neighbor_idx);
            }
        }
    }
}

// Helper function to get 1D index from 3D coords
fn grid_index(x: u32, y: u32, z: u32) -> u32 {
    // Assumes packed u32s for possibilities are handled by multiplying by num_tiles_u32 later
    return z * params.grid_width * params.grid_height + y * params.grid_width + x;
}

// Helper to calculate wrapped coordinate for Periodic boundary mode
fn wrap_coord(coord: i32, max_dim: u32) -> u32 {
    if (max_dim == 0u) { return 0u; } // Avoid modulo by zero
    // Efficient modulo for potentially negative numbers
    let m = coord % i32(max_dim);
    if (m < 0) {
        return u32(m + i32(max_dim));
    } else {
        return u32(m);
    }
}

// First word of the mask of tiles allowed next to `tile` across `axis`. The table holds one
// `words_per_cell()`-word mask per (axis, tile), word-aligned, so a mask can be read directly.
fn rule_row(tile: u32, axis: u32) -> u32 {
    return (axis * params.num_tiles + tile) * words_per_cell();
}

// Helper function to check adjacency rule
fn check_rule(tile1: u32, tile2: u32, axis: u32) -> bool {
    if (tile1 >= params.num_tiles || tile2 >= params.num_tiles || axis >= params.num_axes) {
        return false;
    }
    let index = rule_row(tile1, axis) + tile2 / 32u;
    if (index >= arrayLength(&adjacency_rules)) {
        return false;
    }
    return (adjacency_rules[index] & (1u << (tile2 % 32u))) != 0u;
}

// Helper function to get rule weight
// For non-weighted rules (the default), this returns 1.0
// For weighted rules, this returns a value between 0.0 and 1.0
fn get_rule_weight(tile1: u32, tile2: u32, axis: u32) -> f32 {
    // First check if the rule exists at all
    if (!check_rule(tile1, tile2, axis)) {
        return 0.0;
    }

    // Weights are keyed by the flat rule index, which is independent of how the table is packed.
    let rule_idx = axis * params.num_tiles * params.num_tiles + tile1 * params.num_tiles + tile2;
    
    // Check if this rule has an entry in the weights buffer
    // For now, we'll use a simple linear search approach
    // Future optimization: Use a proper mapping structure
    for (var i = 0u; i < arrayLength(&rule_weights); i += 2u) {
        // Each weight entry consists of two u32s:
        // - rule_idx: The packed rule index
        // - weight_bits: The f32 weight encoded as bits
        
        if (rule_weights[i] == rule_idx) {
            // Found it, get the weight
            // Convert the bits back to float
            return bitcast<f32>(rule_weights[i + 1u]);
        }
    }
    
    // Default weight for valid rules with no explicit weight
    return 1.0;
}

// Union, over every tile still possible in the current cell, of the tiles that tile allows in
// the neighbour along `axis_idx`. Only set bits of the current cell are visited.
fn compute_allowed_neighbor_mask(current_possibilities: ptr<function, PossibilityMask>,
                                 axis_idx: u32) -> PossibilityMask {
    var allowed_neighbor_mask: PossibilityMask;
    let words = words_per_cell();
    for (var w = 0u; w < words; w = w + 1u) {
        allowed_neighbor_mask[w] = 0u;
    }

    for (var w = 0u; w < words; w = w + 1u) {
        var bits = (*current_possibilities)[w];
        while (bits != 0u) {
            let current_tile = w * 32u + countTrailingZeros(bits);
            bits = bits & (bits - 1u);
            if (current_tile >= params.num_tiles) {
                break;
            }
            // Union this tile's allowed neighbours a word at a time, rather than testing every
            // tile pair: at 81 variants that is 3 ORs instead of 81 bit tests.
            let row = rule_row(current_tile, axis_idx);
            for (var w2 = 0u; w2 < words; w2 = w2 + 1u) {
                let index = row + w2;
                if (index < arrayLength(&adjacency_rules)) {
                    allowed_neighbor_mask[w2] = allowed_neighbor_mask[w2] | adjacency_rules[index];
                }
            }
        }
    }

    return allowed_neighbor_mask;
}

// Loads every possibility word of a cell.
fn load_cell_possibilities(cell_idx: u32) -> PossibilityMask {
    var possibilities: PossibilityMask;
    let words = words_per_cell();
    for (var w = 0u; w < words; w = w + 1u) {
        let index = cell_idx * words + w;
        if (index < arrayLength(&grid_possibilities)) {
            possibilities[w] = atomicLoad(&grid_possibilities[index]);
        } else {
            possibilities[w] = 0u;
        }
    }
    return possibilities;
}

// Intersects a neighbour's possibilities with `allowed_neighbor_mask`, storing only words that
// change, and records a contradiction if nothing is left. Returns whether anything changed.
fn update_neighbor(neighbor_idx: u32, allowed_neighbor_mask: PossibilityMask) -> bool {
    let words = words_per_cell();
    var changed = false;
    var any_tiles_possible = false;

    for (var w = 0u; w < words; w = w + 1u) {
        let index = neighbor_idx * words + w;
        if (index >= arrayLength(&grid_possibilities)) {
            break;
        }
        let old_bits = atomicLoad(&grid_possibilities[index]);
        let new_bits = old_bits & allowed_neighbor_mask[w];
        if (new_bits != old_bits) {
            atomicStore(&grid_possibilities[index], new_bits);
            changed = true;
        }
        if (new_bits != 0u) {
            any_tiles_possible = true;
        }
    }

    if (changed && !any_tiles_possible) {
        atomicStore(&contradiction_flag, 1u);
        atomicStore(&contradiction_location, neighbor_idx);
    }

    return changed;
}
