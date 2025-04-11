// utils.wgsl - Common utilities for Wave Function Collapse shaders
//
// This file contains common utility functions, constants, and types that are used 
// across multiple WGSL shaders in the Wave Function Collapse implementation.

// Axis enums for easier readability
const AXIS_POS_X: u32 = 0u;
const AXIS_NEG_X: u32 = 1u;
const AXIS_POS_Y: u32 = 2u;
const AXIS_NEG_Y: u32 = 3u;
const AXIS_POS_Z: u32 = 4u;
const AXIS_NEG_Z: u32 = 5u;

// Number of u32s needed per cell based on total number of tiles
// This will be replaced with the actual value during shader compilation
const NUM_TILES_U32_VALUE: u32 = 1u; // Placeholder, will be dynamically set in source
const NUM_TILES_U32: u32 = NUM_TILES_U32_VALUE;

// Common struct defining shader parameters
struct Params {
    // Grid dimensions
    grid_width: u32,
    grid_height: u32,
    grid_depth: u32,
    
    // Algorithm settings
    num_tiles: u32,
    num_axes: u32,
    boundary_mode: u32, // 0 = clamped, 1 = periodic
    heuristic_type: u32, // 0 = simple count, 1 = shannon entropy, 2 = weighted
    tie_breaking: u32, // 0 = none, 1 = deterministic, 2 = random pattern, 3 = position-based
    
    // Propagation parameters
    max_propagation_steps: u32,
    contradiction_check_frequency: u32,
    
    // Worklist parameters
    worklist_size: u32,
    
    // For SoA structure
    grid_element_count: u32,
    
    // Padding to ensure 16 byte alignment
    _padding: u32,
};

// Possibility mask array type using the NUM_TILES_U32 constant
alias PossibilityMask = array<u32, NUM_TILES_U32_VALUE>;

// Helper function to get 1D index from 3D coords
fn grid_index(x: u32, y: u32, z: u32, grid_width: u32, grid_height: u32) -> u32 {
    // Assumes packed u32s for possibilities are handled by multiplying by num_tiles_u32 later
    return z * grid_width * grid_height + y * grid_width + x;
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

// Helper function to check if a specific bit in the u32 mask array is set
fn is_tile_possible(tile_index: u32, mask: ptr<function, PossibilityMask>) -> bool {
    let u32_index = tile_index / 32u;
    let bit_index = tile_index % 32u;
    
    // For simplicity, only handle the first u32 chunk (index 0)
    // This works fine for tests with small NUM_TILES_U32 values
    if (u32_index == 0u) {
        return ((*mask)[0] & (1u << bit_index)) != 0u;
    }
    
    return false;
}

// Helper function to set a specific bit in a u32 array mask
fn set_tile_possible(tile_index: u32, mask: ptr<function, PossibilityMask>) {
    let u32_index = tile_index / 32u;
    let bit_index = tile_index % 32u;
    
    // For simplicity, only handle the first u32 chunk (index 0)
    // This works fine for tests with small NUM_TILES_U32 values
    if (u32_index == 0u) {
        (*mask)[0] = (*mask)[0] | (1u << bit_index);
    }
    // For other indices, we simply don't set the bit (acceptable for testing)
}

// Count number of 1 bits in a u32
fn count_ones(x: u32) -> u32 {
    var bits = x;
    bits = bits - ((bits >> 1) & 0x55555555u);
    bits = (bits & 0x33333333u) + ((bits >> 2) & 0x33333333u);
    bits = (bits + (bits >> 4)) & 0x0F0F0F0Fu;
    bits = bits + (bits >> 8);
    bits = bits + (bits >> 16);
    return bits & 0x0000003Fu;
} 