// propagate_modular.wgsl - Modular Wave Function Collapse constraint propagation
//
// This compute shader handles the propagation of constraints through the grid
// after a cell's possibilities have been restricted or collapsed. It ensures that
// all neighboring cells maintain consistent possibility states based on the 
// adjacency rules.
//
// This version uses a modular approach, importing common utility functions
// and organizing the code into logical sections for better maintainability.

// Include common utilities
#include "utils.wgsl"

// Include rules handling
#include "rules.wgsl"

// Include coordinate handling
#include "coords.wgsl"

// --- Bindings ---

// Uniform buffer with parameters
@group(0) @binding(0) var<uniform> params: Params;

// Storage buffers for grid data - Structure of Array (SoA) layout
// Each u32 contains up to 32 bits for tile possibilities
@group(0) @binding(1) var<storage, read_write> grid_possibilities: array<atomic<u32>>;

// Storage buffer for adjacency rules (read-only)
// Packed as bits in u32 array
@group(0) @binding(2) var<storage, read> adjacency_rules: array<u32>;

// Storage buffer for rule weights (read-only)
// Contains pairs of (rule_idx, weight_bits) for rules with non-default weights
@group(0) @binding(3) var<storage, read> rule_weights: array<u32>;

// Storage buffer for worklist (read-only)
// Contains 1D indices of cells to process
@group(0) @binding(4) var<storage, read> worklist: array<u32>;

// Storage buffer for output worklist (write-only)
// Will contain indices of cells that need processing in next step
@group(0) @binding(5) var<storage, read_write> output_worklist: array<atomic<u32>>;

// Atomic counter for output worklist
@group(0) @binding(6) var<storage, read_write> output_worklist_count: atomic<u32>;

// Atomic flag for contradiction detection
@group(0) @binding(7) var<storage, read_write> contradiction_flag: atomic<u32>;

// Atomic for tracking contradiction location
@group(0) @binding(8) var<storage, read_write> contradiction_location: atomic<u32>;

// --- Additional buffers for multi-pass propagation ---

// Buffer to track changes per propagation pass
@group(0) @binding(9) var<storage, read_write> pass_statistics: array<atomic<u32>>;

// --- Helper functions specific to propagation ---

// Loads a cell's possibility state from the grid
fn load_cell_possibilities(cell_idx: u32) -> PossibilityMask {
    var possibilities: PossibilityMask;
    
    // Initialize to 0 - use only index 0 for simplicity
    possibilities[0] = 0u;
    
    // Calculate number of cells for SoA indexing
    let num_cells = params.grid_width * params.grid_height * params.grid_depth;
    
    // Load first u32 chunk of possibilities
    let soa_idx = 0u * num_cells + cell_idx;
    if (soa_idx < NUM_TILES_U32 * num_cells) {
        possibilities[0] = atomicLoad(&grid_possibilities[soa_idx]);
    }
    
    return possibilities;
}

// Updates a neighbor cell's possibilities based on allowed tiles
// Returns whether changes were made and if contradiction was found
fn update_neighbor_possibilities(neighbor_idx: u32, 
                                allowed_mask: PossibilityMask) -> vec2<u32> {
    // Calculate number of cells for SoA indexing
    let num_cells = params.grid_width * params.grid_height * params.grid_depth;
    
    // Load current possibilities of neighbor
    var neighbor_possibilities = load_cell_possibilities(neighbor_idx);
    
    // Variables to track changes and contradictions
    var changed = false;
    var any_tiles_possible = false;
    
    // Handle first u32 chunk only for simplicity
    let new_bits = neighbor_possibilities[0] & allowed_mask[0];
    
    // Check if we're making a change
    if (new_bits != neighbor_possibilities[0]) { 
        changed = true; 
        
        // Count bits removed for statistics
        let removed_count = count_bits(neighbor_possibilities[0]) - count_bits(new_bits);
        atomicAdd(&pass_statistics[1], removed_count); // Total possibilities removed
    }
    
    // Check if there are any possible tiles left
    any_tiles_possible = any_tiles_possible || (new_bits != 0u);
    
    // Write the updated possibilities
    let soa_idx = 0u * num_cells + neighbor_idx;
    if (soa_idx < NUM_TILES_U32 * num_cells) {
        // Check for contradiction
        if (new_bits == 0u && neighbor_possibilities[0] != 0u) {
            // Changed from some bits set to no bits - signal contradiction
            atomicStore(&contradiction_flag, 1u);
            
            // Increment contradiction counter
            atomicAdd(&pass_statistics[2], 1u); // Contradiction count
        }
        
        // Update the grid
        atomicStore(&grid_possibilities[soa_idx], new_bits);
    }
    
    // Return (changed, has_contradiction)
    return vec2<u32>(u32(changed), u32(!any_tiles_possible));
}

// Adds a cell to the output worklist for future processing
fn add_to_worklist(cell_idx: u32) {
    // Get current output worklist index
    let worklist_idx = atomicAdd(&output_worklist_count, 1u);
    
    // Bounds check for worklist 
    if (worklist_idx < arrayLength(&output_worklist)) {
        // Add cell to worklist
        output_worklist[worklist_idx] = cell_idx;
        
        // Update statistics for current pass
        atomicAdd(&pass_statistics[0], 1u); // Total cells added
    } else {
        // Worklist overflow - signal for host to resize buffers
        atomicStore(&pass_statistics[3], 1u); // Overflow flag
    }
}

// Process a single cell and propagate constraints to neighbors
fn process_cell(cell_idx: u32, x: u32, y: u32, z: u32) {
    // Load current cell's possibilities
    let current_possibilities = load_cell_possibilities(cell_idx);
    
    // Skip fully collapsed cells (optimization)
    let bit_count = count_bits(current_possibilities[0]);
    if (bit_count <= 1u) {
        return;
    }
    
    // Track neighbors that have changed
    var change_count = 0u;
    
    // Process each axis direction
    for (var axis_idx: u32 = 0u; axis_idx < params.num_axes; axis_idx = axis_idx + 1u) {
        // Get neighbor index and check validity
        let neighbor_info = get_neighbor_index(x, y, z, axis_idx, params);
        let neighbor_idx = neighbor_info.x;
        let is_valid = neighbor_info.y;
        
        // Skip if neighbor is invalid
        if (is_valid == 0u) {
            continue;
        }
        
        // Compute allowed tiles in neighbor based on rules
        let allowed_neighbor_mask = compute_allowed_neighbor_mask(
            &current_possibilities, 
            axis_idx,
            params,
            adjacency_rules
        );
        
        // Update neighbor's possibilities
        let update_result = update_neighbor_possibilities(neighbor_idx, allowed_neighbor_mask);
        let changed = update_result.x;
        let has_contradiction = update_result.y;
        
        // Add to worklist if changes were made
        if (changed == 1u) {
            add_to_worklist(neighbor_idx);
            change_count += 1u;
        }
        
        // Record contradiction location if found
        if (has_contradiction == 1u) {
            atomicMin(&contradiction_location, neighbor_idx);
        }
        
        // Early termination check - only check periodically to reduce overhead
        // Check if this is a contradiction check cycle
        if (params.contradiction_check_frequency > 0u && 
            thread_idx % params.contradiction_check_frequency == 0u) {
            
            // Check for contradiction
            let has_contradiction = atomicLoad(&contradiction_flag);
            if (has_contradiction == 1u) {
                // Exit early if contradiction found
                return;
            }
        }
    }
}

// --- Main compute shader entry point ---

@compute @workgroup_size(64) // Use hardcoded size
fn main_propagate(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    // Use flat 1D indexing
    let thread_idx = global_id.x;
    
    // Bounds check for worklist
    if (thread_idx >= params.worklist_size) {
        return;
    }
    
    // Reset statistics for first thread only
    if (thread_idx == 0u) {
        atomicStore(&pass_statistics[0], 0u); // Reset cells added
        atomicStore(&pass_statistics[1], 0u); // Reset possibilities removed
        atomicStore(&pass_statistics[2], 0u); // Reset contradiction count
        atomicStore(&pass_statistics[3], 0u); // Reset overflow flag
    }
    
    // Wait for all threads to synchronize (using workgroup barrier)
    workgroupBarrier();
    
    // Get the cell to process from the worklist
    let cell_idx = worklist[thread_idx];
    
    // Convert 1D index to 3D coordinates
    let z = cell_idx / (params.grid_width * params.grid_height);
    let temp = cell_idx % (params.grid_width * params.grid_height);
    let y = temp / params.grid_width;
    let x = temp % params.grid_width;
    
    // Process the cell
    process_cell(cell_idx, x, y, z);
} 