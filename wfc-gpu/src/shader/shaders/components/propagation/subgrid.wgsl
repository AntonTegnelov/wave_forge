// Subgrid propagation component for Wave Function Collapse
// This component handles propagation using a subgrid approach for large grids
// by dividing the problem into manageable chunks

// Process a single subgrid cell, calculating its coordinates within the full grid
fn process_subgrid_cell(
    local_idx: u32,            // Local index within the subgrid
    subgrid_origin: vec3<u32>, // Origin (start) of this subgrid in the full grid
    subgrid_dims: vec3<u32>,   // Dimensions of the subgrid
    grid_dims: vec3<u32>,      // Dimensions of the full grid
    u32s_per_cell: u32,        // Number of u32s per cell
    possibilities: array<u32>, // Grid possibilities array
    adjacency_rules: array<u32> // Adjacency rules defining allowed neighbors
) -> bool {
    // Calculate local 3D coordinates within the subgrid
    let local_x = local_idx % subgrid_dims.x;
    let local_y = (local_idx / subgrid_dims.x) % subgrid_dims.y;
    let local_z = local_idx / (subgrid_dims.x * subgrid_dims.y);
    
    // Calculate global coordinates in the full grid
    let global_x = subgrid_origin.x + local_x;
    let global_y = subgrid_origin.y + local_y;
    let global_z = subgrid_origin.z + local_z;
    
    // Skip if outside the full grid bounds
    if (global_x >= grid_dims.x || global_y >= grid_dims.y || global_z >= grid_dims.z) {
        return false;
    }
    
    // Calculate global flat index
    let global_idx = global_z * grid_dims.x * grid_dims.y + global_y * grid_dims.x + global_x;
    
    // Delegate to the standard propagation function for this cell
    return propagate_constraints(
        global_idx,
        grid_dims.x,
        grid_dims.y,
        grid_dims.z,
        u32s_per_cell,
        possibilities,
        adjacency_rules
    );
}

// Process the boundary of a subgrid, focusing on cells that might interact with other subgrids
fn process_subgrid_boundary(
    workgroup_id: vec3<u32>,     // Workgroup ID (identifies the subgrid)
    num_workgroups: vec3<u32>,   // Total number of workgroups/subgrids
    workgroup_size: vec3<u32>,   // Size of each workgroup
    local_id: vec3<u32>,         // Local ID within the workgroup
    grid_dims: vec3<u32>,        // Dimensions of the full grid
    u32s_per_cell: u32,          // Number of u32s per cell
    possibilities: array<u32>,   // Grid possibilities array
    adjacency_rules: array<u32>, // Adjacency rules defining allowed neighbors
    is_boundary_pass: bool       // Whether this is a boundary-only pass
) -> bool {
    // Calculate the origin (starting coordinates) of this subgrid
    let subgrid_origin = workgroup_id * workgroup_size;
    
    // Calculate the global coordinates for this thread
    let global_x = subgrid_origin.x + local_id.x;
    let global_y = subgrid_origin.y + local_id.y;
    let global_z = subgrid_origin.z + local_id.z;
    
    // Skip if outside the grid bounds
    if (global_x >= grid_dims.x || global_y >= grid_dims.y || global_z >= grid_dims.z) {
        return false;
    }
    
    // For boundary pass, only process cells on the boundary of the subgrid
    if (is_boundary_pass) {
        // Check if this cell is on the boundary
        let is_boundary = 
            local_id.x == 0u || local_id.x == workgroup_size.x - 1u ||
            local_id.y == 0u || local_id.y == workgroup_size.y - 1u ||
            local_id.z == 0u || local_id.z == workgroup_size.z - 1u;
            
        // Skip non-boundary cells in boundary pass
        if (!is_boundary) {
            return false;
        }
    }
    
    // Calculate global flat index
    let global_idx = global_z * grid_dims.x * grid_dims.y + global_y * grid_dims.x + global_x;
    
    // Process this cell
    return propagate_constraints(
        global_idx,
        grid_dims.x,
        grid_dims.y,
        grid_dims.z,
        u32s_per_cell,
        possibilities,
        adjacency_rules
    );
}

// Main subgrid propagation function
// This is designed to work with a dispatch that divides the grid into workgroups
fn subgrid_propagate(
    workgroup_id: vec3<u32>,     // Identifies which subgrid this is
    local_id: vec3<u32>,         // Position within the subgrid
    num_workgroups: vec3<u32>,   // Total number of subgrids in each dimension
    grid_dims: vec3<u32>,        // Full grid dimensions
    u32s_per_cell: u32,          // Storage size per cell
    possibilities: array<u32>,   // Grid state
    adjacency_rules: array<u32>, // Rules for constraint propagation
    workgroup_size: vec3<u32>,   // Size of each workgroup/subgrid
    is_boundary_only: bool       // Whether to only process the boundary
) -> bool {
    // Process this cell's constraints
    return process_subgrid_boundary(
        workgroup_id,
        num_workgroups,
        workgroup_size,
        local_id,
        grid_dims,
        u32s_per_cell,
        possibilities,
        adjacency_rules,
        is_boundary_only
    );
} 