#![allow(unused_variables, dead_code)] // Allow unused during development

//! Defines structures and functions for dividing large grids into smaller subgrids
//! for parallel processing.

use wfc_core::grid::PossibilityGrid;

/// Represents a rectangular region within the main grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubgridRegion {
    pub x_offset: usize,
    pub y_offset: usize,
    pub z_offset: usize,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}

/// Configuration for subgrid processing.
#[derive(Clone, Debug, PartialEq)]
pub struct SubgridConfig {
    /// The maximum dimension size (width, height, or depth) for a subgrid.
    pub max_subgrid_size: usize,
    /// The amount of overlap between adjacent subgrids.
    pub overlap_size: usize,
    /// The minimum size of the original grid dimension to trigger subgridding.
    pub min_size: usize,
}

impl Default for SubgridConfig {
    fn default() -> Self {
        Self {
            max_subgrid_size: 64, // Example default
            overlap_size: 2,      // Example default
            min_size: 128,        // Example default
        }
    }
}

/// Divides a large grid into smaller, potentially overlapping subgrids.
///
/// # Arguments
/// * `grid_width`, `grid_height`, `grid_depth` - Dimensions of the original grid.
/// * `config` - Configuration specifying max subgrid size, overlap, and minimum size for subgridding.
///
/// # Returns
/// * `Ok(Vec<SubgridRegion>)` - A vector of subgrid regions if division is successful.
/// * `Err(String)` - If configuration is invalid or grid is too small.
pub fn divide_into_subgrids(
    grid_width: usize,
    grid_height: usize,
    grid_depth: usize,
    config: &SubgridConfig,
) -> Result<Vec<SubgridRegion>, String> {
    if config.max_subgrid_size == 0 {
        return Err("max_subgrid_size must be greater than 0".to_string());
    }
    if config.overlap_size >= config.max_subgrid_size / 2 {
        return Err("overlap_size must be less than half of max_subgrid_size".to_string());
    }

    // Determine if subgridding is needed based on min_size
    if grid_width < config.min_size && grid_height < config.min_size && grid_depth < config.min_size
    {
        // Grid is small, return a single region covering the whole grid
        return Ok(vec![SubgridRegion {
            x_offset: 0,
            y_offset: 0,
            z_offset: 0,
            width: grid_width,
            height: grid_height,
            depth: grid_depth,
        }]);
    }

    let mut subgrids = Vec::new();
    let step_size = config.max_subgrid_size - config.overlap_size;

    for z_start in (0..grid_depth).step_by(step_size) {
        for y_start in (0..grid_height).step_by(step_size) {
            for x_start in (0..grid_width).step_by(step_size) {
                // Calculate base end coordinates (exclusive)
                let base_end_x = std::cmp::min(x_start + config.max_subgrid_size, grid_width);
                let base_end_y = std::cmp::min(y_start + config.max_subgrid_size, grid_height);
                let base_end_z = std::cmp::min(z_start + config.max_subgrid_size, grid_depth);

                // Determine actual region boundaries including overlap
                // Start coordinates remain the same
                let start_x = x_start;
                let start_y = y_start;
                let start_z = z_start;

                // End coordinates extend by overlap, clamped to grid boundaries
                let end_x = std::cmp::min(base_end_x + config.overlap_size, grid_width);
                let end_y = std::cmp::min(base_end_y + config.overlap_size, grid_height);
                let end_z = std::cmp::min(base_end_z + config.overlap_size, grid_depth);

                // Ensure width/height/depth are not zero
                let width = end_x.saturating_sub(start_x);
                let height = end_y.saturating_sub(start_y);
                let depth = end_z.saturating_sub(start_z);

                if width > 0 && height > 0 && depth > 0 {
                    subgrids.push(SubgridRegion {
                        x_offset: start_x,
                        y_offset: start_y,
                        z_offset: start_z,
                        width,
                        height,
                        depth,
                    });
                }
            }
        }
    }

    if subgrids.is_empty() && (grid_width > 0 && grid_height > 0 && grid_depth > 0) {
        // This should only happen if step_size calculation leads to no iterations.
        // Add the whole grid as a single subgrid.
        return Ok(vec![SubgridRegion {
            x_offset: 0,
            y_offset: 0,
            z_offset: 0,
            width: grid_width,
            height: grid_height,
            depth: grid_depth,
        }]);
    }

    Ok(subgrids)
}

/// Extracts a subgrid from the main grid based on the specified region.
///
/// # Arguments
/// * `grid` - The main `PossibilityGrid`.
/// * `region` - The `SubgridRegion` defining the area to extract.
///
/// # Returns
/// * `Ok(PossibilityGrid)` - A new grid containing the data from the specified region.
/// * `Err(String)` - If the region is invalid or out of bounds.
pub fn extract_subgrid(
    grid: &PossibilityGrid,
    region: &SubgridRegion,
) -> Result<PossibilityGrid, String> {
    // Validate region boundaries
    if region.x_offset + region.width > grid.width
        || region.y_offset + region.height > grid.height
        || region.z_offset + region.depth > grid.depth
    {
        return Err(format!(
            "Subgrid region {:?} exceeds main grid dimensions ({}, {}, {})",
            region, grid.width, grid.height, grid.depth
        ));
    }
    if region.width == 0 || region.height == 0 || region.depth == 0 {
        return Err("Subgrid region dimensions cannot be zero".to_string());
    }

    // Create a new grid with the subgrid dimensions
    let mut subgrid =
        PossibilityGrid::new(region.width, region.height, region.depth, grid.num_tiles());

    // Copy data from the main grid to the subgrid
    for z in 0..region.depth {
        for y in 0..region.height {
            for x in 0..region.width {
                let global_x = region.x_offset + x;
                let global_y = region.y_offset + y;
                let global_z = region.z_offset + z;

                // Get possibilities from the main grid
                if let Some(possibilities) = grid.get(global_x, global_y, global_z) {
                    // Get mutable reference to subgrid cell and assign cloned possibilities
                    if let Some(subgrid_cell) = subgrid.get_mut(x, y, z) {
                        *subgrid_cell = possibilities.clone();
                    } else {
                        // This should not happen if dimensions are correct
                        return Err(format!(
                            "Failed to get mutable subgrid cell ({},{},{})",
                            x, y, z
                        ));
                    }
                } else {
                    // Should not happen if region validation passed
                    return Err(format!(
                        "Failed to get main grid cell ({},{},{})",
                        global_x, global_y, global_z
                    ));
                }
            }
        }
    }

    Ok(subgrid)
}

/// Merges processed subgrids back into the main grid.
///
/// Only updates the non-overlapping interior parts of each subgrid region to avoid overwriting
/// changes from adjacent subgrids.
///
/// # Arguments
/// * `grid` - A mutable reference to the main `PossibilityGrid` to merge into.
/// * `subgrids` - A slice of tuples, each containing a `SubgridRegion` and the corresponding processed `PossibilityGrid`.
/// * `config` - The `SubgridConfig` used for division (needed for overlap size).
///
/// # Returns
/// * `Ok(Vec<(usize, usize, usize)>)` - A vector of global coordinates that were updated in the main grid.
/// * `Err(String)` - If there's a dimension mismatch or other error.
pub fn merge_subgrids(
    grid: &mut PossibilityGrid,
    subgrids: &[(SubgridRegion, PossibilityGrid)],
    config: &SubgridConfig,
) -> Result<Vec<(usize, usize, usize)>, String> {
    let mut updated_coords = Vec::new();
    let overlap = config.overlap_size;

    for (region, subgrid) in subgrids {
        // Validate subgrid dimensions match region
        if subgrid.width != region.width
            || subgrid.height != region.height
            || subgrid.depth != region.depth
        {
            return Err(format!(
                "Subgrid dimension mismatch for region {:?}: Expected ({},{},{}), Got ({},{},{})",
                region,
                region.width,
                region.height,
                region.depth,
                subgrid.width,
                subgrid.height,
                subgrid.depth
            ));
        }
        if subgrid.num_tiles() != grid.num_tiles() {
            return Err(format!(
                "Subgrid tile count mismatch: Main grid {}, Subgrid {}",
                grid.num_tiles(),
                subgrid.num_tiles()
            ));
        }

        // Determine the interior region (non-overlapping part) within the subgrid
        let interior_start_x = overlap;
        let interior_start_y = overlap;
        let interior_start_z = overlap;
        // End is calculated carefully to avoid underflow if width/height/depth <= overlap * 2
        let interior_end_x = region.width.saturating_sub(overlap);
        let interior_end_y = region.height.saturating_sub(overlap);
        let interior_end_z = region.depth.saturating_sub(overlap);

        // Copy data from subgrid back to main grid
        for z in 0..region.depth {
            for y in 0..region.height {
                for x in 0..region.width {
                    // Check if this local coordinate (x,y,z) is within the *interior* part of the subgrid
                    let is_interior = x >= interior_start_x
                        && x < interior_end_x
                        && y >= interior_start_y
                        && y < interior_end_y
                        && z >= interior_start_z
                        && z < interior_end_z;

                    // Also consider cells on the actual boundary of the *main* grid as "interior" for merging purposes
                    let global_x = region.x_offset + x;
                    let global_y = region.y_offset + y;
                    let global_z = region.z_offset + z;
                    let is_main_grid_boundary = global_x == 0
                        || global_x == grid.width - 1
                        || global_y == 0
                        || global_y == grid.height - 1
                        || global_z == 0
                        || global_z == grid.depth - 1;

                    // Only merge if it's an interior cell or a main grid boundary cell
                    if is_interior || is_main_grid_boundary {
                        if let Some(subgrid_possibilities) = subgrid.get(x, y, z) {
                            if let Some(main_grid_possibilities) =
                                grid.get_mut(global_x, global_y, global_z)
                            {
                                // Check if the state actually changed
                                if *main_grid_possibilities != *subgrid_possibilities {
                                    *main_grid_possibilities = subgrid_possibilities.clone();
                                    updated_coords.push((global_x, global_y, global_z));
                                }
                            } else {
                                return Err(format!(
                                    "Failed to get mutable main grid cell ({},{},{})",
                                    global_x, global_y, global_z
                                ));
                            }
                        } else {
                            return Err(format!("Failed to get subgrid cell ({},{},{})", x, y, z));
                        }
                    }
                }
            }
        }
    }

    Ok(updated_coords)
}

impl SubgridRegion {
    /// Returns the overlap size (assuming uniform overlap on all sides)
    pub fn overlap_size(&self) -> usize {
        // This is a simplification; in practice you might want to store the overlap explicitly
        // Here we just return a default value
        2
    }

    /// Returns the end coordinates (exclusive) of the region.
    pub fn end_x(&self) -> usize {
        self.x_offset + self.width
    }
    pub fn end_y(&self) -> usize {
        self.y_offset + self.height
    }
    pub fn end_z(&self) -> usize {
        self.z_offset + self.depth
    }

    /// Checks if a global coordinate is contained within this subgrid region.
    pub fn contains(&self, x: usize, y: usize, z: usize) -> bool {
        x >= self.x_offset
            && x < self.end_x()
            && y >= self.y_offset
            && y < self.end_y()
            && z >= self.z_offset
            && z < self.end_z()
    }

    /// Converts global coordinates to local coordinates within this subgrid.
    /// Returns `None` if the global coordinates are outside the region.
    pub fn to_local_coords(
        &self,
        global_x: usize,
        global_y: usize,
        global_z: usize,
    ) -> Option<(usize, usize, usize)> {
        if self.contains(global_x, global_y, global_z) {
            Some((
                global_x - self.x_offset,
                global_y - self.y_offset,
                global_z - self.z_offset,
            ))
        } else {
            None
        }
    }

    /// Converts local coordinates within this subgrid to global coordinates.
    pub fn to_global_coords(
        &self,
        local_x: usize,
        local_y: usize,
        local_z: usize,
    ) -> (usize, usize, usize) {
        // We assume local coordinates are valid (within width/height/depth)
        // No bounds checking needed here, as it's converting *from* local.
        (
            self.x_offset + local_x,
            self.y_offset + local_y,
            self.z_offset + local_z,
        )
    }
}
