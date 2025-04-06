#![allow(unused_variables, dead_code)] // Allow unused during development

//! Defines structures and functions for dividing large grids into smaller subgrids
//! for parallel processing.

use crate::GpuError;
use std::cmp::min;
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
                    // Set possibilities in the subgrid
                    if let Err(e) = subgrid.set(x, y, z, *possibilities) {
                        // This should not happen if dimensions are correct
                        return Err(format!(
                            "Failed to set subgrid cell ({},{},{}): {}",
                            x, y, z, e
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

/// Merges updated subgrids back into the main grid
pub fn merge_subgrids(
    main_grid: &mut PossibilityGrid,
    subgrids: &[(SubgridRegion, PossibilityGrid)],
) -> Result<Vec<(usize, usize, usize)>, GpuError> {
    let mut updated_coords = Vec::new();

    // Process each subgrid
    for (region, subgrid) in subgrids {
        // Copy data from subgrid back to main grid and track updates
        for z in 0..region.depth {
            for y in 0..region.height {
                for x in 0..region.width {
                    let global_x = region.x_offset + x;
                    let global_y = region.y_offset + y;
                    let global_z = region.z_offset + z;

                    // Check if this is an overlap region (not on the edge of a subgrid)
                    let is_interior = x >= region.overlap_size
                        && y >= region.overlap_size
                        && z >= region.overlap_size
                        && x < (region.width - region.overlap_size)
                        && y < (region.height - region.overlap_size)
                        && z < (region.depth - region.overlap_size);

                    // Only update interior cells or cells at the grid boundary
                    if is_interior
                        || global_x == 0
                        || global_x == main_grid.width - 1
                        || global_y == 0
                        || global_y == main_grid.height - 1
                        || global_z == 0
                        || global_z == main_grid.depth - 1
                    {
                        if let Some(subgrid_possibilities) = subgrid.get(x, y, z) {
                            if let Some(main_possibilities) =
                                main_grid.get_mut(global_x, global_y, global_z)
                            {
                                // Check if there's actually a change before updating
                                if *main_possibilities != *subgrid_possibilities {
                                    *main_possibilities = subgrid_possibilities.clone();
                                    updated_coords.push((global_x, global_y, global_z));
                                }
                            }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subgrid_region() {
        let region = SubgridRegion {
            x_offset: 10,
            y_offset: 20,
            z_offset: 30,
            width: 40,
            height: 50,
            depth: 60,
        };

        assert_eq!(region.width, 40);
        assert_eq!(region.height, 50);
        assert_eq!(region.depth, 60);
        assert_eq!(region.width * region.height * region.depth, 120000);

        assert!(region.contains(15, 25, 35));
        assert!(!region.contains(5, 25, 35));

        assert_eq!(region.to_local_coords(15, 25, 35), Some((5, 5, 5)));
        assert_eq!(region.to_local_coords(5, 25, 35), None);

        assert_eq!(region.to_global_coords(5, 5, 5), (15, 25, 35));
    }

    #[test]
    fn test_divide_into_subgrids() {
        let config = SubgridConfig {
            max_subgrid_size: 10,
            overlap_size: 1,
            min_size: 128,
        };

        let subgrids = divide_into_subgrids(25, 15, 5, &config).unwrap();

        // For a 25x15x5 grid with max size 10 and overlap 1, we expect:
        // X: 3 divisions (0-10, 10-20, 20-25)
        // Y: 2 divisions (0-10, 10-15)
        // Z: 1 division (0-5)
        // Total: 3 * 2 * 1 = 6 subgrids
        assert_eq!(subgrids.len(), 6);

        // Check first subgrid (should be 0,0,0 with overlap adjustments)
        let first = &subgrids[0];
        assert_eq!(first.x_offset, 0); // Can't go below 0
        assert_eq!(first.y_offset, 0);
        assert_eq!(first.z_offset, 0);
        assert_eq!(first.width, 11); // 10 + 1 overlap
        assert_eq!(first.height, 11);
        assert_eq!(first.depth, 5); // Only 5 deep total, so no overflow

        // Check last subgrid
        let last = &subgrids[5];
        assert_eq!(last.x_offset, 19); // 20 - 1 overlap
        assert_eq!(last.y_offset, 9); // 10 - 1 overlap
        assert_eq!(last.z_offset, 0); // Can't go below 0
        assert_eq!(last.width, 25); // Grid width
        assert_eq!(last.height, 15); // Grid height
        assert_eq!(last.depth, 5); // Grid depth
    }
}
