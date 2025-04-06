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

/// Divides a grid into subgrids based on the provided configuration
pub fn divide_into_subgrids(
    grid_width: usize,
    grid_height: usize,
    grid_depth: usize,
    config: &SubgridConfig,
) -> Result<Vec<SubgridRegion>, GpuError> {
    // Validate input
    if grid_width == 0 || grid_height == 0 || grid_depth == 0 {
        return Err(GpuError::Other(
            "Cannot divide a grid with zero dimension".to_string(),
        ));
    }

    if config.max_subgrid_size == 0 {
        return Err(GpuError::Other("Subgrid size cannot be zero".to_string()));
    }

    // Calculate number of divisions needed in each dimension
    let x_divisions = grid_width.div_ceil(config.max_subgrid_size);
    let y_divisions = grid_height.div_ceil(config.max_subgrid_size);
    let z_divisions = grid_depth.div_ceil(config.max_subgrid_size);

    // Create subgrids
    let mut subgrids = Vec::new();

    for z_idx in 0..z_divisions {
        for y_idx in 0..y_divisions {
            for x_idx in 0..x_divisions {
                // Calculate base coordinates
                let base_start_x = x_idx * config.max_subgrid_size;
                let base_start_y = y_idx * config.max_subgrid_size;
                let base_start_z = z_idx * config.max_subgrid_size;

                let base_end_x = min(base_start_x + config.max_subgrid_size, grid_width);
                let base_end_y = min(base_start_y + config.max_subgrid_size, grid_height);
                let base_end_z = min(base_start_z + config.max_subgrid_size, grid_depth);

                // Add overlap while respecting grid boundaries
                let start_x = if base_start_x >= config.overlap_size {
                    base_start_x - config.overlap_size
                } else {
                    0
                };

                let start_y = if base_start_y >= config.overlap_size {
                    base_start_y - config.overlap_size
                } else {
                    0
                };

                let start_z = if base_start_z >= config.overlap_size {
                    base_start_z - config.overlap_size
                } else {
                    0
                };

                let end_x = min(base_end_x + config.overlap_size, grid_width);
                let end_y = min(base_end_y + config.overlap_size, grid_height);
                let end_z = min(base_end_z + config.overlap_size, grid_depth);

                subgrids.push(SubgridRegion {
                    x_offset: start_x,
                    y_offset: start_y,
                    z_offset: start_z,
                    width: end_x - start_x,
                    height: end_y - start_y,
                    depth: end_z - start_z,
                });
            }
        }
    }

    Ok(subgrids)
}

/// Extracts a subgrid from the main grid
pub fn extract_subgrid(
    grid: &PossibilityGrid,
    region: &SubgridRegion,
) -> Result<PossibilityGrid, GpuError> {
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

                if let Some(possibilities) = grid.get(global_x, global_y, global_z) {
                    if let Some(subgrid_possibilities) = subgrid.get_mut(x, y, z) {
                        *subgrid_possibilities = possibilities.clone();
                    }
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
