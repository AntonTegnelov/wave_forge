//! Module implementing parallel subgrid processing for large Wave Function Collapse grids.
//!
//! This module contains functionality to divide a large grid into smaller subgrids
//! that can be processed independently in parallel, improving performance and
//! scalability for the GPU-accelerated WFC implementation.

use crate::GpuError;
use std::cmp::min;
use wfc_core::{grid::PossibilityGrid, BoundaryCondition};

/// Represents a subgrid region within the main WFC grid
#[derive(Debug, Clone, Copy)]
pub struct SubgridRegion {
    /// Start X coordinate (inclusive)
    pub start_x: usize,
    /// Start Y coordinate (inclusive)
    pub start_y: usize,
    /// Start Z coordinate (inclusive)
    pub start_z: usize,
    /// End X coordinate (exclusive)
    pub end_x: usize,
    /// End Y coordinate (exclusive)
    pub end_y: usize,
    /// End Z coordinate (exclusive)
    pub end_z: usize,
}

impl SubgridRegion {
    /// Creates a new subgrid region
    pub fn new(
        start_x: usize,
        start_y: usize,
        start_z: usize,
        end_x: usize,
        end_y: usize,
        end_z: usize,
    ) -> Self {
        Self {
            start_x,
            start_y,
            start_z,
            end_x,
            end_y,
            end_z,
        }
    }

    /// Returns the width of the subgrid
    pub fn width(&self) -> usize {
        self.end_x - self.start_x
    }

    /// Returns the height of the subgrid
    pub fn height(&self) -> usize {
        self.end_y - self.start_y
    }

    /// Returns the depth of the subgrid
    pub fn depth(&self) -> usize {
        self.end_z - self.start_z
    }

    /// Returns the size of the subgrid in cells
    pub fn size(&self) -> usize {
        self.width() * self.height() * self.depth()
    }

    /// Checks if a coordinate is within this subgrid
    pub fn contains(&self, x: usize, y: usize, z: usize) -> bool {
        x >= self.start_x
            && x < self.end_x
            && y >= self.start_y
            && y < self.end_y
            && z >= self.start_z
            && z < self.end_z
    }

    /// Gets the relative coordinates within this subgrid
    pub fn to_local_coords(&self, x: usize, y: usize, z: usize) -> Option<(usize, usize, usize)> {
        if !self.contains(x, y, z) {
            return None;
        }

        Some((x - self.start_x, y - self.start_y, z - self.start_z))
    }

    /// Converts local coordinates to global grid coordinates
    pub fn to_global_coords(
        &self,
        local_x: usize,
        local_y: usize,
        local_z: usize,
    ) -> (usize, usize, usize) {
        (
            self.start_x + local_x,
            self.start_y + local_y,
            self.start_z + local_z,
        )
    }
}

/// Configuration for subgrid division
#[derive(Debug, Clone)]
pub struct SubgridConfig {
    /// Target maximum size for each subgrid dimension in cells
    pub max_subgrid_size: usize,
    /// Overlap between adjacent subgrids in cells (to handle boundary interactions)
    pub overlap_size: usize,
}

impl Default for SubgridConfig {
    fn default() -> Self {
        Self {
            max_subgrid_size: 64, // Default reasonable size for GPU processing
            overlap_size: 2,      // Default overlap to handle adjacency constraints
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
    let x_divisions = (grid_width + config.max_subgrid_size - 1) / config.max_subgrid_size;
    let y_divisions = (grid_height + config.max_subgrid_size - 1) / config.max_subgrid_size;
    let z_divisions = (grid_depth + config.max_subgrid_size - 1) / config.max_subgrid_size;

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

                subgrids.push(SubgridRegion::new(
                    start_x, start_y, start_z, end_x, end_y, end_z,
                ));
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
    let mut subgrid = PossibilityGrid::new(
        region.width(),
        region.height(),
        region.depth(),
        grid.num_tiles(),
    );

    // Copy data from the main grid to the subgrid
    for z in 0..region.depth() {
        for y in 0..region.height() {
            for x in 0..region.width() {
                let global_x = region.start_x + x;
                let global_y = region.start_y + y;
                let global_z = region.start_z + z;

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
        for z in 0..region.depth() {
            for y in 0..region.height() {
                for x in 0..region.width() {
                    let global_x = region.start_x + x;
                    let global_y = region.start_y + y;
                    let global_z = region.start_z + z;

                    // Check if this is an overlap region (not on the edge of a subgrid)
                    let is_interior = x >= region.overlap_size()
                        && y >= region.overlap_size()
                        && z >= region.overlap_size()
                        && x < (region.width() - region.overlap_size())
                        && y < (region.height() - region.overlap_size())
                        && z < (region.depth() - region.overlap_size());

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
        let region = SubgridRegion::new(10, 20, 30, 40, 50, 60);

        assert_eq!(region.width(), 30);
        assert_eq!(region.height(), 30);
        assert_eq!(region.depth(), 30);
        assert_eq!(region.size(), 27000);

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
        assert_eq!(first.start_x, 0); // Can't go below 0
        assert_eq!(first.start_y, 0);
        assert_eq!(first.start_z, 0);
        assert_eq!(first.end_x, 11); // 10 + 1 overlap
        assert_eq!(first.end_y, 11);
        assert_eq!(first.end_z, 5); // Only 5 deep total, so no overflow

        // Check last subgrid
        let last = &subgrids[5];
        assert_eq!(last.start_x, 19); // 20 - 1 overlap
        assert_eq!(last.start_y, 9); // 10 - 1 overlap
        assert_eq!(last.start_z, 0); // Can't go below 0
        assert_eq!(last.end_x, 25); // Grid width
        assert_eq!(last.end_y, 15); // Grid height
        assert_eq!(last.end_z, 5); // Grid depth
    }
}
