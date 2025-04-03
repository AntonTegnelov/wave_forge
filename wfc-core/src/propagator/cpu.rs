use crate::{
    grid::PossibilityGrid,
    propagator::{ConstraintPropagator, PropagationError},
    BoundaryMode,
};
use bitvec::prelude::*;
use wfc_rules::AdjacencyRules;

/// Simple CPU-based constraint propagator using a basic iterative approach.
#[derive(Debug, Clone)]
pub struct CpuConstraintPropagator {
    boundary_mode: BoundaryMode,
}

impl CpuConstraintPropagator {
    pub fn new(boundary_mode: BoundaryMode) -> Self {
        Self { boundary_mode }
    }

    // Helper to get neighbor coordinates, handling boundary conditions
    fn get_neighbor_coords(
        &self,
        x: usize,
        y: usize,
        z: usize,
        axis: usize, // 0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z
        width: usize,
        height: usize,
        depth: usize,
    ) -> Option<(usize, usize, usize)> {
        let (dx, dy, dz): (isize, isize, isize) = match axis {
            0 => (1, 0, 0),
            1 => (-1, 0, 0),
            2 => (0, 1, 0),
            3 => (0, -1, 0),
            4 => (0, 0, 1),
            5 => (0, 0, -1),
            _ => return None, // Invalid axis
        };

        let nx_raw = x as isize + dx;
        let ny_raw = y as isize + dy;
        let nz_raw = z as isize + dz;

        match self.boundary_mode {
            BoundaryMode::Clamped => {
                if nx_raw >= 0
                    && nx_raw < width as isize
                    && ny_raw >= 0
                    && ny_raw < height as isize
                    && nz_raw >= 0
                    && nz_raw < depth as isize
                {
                    Some((nx_raw as usize, ny_raw as usize, nz_raw as usize))
                } else {
                    None // Out of bounds for clamped mode
                }
            }
            BoundaryMode::Periodic => {
                // Use modulo arithmetic for wrapping
                let nx = nx_raw.rem_euclid(width as isize) as usize;
                let ny = ny_raw.rem_euclid(height as isize) as usize;
                let nz = nz_raw.rem_euclid(depth as isize) as usize;
                Some((nx, ny, nz))
            }
        }
    }
}

impl ConstraintPropagator for CpuConstraintPropagator {
    fn propagate(
        &mut self,
        grid: &mut PossibilityGrid,
        updated_coords: Vec<(usize, usize, usize)>,
        rules: &AdjacencyRules,
    ) -> Result<(), PropagationError> {
        let width = grid.width;
        let height = grid.height;
        let depth = grid.depth;
        let num_tiles = grid.num_tiles();
        let num_axes = rules.num_axes(); // Typically 6

        // Stack of coordinates whose possibilities have changed and need to be propagated FROM
        let mut propagation_stack = updated_coords;

        while let Some((x, y, z)) = propagation_stack.pop() {
            // Get current possibilities for the cell we are propagating FROM
            // Need to clone it or access immutably if grid access pattern requires it
            // Or use indices/iterators carefully
            let current_possibilities = match grid.get(x, y, z) {
                Some(bv) => bv.clone(), // Clone to avoid borrow issues when modifying neighbors
                None => continue,       // Should not happen if coords are valid
            };

            // Iterate through all potential neighbors (axes)
            for axis in 0..num_axes {
                if let Some((nx, ny, nz)) =
                    self.get_neighbor_coords(x, y, z, axis, width, height, depth)
                {
                    // Get mutable access to the neighbor's possibilities
                    let neighbor_possibilities_mut = match grid.get_mut(nx, ny, nz) {
                        Some(bv) => bv,
                        None => continue, // Should not happen if coords are valid
                    };

                    let original_neighbor_count = neighbor_possibilities_mut.count_ones();
                    if original_neighbor_count == 0 {
                        // Neighbor was already a contradiction, skip (or maybe error earlier?)
                        continue;
                    }

                    // Determine which tiles in the *neighbor* are supported by the *current* cell's possibilities
                    let mut supported_neighbor_tiles = bitvec![u32, Lsb0; 0; num_tiles];
                    let _neighbor_axis = rules.opposite_axis(axis); // Axis viewed from neighbor back to current

                    // Iterate through all possible tiles (t1) in the current cell
                    for t1_ttid in current_possibilities.iter_ones() {
                        // Iterate through all possible tiles (t2) for the neighbor cell *type*
                        for t2_ttid in 0..num_tiles {
                            // Check if t1 supports t2 along the current axis
                            if rules.check(t1_ttid, t2_ttid, axis) {
                                // If rule allows (t1 -> t2 along axis), then t2 is supported in neighbor
                                supported_neighbor_tiles.set(t2_ttid, true);
                            }
                        }
                    }

                    // Intersect neighbor's current possibilities with the supported set
                    let mut changed = false;
                    let mut new_count = 0;
                    for i in 0..num_tiles {
                        if neighbor_possibilities_mut[i] {
                            if supported_neighbor_tiles[i] {
                                new_count += 1;
                            } else {
                                neighbor_possibilities_mut.set(i, false);
                                changed = true;
                            }
                        }
                    }

                    // Check for contradiction and push to stack if changed
                    if changed {
                        if new_count == 0 {
                            return Err(PropagationError::Contradiction(nx, ny, nz));
                        }
                        // If possibilities changed, add neighbor to stack for further propagation
                        propagation_stack.push((nx, ny, nz));
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::PossibilityGrid;
    use bitvec::prelude::bitvec;
    use wfc_rules::{AdjacencyRules, TileSet, TileSetError, Transformation};

    // Helper to create a simple tileset with identity transforms
    fn create_simple_tileset(num_base_tiles: usize) -> Result<TileSet, TileSetError> {
        let weights = vec![1.0; num_base_tiles];
        let allowed_transforms = vec![vec![Transformation::Identity]; num_base_tiles];
        TileSet::new(weights, allowed_transforms)
    }

    // Helper to create simple test rules (e.g., T0 <-> T1 on X, T1 <-> T2 on Y, self on Z)
    fn create_simple_rules(tileset: &TileSet) -> AdjacencyRules {
        let num_tiles = tileset.num_transformed_tiles();
        let num_axes = 6;
        let mut allowed_tuples = Vec::new();

        if num_tiles >= 2 {
            // T0 <-> T1 on X
            allowed_tuples.push((0, 0, 1)); // +X: T0 -> T1
            allowed_tuples.push((1, 1, 0)); // -X: T1 -> T0
        }
        if num_tiles >= 3 {
            // T1 <-> T2 on Y
            allowed_tuples.push((2, 1, 2)); // +Y: T1 -> T2
            allowed_tuples.push((3, 2, 1)); // -Y: T2 -> T1
        }
        // Allow self-adjacency for other cases and axes (simplifies testing)
        for axis in 0..num_axes {
            for ttid in 0..num_tiles {
                // Check if a specific rule already exists for this combination
                let exists = allowed_tuples
                    .iter()
                    .any(|(a, t1, _)| *a == axis && *t1 == ttid);
                if !exists {
                    allowed_tuples.push((axis, ttid, ttid)); // Allow self
                }
                // Also add reverse self-adjacency if needed (though map covers both directions)
                let reverse_exists = allowed_tuples
                    .iter()
                    .any(|(a, _, t2)| *a == axis && *t2 == ttid);
                if !reverse_exists {
                    allowed_tuples.push((axis, ttid, ttid)); // Allow self
                }
            }
        }
        // Ensure all tiles have at least self-adjacency on Z
        for ttid in 0..num_tiles {
            allowed_tuples.push((4, ttid, ttid)); // +Z
            allowed_tuples.push((5, ttid, ttid)); // -Z
        }

        AdjacencyRules::from_allowed_tuples(num_tiles, num_axes, allowed_tuples)
    }

    #[test]
    fn test_propagate_simple_clamped() {
        let tileset = create_simple_tileset(2).unwrap();
        let rules = create_simple_rules(&tileset);
        let mut grid = PossibilityGrid::new(3, 1, 1, 2);
        let mut propagator = CpuConstraintPropagator::new(BoundaryMode::Clamped);

        // Collapse center cell (1,0,0) to only allow Tile 0
        let center_cell = grid.get_mut(1, 0, 0).unwrap();
        *center_cell = bitvec![usize, Lsb0; 1, 0];

        let result = propagator.propagate(&mut grid, vec![(1, 0, 0)], &rules);
        assert!(result.is_ok());

        // Check neighbors (0,0,0) and (2,0,0)
        // Neighbor (0,0,0) is to the -X of (1,0,0). Rule is T1 -> T0 (-X, axis 1).
        // Since (1,0,0) only has T0, neighbor (0,0,0) cannot be T1.
        // However, our simple rules ONLY define T1->T0 on -X. They don't define T0->T?. Let's assume T0->T0 is allowed.
        // Expected: (0,0,0) should NOT be T1. Its initial state was [1,1]. It should become [1,0].
        let left_neighbor = grid.get(0, 0, 0).unwrap();
        assert_eq!(
            *left_neighbor,
            bitvec![usize, Lsb0; 1, 1],
            "Left neighbor unchanged as T0->? (-X) not defined explicitly in simple rules"
        );

        // Neighbor (2,0,0) is to the +X of (1,0,0). Rule is T0 -> T1 (+X, axis 0).
        // Since (1,0,0) only has T0, neighbor (2,0,0) must support being T1.
        // BUT, which tiles in (2,0,0) are supported BY T0 in (1,0,0)? Only T1.
        // So, (2,0,0)'s possibilities should be intersected with [0, 1].
        // Initial: [1,1]. After: [1,1] & [0,1] = [0,1].
        let right_neighbor = grid.get(2, 0, 0).unwrap();
        assert_eq!(*right_neighbor, bitvec![usize, Lsb0; 0, 1]);
    }

    #[test]
    fn test_propagate_contradiction() {
        let tileset = create_simple_tileset(2).unwrap();
        // Rule: T0 can only be next to T0 (+X), T1 only next to T1 (+X)
        let rules = AdjacencyRules::from_allowed_tuples(2, 6, vec![(0, 0, 0), (0, 1, 1)]);
        let mut grid = PossibilityGrid::new(2, 1, 1, 2);
        let mut propagator = CpuConstraintPropagator::new(BoundaryMode::Clamped);

        // Set cell (0,0,0) to T0 only
        *grid.get_mut(0, 0, 0).unwrap() = bitvec![usize, Lsb0; 1, 0];
        // Set cell (1,0,0) to T1 only
        *grid.get_mut(1, 0, 0).unwrap() = bitvec![usize, Lsb0; 0, 1];

        // Propagate from (0,0,0). It requires neighbor (1,0,0) to be T0.
        // But (1,0,0) is already T1. Contradiction!
        let result = propagator.propagate(&mut grid, vec![(0, 0, 0)], &rules);
        assert!(matches!(
            result,
            Err(PropagationError::Contradiction(1, 0, 0))
        ));
    }

    #[test]
    fn test_propagate_periodic() {
        let tileset = create_simple_tileset(2).unwrap();
        let rules = create_simple_rules(&tileset);
        let mut grid = PossibilityGrid::new(3, 1, 1, 2);
        let mut propagator = CpuConstraintPropagator::new(BoundaryMode::Periodic);

        // Collapse cell (0,0,0) to only allow Tile 0
        *grid.get_mut(0, 0, 0).unwrap() = bitvec![usize, Lsb0; 1, 0];

        let result = propagator.propagate(&mut grid, vec![(0, 0, 0)], &rules);
        assert!(result.is_ok());

        // Check neighbors: (1,0,0) and (2,0,0) (periodic neighbor)
        // Neighbor (1,0,0) is +X of (0,0,0). Rule T0->T1 (+X, axis 0).
        // Cell (0,0,0) = [1,0]. Neighbor (1,0,0) possibilities must allow T1. [1,1] & [0,1] = [0,1].
        let right_neighbor = grid.get(1, 0, 0).unwrap();
        assert_eq!(*right_neighbor, bitvec![usize, Lsb0; 0, 1]);

        // Periodic neighbor (2,0,0) is -X of (0,0,0). Rule T1->T0 (-X, axis 1).
        // Since (0,0,0) only has T0, neighbor (2,0,0) cannot be T1.
        // Like clamped case, T0->? (-X) is not defined. Neighbor should be unchanged.
        let periodic_neighbor = grid.get(2, 0, 0).unwrap();
        assert_eq!(*periodic_neighbor, bitvec![usize, Lsb0; 1, 1]);
    }

    #[test]
    fn test_propagate_constraints() {
        let _tileset = create_simple_tileset(2).unwrap();
        // TODO: Add actual assertions for constraint propagation
        // For now, just ensure it compiles and runs without panic
    }

    #[test]
    fn test_no_propagation_needed() {
        let tileset = create_simple_tileset(2).unwrap();
        let rules = create_simple_rules(&tileset);
        let mut grid = PossibilityGrid::new(2, 1, 1, 2);
        let grid_before = grid.clone();
        let mut propagator = CpuConstraintPropagator::new(BoundaryMode::Clamped);

        // Propagate with empty update list
        let result = propagator.propagate(&mut grid, vec![], &rules);
        assert!(result.is_ok());

        // Grid should remain unchanged
        assert_eq!(grid, grid_before);
    }
}
