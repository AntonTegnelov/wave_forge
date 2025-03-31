use crate::grid::PossibilityGrid;
use crate::rules::AdjacencyRules;
use crate::tile::TileId;
use bitvec::prelude::*;
use std::collections::{HashSet, VecDeque};
use std::default::Default;
use thiserror::Error;

/// Errors that can occur during the constraint propagation phase of WFC.
#[derive(Error, Debug, Clone)]
pub enum PropagationError {
    /// Indicates that a cell's possibility set became empty during propagation,
    /// meaning no tile can satisfy the constraints at this location.
    /// Contains the (x, y, z) coordinates of the contradictory cell.
    #[error("Contradiction detected during propagation at ({0}, {1}, {2})")]
    Contradiction(usize, usize, usize),
    /// Represents an error during the setup phase of GPU-based propagation (if used).
    #[error("GPU setup error during propagation: {0}")]
    GpuSetupError(String),
    /// Represents an error during communication with the GPU during propagation (if used).
    #[error("GPU communication error during propagation: {0}")]
    GpuCommunicationError(String),
}

/// Trait defining the interface for a constraint propagation algorithm.
///
/// Implementors of this trait are responsible for updating the `PossibilityGrid`
/// based on changes initiated (e.g., collapsing a cell) and the defined `AdjacencyRules`.
#[must_use] // Propagation results in a Result that should be checked
pub trait ConstraintPropagator {
    /// Performs constraint propagation on the grid.
    ///
    /// Takes the current state of the `PossibilityGrid`, a list of coordinates
    /// `updated_coords` where possibilities have recently changed (e.g., due to a collapse),
    /// and the `AdjacencyRules`.
    ///
    /// Modifies the `grid` in place by removing possibilities that violate the rules
    /// based on the updates.
    ///
    /// # Arguments
    ///
    /// * `grid` - The mutable `PossibilityGrid` to operate on.
    /// * `updated_coords` - A vector of (x, y, z) coordinates indicating cells whose possibilities have changed.
    /// * `rules` - The `AdjacencyRules` defining valid neighbor relationships.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if propagation completed successfully without contradictions.
    /// * `Err(PropagationError::Contradiction(x, y, z))` if a cell at (x, y, z) becomes empty (contradiction).
    /// * Other `Err(PropagationError)` variants for different propagation issues (e.g., GPU errors).
    fn propagate(
        &mut self,
        grid: &mut PossibilityGrid,
        updated_coords: Vec<(usize, usize, usize)>,
        rules: &AdjacencyRules,
    ) -> Result<(), PropagationError>;
}

/// A basic, single-threaded CPU implementation of the `ConstraintPropagator` trait.
///
/// This propagator uses a worklist (queue) to manage cells whose possibilities
/// need to be re-evaluated and propagated to their neighbors.
#[derive(Debug, Clone)]
pub struct CpuConstraintPropagator;

impl CpuConstraintPropagator {
    /// Creates a new `CpuConstraintPropagator`.
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuConstraintPropagator {
    /// Creates a default `CpuConstraintPropagator`.
    fn default() -> Self {
        Self::new()
    }
}

// --- Axis Definitions (Internal Detail but useful for understanding) ---

/// Index for the positive X-axis direction (+1, 0, 0).
const AXIS_POS_X: usize = 0;
/// Index for the negative X-axis direction (-1, 0, 0).
const AXIS_NEG_X: usize = 1;
/// Index for the positive Y-axis direction (0, +1, 0).
const AXIS_POS_Y: usize = 2;
/// Index for the negative Y-axis direction (0, -1, 0).
const AXIS_NEG_Y: usize = 3;
/// Index for the positive Z-axis direction (0, 0, +1).
const AXIS_POS_Z: usize = 4;
/// Index for the negative Z-axis direction (0, 0, -1).
const AXIS_NEG_Z: usize = 5;
/// Total number of axes (should match `AdjacencyRules::num_axes`).
const NUM_AXES: usize = 6;

/// Array defining the properties of each axis for neighbor checking.
/// Format: `(dx, dy, dz, axis_index, opposite_axis_index)`
/// Used to iterate through neighbors and apply the correct adjacency rule direction.
const AXES: [(isize, isize, isize, usize, usize); NUM_AXES] = [
    (1, 0, 0, AXIS_POS_X, AXIS_NEG_X),  // +X
    (-1, 0, 0, AXIS_NEG_X, AXIS_POS_X), // -X
    (0, 1, 0, AXIS_POS_Y, AXIS_NEG_Y),  // +Y
    (0, -1, 0, AXIS_NEG_Y, AXIS_POS_Y), // -Y
    (0, 0, 1, AXIS_POS_Z, AXIS_NEG_Z),  // +Z
    (0, 0, -1, AXIS_NEG_Z, AXIS_POS_Z), // -Z
];

// --- CPU Propagator Implementation ---

impl ConstraintPropagator for CpuConstraintPropagator {
    /// CPU implementation of the constraint propagation algorithm.
    ///
    /// Uses a worklist approach (VecDeque and HashSet) to process updated cells
    /// and their neighbors iteratively until no further changes occur or a
    /// contradiction is found.
    ///
    /// # Algorithm
    ///
    /// 1. Initialize a worklist queue and set with the initial `updated_coords`.
    /// 2. While the worklist is not empty:
    ///    a. Dequeue a cell coordinate (x, y, z).
    ///    b. Retrieve the current possibilities `P` for cell (x, y, z).
    ///    c. For each neighbor (nx, ny, nz) of (x, y, z) along each `axis`:
    ///       i. Determine the set of allowed tiles `A` for the neighbor based on `P` and `rules.check(..., axis)`.
    ///       ii. Get the neighbor's current possibilities `N`.
    ///       iii. Calculate the intersection `I = N & A`.
    ///       iv. If `I` is empty, return `Err(PropagationError::Contradiction)`.
    ///       v. If `I` is different from `N`:
    ///          - Update the neighbor's possibilities in the `grid` to `I`.
    ///          - Enqueue (nx, ny, nz) to the worklist if it wasn't already present.
    /// 3. If the loop finishes without contradiction, return `Ok(())`.
    ///
    /// # Panics
    ///
    /// Panics if `rules.num_axes()` does not match the internal `NUM_AXES` constant (currently 6).
    /// This indicates an inconsistency between the rule definition and the propagator's expectation.
    fn propagate(
        &mut self,
        grid: &mut PossibilityGrid,
        updated_coords: Vec<(usize, usize, usize)>,
        rules: &AdjacencyRules,
    ) -> Result<(), PropagationError> {
        assert_eq!(
            rules.num_axes(),
            NUM_AXES,
            "AdjacencyRules has unexpected number of axes"
        );

        let num_tiles = rules.num_tiles();
        let mut worklist: VecDeque<(usize, usize, usize)> = updated_coords.into();
        let mut worklist_set: HashSet<(usize, usize, usize)> = worklist.iter().copied().collect();

        while let Some((x, y, z)) = worklist.pop_front() {
            worklist_set.remove(&(x, y, z));

            // Get current possibilities for the cell that caused the update
            // Clone needed as we might borrow grid mutably later for neighbors
            let current_possibilities = match grid.get(x, y, z) {
                Some(p) => p.clone(),
                None => continue, // Should not happen if coords are valid
            };

            // If this cell is already a contradiction (empty), we shouldn't process it,
            // but the contradiction should have been caught when it *became* empty.
            // If it starts empty (initial state?), that's an issue upstream.
            if current_possibilities.not_any() {
                // This might indicate an issue, but let's assume initial state is valid.
                // A contradiction is only returned when a CHANGE results in empty.
                continue;
            }

            // Check all neighbors
            for (dx, dy, dz, axis, _opposite_axis) in AXES.iter().cloned() {
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                let nz = z as isize + dz;

                // Check bounds
                if nx < 0
                    || nx >= grid.width as isize
                    || ny < 0
                    || ny >= grid.height as isize
                    || nz < 0
                    || nz >= grid.depth as isize
                {
                    continue;
                }
                let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);

                // Calculate the set of tiles allowed in the neighbor based on the current cell's possibilities
                let mut allowed_neighbor_tiles = bitvec![0; num_tiles];
                for tile1_idx in current_possibilities.iter_ones() {
                    for tile2_idx in 0..num_tiles {
                        // Use the *axis* rule when checking from current -> neighbor
                        // (Corrected logic: We need to know what tile2 can be placed next to tile1 along 'axis')
                        if rules.check(TileId(tile1_idx), TileId(tile2_idx), axis) {
                            allowed_neighbor_tiles.set(tile2_idx, true);
                        }
                    }
                }

                // Get the neighbor's current possibilities and update them
                if let Some(neighbor_possibilities) = grid.get_mut(nx, ny, nz) {
                    let original_neighbor_possibilities = neighbor_possibilities.clone();

                    // Calculate the intersection
                    let intersection =
                        original_neighbor_possibilities.clone() & &allowed_neighbor_tiles;

                    // REMOVE DEBUG PRINT (Temporary)
                    // if (x, y, z) == (1, 0, 0) && (nx, ny, nz) == (2, 0, 0) { ... }
                    // END REMOVE DEBUG PRINT

                    // Check for contradiction first
                    if intersection.not_any() {
                        return Err(PropagationError::Contradiction(nx, ny, nz));
                    }

                    // If the intersection resulted in a change, update the grid and add to worklist
                    if intersection != original_neighbor_possibilities {
                        *neighbor_possibilities = intersection; // Update the grid cell
                        if worklist_set.insert((nx, ny, nz)) {
                            // Avoid adding duplicates to queue
                            worklist.push_back((nx, ny, nz));
                        }
                    }
                    // No else needed: if no change, do nothing
                }
            }
        }

        // TODO: Explore rayon for parallelism (complex due to potential simultaneous writes to neighbors)
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        grid::PossibilityGrid,
        rules::AdjacencyRules,
        // Removed: tile::{TileId, TileSet},
    };
    // Removed: use bitvec::prelude::*;
    // Removed: use bitvec::vec::BitVec; // Keep specific BitVec import if needed

    // Helper to create simple rules: Tile `i` can only be adjacent to Tile `i`.
    fn create_identity_rules(num_tiles: usize) -> AdjacencyRules {
        let num_axes = 6;
        let mut allowed = vec![false; num_axes * num_tiles * num_tiles];
        for axis in 0..num_axes {
            for tile_idx in 0..num_tiles {
                let index = axis * num_tiles * num_tiles + tile_idx * num_tiles + tile_idx;
                allowed[index] = true;
            }
        }
        AdjacencyRules::new(num_tiles, num_axes, allowed)
    }

    // Helper: Creates rules where Tile `i` can only be adjacent to Tile `i+1` (and `n-1` to `0`)
    // Updated logic AGAIN to correctly handle symmetric rules across axes.
    fn create_sequential_rules(num_tiles: usize) -> AdjacencyRules {
        let num_axes = 6;
        let n_sq = num_tiles * num_tiles;
        let mut allowed = vec![false; num_axes * n_sq];
        // Define pairs of positive/negative axes
        let axis_pairs: [(usize, usize); 3] = [
            (AXIS_POS_X, AXIS_NEG_X), // (0, 1)
            (AXIS_POS_Y, AXIS_NEG_Y), // (2, 3)
            (AXIS_POS_Z, AXIS_NEG_Z), // (4, 5)
        ];

        for t1_idx in 0..num_tiles {
            let t2_idx = (t1_idx + 1) % num_tiles; // Pair (t1, t2) for sequential

            for &(pos_axis, neg_axis) in &axis_pairs {
                // Rule: t1 -> t2 along positive axis
                let index_fwd = pos_axis * n_sq + t1_idx * num_tiles + t2_idx;
                if index_fwd < allowed.len() {
                    allowed[index_fwd] = true;
                }

                // Rule: t2 -> t1 along negative axis (opposite direction)
                let index_bwd = neg_axis * n_sq + t2_idx * num_tiles + t1_idx;
                if index_bwd < allowed.len() {
                    allowed[index_bwd] = true;
                }
            }
        }
        AdjacencyRules::new(num_tiles, num_axes, allowed)
    }

    #[test]
    fn test_propagate_simple_reduction() {
        let mut propagator = CpuConstraintPropagator::new();
        let num_tiles = 2;
        let rules = create_identity_rules(num_tiles);
        let mut grid = PossibilityGrid::new(2, 1, 1, num_tiles);

        // Collapse cell (0,0,0) to Tile 0
        *grid.get_mut(0, 0, 0).unwrap() = bitvec![1, 0];

        // Propagate from the updated cell
        let result = propagator.propagate(&mut grid, vec![(0, 0, 0)], &rules);

        assert!(result.is_ok());
        // Cell (1,0,0) should now only allow Tile 0 due to identity rule
        assert_eq!(grid.get(1, 0, 0), Some(&bitvec![1, 0]));
        // Cell (0,0,0) remains unchanged
        assert_eq!(grid.get(0, 0, 0), Some(&bitvec![1, 0]));
    }

    #[test]
    fn test_propagate_no_change() {
        let mut propagator = CpuConstraintPropagator::new();
        let num_tiles = 2;
        let rules = create_identity_rules(num_tiles);
        let mut grid = PossibilityGrid::new(2, 1, 1, num_tiles);

        // Set both cells to Tile 0 already
        *grid.get_mut(0, 0, 0).unwrap() = bitvec![1, 0];
        *grid.get_mut(1, 0, 0).unwrap() = bitvec![1, 0];
        let original_grid = grid.clone();

        // Propagate from cell (0,0,0)
        let result = propagator.propagate(&mut grid, vec![(0, 0, 0)], &rules);

        assert!(result.is_ok());
        // Grid should not have changed
        assert_eq!(grid.get(0, 0, 0), original_grid.get(0, 0, 0));
        assert_eq!(grid.get(1, 0, 0), original_grid.get(1, 0, 0));
    }

    #[test]
    fn test_propagate_contradiction() {
        let mut propagator = CpuConstraintPropagator::new();
        let num_tiles = 2;
        let rules = create_identity_rules(num_tiles); // Tile 0 needs Tile 0, Tile 1 needs Tile 1
        let mut grid = PossibilityGrid::new(2, 1, 1, num_tiles);

        // Set cell (0,0,0) to Tile 0
        *grid.get_mut(0, 0, 0).unwrap() = bitvec![1, 0];
        // Set cell (1,0,0) to Tile 1 - this creates the contradiction potential
        *grid.get_mut(1, 0, 0).unwrap() = bitvec![0, 1];

        // Propagate from cell (0,0,0). This should try to force cell (1,0,0) to Tile 0.
        let result = propagator.propagate(&mut grid, vec![(0, 0, 0)], &rules);

        // Expect a contradiction error at cell (1,0,0)
        assert!(matches!(
            result,
            Err(PropagationError::Contradiction(1, 0, 0))
        ));
    }

    #[test]
    fn test_propagate_multiple_steps() {
        let mut propagator = CpuConstraintPropagator::new();
        let num_tiles = 2;
        let rules = create_identity_rules(num_tiles);
        let mut grid = PossibilityGrid::new(3, 1, 1, num_tiles);

        // Collapse cell (0,0,0) to Tile 0
        *grid.get_mut(0, 0, 0).unwrap() = bitvec![1, 0];

        // Propagate from the updated cell
        let result = propagator.propagate(&mut grid, vec![(0, 0, 0)], &rules);
        assert!(result.is_ok());

        // Cell (1,0,0) should be reduced to Tile 0
        assert_eq!(grid.get(1, 0, 0), Some(&bitvec![1, 0]));
        // Cell (2,0,0) should ALSO be reduced to Tile 0 due to propagation from (1,0,0)
        assert_eq!(grid.get(2, 0, 0), Some(&bitvec![1, 0]));
    }

    #[test]
    fn test_propagate_empty_update() {
        let mut propagator = CpuConstraintPropagator::new();
        let num_tiles = 2;
        let rules = create_identity_rules(num_tiles);
        let mut grid = PossibilityGrid::new(2, 1, 1, num_tiles);
        let original_grid = grid.clone();

        // Propagate with no updated coords
        let result = propagator.propagate(&mut grid, vec![], &rules);

        assert!(result.is_ok());
        // Grid should be unchanged
        assert_eq!(grid.get(0, 0, 0), original_grid.get(0, 0, 0));
        assert_eq!(grid.get(1, 0, 0), original_grid.get(1, 0, 0));
    }

    #[test]
    fn test_propagate_complex_sequential_contradiction() {
        let mut propagator = CpuConstraintPropagator::new();
        let num_tiles = 3;
        // Rules: 0->1, 1->2, 2->0 (and vice-versa)
        let rules = create_sequential_rules(num_tiles);
        let mut grid = PossibilityGrid::new(3, 1, 1, num_tiles);

        // Initial state: [Tile 0] - [All Tiles] - [Tile 0]
        // This state inherently leads to a contradiction via propagation
        *grid.get_mut(0, 0, 0).unwrap() = bitvec![1, 0, 0];
        *grid.get_mut(2, 0, 0).unwrap() = bitvec![1, 0, 0];

        // Propagate starting from cell (0,0,0) (Tile 0).
        // - (0,0,0) forces (1,0,0) to be Tile 1.
        // - Worklist adds (1,0,0).
        // - Propagate from (1,0,0) (Tile 1).
        // - (1,0,0) forces (2,0,0) to be Tile 2.
        // - Neighbor (2,0,0) state [1,0,0] intersected with allowed [0,0,1] becomes [0,0,0] -> Contradiction.
        let result = propagator.propagate(&mut grid, vec![(0, 0, 0)], &rules);

        assert!(
            matches!(
                result,
                Err(PropagationError::Contradiction(2, 0, 0)) // Expect contradiction at (2,0,0)
            ),
            "Expected contradiction at (2,0,0), but got: {:?}",
            result
        );

        // Add another scenario: Force a contradiction by manual edit after initial state settles
        let mut grid = PossibilityGrid::new(3, 1, 1, num_tiles);
        // State: [0] - [1] - [2] (Consistent initial state)
        *grid.get_mut(0, 0, 0).unwrap() = bitvec![1, 0, 0];
        *grid.get_mut(1, 0, 0).unwrap() = bitvec![0, 1, 0];
        *grid.get_mut(2, 0, 0).unwrap() = bitvec![0, 0, 1];

        // Force cell (2,0,0) to Tile 1. State: [0] - [1] - [1]
        *grid.get_mut(2, 0, 0).unwrap() = bitvec![0, 1, 0];

        // Propagate from the changed cell (2,0,0) (Tile 1).
        // - Checks neighbor (1,0,0) along -X.
        // - Source (2,0,0) has Tile 1.
        // - Rule `rules.check(TileId(1), TileId(tile2_idx), AXIS_NEG_X)` allows only Tile 2.
        // - Neighbor (1,0,0) state [0,1,0] intersected with allowed [0,0,1] becomes [0,0,0] -> Contradiction.
        let result = propagator.propagate(&mut grid, vec![(2, 0, 0)], &rules);
        assert!(
            matches!(
                result,
                Err(PropagationError::Contradiction(1, 0, 0)) // Expect contradiction at (1,0,0)
            ),
            "Expected contradiction at (1,0,0) after manual change, but got: {:?}",
            result
        );
    }

    #[test]
    fn test_sequential_rules_check() {
        let num_tiles = 3;
        let rules = create_sequential_rules(num_tiles);
        // Removed direct access to private rules.allowed
        // Rely on the public check() method for verification.

        // Test using the check function
        // Axis 0 (+X)
        assert!(rules.check(TileId(0), TileId(1), 0), "Check +X: 0->1");
        assert!(rules.check(TileId(1), TileId(2), 0), "Check +X: 1->2");
        assert!(rules.check(TileId(2), TileId(0), 0), "Check +X: 2->0");
        assert!(!rules.check(TileId(1), TileId(0), 0), "Check +X: 1->0");
        assert!(!rules.check(TileId(2), TileId(1), 0), "Check +X: 2->1");

        // Axis 1 (-X)
        assert!(rules.check(TileId(1), TileId(0), 1), "Check -X: 1->0");
        assert!(rules.check(TileId(2), TileId(1), 1), "Check -X: 2->1");
        assert!(rules.check(TileId(0), TileId(2), 1), "Check -X: 0->2");
        assert!(!rules.check(TileId(0), TileId(1), 1), "Check -X: 0->1");
        assert!(!rules.check(TileId(1), TileId(2), 1), "Check -X: 1->2");
    }

    #[test]
    fn test_propagation_error_variants() {
        let err1 = PropagationError::Contradiction(1, 2, 3);
        let err2 = PropagationError::GpuSetupError("Setup failed".to_string());
        let err3 = PropagationError::GpuCommunicationError("Comm failed".to_string());

        assert!(matches!(err1, PropagationError::Contradiction(1, 2, 3)));
        match err2 {
            PropagationError::GpuSetupError(msg) => assert_eq!(msg, "Setup failed"),
            _ => panic!("Expected GpuSetupError"),
        }
        match err3 {
            PropagationError::GpuCommunicationError(msg) => assert_eq!(msg, "Comm failed"),
            _ => panic!("Expected GpuCommunicationError"),
        }
    }
}
