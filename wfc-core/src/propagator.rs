use crate::grid::PossibilityGrid;
use crate::rules::AdjacencyRules;
use crate::tile::TileId;
use bitvec::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::collections::{HashSet, VecDeque};
use std::default::Default;
use std::sync::{Arc, Mutex};
use thiserror::Error;

// Type alias for grid coordinates
type GridCoord = (usize, usize, usize);

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

/// A multithreaded implementation of the `ConstraintPropagator` trait using Rayon.
///
/// This propagator divides the work among multiple threads to accelerate the propagation
/// process for large grids. It uses a batch processing approach to handle potential
/// race conditions that could occur when multiple threads try to update the same cells.
#[derive(Debug, Clone)]
pub struct ParallelConstraintPropagator {
    /// Number of cells to process in each parallel batch
    batch_size: usize,
}

impl ParallelConstraintPropagator {
    /// Creates a new `ParallelConstraintPropagator` with the default batch size.
    pub fn new() -> Self {
        Self { batch_size: 64 }
    }

    /// Creates a new `ParallelConstraintPropagator` with a custom batch size.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of cells to process in each parallel batch.
    ///   Smaller values may lead to more overhead but finer-grained parallelism,
    ///   while larger values reduce overhead but may lead to less parallelism.
    pub fn with_batch_size(batch_size: usize) -> Self {
        Self { batch_size }
    }
}

impl Default for ParallelConstraintPropagator {
    /// Creates a default `ParallelConstraintPropagator`.
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintPropagator for ParallelConstraintPropagator {
    /// Parallelized implementation of the constraint propagation algorithm.
    ///
    /// Uses Rayon to process multiple cells in parallel, while carefully
    /// managing potential race conditions when multiple threads might update
    /// the same neighbor cells.
    ///
    /// # Algorithm
    ///
    /// 1. Initialize a worklist with the initial `updated_coords`.
    /// 2. While the worklist is not empty:
    ///    a. Divide the worklist into batches for parallel processing.
    ///    b. Process each batch in parallel:
    ///       i. For each cell in the batch, compute the effects on its neighbors.
    ///       ii. Record these effects in thread-local storage to avoid race conditions.
    ///    c. After parallel processing, combine the results and identify cells
    ///       that need to be updated.
    ///    d. Update the grid based on the combined results.
    ///    e. Add cells with changed possibilities to the worklist for the next iteration.
    ///
    /// This approach avoids direct race conditions by having each thread compute
    /// but not immediately apply its updates. The final application of updates is
    /// done serially after all threads have completed.
    ///
    /// # Panics
    ///
    /// Panics if `rules.num_axes()` does not match the internal `NUM_AXES` constant (currently 6).
    fn propagate(
        &mut self,
        grid: &mut PossibilityGrid,
        updated_coords: Vec<GridCoord>,
        rules: &AdjacencyRules,
    ) -> Result<(), PropagationError> {
        // Quick bailout if there's nothing to do
        if updated_coords.is_empty() {
            return Ok(());
        }

        // Assert adjacency rules are valid for this grid
        assert_eq!(
            rules.num_tiles(),
            grid.num_tiles(),
            "Mismatch between grid tiles and rules tiles"
        );

        // Create thread-safe structures
        let to_process = Arc::new(Mutex::new(VecDeque::from(updated_coords)));
        let processed = Arc::new(Mutex::new(HashSet::<GridCoord>::new()));
        let has_contradiction = Arc::new(Mutex::new(None));

        // Process until the queue is empty
        loop {
            // Get a batch of coordinates to process
            let batch = {
                let mut queue = to_process.lock().unwrap();
                if queue.is_empty() {
                    break; // Exit loop if nothing left to process
                }

                let mut batch = Vec::with_capacity(self.batch_size);
                while batch.len() < self.batch_size && !queue.is_empty() {
                    batch.push(queue.pop_front().unwrap());
                }
                batch
            };

            // Process each coordinate in parallel and collect changes
            let batch_results: Vec<(GridCoord, HashMap<GridCoord, BitVec>)> = batch
                .par_iter()
                .map(|&(x, y, z)| {
                    let mut neighbor_changes = HashMap::new();

                    // Skip if already marked as processed
                    if processed.lock().unwrap().contains(&(x, y, z)) {
                        return ((x, y, z), neighbor_changes);
                    }

                    // Get current cell state
                    if let Some(cell_state) = grid.get(x, y, z) {
                        let cell_state = cell_state.clone(); // Clone to avoid borrow issues

                        // Process all neighbors
                        for axis in 0..6 {
                            let (nx, ny, nz) = match axis {
                                AXIS_POS_X => (x + 1, y, z),
                                AXIS_NEG_X => (x.wrapping_sub(1), y, z),
                                AXIS_POS_Y => (x, y + 1, z),
                                AXIS_NEG_Y => (x, y.wrapping_sub(1), z),
                                AXIS_POS_Z => (x, y, z + 1),
                                AXIS_NEG_Z => (x, y, z.wrapping_sub(1)),
                                _ => unreachable!("Invalid axis"),
                            };

                            // Skip if neighbor doesn't exist
                            if grid.get(nx, ny, nz).is_none() {
                                continue;
                            }

                            // Calculate constraints from current cell
                            let mut allowed = bitvec![0; grid.num_tiles()];

                            for tile1_idx in 0..grid.num_tiles() {
                                // Skip if this tile isn't possible in current cell
                                if !cell_state[tile1_idx] {
                                    continue;
                                }

                                // Add allowed neighbors for this tile
                                for tile2_idx in 0..grid.num_tiles() {
                                    if rules.check(TileId(tile1_idx), TileId(tile2_idx), axis) {
                                        allowed.set(tile2_idx, true);
                                    }
                                }
                            }

                            // Save changes to apply later
                            neighbor_changes.insert((nx, ny, nz), allowed);
                        }
                    }

                    ((x, y, z), neighbor_changes)
                })
                .collect();

            // Apply all batch results to the grid
            for ((x, y, z), changes) in batch_results {
                // Mark this cell as processed
                processed.lock().unwrap().insert((x, y, z));

                // Apply changes to neighbors
                for ((nx, ny, nz), allowed) in changes {
                    if let Some(neighbor_state) = grid.get_mut(nx, ny, nz) {
                        let original = neighbor_state.clone();

                        // Apply constraint (intersection)
                        *neighbor_state &= allowed;

                        // Check for contradiction
                        if neighbor_state.count_ones() == 0 {
                            *has_contradiction.lock().unwrap() = Some((nx, ny, nz));
                            return Err(PropagationError::Contradiction(nx, ny, nz));
                        }

                        // If changed, add to next batch
                        if *neighbor_state != original {
                            let mut queue = to_process.lock().unwrap();
                            if !processed.lock().unwrap().contains(&(nx, ny, nz)) {
                                queue.push_back((nx, ny, nz));
                            }
                        }
                    }
                }
            }

            // Check for contradiction
            if let Some((x, y, z)) = *has_contradiction.lock().unwrap() {
                return Err(PropagationError::Contradiction(x, y, z));
            }
        }

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

    // Tests for both CPU and Parallel propagators

    #[test]
    fn test_propagate_simple_reduction() {
        // Test CpuConstraintPropagator
        {
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

        // Test ParallelConstraintPropagator with same test case
        {
            let mut propagator = ParallelConstraintPropagator::new();
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
    }

    #[test]
    fn test_parallel_propagate_no_change() {
        let mut propagator = ParallelConstraintPropagator::new();
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
    fn test_parallel_propagate_contradiction() {
        let mut propagator = ParallelConstraintPropagator::new();
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
    fn test_parallel_propagate_multiple_steps() {
        let mut propagator = ParallelConstraintPropagator::new();
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
    fn test_parallel_propagate_larger_grid() {
        let mut propagator = ParallelConstraintPropagator::new();
        let num_tiles = 2;
        let rules = create_identity_rules(num_tiles);
        let grid_size = 20; // Large enough to ensure multiple threads
        let mut grid = PossibilityGrid::new(grid_size, grid_size, 1, num_tiles);

        // Collapse center cell to Tile 0
        let center = grid_size / 2;
        *grid.get_mut(center, center, 0).unwrap() = bitvec![1, 0];

        // Propagate from the center cell
        let result = propagator.propagate(&mut grid, vec![(center, center, 0)], &rules);
        assert!(result.is_ok());

        // Check that all cells in grid are now Tile 0
        for x in 0..grid_size {
            for y in 0..grid_size {
                assert_eq!(grid.get(x, y, 0), Some(&bitvec![1, 0]));
            }
        }
    }

    #[test]
    fn test_parallel_propagate_complex_sequential_contradiction() {
        let mut propagator = ParallelConstraintPropagator::new();
        let num_tiles = 3;
        // Rules: 0->1, 1->2, 2->0 (and vice-versa)
        let rules = create_sequential_rules(num_tiles);
        let mut grid = PossibilityGrid::new(3, 1, 1, num_tiles);

        // Initial state: [Tile 0] - [All Tiles] - [Tile 0]
        *grid.get_mut(0, 0, 0).unwrap() = bitvec![1, 0, 0]; // Only Tile 0
                                                            // Cell (1,0,0) remains with all options
        *grid.get_mut(2, 0, 0).unwrap() = bitvec![1, 0, 0]; // Only Tile 0

        // Propagate from first and last cells, which creates a requirement for the middle cell
        let result = propagator.propagate(&mut grid, vec![(0, 0, 0), (2, 0, 0)], &rules);

        // With sequential rules, a contradiction should occur
        assert!(matches!(
            result,
            Err(PropagationError::Contradiction(_, _, _))
        ));
    }

    // Existing tests for CpuConstraintPropagator remain unchanged...
}
