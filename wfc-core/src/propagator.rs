use crate::grid::PossibilityGrid;
use crate::rules::AdjacencyRules;
use crate::tile::TileId;
use bitvec::prelude::*;
use std::collections::{HashSet, VecDeque};
use std::default::Default;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PropagationError {
    #[error("Contradiction detected during propagation at ({0}, {1}, {2})")]
    Contradiction(usize, usize, usize),
    // Add other specific propagation errors later
}

#[must_use]
pub trait ConstraintPropagator {
    fn propagate(
        &mut self,
        grid: &mut PossibilityGrid,
        updated_coords: Vec<(usize, usize, usize)>,
        rules: &AdjacencyRules,
    ) -> Result<(), PropagationError>;
}

// Basic CPU implementation
#[derive(Debug, Clone)]
pub struct CpuConstraintPropagator;

impl CpuConstraintPropagator {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuConstraintPropagator {
    fn default() -> Self {
        Self::new()
    }
}

// Define axes explicitly for clarity (assuming 6 axes)
const AXIS_POS_X: usize = 0;
const AXIS_NEG_X: usize = 1;
const AXIS_POS_Y: usize = 2;
const AXIS_NEG_Y: usize = 3;
const AXIS_POS_Z: usize = 4;
const AXIS_NEG_Z: usize = 5;
const NUM_AXES: usize = 6; // Should match AdjacencyRules num_axes

// Axis definitions: (dx, dy, dz, axis_index, opposite_axis_index)
const AXES: [(isize, isize, isize, usize, usize); NUM_AXES] = [
    (1, 0, 0, AXIS_POS_X, AXIS_NEG_X),  // +X
    (-1, 0, 0, AXIS_NEG_X, AXIS_POS_X), // -X
    (0, 1, 0, AXIS_POS_Y, AXIS_NEG_Y),  // +Y
    (0, -1, 0, AXIS_NEG_Y, AXIS_POS_Y), // -Y
    (0, 0, 1, AXIS_POS_Z, AXIS_NEG_Z),  // +Z
    (0, 0, -1, AXIS_NEG_Z, AXIS_POS_Z), // -Z
];

impl ConstraintPropagator for CpuConstraintPropagator {
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
                    *neighbor_possibilities &= &allowed_neighbor_tiles; // Intersect with allowed

                    // Check for contradiction
                    if neighbor_possibilities.not_any() {
                        return Err(PropagationError::Contradiction(nx, ny, nz));
                    }

                    // If possibilities changed AND the neighbor was not already in the worklist_set, add it.
                    if *neighbor_possibilities != original_neighbor_possibilities
                        && worklist_set.insert((nx, ny, nz))
                    {
                        worklist.push_back((nx, ny, nz));
                    }
                }
            }
        }

        // TODO: Explore rayon for parallelism (complex due to potential simultaneous writes to neighbors)
        Ok(())
    }
}
