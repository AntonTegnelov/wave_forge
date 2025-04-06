/// Constraint propagation logic and traits.
/// Make propagator module public
use crate::grid::PossibilityGrid;
use async_trait::async_trait;
use thiserror::Error;
use wfc_rules::AdjacencyRules;

/// Errors that can occur during the constraint propagation phase of WFC.
#[derive(Debug, Error)]
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
    /// An internal error within the propagation logic.
    #[error("Internal propagation error: {0}")]
    InternalError(String),
}

/// Trait defining the interface for a constraint propagation algorithm.
///
/// Implementors of this trait are responsible for updating the `PossibilityGrid`
/// based on changes initiated (e.g., collapsing a cell) and the defined `AdjacencyRules`.
#[async_trait]
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
    async fn propagate(
        &mut self,
        grid: &mut PossibilityGrid,
        updated_coords: Vec<(usize, usize, usize)>,
        rules: &AdjacencyRules,
    ) -> Result<(), PropagationError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::PossibilityGrid;
    use wfc_rules::AdjacencyRules;

    // Helper function to create a simple 2-tile, 1-axis ruleset (3D compatible)
    fn setup_simple_rules_3d() -> AdjacencyRules {
        let num_tiles = 2; // These are transformed tile indices (assuming Identity only)
        let num_axes = 6;
        // Specify allowed rules as tuples (axis, ttid1, ttid2)
        let mut allowed_tuples = Vec::new();

        // Allow 0 -> 1 along +X (axis 0)
        allowed_tuples.push((0, 0, 1));
        // Allow 1 -> 0 along -X (axis 1)
        allowed_tuples.push((1, 1, 0));

        // Allow self-adjacency along other axes (Y, Z) for simplicity
        for axis in 2..num_axes {
            for tile_idx in 0..num_tiles {
                // tile_idx is the transformed tile index (0 or 1)
                allowed_tuples.push((axis, tile_idx, tile_idx));
            }
        }
        // Use the new constructor
        AdjacencyRules::from_allowed_tuples(num_tiles, num_axes, allowed_tuples)
    }

    // --- Test cases for propagation logic (GPU or GPU) ---
    // Note: These tests focus on the expected outcome (grid state changes),
    // not the specific implementation (GPU). They require a concrete
    // propagator implementation to run.

    // Mock Propagator (if needed for tests independent of GPU impl)
    struct MockPropagator;
    #[async_trait]
    impl ConstraintPropagator for MockPropagator {
        async fn propagate(
            &mut self,
            _grid: &mut PossibilityGrid,
            _updated_coords: Vec<(usize, usize, usize)>,
            _rules: &AdjacencyRules,
        ) -> Result<(), PropagationError> {
            // No-op for testing trait bounds, etc.
            Ok(())
        }
    }

    // A basic test that uses the MockPropagator (if you need a placeholder)
    #[tokio::test]
    async fn test_mock_propagator() {
        let mut grid = PossibilityGrid::new(2, 1, 1, 2);
        let rules = setup_simple_rules_3d();
        let mut propagator = MockPropagator;
        let result = propagator
            .propagate(&mut grid, vec![(0, 0, 0)], &rules)
            .await;
        assert!(result.is_ok());
    }

    // Add more tests here that verify the *behavior* of propagation,
    // potentially using setup functions that instantiate the propagator
    // under test (e.g., create a test-specific GpuAccelerator instance?).
    // Testing GPU logic directly often requires integration tests or specific
    // GPU testing frameworks/setups.
}
