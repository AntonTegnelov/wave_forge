// wfc-gpu/src/coordination/propagation.rs

//! Module responsible for coordinating different constraint propagation strategies.

use crate::{GpuConstraintPropagator, GpuError};
use async_trait::async_trait;
use wfc_core::{
    grid::PossibilityGrid,
    propagator::{ConstraintPropagator, PropagationError},
};
use wfc_rules::AdjacencyRules;

// --- Traits --- //

/// Defines the interface for coordinating propagation.
/// Allows switching between different propagation strategies (e.g., direct, subgrid).
#[async_trait]
pub trait PropagationCoordinator {
    /// Executes the chosen propagation strategy.
    ///
    /// # Arguments
    /// * `propagator` - The underlying GPU propagator implementation.
    /// * `grid` - The mutable possibility grid.
    /// * `updated_coords` - Coordinates of cells that were initially updated.
    /// * `rules` - The adjacency rules.
    ///
    /// # Returns
    /// * `Ok(())` if propagation succeeds without contradiction.
    /// * `Err(PropagationError)` if a contradiction or other error occurs.
    async fn coordinate_propagation(
        &mut self,
        propagator: &mut GpuConstraintPropagator, // Pass the concrete type for now
        grid: &mut PossibilityGrid,
        updated_coords: Vec<(usize, usize, usize)>,
        rules: &AdjacencyRules,
    ) -> Result<(), PropagationError>;
}

// --- Structs --- //

/// Coordinates propagation using the standard direct approach.
#[derive(Debug, Default)]
pub struct DirectPropagationCoordinator;

#[async_trait]
impl PropagationCoordinator for DirectPropagationCoordinator {
    async fn coordinate_propagation(
        &mut self,
        propagator: &mut GpuConstraintPropagator,
        grid: &mut PossibilityGrid,
        updated_coords: Vec<(usize, usize, usize)>,
        rules: &AdjacencyRules,
    ) -> Result<(), PropagationError> {
        // Simply delegate to the propagator's default implementation
        propagator.propagate(grid, updated_coords, rules).await
    }
}

/// Coordinates propagation using the subgrid strategy.
#[derive(Debug, Default)]
pub struct SubgridPropagationCoordinator; // Requires SubgridConfig

#[async_trait]
impl PropagationCoordinator for SubgridPropagationCoordinator {
    async fn coordinate_propagation(
        &mut self,
        propagator: &mut GpuConstraintPropagator,
        grid: &mut PossibilityGrid,
        updated_coords: Vec<(usize, usize, usize)>,
        rules: &AdjacencyRules,
    ) -> Result<(), PropagationError> {
        // Delegate to the propagator's subgrid-specific logic
        // This assumes the propagator was configured with subgrid settings.
        // TODO: Refine this - maybe the coordinator holds the config?
        propagator.propagate(grid, updated_coords, rules).await // Propagator handles subgridding internally if configured
    }
}
