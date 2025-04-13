// wfc-gpu/src/coordination/propagation.rs

//! Module responsible for coordinating different constraint propagation strategies.

use crate::propagator::GpuConstraintPropagator;
use crate::utils::RwLock; // Use the type alias
use async_trait::async_trait;
use std::fmt::Debug;
use std::sync::Arc;
use wfc_core::{
    grid::PossibilityGrid,
    propagator::{ConstraintPropagator, PropagationError},
};
use wfc_rules::AdjacencyRules;

// --- Traits --- //

/// Defines the strategy interface for propagation coordination.
/// Implementations provide different approaches to coordinating constraint
/// propagation for the Wave Function Collapse algorithm.
#[async_trait]
pub trait PropagationCoordinationStrategy: Debug + Send + Sync {
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
        propagator: &Arc<RwLock<GpuConstraintPropagator>>,
        grid: &mut PossibilityGrid,
        updated_coords: Vec<(usize, usize, usize)>,
        rules: &AdjacencyRules,
    ) -> Result<(), PropagationError>;

    /// Clone this strategy into a boxed trait object.
    fn clone_box(&self) -> Box<dyn PropagationCoordinationStrategy>;
}

impl Clone for Box<dyn PropagationCoordinationStrategy> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Legacy interface for coordinating propagation.
/// This is kept for backward compatibility and delegates to the new strategy pattern.
/// New code should prefer using PropagationCoordinationStrategy directly.
#[async_trait]
#[deprecated(
    since = "0.1.0",
    note = "Use PropagationCoordinationStrategy instead which provides a more comprehensive interface"
)]
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

/// Factory for creating propagation coordination strategies.
pub struct PropagationCoordinationStrategyFactory;

impl PropagationCoordinationStrategyFactory {
    /// Creates a direct propagation coordination strategy.
    pub fn create_direct() -> Box<dyn PropagationCoordinationStrategy> {
        Box::new(DirectPropagationCoordinationStrategy::new())
    }

    /// Creates a subgrid propagation coordination strategy.
    pub fn create_subgrid() -> Box<dyn PropagationCoordinationStrategy> {
        Box::new(SubgridPropagationCoordinationStrategy::new())
    }

    /// Creates an adaptive propagation coordination strategy based on grid size.
    pub fn create_adaptive(
        grid_size: (usize, usize, usize),
    ) -> Box<dyn PropagationCoordinationStrategy> {
        // Choose strategy based on grid size
        if grid_size.0 * grid_size.1 * grid_size.2 > 1_000_000 {
            Self::create_subgrid()
        } else {
            Self::create_direct()
        }
    }
}

// --- Structs --- //

/// Coordinates propagation using the standard direct approach.
#[derive(Debug, Default, Clone)]
pub struct DirectPropagationCoordinationStrategy;

impl DirectPropagationCoordinationStrategy {
    /// Creates a new direct propagation coordinator.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl PropagationCoordinationStrategy for DirectPropagationCoordinationStrategy {
    async fn coordinate_propagation(
        &mut self,
        propagator: &Arc<RwLock<GpuConstraintPropagator>>,
        grid: &mut PossibilityGrid,
        updated_coords: Vec<(usize, usize, usize)>,
        rules: &AdjacencyRules,
    ) -> Result<(), PropagationError> {
        // Get a write lock asynchronously. The tokio lock guard is Send,
        // so it can be held across the await point
        let mut propagator_guard = propagator.write().await;

        // Call propagate and await the result while holding the lock
        propagator_guard
            .propagate(grid, updated_coords.clone(), rules)
            .await
    }

    fn clone_box(&self) -> Box<dyn PropagationCoordinationStrategy> {
        Box::new(self.clone())
    }
}

/// Coordinates propagation using the subgrid strategy.
#[derive(Debug, Default, Clone)]
pub struct SubgridPropagationCoordinationStrategy;

impl SubgridPropagationCoordinationStrategy {
    /// Creates a new subgrid propagation coordinator.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl PropagationCoordinationStrategy for SubgridPropagationCoordinationStrategy {
    async fn coordinate_propagation(
        &mut self,
        propagator: &Arc<RwLock<GpuConstraintPropagator>>,
        grid: &mut PossibilityGrid,
        updated_coords: Vec<(usize, usize, usize)>,
        rules: &AdjacencyRules,
    ) -> Result<(), PropagationError> {
        // Get a write lock asynchronously. The tokio lock guard is Send,
        // so it can be held across the await point
        let mut propagator_guard = propagator.write().await;

        // Call propagate and await the result while holding the lock
        propagator_guard
            .propagate(grid, updated_coords.clone(), rules)
            .await
    }

    fn clone_box(&self) -> Box<dyn PropagationCoordinationStrategy> {
        Box::new(self.clone())
    }
}

// Legacy implementations for backward compatibility

/// Coordinates propagation using the standard direct approach.
/// Legacy implementation that delegates to the new strategy.
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
/// Legacy implementation that delegates to the new strategy.
#[derive(Debug, Default)]
pub struct SubgridPropagationCoordinator;

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
        propagator.propagate(grid, updated_coords, rules).await
    }
}
