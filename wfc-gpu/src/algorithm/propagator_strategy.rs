//! Propagation strategy implementations for the WFC algorithm.
//! This module provides different strategies for propagating constraints
//! after a cell collapse in the Wave Function Collapse algorithm.

use crate::{buffers::GpuBuffers, error_recovery::GridCoord, sync::GpuSynchronizer, GpuError};
use std::sync::Arc;
use wfc_core::{grid::PossibilityGrid, propagator::PropagationError};

/// Strategy trait for constraint propagation in WFC algorithm.
/// This defines the interface for different propagation strategies,
/// allowing the propagator to be adapted for different use cases.
pub trait PropagationStrategy: Send + Sync {
    /// Get the name of this propagation strategy
    fn name(&self) -> &str;

    /// Prepare for propagation by initializing any necessary buffers or state
    fn prepare(&self, synchronizer: &GpuSynchronizer) -> Result<(), PropagationError>;

    /// Propagate constraints from the specified cells
    fn propagate(
        &self,
        grid: &mut PossibilityGrid,
        updated_cells: &[GridCoord],
        buffers: &Arc<GpuBuffers>,
        synchronizer: &GpuSynchronizer,
    ) -> Result<(), PropagationError>;

    /// Clean up any resources used during propagation
    fn cleanup(&self, synchronizer: &GpuSynchronizer) -> Result<(), PropagationError>;
}

/// Direct propagation strategy - propagates constraints directly across
/// the entire grid without any partitioning or optimization.
pub struct DirectPropagationStrategy {
    name: String,
    max_iterations: u32,
}

impl DirectPropagationStrategy {
    /// Create a new direct propagation strategy
    pub fn new(max_iterations: u32) -> Self {
        Self {
            name: "Direct Propagation".to_string(),
            max_iterations,
        }
    }

    /// Create a new strategy with default settings
    pub fn default() -> Self {
        Self::new(1000)
    }
}

impl PropagationStrategy for DirectPropagationStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn prepare(&self, _synchronizer: &GpuSynchronizer) -> Result<(), PropagationError> {
        // Direct propagation doesn't need special preparation
        Ok(())
    }

    fn propagate(
        &self,
        _grid: &mut PossibilityGrid,
        _updated_cells: &[GridCoord],
        _buffers: &Arc<GpuBuffers>,
        _synchronizer: &GpuSynchronizer,
    ) -> Result<(), PropagationError> {
        // Placeholder - will be implemented when extracting logic from propagator.rs
        Ok(())
    }

    fn cleanup(&self, _synchronizer: &GpuSynchronizer) -> Result<(), PropagationError> {
        // Direct propagation doesn't need special cleanup
        Ok(())
    }
}

/// Subgrid propagation strategy - divides the grid into smaller subgrids
/// to optimize propagation for large grids.
pub struct SubgridPropagationStrategy {
    name: String,
    max_iterations: u32,
    subgrid_size: u32,
}

impl SubgridPropagationStrategy {
    /// Create a new subgrid propagation strategy
    pub fn new(max_iterations: u32, subgrid_size: u32) -> Self {
        Self {
            name: "Subgrid Propagation".to_string(),
            max_iterations,
            subgrid_size,
        }
    }

    /// Create a new strategy with default settings
    pub fn default() -> Self {
        Self::new(1000, 16)
    }
}

impl PropagationStrategy for SubgridPropagationStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn prepare(&self, _synchronizer: &GpuSynchronizer) -> Result<(), PropagationError> {
        // Subgrid propagation may need to initialize subgrid buffers
        // Placeholder - will be implemented when extracting logic from subgrid.rs
        Ok(())
    }

    fn propagate(
        &self,
        _grid: &mut PossibilityGrid,
        _updated_cells: &[GridCoord],
        _buffers: &Arc<GpuBuffers>,
        _synchronizer: &GpuSynchronizer,
    ) -> Result<(), PropagationError> {
        // Placeholder - will be implemented when extracting logic from propagator.rs and subgrid.rs
        Ok(())
    }

    fn cleanup(&self, _synchronizer: &GpuSynchronizer) -> Result<(), PropagationError> {
        // Subgrid propagation may need to clean up temporary buffers
        Ok(())
    }
}

/// Adaptive propagation strategy - automatically selects the best strategy
/// based on grid size and other factors.
pub struct AdaptivePropagationStrategy {
    name: String,
    direct_strategy: DirectPropagationStrategy,
    subgrid_strategy: SubgridPropagationStrategy,
    size_threshold: usize, // Grid size threshold for switching strategies
}

impl AdaptivePropagationStrategy {
    /// Create a new adaptive propagation strategy
    pub fn new(max_iterations: u32, subgrid_size: u32, size_threshold: usize) -> Self {
        Self {
            name: "Adaptive Propagation".to_string(),
            direct_strategy: DirectPropagationStrategy::new(max_iterations),
            subgrid_strategy: SubgridPropagationStrategy::new(max_iterations, subgrid_size),
            size_threshold,
        }
    }

    /// Create a new strategy with default settings
    pub fn default() -> Self {
        Self::new(1000, 16, 4096) // Use subgrid for grids larger than 64x64
    }

    /// Determine which strategy to use based on grid size
    fn select_strategy(&self, grid: &PossibilityGrid) -> &dyn PropagationStrategy {
        let total_cells = grid.width * grid.height * grid.depth;
        if total_cells > self.size_threshold {
            &self.subgrid_strategy as &dyn PropagationStrategy
        } else {
            &self.direct_strategy as &dyn PropagationStrategy
        }
    }
}

impl PropagationStrategy for AdaptivePropagationStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn prepare(&self, synchronizer: &GpuSynchronizer) -> Result<(), PropagationError> {
        // Prepare both strategies since we don't know which will be used
        self.direct_strategy.prepare(synchronizer)?;
        self.subgrid_strategy.prepare(synchronizer)?;
        Ok(())
    }

    fn propagate(
        &self,
        grid: &mut PossibilityGrid,
        updated_cells: &[GridCoord],
        buffers: &Arc<GpuBuffers>,
        synchronizer: &GpuSynchronizer,
    ) -> Result<(), PropagationError> {
        // Delegate to the appropriate strategy based on grid size
        let strategy = self.select_strategy(grid);
        strategy.propagate(grid, updated_cells, buffers, synchronizer)
    }

    fn cleanup(&self, synchronizer: &GpuSynchronizer) -> Result<(), PropagationError> {
        // Clean up both strategies
        self.direct_strategy.cleanup(synchronizer)?;
        self.subgrid_strategy.cleanup(synchronizer)?;
        Ok(())
    }
}

/// Factory for creating propagation strategy instances
pub struct PropagationStrategyFactory;

impl PropagationStrategyFactory {
    /// Create a direct propagation strategy with custom settings
    pub fn create_direct(max_iterations: u32) -> Box<dyn PropagationStrategy> {
        Box::new(DirectPropagationStrategy::new(max_iterations))
    }

    /// Create a subgrid propagation strategy with custom settings
    pub fn create_subgrid(max_iterations: u32, subgrid_size: u32) -> Box<dyn PropagationStrategy> {
        Box::new(SubgridPropagationStrategy::new(
            max_iterations,
            subgrid_size,
        ))
    }

    /// Create an adaptive propagation strategy with custom settings
    pub fn create_adaptive(
        max_iterations: u32,
        subgrid_size: u32,
        size_threshold: usize,
    ) -> Box<dyn PropagationStrategy> {
        Box::new(AdaptivePropagationStrategy::new(
            max_iterations,
            subgrid_size,
            size_threshold,
        ))
    }

    /// Create a strategy based on the grid size
    pub fn create_for_grid(grid: &PossibilityGrid) -> Box<dyn PropagationStrategy> {
        let total_cells = grid.width * grid.height * grid.depth;
        if total_cells > 4096 {
            Self::create_subgrid(1000, 16)
        } else {
            Self::create_direct(1000)
        }
    }
}
