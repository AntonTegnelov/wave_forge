use crate::{
    buffers::GpuBuffers, gpu::sync::GpuSynchronizer, shader::pipeline::ComputePipelines,
    utils::error_recovery::GridCoord,
};
use async_trait;
use std::default::Default;
use std::sync::Arc;
use wfc_core::{grid::PossibilityGrid, propagator::PropagationError};

use super::{
    direct_strategy::DirectPropagationStrategy, subgrid_strategy::SubgridPropagationStrategy,
};

/// Adaptive propagation strategy that automatically selects between
/// direct and subgrid propagation based on grid size.
#[derive(Debug)]
pub struct AdaptivePropagationStrategy {
    name: String,
    direct_strategy: DirectPropagationStrategy,
    subgrid_strategy: SubgridPropagationStrategy,
    size_threshold: usize, // Grid size threshold for switching strategies
}

impl AdaptivePropagationStrategy {
    /// Create a new adaptive propagation strategy
    pub fn new(
        max_iterations: u32,
        subgrid_size: u32,
        size_threshold: usize,
        pipelines: Arc<ComputePipelines>,
    ) -> Self {
        Self {
            name: "Adaptive Propagation".to_string(),
            direct_strategy: DirectPropagationStrategy::new(max_iterations, pipelines.clone()),
            subgrid_strategy: SubgridPropagationStrategy::new(
                max_iterations,
                subgrid_size,
                pipelines,
            ),
            size_threshold,
        }
    }

    /// Select the appropriate strategy based on grid size
    fn select_strategy(&self, grid: &PossibilityGrid) -> Strategy {
        let total_cells = grid.width * grid.height * grid.depth;
        if total_cells <= self.size_threshold {
            Strategy::Direct
        } else {
            Strategy::Subgrid
        }
    }
}

/// Enum representing the available strategies
#[derive(Debug)]
enum Strategy {
    Direct,
    Subgrid,
}

impl crate::propagator::PropagationStrategy for AdaptivePropagationStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn prepare(&self, synchronizer: &GpuSynchronizer) -> Result<(), PropagationError> {
        // Prepare both strategies
        self.direct_strategy.prepare(synchronizer)?;
        self.subgrid_strategy.prepare(synchronizer)?;
        Ok(())
    }

    fn cleanup(&self, synchronizer: &GpuSynchronizer) -> Result<(), PropagationError> {
        // Cleanup both strategies
        self.direct_strategy.cleanup(synchronizer)?;
        self.subgrid_strategy.cleanup(synchronizer)?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl crate::propagator::AsyncPropagationStrategy for AdaptivePropagationStrategy {
    async fn propagate(
        &self,
        grid: &mut PossibilityGrid,
        updated_cells: &[GridCoord],
        buffers: &Arc<GpuBuffers>,
        synchronizer: &GpuSynchronizer,
    ) -> Result<(), PropagationError> {
        match self.select_strategy(grid) {
            Strategy::Direct => {
                self.direct_strategy
                    .propagate(grid, updated_cells, buffers, synchronizer)
                    .await
            }
            Strategy::Subgrid => {
                self.subgrid_strategy
                    .propagate(grid, updated_cells, buffers, synchronizer)
                    .await
            }
        }
    }
}

/// Implement Default trait for AdaptivePropagationStrategy
impl Default for AdaptivePropagationStrategy {
    fn default() -> Self {
        unimplemented!(
            "AdaptivePropagationStrategy requires pipelines, cannot be created with default()"
        )
    }
}
