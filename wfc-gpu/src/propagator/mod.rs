//! Propagation strategy implementations for the WFC algorithm.
//! This module provides different strategies for propagating constraints
//! after a cell collapse in the Wave Function Collapse algorithm.

// Re-export propagator components
pub mod gpu_constraint_propagator;
pub use gpu_constraint_propagator::GpuConstraintPropagator;

// Strategy modules
mod adaptive_strategy;
mod direct_strategy;
mod subgrid_strategy;

// Re-export strategy types
pub use adaptive_strategy::AdaptivePropagationStrategy;
pub use direct_strategy::DirectPropagationStrategy;
pub use subgrid_strategy::SubgridPropagationStrategy;

// These traits are defined directly in this module, no need to import them
// pub use direct_strategy::{AsyncPropagationStrategy, PropagationStrategy};

use crate::{
    buffers::GpuBuffers,
    gpu::sync::GpuSynchronizer,
    shader::pipeline::ComputePipelines,
    utils::error_recovery::{GpuError, GridCoord},
};
use async_trait;
use std::sync::Arc;
use wfc_core::{grid::PossibilityGrid, propagator::PropagationError};

/// Strategy trait for constraint propagation in WFC algorithm.
/// This trait contains only synchronous methods for object safety.
pub trait PropagationStrategy: Send + Sync + std::fmt::Debug {
    /// Get the name of this propagation strategy
    fn name(&self) -> &str;

    /// Prepare for propagation by initializing any necessary buffers or state
    fn prepare(&self, synchronizer: &GpuSynchronizer) -> Result<(), PropagationError>;

    /// Clean up any resources used during propagation
    fn cleanup(&self, synchronizer: &GpuSynchronizer) -> Result<(), PropagationError>;
}

/// Async extension of the PropagationStrategy trait.
/// This trait contains the async propagate method which can't be part of
/// the object-safe PropagationStrategy trait.
#[async_trait::async_trait]
pub trait AsyncPropagationStrategy: PropagationStrategy {
    /// Propagate constraints from the specified cells
    async fn propagate(
        &self,
        grid: &mut PossibilityGrid,
        updated_cells: &[GridCoord],
        buffers: &Arc<GpuBuffers>,
        synchronizer: &GpuSynchronizer,
    ) -> Result<(), PropagationError>;
}

/// Factory for creating propagation strategy instances
pub struct PropagationStrategyFactory;

impl PropagationStrategyFactory {
    /// Create a direct propagation strategy with custom settings
    pub fn create_direct(
        max_iterations: u32,
        pipelines: Arc<ComputePipelines>,
    ) -> Box<dyn PropagationStrategy + Send + Sync> {
        Box::new(DirectPropagationStrategy::new(max_iterations, pipelines))
    }

    /// Create a subgrid propagation strategy with custom settings
    pub fn create_subgrid(
        max_iterations: u32,
        subgrid_size: u32,
        pipelines: Arc<ComputePipelines>,
    ) -> Box<dyn PropagationStrategy + Send + Sync> {
        Box::new(SubgridPropagationStrategy::new(
            max_iterations,
            subgrid_size,
            pipelines,
        ))
    }

    /// Create an adaptive propagation strategy with custom settings
    pub fn create_adaptive(
        max_iterations: u32,
        subgrid_size: u32,
        size_threshold: usize,
        pipelines: Arc<ComputePipelines>,
    ) -> Box<dyn PropagationStrategy + Send + Sync> {
        Box::new(AdaptivePropagationStrategy::new(
            max_iterations,
            subgrid_size,
            size_threshold,
            pipelines,
        ))
    }

    /// Create a strategy based on the grid size
    pub fn create_for_grid(
        grid: &PossibilityGrid,
        pipelines: Arc<ComputePipelines>,
    ) -> Box<dyn PropagationStrategy + Send + Sync> {
        let total_cells = grid.width * grid.height * grid.depth;
        if total_cells > 4096 {
            Self::create_subgrid(1000, 16, pipelines.clone())
        } else {
            Self::create_direct(1000, pipelines)
        }
    }

    /// Create a direct strategy that also implements AsyncPropagationStrategy
    pub fn create_direct_async(
        max_iterations: u32,
        pipelines: Arc<ComputePipelines>,
    ) -> Box<dyn AsyncPropagationStrategy + Send + Sync> {
        Box::new(DirectPropagationStrategy::new(max_iterations, pipelines))
    }

    /// Create a subgrid strategy that also implements AsyncPropagationStrategy
    pub fn create_subgrid_async(
        max_iterations: u32,
        subgrid_size: u32,
        pipelines: Arc<ComputePipelines>,
    ) -> Box<dyn AsyncPropagationStrategy + Send + Sync> {
        Box::new(SubgridPropagationStrategy::new(
            max_iterations,
            subgrid_size,
            pipelines,
        ))
    }

    /// Create an adaptive strategy that also implements AsyncPropagationStrategy
    pub fn create_adaptive_async(
        max_iterations: u32,
        subgrid_size: u32,
        size_threshold: usize,
        pipelines: Arc<ComputePipelines>,
    ) -> Box<dyn AsyncPropagationStrategy + Send + Sync> {
        Box::new(AdaptivePropagationStrategy::new(
            max_iterations,
            subgrid_size,
            size_threshold,
            pipelines,
        ))
    }

    /// Create a strategy based on the grid size that also implements AsyncPropagationStrategy
    pub fn create_for_grid_async(
        grid: &PossibilityGrid,
        pipelines: Arc<ComputePipelines>,
    ) -> Box<dyn AsyncPropagationStrategy + Send + Sync> {
        let total_cells = grid.width * grid.height * grid.depth;
        if total_cells > 4096 {
            Self::create_subgrid_async(1000, 16, pipelines.clone())
        } else {
            Self::create_direct_async(1000, pipelines)
        }
    }
}

// Utility functions
mod utils;
pub use utils::*;

/// Convert a GPU error to a propagation error
pub fn gpu_error_to_propagation_error(error: GpuError) -> PropagationError {
    PropagationError::InternalError(format!("GPU error: {}", error))
}
