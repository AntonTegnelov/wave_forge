use std::sync::Arc;

use log::{debug, warn};
use wgpu::{Adapter, Device, Queue};

use crate::gpu::backend::GpuBackend;
use crate::propagator::propagator_strategy::{
    AdaptivePropagationStrategy, DirectPropagationStrategy, PropagationStrategy,
    PropagationStrategyType, SubgridPropagationStrategy,
};
use crate::utils::error::WfcError;
use crate::utils::subgrid::SubgridConfig;

/// Unit tests for all propagation strategies
#[cfg(test)]
mod propagation_strategy_tests {
    use super::*;

    /// Initialize a GPU context for testing
    async fn init_gpu() -> Result<(Adapter, Device, Queue), WfcError> {
        let backend = GpuBackend::new().await?;
        let (adapter, device, queue) = backend.get_context();
        Ok((adapter.clone(), device.clone(), queue.clone()))
    }

    /// Test that the direct propagation strategy initializes correctly
    #[tokio::test]
    async fn test_direct_propagation_strategy() -> Result<(), WfcError> {
        let (_adapter, device, queue) = init_gpu().await?;

        // Create a direct propagation strategy
        let strategy = DirectPropagationStrategy::new(Arc::new(device), Arc::new(queue))?;

        // Check that the strategy returns the expected type
        assert_eq!(strategy.strategy_type(), PropagationStrategyType::Direct);

        // This only tests API surface - actual computation testing requires
        // more complex setup with buffers and grid state

        Ok(())
    }

    /// Test that the subgrid propagation strategy calculates correctly
    #[tokio::test]
    async fn test_subgrid_propagation_strategy() -> Result<(), WfcError> {
        let (_adapter, device, queue) = init_gpu().await?;

        // Create a subgrid propagation strategy with default config
        let config = SubgridConfig::default();
        let strategy = SubgridPropagationStrategy::new(Arc::new(device), Arc::new(queue), config)?;

        // Check that the strategy returns the expected type
        assert_eq!(strategy.strategy_type(), PropagationStrategyType::Subgrid);

        // Verify the subgrid config is as expected
        let retrieved_config = strategy.subgrid_config();
        assert_eq!(
            retrieved_config.min_cells_per_subgrid,
            config.min_cells_per_subgrid
        );
        assert_eq!(retrieved_config.max_subgrid_count, config.max_subgrid_count);

        Ok(())
    }

    /// Test that the adaptive propagation strategy selects correctly based on grid size
    #[tokio::test]
    async fn test_adaptive_propagation_strategy() -> Result<(), WfcError> {
        let (_adapter, device, queue) = init_gpu().await?;

        // Create an adaptive propagation strategy
        let strategy = AdaptivePropagationStrategy::new(Arc::new(device), Arc::new(queue))?;

        // Check that the strategy returns the expected type
        assert_eq!(strategy.strategy_type(), PropagationStrategyType::Adaptive);

        // Test strategy selection based on grid size
        // Small grid should use direct strategy
        let small_grid_dims = (10, 10, 1);
        let small_strategy = strategy.select_strategy_for_dimensions(small_grid_dims)?;
        assert_eq!(
            small_strategy.strategy_type(),
            PropagationStrategyType::Direct
        );

        // Very large grid should use subgrid strategy
        let large_grid_dims = (1000, 1000, 1);
        let large_strategy = strategy.select_strategy_for_dimensions(large_grid_dims)?;
        assert_eq!(
            large_strategy.strategy_type(),
            PropagationStrategyType::Subgrid
        );

        Ok(())
    }

    /// Test strategy factory to ensure it creates the right strategies
    #[tokio::test]
    async fn test_strategy_factory() -> Result<(), WfcError> {
        let (_adapter, device, queue) = init_gpu().await?;

        // Create all strategies from factory method
        let device_arc = Arc::new(device);
        let queue_arc = Arc::new(queue);

        let direct = PropagationStrategy::create_strategy(
            PropagationStrategyType::Direct,
            device_arc.clone(),
            queue_arc.clone(),
            None, // no config needed for direct
        )?;

        let config = SubgridConfig::default();
        let subgrid = PropagationStrategy::create_strategy(
            PropagationStrategyType::Subgrid,
            device_arc.clone(),
            queue_arc.clone(),
            Some(config.clone()),
        )?;

        let adaptive = PropagationStrategy::create_strategy(
            PropagationStrategyType::Adaptive,
            device_arc.clone(),
            queue_arc.clone(),
            None, // no config needed for adaptive
        )?;

        // Verify all strategies created correctly
        assert_eq!(direct.strategy_type(), PropagationStrategyType::Direct);
        assert_eq!(subgrid.strategy_type(), PropagationStrategyType::Subgrid);
        assert_eq!(adaptive.strategy_type(), PropagationStrategyType::Adaptive);

        Ok(())
    }
}
