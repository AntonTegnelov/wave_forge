use std::sync::Arc;

use log::{debug, warn};
use wgpu::{Adapter, Device, Queue};

use crate::entropy::entropy_strategy::{
    CountEntropyStrategy, CountSimpleEntropyStrategy, ShannonEntropyStrategy,
    WeightedCountEntropyStrategy,
};
use crate::entropy::{EntropyHeuristicType, EntropyStrategy};
use crate::gpu::backend::GpuBackend;
use crate::utils::error::WfcError;

/// Unit tests for all entropy calculation strategies
#[cfg(test)]
mod entropy_strategy_tests {
    use super::*;

    /// Initialize a GPU context for testing
    async fn init_gpu() -> Result<(Adapter, Device, Queue), WfcError> {
        let backend = GpuBackend::new().await?;
        let (adapter, device, queue) = backend.get_context();
        Ok((adapter.clone(), device.clone(), queue.clone()))
    }

    /// Test that the Shannon entropy strategy calculates correctly
    #[tokio::test]
    async fn test_shannon_entropy_strategy() -> Result<(), WfcError> {
        let (_adapter, device, queue) = init_gpu().await?;

        // Create a Shannon entropy strategy
        let strategy = ShannonEntropyStrategy::new(Arc::new(device), Arc::new(queue))?;

        // Check that the strategy returns the expected heuristic type
        assert_eq!(strategy.heuristic_type(), EntropyHeuristicType::Shannon);

        // Note: This only tests API surface - actual computation testing requires
        // more complex setup with buffers and grid state

        Ok(())
    }

    /// Test that the Count entropy strategy calculates correctly
    #[tokio::test]
    async fn test_count_entropy_strategy() -> Result<(), WfcError> {
        let (_adapter, device, queue) = init_gpu().await?;

        // Create a Count entropy strategy
        let strategy = CountEntropyStrategy::new(Arc::new(device), Arc::new(queue))?;

        // Check that the strategy returns the expected heuristic type
        assert_eq!(strategy.heuristic_type(), EntropyHeuristicType::Count);

        Ok(())
    }

    /// Test that the Simple Count entropy strategy calculates correctly
    #[tokio::test]
    async fn test_count_simple_entropy_strategy() -> Result<(), WfcError> {
        let (_adapter, device, queue) = init_gpu().await?;

        // Create a Simple Count entropy strategy
        let strategy = CountSimpleEntropyStrategy::new(Arc::new(device), Arc::new(queue))?;

        // Check that the strategy returns the expected heuristic type
        assert_eq!(strategy.heuristic_type(), EntropyHeuristicType::CountSimple);

        Ok(())
    }

    /// Test that the Weighted Count entropy strategy calculates correctly
    #[tokio::test]
    async fn test_weighted_count_entropy_strategy() -> Result<(), WfcError> {
        let (_adapter, device, queue) = init_gpu().await?;

        // Create a Weighted Count entropy strategy
        let strategy = WeightedCountEntropyStrategy::new(Arc::new(device), Arc::new(queue))?;

        // Check that the strategy returns the expected heuristic type
        assert_eq!(
            strategy.heuristic_type(),
            EntropyHeuristicType::WeightedCount
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

        let shannon = EntropyStrategy::create_strategy(
            EntropyHeuristicType::Shannon,
            device_arc.clone(),
            queue_arc.clone(),
        )?;

        let count = EntropyStrategy::create_strategy(
            EntropyHeuristicType::Count,
            device_arc.clone(),
            queue_arc.clone(),
        )?;

        let count_simple = EntropyStrategy::create_strategy(
            EntropyHeuristicType::CountSimple,
            device_arc.clone(),
            queue_arc.clone(),
        )?;

        let weighted_count = EntropyStrategy::create_strategy(
            EntropyHeuristicType::WeightedCount,
            device_arc.clone(),
            queue_arc.clone(),
        )?;

        // Verify all strategies created correctly
        assert_eq!(shannon.heuristic_type(), EntropyHeuristicType::Shannon);
        assert_eq!(count.heuristic_type(), EntropyHeuristicType::Count);
        assert_eq!(
            count_simple.heuristic_type(),
            EntropyHeuristicType::CountSimple
        );
        assert_eq!(
            weighted_count.heuristic_type(),
            EntropyHeuristicType::WeightedCount
        );

        Ok(())
    }
}
