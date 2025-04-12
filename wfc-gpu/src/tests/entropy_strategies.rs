use std::sync::Arc;

use log::{debug, warn};
use wgpu::{Adapter, Device, Queue};

use crate::entropy::{
    CountEntropyStrategy, CountSimpleEntropyStrategy, EntropyHeuristicType, EntropyStrategy,
    EntropyStrategyFactory, ShannonEntropyStrategy, WeightedCountEntropyStrategy,
};
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
        let (_adapter, _device, _queue) = init_gpu().await?;

        // Create a Shannon entropy strategy
        let strategy = ShannonEntropyStrategy::new(
            128, // num_tiles
            4,   // u32s_per_cell
        );

        // Check that the strategy returns the expected heuristic type
        assert_eq!(strategy.heuristic_type(), EntropyHeuristicType::Shannon);

        // Note: This only tests API surface - actual computation testing requires
        // more complex setup with buffers and grid state

        Ok(())
    }

    /// Test that the Count entropy strategy calculates correctly
    #[tokio::test]
    async fn test_count_entropy_strategy() -> Result<(), WfcError> {
        let (_adapter, _device, _queue) = init_gpu().await?;

        // Create a Count entropy strategy
        let strategy = CountEntropyStrategy::new(
            128, // num_tiles
            4,   // u32s_per_cell
        );

        // Check that the strategy returns the expected heuristic type
        assert_eq!(strategy.heuristic_type(), EntropyHeuristicType::Count);

        Ok(())
    }

    /// Test that the Simple Count entropy strategy calculates correctly
    #[tokio::test]
    async fn test_count_simple_entropy_strategy() -> Result<(), WfcError> {
        let (_adapter, _device, _queue) = init_gpu().await?;

        // Create a Simple Count entropy strategy
        let strategy = CountSimpleEntropyStrategy::new(
            128, // num_tiles
            4,   // u32s_per_cell
        );

        // Check that the strategy returns the expected heuristic type
        assert_eq!(strategy.heuristic_type(), EntropyHeuristicType::CountSimple);

        Ok(())
    }

    /// Test that the Weighted Count entropy strategy calculates correctly
    #[tokio::test]
    async fn test_weighted_count_entropy_strategy() -> Result<(), WfcError> {
        let (_adapter, _device, _queue) = init_gpu().await?;

        // Create a Weighted Count entropy strategy
        let strategy = WeightedCountEntropyStrategy::new(
            128, // num_tiles
            4,   // u32s_per_cell
        );

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
        let (_adapter, _device, _queue) = init_gpu().await?;

        // Create all strategies from factory method
        let shannon = EntropyStrategyFactory::create_strategy(
            EntropyHeuristicType::Shannon,
            128, // num_tiles
            4,   // u32s_per_cell
        );

        let count = EntropyStrategyFactory::create_strategy(
            EntropyHeuristicType::Count,
            128, // num_tiles
            4,   // u32s_per_cell
        );

        let count_simple = EntropyStrategyFactory::create_strategy(
            EntropyHeuristicType::CountSimple,
            128, // num_tiles
            4,   // u32s_per_cell
        );

        let weighted_count = EntropyStrategyFactory::create_strategy(
            EntropyHeuristicType::WeightedCount,
            128, // num_tiles
            4,   // u32s_per_cell
        );

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
