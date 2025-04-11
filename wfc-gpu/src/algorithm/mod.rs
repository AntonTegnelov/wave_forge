//! Algorithm module containing strategy implementations for the WFC algorithm.
//! This module houses various strategy implementations for different aspects
//! of the Wave Function Collapse algorithm, following the Strategy pattern.

pub mod entropy_strategy;
pub mod propagator_strategy;

// Re-export key types for convenience
pub use entropy_strategy::{EntropyStrategy, EntropyStrategyFactory};
pub use propagator_strategy::{PropagationStrategy, PropagationStrategyFactory};
