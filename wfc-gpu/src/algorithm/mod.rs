//! Algorithm module containing strategy implementations for the WFC algorithm.
//! This module houses various strategy implementations for different aspects
//! of the Wave Function Collapse algorithm, following the Strategy pattern.

pub mod entropy_strategy;

// Re-export key types for convenience
pub use entropy_strategy::{EntropyStrategy, EntropyStrategyFactory};
