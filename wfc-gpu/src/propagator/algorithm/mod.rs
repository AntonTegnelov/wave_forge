//! Algorithm module contains strategy implementations for the WFC algorithm.
//! This includes different approaches to entropy calculation and propagation.

// Re-export strategy interfaces and implementations
pub mod entropy_strategy;
pub mod propagator_strategy;

// Re-export core types
pub use entropy_strategy::{EntropyStrategy, EntropyStrategyFactory};
pub use propagator_strategy::{PropagationStrategy, PropagationStrategyFactory};
