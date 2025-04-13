// Re-export propagator components
pub mod gpu_constraint_propagator;
pub mod propagator_strategy;

pub use gpu_constraint_propagator::GpuConstraintPropagator;
pub use propagator_strategy::{
    AdaptivePropagationStrategy, AsyncPropagationStrategy, DirectPropagationStrategy,
    PropagationStrategy, PropagationStrategyFactory, SubgridPropagationStrategy,
};

// Utility functions
mod utils;
pub use utils::*;
