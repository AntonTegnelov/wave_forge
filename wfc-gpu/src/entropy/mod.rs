// Entropy module for Wave Function Collapse GPU implementation
// Handles calculation of entropy and cell selection in the WFC algorithm

mod calculator;
mod entropy_strategy;

// Re-export main types from calculator
pub use calculator::{GpuEntropyCalculator, GpuEntropyCalculatorExt};

// Re-export entropy types from core
pub use wfc_core::entropy::EntropyHeuristicType;

// Re-export entropy strategy types
pub use entropy_strategy::{
    CountEntropyStrategy, CountSimpleEntropyStrategy, EntropyStrategy, EntropyStrategyFactory,
    ShannonEntropyStrategy, WeightedCountEntropyStrategy,
};
