//! Error recovery mechanisms for the WFC-GPU module.
//!
//! This module provides strategies and utilities for recovering from various error conditions
//! that may occur during algorithm execution.

pub mod strategies;

use std::time::Duration;

pub use crate::utils::error_recovery::{
    AdaptiveTimeoutConfig, GpuErrorRecovery, GridCoord, RecoverableGpuOp,
};
pub use strategies::{
    FallbackStrategy, GracefulDegradationStrategy, RecoveryStrategy, RetryStrategy,
};

// Forward the legacy error recovery module for now
// FIXME: Eventually we will remove the old system entirely
pub use super::error_recovery::{
    AdaptiveTimeoutConfig, GpuErrorRecovery, GridCoord, RecoverableGpuOp,
};

/// Manages the application of recovery strategies.
pub struct ErrorRecoveryManager {
    /// Available recovery strategies
    strategies: Vec<Box<dyn RecoveryStrategy>>,

    /// Whether to use legacy error recovery
    use_legacy: bool,

    /// Optional fallback to legacy error recovery system
    legacy_recovery: Option<GpuErrorRecovery>,
}

impl Default for ErrorRecoveryManager {
    fn default() -> Self {
        Self {
            strategies: Vec::new(),
            use_legacy: true,
            legacy_recovery: Some(GpuErrorRecovery::default()),
        }
    }
}

impl ErrorRecoveryManager {
    /// Create a new error recovery manager
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a recovery strategy
    pub fn add_strategy<S>(&mut self, strategy: S) -> &mut Self
    where
        S: RecoveryStrategy + 'static,
    {
        self.strategies.push(Box::new(strategy));
        self
    }

    /// Enable or disable legacy error recovery
    pub fn with_legacy_recovery(mut self, use_legacy: bool) -> Self {
        self.use_legacy = use_legacy;
        self
    }

    /// Add a retry strategy with exponential backoff
    pub fn with_retry_strategy(
        mut self,
        max_retries: u32,
        initial_delay: Duration,
        max_delay: Duration,
    ) -> Self {
        self.add_strategy(RetryStrategy::new(max_retries, initial_delay, max_delay));
        self
    }

    /// Add a fallback strategy that provides alternative solutions
    pub fn with_fallback_strategy(mut self) -> Self {
        self.add_strategy(FallbackStrategy::new());
        self
    }

    /// Add a strategy for graceful degradation
    pub fn with_graceful_degradation(mut self) -> Self {
        self.add_strategy(GracefulDegradationStrategy::new());
        self
    }

    /// Get the legacy error recovery system, if enabled
    pub fn legacy_recovery(&self) -> Option<&GpuErrorRecovery> {
        if self.use_legacy {
            self.legacy_recovery.as_ref()
        } else {
            None
        }
    }
}
