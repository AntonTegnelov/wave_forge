//! Recovery strategies for handling various error conditions.
//!
//! This module provides different approaches to error recovery that can
//! be composed and applied based on the specific error context.

use crate::utils::error::{ErrorSeverity, ErrorWithContext, WfcError};
use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

/// Trait for error recovery strategies
pub trait RecoveryStrategy: fmt::Debug + Send + Sync {
    /// Attempt to recover from an error.
    ///
    /// Returns:
    /// - `Ok(RecoveryAction)`: Recovery attempt result and recommended action
    /// - `Err(Box<WfcError>)`: If recovery failed or is not possible
    fn attempt_recovery(&self, error: &WfcError) -> Result<RecoveryAction, Box<WfcError>>;

    /// Check if this strategy is applicable to the given error.
    fn is_applicable(&self, error: &WfcError) -> bool;

    /// Get a descriptive name for this strategy.
    fn name(&self) -> &'static str;
}

/// Actions that can be taken after a recovery attempt
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryAction {
    /// Retry the operation that failed
    Retry,

    /// Skip the operation and continue execution
    Skip,

    /// Use alternative implementation or approach
    UseAlternative,

    /// Reinitialize the affected component
    Reinitialize,

    /// Abort the current algorithm iteration and restart
    Abort,
}

/// Strategy that attempts to retry operations with exponential backoff
#[derive(Debug)]
pub struct RetryStrategy {
    /// Maximum number of retry attempts
    max_retries: u32,

    /// Initial delay between retries (milliseconds)
    initial_delay: Duration,

    /// Maximum delay between retries (milliseconds)
    max_delay: Duration,

    /// Counter for retry attempts
    retry_counter: AtomicU32,
}

impl RetryStrategy {
    /// Create a new retry strategy
    pub fn new(max_retries: u32, initial_delay: Duration, max_delay: Duration) -> Self {
        Self {
            max_retries,
            initial_delay,
            max_delay,
            retry_counter: AtomicU32::new(0),
        }
    }

    /// Get the next backoff delay
    fn get_backoff_delay(&self) -> Duration {
        let retry_count = self.retry_counter.load(Ordering::Relaxed);

        // Calculate delay with exponential backoff: initial_delay * 2^retry_count
        let delay_ms = self.initial_delay.as_millis() * (1 << retry_count.min(10));

        // Clamp to max_delay
        Duration::from_millis(delay_ms.min(self.max_delay.as_millis()) as u64)
    }

    /// Reset the retry counter
    pub fn reset(&self) {
        self.retry_counter.store(0, Ordering::Relaxed);
    }
}

impl RecoveryStrategy for RetryStrategy {
    fn attempt_recovery(&self, error: &WfcError) -> Result<RecoveryAction, Box<WfcError>> {
        if !error.is_recoverable() {
            return Err(Box::new(error.clone()));
        }

        let retry_count = self.retry_counter.fetch_add(1, Ordering::Relaxed);

        if retry_count >= self.max_retries {
            // Max retries exceeded
            self.reset();
            return Err(Box::new(error.clone()));
        }

        // Sleep for backoff delay
        let delay = self.get_backoff_delay();
        std::thread::sleep(delay);

        Ok(RecoveryAction::Retry)
    }

    fn is_applicable(&self, error: &WfcError) -> bool {
        // Retry strategy applies to any error that's not fatal
        error.severity() != ErrorSeverity::Fatal
    }

    fn name(&self) -> &'static str {
        "Retry"
    }
}

/// Strategy that provides fallback alternatives for specific operations
#[derive(Debug, Default)]
pub struct FallbackStrategy {
    // Could include a map of operation types to fallback functions
}

impl FallbackStrategy {
    /// Create a new fallback strategy
    pub fn new() -> Self {
        Self {}
    }
}

impl RecoveryStrategy for FallbackStrategy {
    fn attempt_recovery(&self, error: &WfcError) -> Result<RecoveryAction, Box<WfcError>> {
        // Based on error type, suggest a fallback approach
        match error {
            // For example, if a buffer failed, we might suggest reinitialization
            WfcError::Gpu(gpu_error) => {
                // Simplified example - would need more sophistication in practice
                match gpu_error.severity() {
                    ErrorSeverity::Recoverable => Ok(RecoveryAction::UseAlternative),
                    _ => Err(Box::new(error.clone())),
                }
            }
            _ => Err(Box::new(error.clone())),
        }
    }

    fn is_applicable(&self, error: &WfcError) -> bool {
        // In this basic implementation, we only handle GPU errors
        matches!(error, WfcError::Gpu(_))
    }

    fn name(&self) -> &'static str {
        "Fallback"
    }
}

/// Strategy that attempts to continue execution with reduced quality/functionality
#[derive(Debug, Default)]
pub struct GracefulDegradationStrategy {
    // Could track degradation level, available fallbacks, etc.
}

impl GracefulDegradationStrategy {
    /// Create a new graceful degradation strategy
    pub fn new() -> Self {
        Self {}
    }
}

impl RecoveryStrategy for GracefulDegradationStrategy {
    fn attempt_recovery(&self, error: &WfcError) -> Result<RecoveryAction, Box<WfcError>> {
        // This would analyze the error and suggest a way to continue
        // with reduced functionality or quality

        // Example implementation
        match error {
            WfcError::Gpu(_) => {
                // For GPU errors, we might suggest skipping certain operations
                Ok(RecoveryAction::Skip)
            }
            _ => Err(Box::new(error.clone())),
        }
    }

    fn is_applicable(&self, error: &WfcError) -> bool {
        // In a real implementation, would check if degradation is possible
        // for this specific error type
        error.severity() == ErrorSeverity::Recoverable
    }

    fn name(&self) -> &'static str {
        "Graceful Degradation"
    }
}

/// A composite strategy that tries multiple recovery approaches
#[derive(Debug, Default)]
pub struct CompositeRecoveryStrategy {
    strategies: Vec<Box<dyn RecoveryStrategy>>,
}

impl CompositeRecoveryStrategy {
    /// Create a new composite recovery strategy
    pub fn new() -> Self {
        Self {
            strategies: Vec::new(),
        }
    }

    /// Add a strategy to the composite
    pub fn add_strategy<S>(&mut self, strategy: S) -> &mut Self
    where
        S: RecoveryStrategy + 'static,
    {
        self.strategies.push(Box::new(strategy));
        self
    }
}

impl RecoveryStrategy for CompositeRecoveryStrategy {
    fn attempt_recovery(&self, error: &WfcError) -> Result<RecoveryAction, Box<WfcError>> {
        // Try each applicable strategy in order
        let mut last_error: Option<Box<WfcError>> = None;

        for strategy in &self.strategies {
            if strategy.is_applicable(error) {
                match strategy.attempt_recovery(error) {
                    Ok(action) => return Ok(action),
                    Err(err) => {
                        last_error = Some(err);
                        // Continue to the next strategy
                    }
                }
            }
        }

        // If we got here, all strategies failed or none were applicable
        match last_error {
            Some(err) => Err(err),
            None => Err(Box::new(error.clone())),
        }
    }

    fn is_applicable(&self, error: &WfcError) -> bool {
        // The composite strategy is applicable if any of its
        // contained strategies are applicable
        self.strategies.iter().any(|s| s.is_applicable(error))
    }

    fn name(&self) -> &'static str {
        "Composite"
    }
}
