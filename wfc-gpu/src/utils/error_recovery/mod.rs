//! Error recovery mechanisms for the WFC-GPU module.
//!
//! This module provides strategies and utilities for recovering from various error conditions
//! that may occur during algorithm execution.

pub mod strategies;

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

use crate::utils::error::{ErrorSeverity, WfcError};

// Re-export strategy types
pub use strategies::{
    FallbackStrategy, GracefulDegradationStrategy, RecoveryAction, RecoveryStrategy, RetryStrategy,
};

/// Represents a coordinate in the WFC grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridCoord {
    pub x: usize,
    pub y: usize,
}

impl fmt::Display for GridCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

/// Enum of all possible GPU errors that can occur during WFC execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuError {
    /// Failed to allocate GPU memory
    MemoryAllocation(String),

    /// GPU computation timeout
    ComputationTimeout {
        grid_size: (usize, usize),
        duration: Duration,
    },

    /// Kernel execution error
    KernelExecution(String),

    /// Queue submission error
    QueueSubmission(String),

    /// Device lost or crashed
    DeviceLost(String),

    /// Invalid state encountered
    InvalidState(String),

    /// Barrier synchronization error
    BarrierSynchronization(String),

    /// Error in buffer copy operation
    BufferCopy(String),

    /// Buffer mapping error
    BufferMapping(String),

    /// Other GPU error
    Other(String),
}

/// Operations that can be retried if they fail
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecoverableGpuOp {
    /// Collapse a cell to a specific state
    Collapse,

    /// Propagate constraints through the grid
    Propagate,

    /// Observe the grid state
    Observe,

    /// Clear the grid to initial state
    Clear,

    /// Copy data between buffers
    BufferCopy,

    /// General compute shader execution
    Compute,
}

/// Configuration for adaptive timeout calculations
#[derive(Debug, Clone)]
pub struct AdaptiveTimeoutConfig {
    /// Base timeout duration for small grids (in milliseconds)
    pub base_timeout_ms: u64,

    /// Factor to increase timeout per thousand cells
    pub size_factor: f32,

    /// Factor to increase timeout based on complexity (entropy states)
    pub complexity_factor: f32,

    /// Maximum timeout duration (in milliseconds)
    pub max_timeout_ms: u64,
}

impl Default for AdaptiveTimeoutConfig {
    fn default() -> Self {
        Self {
            base_timeout_ms: 500,
            size_factor: 1.5,
            complexity_factor: 0.8,
            max_timeout_ms: 30000, // 30 seconds max timeout
        }
    }
}

/// Main error recovery struct for handling GPU errors during WFC execution
#[derive(Debug, Clone)]
pub struct GpuErrorRecovery {
    /// Maximum number of retries for recoverable operations
    max_retries: HashMap<RecoverableGpuOp, u32>,

    /// Current retry count for each operation type
    retry_count: HashMap<RecoverableGpuOp, u32>,

    /// Adaptive timeout configuration
    timeout_config: AdaptiveTimeoutConfig,

    /// Tracks if the most recent error was fatal
    had_fatal_error: bool,
}

impl Default for GpuErrorRecovery {
    fn default() -> Self {
        let mut max_retries = HashMap::new();
        // Default retry counts for different operations
        max_retries.insert(RecoverableGpuOp::Collapse, 3);
        max_retries.insert(RecoverableGpuOp::Propagate, 5);
        max_retries.insert(RecoverableGpuOp::Observe, 2);
        max_retries.insert(RecoverableGpuOp::Clear, 2);
        max_retries.insert(RecoverableGpuOp::BufferCopy, 3);
        max_retries.insert(RecoverableGpuOp::Compute, 4);

        Self {
            max_retries,
            retry_count: HashMap::new(),
            timeout_config: AdaptiveTimeoutConfig::default(),
            had_fatal_error: false,
        }
    }
}

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

impl GpuErrorRecovery {
    /// Create a new GPU error recovery handler with custom configuration
    pub fn new(
        max_retries: HashMap<RecoverableGpuOp, u32>,
        timeout_config: AdaptiveTimeoutConfig,
    ) -> Self {
        Self {
            max_retries,
            retry_count: HashMap::new(),
            timeout_config,
            had_fatal_error: false,
        }
    }

    /// Calculate adaptive timeout based on grid size and complexity
    pub fn calculate_timeout(&self, grid_size: (usize, usize), num_patterns: usize) -> Duration {
        let total_cells = grid_size.0 * grid_size.1;

        // Base calculation
        let size_factor = total_cells as f32 / 1000.0 * self.timeout_config.size_factor;
        let complexity_factor =
            (num_patterns as f32).log2() * self.timeout_config.complexity_factor;

        let timeout_ms =
            self.timeout_config.base_timeout_ms as f32 * (1.0 + size_factor + complexity_factor);
        let timeout_ms = timeout_ms.min(self.timeout_config.max_timeout_ms as f32) as u64;

        Duration::from_millis(timeout_ms)
    }

    /// Determine if an error is fatal or recoverable
    pub fn classify_error(&self, error: &GpuError) -> ErrorSeverity {
        match error {
            GpuError::DeviceLost(_) => ErrorSeverity::Fatal,
            GpuError::MemoryAllocation(_) => ErrorSeverity::Fatal,
            GpuError::InvalidState(_) => ErrorSeverity::Fatal,
            GpuError::ComputationTimeout { .. } => ErrorSeverity::Recoverable,
            GpuError::QueueSubmission(_) => ErrorSeverity::Recoverable,
            GpuError::KernelExecution(_) => ErrorSeverity::Recoverable,
            GpuError::BarrierSynchronization(_) => ErrorSeverity::Recoverable,
            GpuError::BufferCopy(_) => ErrorSeverity::Recoverable,
            GpuError::BufferMapping(_) => ErrorSeverity::Recoverable,
            GpuError::Other(_) => ErrorSeverity::Recoverable,
        }
    }

    /// Check if we can retry a failed operation
    pub fn can_retry(&mut self, op: RecoverableGpuOp) -> bool {
        let count = self.retry_count.entry(op).or_insert(0);
        let max = self.max_retries.get(&op).copied().unwrap_or(0);

        if *count < max {
            *count += 1;
            true
        } else {
            false
        }
    }

    /// Reset retry count for a specific operation
    pub fn reset_retries(&mut self, op: RecoverableGpuOp) {
        self.retry_count.insert(op, 0);
    }

    /// Reset all retry counters
    pub fn reset_all_retries(&mut self) {
        self.retry_count.clear();
    }

    /// Set whether we encountered a fatal error
    pub fn set_fatal_error(&mut self, had_fatal: bool) {
        self.had_fatal_error = had_fatal;
    }

    /// Check if we had a fatal error
    pub fn had_fatal_error(&self) -> bool {
        self.had_fatal_error
    }

    /// Update the maximum number of retries for a specific operation
    pub fn set_max_retries(&mut self, op: RecoverableGpuOp, max: u32) {
        self.max_retries.insert(op, max);
    }

    /// Get the current retry count for an operation
    pub fn get_retry_count(&self, op: RecoverableGpuOp) -> u32 {
        *self.retry_count.get(&op).unwrap_or(&0)
    }

    /// Update the timeout configuration
    pub fn set_timeout_config(&mut self, config: AdaptiveTimeoutConfig) {
        self.timeout_config = config;
    }

    /// Get the current timeout configuration
    pub fn timeout_config(&self) -> &AdaptiveTimeoutConfig {
        &self.timeout_config
    }
}
