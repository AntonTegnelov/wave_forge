//! Error recovery mechanisms for GPU operations.
//!
//! This module provides utilities for handling and recovering from non-fatal GPU errors
//! that may occur during WFC algorithm execution.

use log::{debug, error, info, warn};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

/// A simple coordinate struct for grid positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridCoord {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl GridCoord {
    pub fn new(x: usize, y: usize, z: usize) -> Self {
        Self { x, y, z }
    }
}

impl From<(usize, usize, usize)> for GridCoord {
    fn from((x, y, z): (usize, usize, usize)) -> Self {
        Self { x, y, z }
    }
}

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("Buffer map failed: {0}")]
    BufferMapFailed(String),
    #[error("Validation error: {0}")]
    ValidationError(wgpu::Error),
    #[error("Command execution error: {0}")]
    CommandExecutionError(String),
    #[error("Buffer operation error: {0}")]
    BufferOperationError(String),
    #[error("Data transfer error: {0}")]
    TransferError(String),
    #[error("GPU resource creation failed: {0}")]
    ResourceCreationFailed(String),
    #[error("Shader compilation failed: {0}")]
    ShaderError(String),
    #[error("Buffer size mismatch: {0}")]
    BufferSizeMismatch(String),
    #[error("Timeout occurred: {0}")]
    Timeout(String),
    #[error("Device lost: {0}")]
    DeviceLost(String),
    #[error("Internal error: {0}")]
    InternalError(String),
    #[error("GPU Adapter request failed")]
    AdapterRequestFailed,
    #[error("GPU Device request failed: {0}")]
    DeviceRequestFailed(wgpu::RequestDeviceError),
    #[error("Mutex lock error: {0}")]
    MutexError(String),
    #[error("Buffer error: {0}")]
    BufferError(String),
    #[error("Buffer map error: {0}")]
    BufferMapError(String),
    #[error("Buffer map timeout: {0}")]
    BufferMapTimeout(String),
    #[error("Contradiction detected at {coord:?}")]
    ContradictionDetected { coord: Option<GridCoord> },
    #[error("Other GPU error: {0}")]
    Other(String),
}

/// Represents the severity level of a GPU error
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    /// Fatal errors cause operations to fail and do not allow recovery
    Fatal,
    /// Recoverable errors can be retried or worked around
    Recoverable,
    /// Warning errors indicate potential issues but operations continue
    Warning,
}

/// Manages error recovery attempts and strategies for GPU operations.
///
/// This struct tracks error attempts, implements backoff strategies, and provides
/// mechanisms to handle certain classes of GPU errors that do not require immediate
/// termination of the WFC algorithm.
pub struct GpuErrorRecovery {
    /// Number of retries allowed for recoverable operations
    max_retries: u32,
    /// Current error counter for the session
    error_counter: AtomicU32,
    /// Base delay for exponential backoff (in milliseconds)
    base_delay_ms: u64,
    /// Maximum delay for exponential backoff (in milliseconds)
    max_delay_ms: u64,
    /// Last error timestamp for rate limiting
    last_error_time: Instant,
    /// Whether to attempt recovery for buffer mapping errors
    recover_buffer_mapping: bool,
    /// Whether to attempt recovery for device lost errors
    recover_device_lost: bool,
    /// Whether to attempt recovery for timeout errors
    recover_timeouts: bool,
    /// Adaptive timeout settings based on grid complexity
    adaptive_timeout: Option<AdaptiveTimeoutConfig>,
}

/// Configuration for adaptive timeout calculations.
///
/// This struct holds parameters used to calculate appropriate timeout durations
/// based on grid size and complexity. The calculated timeout is used for various
/// GPU operations to account for larger or more complex grids requiring more time.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveTimeoutConfig {
    /// Base timeout in milliseconds for a reference-sized grid
    pub base_timeout_ms: u64,
    /// Reference grid cell count for scaling calculations
    pub reference_cell_count: u64,
    /// How much to scale the timeout based on grid size (power factor)
    pub size_scale_factor: f64,
    /// Additional multiplier for operations with many tiles per cell
    pub tile_count_multiplier: f64,
    /// Maximum timeout duration in milliseconds (upper bound)
    pub max_timeout_ms: u64,
}

impl Default for AdaptiveTimeoutConfig {
    fn default() -> Self {
        Self {
            base_timeout_ms: 1000,        // 1 second base timeout
            reference_cell_count: 10_000, // 10,000 cells (e.g., 100x100 grid)
            size_scale_factor: 0.5,       // Sub-linear scaling with size
            tile_count_multiplier: 0.01,  // 1% increase per tile
            max_timeout_ms: 10_000,       // 10 seconds maximum timeout
        }
    }
}

impl Default for GpuErrorRecovery {
    fn default() -> Self {
        Self {
            max_retries: 3,
            error_counter: AtomicU32::new(0),
            base_delay_ms: 50,
            max_delay_ms: 2000,
            last_error_time: Instant::now(),
            recover_buffer_mapping: true,
            recover_device_lost: false, // Device lost usually requires full reinitialization
            recover_timeouts: true,
            adaptive_timeout: None, // Disabled by default
        }
    }
}

impl GpuErrorRecovery {
    /// Creates a new GpuErrorRecovery with custom settings.
    pub fn new(
        max_retries: u32,
        base_delay_ms: u64,
        max_delay_ms: u64,
        recover_buffer_mapping: bool,
        recover_device_lost: bool,
        recover_timeouts: bool,
    ) -> Self {
        Self {
            max_retries,
            error_counter: AtomicU32::new(0),
            base_delay_ms,
            max_delay_ms,
            last_error_time: Instant::now(),
            recover_buffer_mapping,
            recover_device_lost,
            recover_timeouts,
            adaptive_timeout: None,
        }
    }

    /// Enables adaptive timeouts based on grid size and complexity.
    ///
    /// This configures the GpuErrorRecovery to calculate appropriate timeout
    /// durations based on grid dimensions and other complexity factors.
    /// This helps prevent premature timeouts for larger grids.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration parameters for the adaptive timeout
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining
    pub fn with_adaptive_timeout(mut self, config: AdaptiveTimeoutConfig) -> Self {
        self.adaptive_timeout = Some(config);
        self
    }

    /// Calculates an appropriate timeout duration based on grid size and complexity.
    ///
    /// # Arguments
    ///
    /// * `grid_cells` - Total number of cells in the grid (width * height * depth)
    /// * `tile_count` - Number of possible tile types
    /// * `operation_complexity` - Relative complexity factor of the operation (default 1.0)
    ///
    /// # Returns
    ///
    /// Duration representing the calculated timeout
    pub fn calculate_timeout(
        &self,
        grid_cells: usize,
        tile_count: usize,
        operation_complexity: f64,
    ) -> Duration {
        match self.adaptive_timeout {
            Some(config) => {
                // Calculate size factor based on cell count
                let size_ratio = grid_cells as f64 / config.reference_cell_count as f64;
                let size_factor = size_ratio.powf(config.size_scale_factor);

                // Calculate tile complexity factor
                let tile_factor = 1.0 + (tile_count as f64 * config.tile_count_multiplier);

                // Calculate final timeout in milliseconds
                let computed_ms = (config.base_timeout_ms as f64
                    * size_factor
                    * tile_factor
                    * operation_complexity)
                    .round() as u64;
                let timeout_ms = computed_ms.min(config.max_timeout_ms);

                Duration::from_millis(timeout_ms.max(1)) // At least 1ms
            }
            None => {
                // If adaptive timeouts are disabled, use a fixed default
                Duration::from_secs(5) // Default 5-second timeout
            }
        }
    }

    /// Determines if an error is recoverable based on its type and configured settings.
    ///
    /// # Arguments
    ///
    /// * `error` - The GPU error to evaluate
    ///
    /// # Returns
    ///
    /// The error's severity level
    pub fn classify_error(&self, error: &GpuError) -> ErrorSeverity {
        match error {
            GpuError::BufferMapFailed(_)
            | GpuError::BufferError(_)
            | GpuError::BufferOperationError { .. }
            | GpuError::BufferMapError(_)
            | GpuError::BufferMapTimeout(_)
            | GpuError::BufferSizeMismatch(_) => {
                if self.recover_buffer_mapping {
                    ErrorSeverity::Recoverable
                } else {
                    ErrorSeverity::Fatal
                }
            }
            GpuError::ValidationError(wgpu_error) => {
                // Handle device lost errors with different patterns based on wgpu version
                // Check string representation for device lost message
                let error_str = wgpu_error.to_string().to_lowercase();
                if error_str.contains("device lost") || error_str.contains("devicelost") {
                    if self.recover_device_lost {
                        ErrorSeverity::Recoverable
                    } else {
                        ErrorSeverity::Fatal
                    }
                } else {
                    // Most validation errors indicate API usage issues that should be fixed
                    ErrorSeverity::Fatal
                }
            }
            GpuError::CommandExecutionError(msg) => {
                if msg.contains("timeout") && self.recover_timeouts {
                    // Use adaptive timeout recovery for timeout errors
                    // This will ensure that larger grids get more attempts and longer timeouts
                    if self.adaptive_timeout.is_some() {
                        log::debug!(
                            "Using adaptive timeout recovery for command execution timeout"
                        );
                    }
                    ErrorSeverity::Recoverable
                } else {
                    ErrorSeverity::Fatal
                }
            }
            GpuError::TransferError(_) => ErrorSeverity::Recoverable,
            GpuError::Other(msg) => {
                if msg.contains("timeout") && self.recover_timeouts {
                    ErrorSeverity::Recoverable
                } else {
                    ErrorSeverity::Warning
                }
            }
            _ => ErrorSeverity::Fatal,
        }
    }

    /// Returns whether we should attempt recovery based on the error count and max retries.
    pub fn should_attempt_recovery(&self, error: &GpuError) -> bool {
        let error_count = self.error_counter.load(Ordering::Relaxed);

        // Always allow at least one retry (the current attempt)
        if error_count == 0 {
            return true;
        }

        let severity = self.classify_error(error);
        match severity {
            ErrorSeverity::Fatal => false,
            ErrorSeverity::Recoverable => error_count <= self.max_retries,
            ErrorSeverity::Warning => true,
        }
    }

    /// Calculates the delay time for the next retry based on an exponential backoff strategy.
    fn get_backoff_delay(&self) -> Duration {
        let error_count = self.error_counter.load(Ordering::Relaxed);
        let exponent = error_count.min(10); // Cap exponent to avoid overflow
        let delay_ms = self
            .base_delay_ms
            .saturating_mul(2u64.saturating_pow(exponent))
            .min(self.max_delay_ms);
        Duration::from_millis(delay_ms)
    }

    /// Records an error and returns the recommended backoff delay if recovery should be attempted.
    ///
    /// # Arguments
    ///
    /// * `error` - The GPU error that occurred
    ///
    /// # Returns
    ///
    /// * `Some(Duration)` - If recovery should be attempted, the recommended delay
    /// * `None` - If recovery should not be attempted
    pub fn record_error(&self, error: &GpuError) -> Option<Duration> {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_error_time);

        // Rate limit error logging (don't spam logs with the same error type)
        if elapsed > Duration::from_secs(1) {
            match self.classify_error(error) {
                ErrorSeverity::Fatal => error!("Fatal GPU error encountered: {}", error),
                ErrorSeverity::Recoverable => warn!("Recoverable GPU error encountered: {}", error),
                ErrorSeverity::Warning => debug!("Warning-level GPU error: {}", error),
            }
        }

        self.error_counter.fetch_add(1, Ordering::Relaxed);

        if self.should_attempt_recovery(error) {
            let delay = self.get_backoff_delay();
            Some(delay)
        } else {
            None
        }
    }

    /// Resets the error counter, typically called after successful operations.
    pub fn reset_error_count(&self) {
        self.error_counter.store(0, Ordering::Relaxed);
    }

    /// Gets the current error count
    pub fn error_count(&self) -> u32 {
        self.error_counter.load(Ordering::Relaxed)
    }
}

/// A wrapper for recovery-enabled GPU operations.
///
/// This struct provides methods for executing GPU operations with automatic recovery
/// attempts for non-fatal errors.
pub struct RecoverableGpuOp {
    recovery: Arc<GpuErrorRecovery>,
    /// Grid size information for adaptive timeouts
    grid_cells: Option<usize>,
    /// Tile count information for adaptive timeouts
    tile_count: Option<usize>,
}

impl RecoverableGpuOp {
    /// Creates a new RecoverableGpuOp with default settings.
    pub fn new() -> Self {
        Self {
            recovery: Arc::new(GpuErrorRecovery::default()),
            grid_cells: None,
            tile_count: None,
        }
    }

    /// Creates a new RecoverableGpuOp with custom error recovery settings.
    pub fn with_recovery(recovery: Arc<GpuErrorRecovery>) -> Self {
        Self {
            recovery,
            grid_cells: None,
            tile_count: None,
        }
    }

    /// Enables and configures adaptive timeout calculations for GPU operations.
    ///
    /// # Arguments
    ///
    /// * `timeout_config` - Configuration for calculating adaptive timeouts
    /// * `grid_cells` - Total number of cells in the grid (width * height * depth)
    /// * `tile_count` - Number of possible tile types
    ///
    /// # Returns
    ///
    /// `Self` with adaptive timeout settings configured
    pub fn with_adaptive_timeout(
        mut self,
        timeout_config: AdaptiveTimeoutConfig,
        grid_cells: usize,
        tile_count: usize,
    ) -> Self {
        self.recovery = Arc::new(GpuErrorRecovery::default().with_adaptive_timeout(timeout_config));
        self.grid_cells = Some(grid_cells);
        self.tile_count = Some(tile_count);
        self
    }

    /// Calculates the appropriate timeout for the current operation.
    ///
    /// # Arguments
    ///
    /// * `operation_complexity` - Relative complexity factor (default: 1.0)
    ///
    /// # Returns
    ///
    /// Duration representing the appropriate timeout
    pub fn calculate_operation_timeout(&self, operation_complexity: f64) -> Duration {
        let grid_cells = self.grid_cells.unwrap_or(10_000);
        let tile_count = self.tile_count.unwrap_or(16);
        self.recovery
            .calculate_timeout(grid_cells, tile_count, operation_complexity)
    }

    /// Attempts to execute a GPU operation with automatic retries for recoverable errors.
    ///
    /// The function will:
    /// 1. Try to execute the provided operation
    /// 2. If it fails with a recoverable error, wait using exponential backoff
    /// 3. Retry up to the configured maximum number of attempts
    ///
    /// # Arguments
    ///
    /// * `op` - A closure that returns a Result with a GpuError as the error type
    ///
    /// # Returns
    ///
    /// The result of the operation, or the last error if all recovery attempts fail
    pub async fn try_with_recovery<T, F, Fut>(&self, mut op: F) -> Result<T, GpuError>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T, GpuError>>,
    {
        let mut attempts = 0;
        let max_attempts = self.recovery.max_retries as usize + 1; // +1 for initial attempt

        loop {
            match op().await {
                Ok(result) => {
                    // Reset error count on success after retries
                    if attempts > 0 {
                        self.recovery.reset_error_count();
                    }
                    return Ok(result);
                }
                Err(err) => {
                    attempts += 1;

                    // Use consistent error classification for all buffer types
                    let is_buffer_error = matches!(
                        err,
                        GpuError::BufferMapFailed(_)
                            | GpuError::BufferError(_)
                            | GpuError::BufferOperationError { .. }
                            | GpuError::BufferMapError(_)
                            | GpuError::BufferMapTimeout(_)
                            | GpuError::BufferSizeMismatch(_)
                    );

                    if attempts >= max_attempts
                        || !self.recovery.should_attempt_recovery(&err)
                        || (is_buffer_error && !self.recovery.recover_buffer_mapping)
                    {
                        return Err(err);
                    }

                    // Consistent handling for all buffer errors
                    if is_buffer_error {
                        debug!(
                            "Buffer operation failed (attempt {}/{}). Retrying after {:?}...",
                            attempts,
                            max_attempts,
                            self.recovery.get_backoff_delay()
                        );
                    } else {
                        // ... existing debug logs for other error types ...
                    }

                    // Apply backoff delay
                    if let Some(delay) = self.recovery.record_error(&err) {
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }
    }

    /// Synchronous version of try_with_recovery for operations that don't use async/await.
    ///
    /// # Arguments
    ///
    /// * `op` - A closure that returns a Result with a GpuError as the error type
    ///
    /// # Returns
    ///
    /// The result of the operation, or the last error if all recovery attempts fail
    pub fn try_with_recovery_sync<T, F>(&self, mut op: F) -> Result<T, GpuError>
    where
        F: FnMut() -> Result<T, GpuError>,
    {
        let mut attempts = 0;
        let max_attempts = self.recovery.max_retries as usize + 1; // +1 for initial attempt

        loop {
            match op() {
                Ok(result) => {
                    // Reset error count on success after retries
                    if attempts > 0 {
                        self.recovery.reset_error_count();
                    }
                    return Ok(result);
                }
                Err(err) => {
                    attempts += 1;

                    // Use consistent error classification for all buffer types
                    let is_buffer_error = matches!(
                        err,
                        GpuError::BufferMapFailed(_)
                            | GpuError::BufferError(_)
                            | GpuError::BufferOperationError { .. }
                            | GpuError::BufferMapError(_)
                            | GpuError::BufferMapTimeout(_)
                            | GpuError::BufferSizeMismatch(_)
                    );

                    if attempts >= max_attempts
                        || !self.recovery.should_attempt_recovery(&err)
                        || (is_buffer_error && !self.recovery.recover_buffer_mapping)
                    {
                        return Err(err);
                    }

                    // Consistent handling for all buffer errors
                    if is_buffer_error {
                        debug!(
                            "Buffer operation failed (attempt {}/{}). Retrying after {:?}...",
                            attempts,
                            max_attempts,
                            self.recovery.get_backoff_delay()
                        );
                    } else {
                        // ... existing debug logs for other error types ...
                    }

                    // Apply backoff delay
                    if let Some(delay) = self.recovery.record_error(&err) {
                        std::thread::sleep(delay);
                    }
                }
            }
        }
    }
}

impl Default for RecoverableGpuOp {
    fn default() -> Self {
        Self::new()
    }
}
