//! Unified error system for the WFC-GPU module.
//!
//! This module provides a structured approach to error handling and recovery,
//! with specialized error types and recovery strategies for various failure scenarios.

pub mod gpu_error;
pub mod io_error;

use std::fmt;
use thiserror::Error;

pub use gpu_error::{GpuError, GpuErrorContext, GpuResourceType};
pub use io_error::{IoError, IoResourceType};

/// Common trait for all error types in the WFC-GPU module.
pub trait ErrorWithContext: std::error::Error {
    /// Returns a descriptive context string for this error
    fn context(&self) -> String;

    /// Returns a suggested solution for this error, if available
    fn suggested_solution(&self) -> Option<String>;

    /// Returns the severity level of this error
    fn severity(&self) -> ErrorSeverity;

    /// Returns true if this error is potentially recoverable
    fn is_recoverable(&self) -> bool {
        self.severity() != ErrorSeverity::Fatal
    }
}

/// Main error type for the WFC-GPU library, encompassing all possible error scenarios.
#[derive(Error, Debug)]
pub enum WfcError {
    /// GPU-related errors
    #[error("GPU error: {0}")]
    Gpu(#[from] GpuError),

    /// I/O and resource errors
    #[error("I/O error: {0}")]
    Io(#[from] IoError),

    /// Algorithm-specific errors
    #[error("Algorithm error: {0}")]
    Algorithm(String),

    /// Validation errors
    #[error("Validation error: {0}")]
    Validation(String),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Other errors
    #[error("Error: {0}")]
    Other(String),
}

impl ErrorWithContext for WfcError {
    fn context(&self) -> String {
        match self {
            WfcError::Gpu(err) => err.context(),
            WfcError::Io(err) => err.context(),
            WfcError::Algorithm(msg) => format!("Algorithm error: {}", msg),
            WfcError::Validation(msg) => format!("Validation error: {}", msg),
            WfcError::Configuration(msg) => format!("Configuration error: {}", msg),
            WfcError::Other(msg) => format!("Other error: {}", msg),
        }
    }

    fn suggested_solution(&self) -> Option<String> {
        match self {
            WfcError::Gpu(err) => err.suggested_solution(),
            WfcError::Io(err) => err.suggested_solution(),
            WfcError::Algorithm(_) => Some("Check algorithm parameters and input data".to_string()),
            WfcError::Validation(_) => Some("Validate input data and configuration".to_string()),
            WfcError::Configuration(_) => Some("Review configuration settings".to_string()),
            WfcError::Other(_) => None,
        }
    }

    fn severity(&self) -> ErrorSeverity {
        match self {
            WfcError::Gpu(err) => err.severity(),
            WfcError::Io(err) => err.severity(),
            WfcError::Algorithm(_) => ErrorSeverity::Fatal,
            WfcError::Validation(_) => ErrorSeverity::Fatal,
            WfcError::Configuration(_) => ErrorSeverity::Fatal,
            WfcError::Other(_) => ErrorSeverity::Fatal,
        }
    }
}

/// Represents the severity level of an error
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    /// Fatal errors cause operations to fail and do not allow recovery
    Fatal,
    /// Recoverable errors can be retried or worked around
    Recoverable,
    /// Warning errors indicate potential issues but operations continue
    Warning,
}

/// Location information for errors
#[derive(Debug, Clone)]
pub struct ErrorLocation {
    /// File where the error occurred
    pub file: &'static str,
    /// Line number where the error occurred
    pub line: u32,
    /// Function name where the error occurred
    pub function: &'static str,
}

impl ErrorLocation {
    /// Create a new ErrorLocation
    pub const fn new(file: &'static str, line: u32, function: &'static str) -> Self {
        Self {
            file,
            line,
            function,
        }
    }
}

impl fmt::Display for ErrorLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{} in {}", self.file, self.line, self.function)
    }
}

/// Macro to create an ErrorLocation with the current source location
#[macro_export]
macro_rules! error_location {
    () => {
        $crate::utils::error::ErrorLocation::new(file!(), line!(), function_name!())
    };
}

// Re-export function_name for use with error_location! macro
use function_name::named;
