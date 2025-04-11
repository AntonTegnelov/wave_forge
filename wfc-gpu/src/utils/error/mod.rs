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
            WfcError::Algorithm(msg) => Some(format!(
                concat!(
                    "Algorithm error: {}. Try the following:\n",
                    "1. Check algorithm parameters for validity\n",
                    "2. Verify input data matches algorithm requirements\n",
                    "3. Ensure input grid dimensions are compatible with algorithm\n",
                    "4. Consider using a different algorithm variant if available\n",
                    "5. Review the implementation for logical errors"
                ),
                msg
            )),
            WfcError::Validation(msg) => Some(format!(
                concat!(
                    "Validation error: {}. Try the following:\n",
                    "1. Check input values against allowed ranges\n",
                    "2. Verify data format and structure\n",
                    "3. Ensure all required fields are populated\n",
                    "4. Check for inconsistencies in configuration\n",
                    "5. Add pre-validation steps to catch issues earlier"
                ),
                msg
            )),
            WfcError::Configuration(msg) => Some(format!(
                concat!(
                    "Configuration error: {}. Try the following:\n",
                    "1. Review configuration settings for validity\n",
                    "2. Check for conflicting or incompatible options\n",
                    "3. Ensure required configuration properties are set\n",
                    "4. Verify that feature requirements match capabilities\n",
                    "5. Consider using default configuration if unsure"
                ),
                msg
            )),
            WfcError::Other(msg) => Some(format!(
                concat!(
                    "Error: {}. Try the following:\n",
                    "1. Check logs for additional error details\n",
                    "2. Review recent code changes that might affect functionality\n",
                    "3. Verify system resources are sufficient\n",
                    "4. Consider restarting the application\n",
                    "5. Report the issue if it persists"
                ),
                msg
            )),
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
