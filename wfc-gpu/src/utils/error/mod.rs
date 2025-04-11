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

    /// Formats a detailed diagnostic report for this error
    fn diagnostic_report(&self) -> String {
        let mut report = String::new();

        // Error type and message
        report.push_str(&format!("ERROR: {}\n", self));

        // Context information
        report.push_str(&format!("CONTEXT: {}\n", self.context()));

        // Severity
        report.push_str(&format!("SEVERITY: {:?}\n", self.severity()));
        report.push_str(&format!("RECOVERABLE: {}\n", self.is_recoverable()));

        // Suggested solution
        if let Some(solution) = self.suggested_solution() {
            report.push_str(&format!("\nSUGGESTED SOLUTION:\n{}\n", solution));
        }

        report
    }

    /// Logs the error diagnostic information to a provided function
    fn log_diagnostic<F>(&self, log_fn: F)
    where
        F: FnOnce(&str),
    {
        log_fn(&self.diagnostic_report());
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

impl WfcError {
    /// Create a new algorithm error
    pub fn algorithm<S: Into<String>>(msg: S) -> Self {
        Self::Algorithm(msg.into())
    }

    /// Create a new validation error
    pub fn validation<S: Into<String>>(msg: S) -> Self {
        Self::Validation(msg.into())
    }

    /// Create a new configuration error
    pub fn configuration<S: Into<String>>(msg: S) -> Self {
        Self::Configuration(msg.into())
    }

    /// Create a new general error
    pub fn other<S: Into<String>>(msg: S) -> Self {
        Self::Other(msg.into())
    }

    /// Format diagnostic information as JSON for easier parsing
    pub fn as_json(&self) -> String {
        let severity = match self.severity() {
            ErrorSeverity::Fatal => "fatal",
            ErrorSeverity::Recoverable => "recoverable",
            ErrorSeverity::Warning => "warning",
        };

        let error_type = match self {
            WfcError::Gpu(_) => "gpu",
            WfcError::Io(_) => "io",
            WfcError::Algorithm(_) => "algorithm",
            WfcError::Validation(_) => "validation",
            WfcError::Configuration(_) => "configuration",
            WfcError::Other(_) => "other",
        };

        let solution = match self.suggested_solution() {
            Some(s) => format!(
                "\"solution\": \"{}\"",
                s.replace('"', "\\\"").replace('\n', "\\n")
            ),
            None => "\"solution\": null".to_string(),
        };

        format!(
            "{{\"error\": \"{}\", \"type\": \"{}\", \"context\": \"{}\", \"severity\": \"{}\", \"recoverable\": {}, {}}}",
            self.to_string().replace('"', "\\\""),
            error_type,
            self.context().replace('"', "\\\"").replace('\n', "\\n"),
            severity,
            self.is_recoverable(),
            solution
        )
    }

    /// Returns a user-friendly recovery action suggestion
    ///
    /// This method suggests concrete actions that user code can take to
    /// recover from or respond to the error.
    pub fn suggested_action(&self) -> RecoveryAction {
        match self {
            WfcError::Gpu(err) => {
                match err.severity() {
                    ErrorSeverity::Recoverable => {
                        // For recoverable GPU errors, suggest appropriate actions
                        if err.to_string().contains("timeout") {
                            RecoveryAction::Retry
                        } else if err.to_string().contains("device lost") {
                            RecoveryAction::ReportError
                        } else if err.to_string().contains("contradiction") {
                            RecoveryAction::UseAlternative
                        } else {
                            RecoveryAction::RetryWithModifiedParams
                        }
                    }
                    ErrorSeverity::Warning => RecoveryAction::Retry,
                    ErrorSeverity::Fatal => RecoveryAction::ReportError,
                }
            }
            WfcError::Io(err) => match err.severity() {
                ErrorSeverity::Recoverable => RecoveryAction::Retry,
                _ => RecoveryAction::ReportError,
            },
            WfcError::Algorithm(_) => RecoveryAction::UseAlternative,
            WfcError::Validation(_) => RecoveryAction::RetryWithModifiedParams,
            WfcError::Configuration(_) => RecoveryAction::RetryWithModifiedParams,
            WfcError::Other(_) => RecoveryAction::ReportError,
        }
    }

    /// Returns true if this error is expected during normal operation
    ///
    /// Some errors, like WFC contradictions, are expected parts of the algorithm.
    /// This helper method helps distinguish between expected errors and true failures.
    pub fn is_expected(&self) -> bool {
        match self {
            WfcError::Gpu(err) => {
                matches!(err, GpuError::ContradictionDetected { .. })
            }
            _ => false,
        }
    }

    /// Returns true if this error suggests the operation should be retried
    pub fn should_retry(&self) -> bool {
        matches!(
            self.suggested_action(),
            RecoveryAction::Retry | RecoveryAction::RetryWithModifiedParams
        )
    }

    /// Returns true if this error suggests using an alternative approach
    pub fn should_use_alternative(&self) -> bool {
        matches!(self.suggested_action(), RecoveryAction::UseAlternative)
    }

    /// Returns true if the application should continue with reduced quality
    pub fn should_reduce_quality(&self) -> bool {
        matches!(self.suggested_action(), RecoveryAction::ReduceQuality)
    }

    /// Extract GPU-specific error details if this is a GPU error
    pub fn gpu_error(&self) -> Option<&GpuError> {
        match self {
            WfcError::Gpu(err) => Some(err),
            _ => None,
        }
    }

    /// Extract I/O-specific error details if this is an I/O error
    pub fn io_error(&self) -> Option<&IoError> {
        match self {
            WfcError::Io(err) => Some(err),
            _ => None,
        }
    }

    /// Returns a more detailed description of actions user code can take to handle this error
    pub fn recovery_instructions(&self) -> String {
        match self.suggested_action() {
            RecoveryAction::Retry => format!(
                "This error is potentially transient. Retry the operation.\n\
                Consider implementing an exponential backoff strategy if retrying multiple times."
            ),
            RecoveryAction::RetryWithModifiedParams => {
                let mut instructions =
                    String::from("Retry the operation with adjusted parameters:\n");

                match self {
                    WfcError::Gpu(err) => {
                        if err.to_string().contains("memory")
                            || err.to_string().contains("allocation")
                        {
                            instructions.push_str(
                                "- Reduce grid size or complexity\n\
                                 - Consider using subgrid processing\n\
                                 - Check for memory leaks in your application",
                            );
                        } else {
                            instructions.push_str(
                                "- Adjust timeouts or buffer sizes\n\
                                 - Consider simplifying the workload\n\
                                 - Check GPU driver status",
                            );
                        }
                    }
                    WfcError::Configuration(msg) => {
                        instructions.push_str(
                            "- Review configuration parameters\n\
                             - Check for incompatible settings\n\
                             - Consider using default values",
                        );
                    }
                    WfcError::Validation(msg) => {
                        instructions.push_str(
                            "- Validate input data before passing to the algorithm\n\
                             - Check value ranges and input format\n\
                             - Ensure all required fields are populated",
                        );
                    }
                    _ => {
                        instructions.push_str(
                            "- Review operation parameters\n\
                             - Check logs for additional details\n\
                             - Consider using alternative approaches",
                        );
                    }
                }

                instructions
            }
            RecoveryAction::UseAlternative => format!(
                "Consider an alternative approach:\n\
                - If this is a contradiction in WFC, try different starting constraints\n\
                - If algorithm-specific, consider a different algorithm variant\n\
                - For GPU-specific issues, consider CPU fallback if available"
            ),
            RecoveryAction::ReduceQuality => format!(
                "Continue with reduced quality or functionality:\n\
                - Reduce grid resolution or detail level\n\
                - Simplify ruleset or constraints\n\
                - Disable advanced features"
            ),
            RecoveryAction::ReportError => format!(
                "This error requires intervention:\n\
                - Log detailed error information\n\
                - Check system requirements and GPU compatibility\n\
                - Report the issue if it persists"
            ),
            RecoveryAction::NoAction => format!(
                "No recovery action is possible:\n\
                - Operation cannot continue\n\
                - Review logs and error details for diagnostic information"
            ),
        }
    }
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

/// Logs an error with detailed diagnostic information
#[macro_export]
macro_rules! log_error {
    ($error:expr, $log_fn:expr) => {{
        use $crate::utils::error::ErrorWithContext;
        $error.log_diagnostic($log_fn);
    }};
}

/// Creates a diagnostic report from the error
#[macro_export]
macro_rules! error_report {
    ($error:expr) => {{
        use $crate::utils::error::ErrorWithContext;
        $error.diagnostic_report()
    }};
}

/// Represents possible user-friendly recovery actions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryAction {
    /// Retry the operation that failed
    Retry,
    /// Retry with modified parameters
    RetryWithModifiedParams,
    /// Use an alternative approach
    UseAlternative,
    /// Reduce functionality or quality to proceed
    ReduceQuality,
    /// Report the error and seek assistance
    ReportError,
    /// No action is possible - operation cannot continue
    NoAction,
}

/// A type for user-defined recovery hooks that can be registered to handle specific errors
pub type RecoveryHookFn = Box<dyn Fn(&WfcError) -> Option<RecoveryAction> + Send + Sync>;

/// Registry for user-defined error recovery hooks
#[derive(Default)]
pub struct RecoveryHookRegistry {
    /// Hooks registered for specific error types
    hooks: Vec<(Box<dyn Fn(&WfcError) -> bool + Send + Sync>, RecoveryHookFn)>,
}

impl RecoveryHookRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    /// Register a hook that will be called for errors matching the predicate
    pub fn register<P, F>(&mut self, predicate: P, hook: F)
    where
        P: Fn(&WfcError) -> bool + Send + Sync + 'static,
        F: Fn(&WfcError) -> Option<RecoveryAction> + Send + Sync + 'static,
    {
        self.hooks.push((Box::new(predicate), Box::new(hook)));
    }

    /// Register a hook specifically for GPU errors
    pub fn register_for_gpu_errors<F>(&mut self, hook: F)
    where
        F: Fn(&GpuError) -> Option<RecoveryAction> + Send + Sync + 'static,
    {
        self.register(
            |err| matches!(err, WfcError::Gpu(_)),
            move |err| {
                if let WfcError::Gpu(gpu_err) = err {
                    hook(gpu_err)
                } else {
                    None
                }
            },
        );
    }

    /// Register a hook for algorithm errors
    pub fn register_for_algorithm_errors<F>(&mut self, hook: F)
    where
        F: Fn(&str) -> Option<RecoveryAction> + Send + Sync + 'static,
    {
        self.register(
            |err| matches!(err, WfcError::Algorithm(_)),
            move |err| {
                if let WfcError::Algorithm(msg) = err {
                    hook(msg)
                } else {
                    None
                }
            },
        );
    }

    /// Register a hook for validation errors
    pub fn register_for_validation_errors<F>(&mut self, hook: F)
    where
        F: Fn(&str) -> Option<RecoveryAction> + Send + Sync + 'static,
    {
        self.register(
            |err| matches!(err, WfcError::Validation(_)),
            move |err| {
                if let WfcError::Validation(msg) = err {
                    hook(msg)
                } else {
                    None
                }
            },
        );
    }

    /// Register a hook for configuration errors
    pub fn register_for_configuration_errors<F>(&mut self, hook: F)
    where
        F: Fn(&str) -> Option<RecoveryAction> + Send + Sync + 'static,
    {
        self.register(
            |err| matches!(err, WfcError::Configuration(_)),
            move |err| {
                if let WfcError::Configuration(msg) = err {
                    hook(msg)
                } else {
                    None
                }
            },
        );
    }

    /// Try to handle an error with registered hooks
    pub fn try_handle(&self, error: &WfcError) -> Option<RecoveryAction> {
        for (predicate, hook) in &self.hooks {
            if predicate(error) {
                if let Some(action) = hook(error) {
                    return Some(action);
                }
            }
        }
        None
    }
}
