//! I/O and resource-related error types and handling logic.
//!
//! This module defines specialized error types for I/O operations,
//! with detailed context information to aid in debugging and recovery.

use super::{ErrorLocation, ErrorSeverity, ErrorWithContext};
use std::fmt;
use std::path::PathBuf;
use thiserror::Error;

/// Types of I/O resources that can be involved in errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoResourceType {
    File,
    Directory,
    Network,
    Memory,
    Resource,
    Other,
}

impl fmt::Display for IoResourceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IoResourceType::File => write!(f, "File"),
            IoResourceType::Directory => write!(f, "Directory"),
            IoResourceType::Network => write!(f, "Network"),
            IoResourceType::Memory => write!(f, "Memory"),
            IoResourceType::Resource => write!(f, "Resource"),
            IoResourceType::Other => write!(f, "Other"),
        }
    }
}

/// Contains contextual information about an I/O error
#[derive(Debug, Clone)]
pub struct IoErrorContext {
    /// Type of I/O resource involved
    pub resource_type: IoResourceType,
    /// Path of the resource, if applicable
    pub path: Option<PathBuf>,
    /// Additional details about the error
    pub details: Option<String>,
    /// Source location information
    pub location: Option<ErrorLocation>,
    /// Suggested solution or recovery strategy
    pub suggested_solution: Option<String>,
}

impl Default for IoErrorContext {
    fn default() -> Self {
        Self {
            resource_type: IoResourceType::Other,
            path: None,
            details: None,
            location: None,
            suggested_solution: None,
        }
    }
}

impl IoErrorContext {
    /// Create a new I/O error context
    pub fn new(resource_type: IoResourceType) -> Self {
        Self {
            resource_type,
            ..Default::default()
        }
    }

    /// Add a path to the context
    pub fn with_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Add detailed information to the context
    pub fn with_details<S: Into<String>>(mut self, details: S) -> Self {
        self.details = Some(details.into());
        self
    }

    /// Add source location information to the context
    pub fn with_location(mut self, location: ErrorLocation) -> Self {
        self.location = Some(location);
        self
    }

    /// Add a suggested solution to the context
    pub fn with_solution<S: Into<String>>(mut self, solution: S) -> Self {
        self.suggested_solution = Some(solution.into());
        self
    }
}

impl fmt::Display for IoErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "I/O {} error", self.resource_type)?;

        if let Some(path) = &self.path {
            write!(f, " with '{}'", path.display())?;
        }

        if let Some(details) = &self.details {
            write!(f, ": {}", details)?;
        }

        if let Some(location) = &self.location {
            write!(f, " [at {}]", location)?;
        }

        Ok(())
    }
}

/// Enumeration of all possible I/O errors
#[derive(Error, Debug)]
pub enum IoError {
    #[error("File not found")]
    NotFound { context: IoErrorContext },

    #[error("Permission denied")]
    PermissionDenied { context: IoErrorContext },

    #[error("I/O error: {0}")]
    Io(std::io::Error, IoErrorContext),

    #[error("Serialization error: {msg}")]
    Serialization {
        msg: String,
        context: IoErrorContext,
    },

    #[error("Deserialization error: {msg}")]
    Deserialization {
        msg: String,
        context: IoErrorContext,
    },

    #[error("Resource not available: {msg}")]
    ResourceNotAvailable {
        msg: String,
        context: IoErrorContext,
    },

    #[error("Memory allocation error: {msg}")]
    MemoryError {
        msg: String,
        context: IoErrorContext,
    },

    #[error("Other I/O error: {msg}")]
    Other {
        msg: String,
        context: IoErrorContext,
    },

    /// Create an error for file or resource loading failures
    #[error("Resource error: {message}")]
    ResourceError {
        kind: IoResourceType,
        message: String,
        context: IoErrorContext,
    },
}

impl IoError {
    /// Create a new file not found error
    pub fn not_found(context: IoErrorContext) -> Self {
        Self::NotFound { context }
    }

    /// Create a new permission denied error
    pub fn permission_denied(context: IoErrorContext) -> Self {
        Self::PermissionDenied { context }
    }

    /// Create a new I/O error
    pub fn io(err: std::io::Error, context: IoErrorContext) -> Self {
        Self::Io(err, context)
    }

    /// Create a new serialization error
    pub fn serialization<S: Into<String>>(msg: S, context: IoErrorContext) -> Self {
        Self::Serialization {
            msg: msg.into(),
            context,
        }
    }

    /// Create a new deserialization error
    pub fn deserialization<S: Into<String>>(msg: S, context: IoErrorContext) -> Self {
        Self::Deserialization {
            msg: msg.into(),
            context,
        }
    }

    /// Create a new resource not available error
    pub fn resource_not_available<S: Into<String>>(msg: S, context: IoErrorContext) -> Self {
        Self::ResourceNotAvailable {
            msg: msg.into(),
            context,
        }
    }

    /// Create a new memory error
    pub fn memory_error<S: Into<String>>(msg: S, context: IoErrorContext) -> Self {
        Self::MemoryError {
            msg: msg.into(),
            context,
        }
    }

    /// Create a new other error
    pub fn other<S: Into<String>>(msg: S, context: IoErrorContext) -> Self {
        Self::Other {
            msg: msg.into(),
            context,
        }
    }

    /// Create an error for file or resource loading failures
    pub fn loading<S: Into<String>>(msg: S) -> Self {
        Self::ResourceError {
            kind: IoResourceType::File,
            message: msg.into(),
            context: IoErrorContext::default(),
        }
    }

    /// Get the context information for this error
    pub fn get_context(&self) -> &IoErrorContext {
        match self {
            Self::NotFound { context } => context,
            Self::PermissionDenied { context } => context,
            Self::Io(_, context) => context,
            Self::Serialization { context, .. } => context,
            Self::Deserialization { context, .. } => context,
            Self::ResourceNotAvailable { context, .. } => context,
            Self::MemoryError { context, .. } => context,
            Self::Other { context, .. } => context,
            Self::ResourceError { context, .. } => context,
        }
    }
}

impl ErrorWithContext for IoError {
    fn context(&self) -> String {
        match self {
            Self::NotFound { context } => format!("{}", context),
            Self::PermissionDenied { context } => format!("{}", context),
            Self::Io(err, context) => format!("{}: {}", context, err),
            Self::Serialization { msg, context } => format!("{}: {}", context, msg),
            Self::Deserialization { msg, context } => format!("{}: {}", context, msg),
            Self::ResourceNotAvailable { msg, context } => format!("{}: {}", context, msg),
            Self::MemoryError { msg, context } => format!("{}: {}", context, msg),
            Self::Other { msg, context } => format!("{}: {}", context, msg),
            Self::ResourceError {
                message, context, ..
            } => format!("{}: {}", context, message),
        }
    }

    fn suggested_solution(&self) -> Option<String> {
        let context = self.get_context();

        // Return custom solution if available
        if let Some(solution) = &context.suggested_solution {
            return Some(solution.clone());
        }

        // Otherwise return default solutions based on error type
        match self {
            Self::NotFound { .. } => Some(
                concat!(
                    "File not found. Try the following:\n",
                    "1. Check that the file path is correct\n",
                    "2. Verify that the file exists at the specified location\n",
                    "3. Ensure that external resources have been downloaded or generated\n",
                    "4. Check for typos in file paths or names\n",
                    "5. Verify working directory if using relative paths"
                )
                .to_string(),
            ),
            Self::PermissionDenied { .. } => Some(
                concat!(
                    "Permission denied. Try the following:\n",
                    "1. Check file permissions for current user\n",
                    "2. Run the application with elevated privileges if necessary\n",
                    "3. Verify that the file is not locked by another process\n",
                    "4. Check if the resource is read-only\n",
                    "5. Ensure proper access rights on network resources"
                )
                .to_string(),
            ),
            Self::Io(err, _) => {
                let default_msg = concat!(
                    "I/O error occurred. Try the following:\n",
                    "1. Check if the device or resource is available\n",
                    "2. Verify network connectivity for remote resources\n",
                    "3. Ensure sufficient disk space is available\n",
                    "4. Close other applications that might be using the resource\n",
                    "5. Retry the operation after a short delay"
                );

                // Add more specific advice based on the error kind
                match err.kind() {
                    std::io::ErrorKind::AlreadyExists => Some(
                        concat!(
                            "Resource already exists. Try the following:\n",
                            "1. Use a different name for the resource\n",
                            "2. Delete or move the existing resource first\n",
                            "3. Check if the application can overwrite existing resources\n",
                            "4. Use a unique naming scheme for resources\n",
                            "5. Consider implementing versioning for resources"
                        )
                        .to_string(),
                    ),
                    std::io::ErrorKind::Interrupted => Some(
                        concat!(
                            "Operation interrupted. Try the following:\n",
                            "1. Retry the operation\n",
                            "2. Implement retry logic with backoff\n",
                            "3. Check for system signals that might be interrupting operation\n",
                            "4. Handle interruption gracefully in the application\n",
                            "5. Consider using async I/O for better handling of interruptions"
                        )
                        .to_string(),
                    ),
                    std::io::ErrorKind::Unsupported => Some(
                        concat!(
                            "Unsupported operation. Try the following:\n",
                            "1. Check if the operation is supported on this platform\n",
                            "2. Use an alternative approach that is supported\n",
                            "3. Verify feature requirements for the operation\n",
                            "4. Check documentation for platform-specific limitations\n",
                            "5. Consider using abstraction libraries for cross-platform support"
                        )
                        .to_string(),
                    ),
                    std::io::ErrorKind::ConnectionRefused => Some(
                        concat!(
                            "Connection refused. Try the following:\n",
                            "1. Verify the server is running and accessible\n",
                            "2. Check network connectivity and firewall settings\n",
                            "3. Verify connection parameters (host, port)\n",
                            "4. Ensure the service is accepting connections\n",
                            "5. Implement connection retry with exponential backoff"
                        )
                        .to_string(),
                    ),
                    std::io::ErrorKind::TimedOut => Some(
                        concat!(
                            "Operation timed out. Try the following:\n",
                            "1. Increase timeout duration if possible\n",
                            "2. Check if the resource is overloaded or slow\n",
                            "3. Verify network connectivity for remote resources\n",
                            "4. Implement retry logic with backoff\n",
                            "5. Consider breaking the operation into smaller chunks"
                        )
                        .to_string(),
                    ),
                    _ => Some(default_msg.to_string()),
                }
            }
            Self::Serialization { .. } => Some(
                concat!(
                    "Serialization error. Try the following:\n",
                    "1. Check the data structure for incompatible types\n",
                    "2. Verify that all fields are serializable\n",
                    "3. Consider using a different serialization format\n",
                    "4. Add robust error handling for serialization failures\n",
                    "5. Validate data before serialization"
                )
                .to_string(),
            ),
            Self::Deserialization { .. } => Some(
                concat!(
                    "Deserialization error. Try the following:\n",
                    "1. Verify the format of the input data\n",
                    "2. Check for format version mismatches\n",
                    "3. Ensure the data structure matches the serialized format\n",
                    "4. Validate input data before attempting deserialization\n",
                    "5. Consider implementing migration for older formats"
                )
                .to_string(),
            ),
            Self::ResourceNotAvailable { .. } => Some(
                concat!(
                    "Resource not available. Try the following:\n",
                    "1. Check if the resource exists and is accessible\n",
                    "2. Verify that external dependencies are installed\n",
                    "3. Ensure sufficient system resources (memory, disk space)\n",
                    "4. Check if resource is locked by another process\n",
                    "5. Implement resource availability checking before use"
                )
                .to_string(),
            ),
            Self::MemoryError { .. } => Some(
                concat!(
                    "Memory allocation error. Try the following:\n",
                    "1. Reduce memory usage by processing data in smaller chunks\n",
                    "2. Close other applications to free up memory\n",
                    "3. Check for memory leaks in the application\n",
                    "4. Ensure system has sufficient RAM and swap space\n",
                    "5. Consider using memory-mapped files for large datasets"
                )
                .to_string(),
            ),
            Self::Other { .. } => Some(
                concat!(
                    "I/O error occurred. Try the following:\n",
                    "1. Check system logs for detailed error information\n",
                    "2. Verify system resources and permissions\n",
                    "3. Restart the application or service\n",
                    "4. Check for operating system or hardware issues\n",
                    "5. Consider updating drivers or system software"
                )
                .to_string(),
            ),
            Self::ResourceError { message, .. } => Some(message.clone()),
        }
    }

    fn severity(&self) -> ErrorSeverity {
        match self {
            // Most I/O errors are recoverable since they often involve
            // external resources that might become available later
            Self::NotFound { .. } => ErrorSeverity::Recoverable,
            Self::PermissionDenied { .. } => ErrorSeverity::Recoverable,
            Self::Io(_, _) => ErrorSeverity::Recoverable,
            Self::ResourceNotAvailable { .. } => ErrorSeverity::Recoverable,

            // Format/parsing errors are usually fatal since they indicate
            // data is corrupt or incompatible
            Self::Serialization { .. } => ErrorSeverity::Fatal,
            Self::Deserialization { .. } => ErrorSeverity::Fatal,

            // Memory errors are usually fatal
            Self::MemoryError { .. } => ErrorSeverity::Fatal,

            // Other errors are fatal by default
            Self::Other { .. } => ErrorSeverity::Fatal,

            // Resource errors are fatal by default
            Self::ResourceError { .. } => ErrorSeverity::Fatal,
        }
    }
}
