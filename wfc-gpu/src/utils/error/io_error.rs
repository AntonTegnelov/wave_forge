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
            Self::NotFound { .. } => {
                Some("Check the file path and ensure the file exists".to_string())
            }
            Self::PermissionDenied { .. } => {
                Some("Check file permissions or run with elevated privileges".to_string())
            }
            Self::MemoryError { .. } => {
                Some("Consider reducing memory usage or increasing available memory".to_string())
            }
            _ => None,
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
        }
    }
}
