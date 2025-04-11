//! GPU-specific error types and handling logic.
//!
//! This module defines specialized error types for GPU operations,
//! with detailed context information to aid in debugging and recovery.

use super::{ErrorLocation, ErrorSeverity, ErrorWithContext};
use crate::utils::error_recovery::GridCoord;
use std::fmt;
use thiserror::Error;

/// Types of GPU resources that can be involved in errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuResourceType {
    Buffer,
    Texture,
    Shader,
    Pipeline,
    Adapter,
    Device,
    Queue,
    Sampler,
    CommandEncoder,
    BindGroup,
    PipelineLayout,
    BindGroupLayout,
    Other,
}

impl fmt::Display for GpuResourceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuResourceType::Buffer => write!(f, "Buffer"),
            GpuResourceType::Texture => write!(f, "Texture"),
            GpuResourceType::Shader => write!(f, "Shader"),
            GpuResourceType::Pipeline => write!(f, "Pipeline"),
            GpuResourceType::Adapter => write!(f, "Adapter"),
            GpuResourceType::Device => write!(f, "Device"),
            GpuResourceType::Queue => write!(f, "Queue"),
            GpuResourceType::Sampler => write!(f, "Sampler"),
            GpuResourceType::CommandEncoder => write!(f, "Command Encoder"),
            GpuResourceType::BindGroup => write!(f, "Bind Group"),
            GpuResourceType::PipelineLayout => write!(f, "Pipeline Layout"),
            GpuResourceType::BindGroupLayout => write!(f, "Bind Group Layout"),
            GpuResourceType::Other => write!(f, "Other Resource"),
        }
    }
}

/// Contains contextual information about a GPU error
#[derive(Debug, Clone)]
pub struct GpuErrorContext {
    /// Type of GPU resource involved
    pub resource_type: GpuResourceType,
    /// Optional name or label of the resource
    pub resource_label: Option<String>,
    /// Optional grid coordinate where the error occurred
    pub grid_coord: Option<GridCoord>,
    /// Additional details about the error
    pub details: Option<String>,
    /// Source location information
    pub location: Option<ErrorLocation>,
    /// Suggested solution or recovery strategy
    pub suggested_solution: Option<String>,
}

impl Default for GpuErrorContext {
    fn default() -> Self {
        Self {
            resource_type: GpuResourceType::Other,
            resource_label: None,
            grid_coord: None,
            details: None,
            location: None,
            suggested_solution: None,
        }
    }
}

impl GpuErrorContext {
    /// Create a new GPU error context
    pub fn new(resource_type: GpuResourceType) -> Self {
        Self {
            resource_type,
            ..Default::default()
        }
    }

    /// Add a resource label to the context
    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self {
        self.resource_label = Some(label.into());
        self
    }

    /// Add a grid coordinate to the context
    pub fn with_grid_coord(mut self, coord: GridCoord) -> Self {
        self.grid_coord = Some(coord);
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

impl fmt::Display for GpuErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GPU {} error",
            match &self.resource_label {
                Some(label) => format!("{} '{}'", self.resource_type, label),
                None => format!("{}", self.resource_type),
            }
        )?;

        if let Some(coord) = &self.grid_coord {
            write!(f, " at ({}, {}, {})", coord.x, coord.y, coord.z)?;
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

/// Enumeration of all possible GPU errors
#[derive(Error, Debug)]
pub enum GpuError {
    #[error("Buffer mapping failed: {msg}")]
    BufferMapFailed {
        msg: String,
        context: GpuErrorContext,
    },

    #[error("Validation error: {0}")]
    ValidationError(wgpu::Error, GpuErrorContext),

    #[error("Command execution error: {msg}")]
    CommandExecutionError {
        msg: String,
        context: GpuErrorContext,
    },

    #[error("Buffer operation error: {msg}")]
    BufferOperationError {
        msg: String,
        context: GpuErrorContext,
    },

    #[error("Data transfer error: {msg}")]
    TransferError {
        msg: String,
        context: GpuErrorContext,
    },

    #[error("GPU resource creation failed: {msg}")]
    ResourceCreationFailed {
        msg: String,
        context: GpuErrorContext,
    },

    #[error("Shader compilation failed: {msg}")]
    ShaderError {
        msg: String,
        context: GpuErrorContext,
    },

    #[error("Buffer size mismatch: {msg}")]
    BufferSizeMismatch {
        msg: String,
        context: GpuErrorContext,
    },

    #[error("Timeout occurred: {msg}")]
    Timeout {
        msg: String,
        context: GpuErrorContext,
    },

    #[error("Device lost: {msg}")]
    DeviceLost {
        msg: String,
        context: GpuErrorContext,
    },

    #[error("GPU Adapter request failed")]
    AdapterRequestFailed { context: GpuErrorContext },

    #[error("GPU Device request failed: {0}")]
    DeviceRequestFailed(wgpu::RequestDeviceError, GpuErrorContext),

    #[error("Mutex lock error: {msg}")]
    MutexError {
        msg: String,
        context: GpuErrorContext,
    },

    #[error("Contradiction detected")]
    ContradictionDetected { context: GpuErrorContext },

    #[error("Other GPU error: {msg}")]
    Other {
        msg: String,
        context: GpuErrorContext,
    },
}

impl GpuError {
    /// Create a new buffer map failed error
    pub fn buffer_map_failed<S: Into<String>>(msg: S, context: GpuErrorContext) -> Self {
        Self::BufferMapFailed {
            msg: msg.into(),
            context,
        }
    }

    /// Create a new validation error
    pub fn validation_error(err: wgpu::Error, context: GpuErrorContext) -> Self {
        Self::ValidationError(err, context)
    }

    /// Create a new command execution error
    pub fn command_execution_error<S: Into<String>>(msg: S, context: GpuErrorContext) -> Self {
        Self::CommandExecutionError {
            msg: msg.into(),
            context,
        }
    }

    /// Create a new buffer operation error
    pub fn buffer_operation_error<S: Into<String>>(msg: S, context: GpuErrorContext) -> Self {
        Self::BufferOperationError {
            msg: msg.into(),
            context,
        }
    }

    /// Create a new transfer error
    pub fn transfer_error<S: Into<String>>(msg: S, context: GpuErrorContext) -> Self {
        Self::TransferError {
            msg: msg.into(),
            context,
        }
    }

    /// Create a new resource creation failed error
    pub fn resource_creation_failed<S: Into<String>>(msg: S, context: GpuErrorContext) -> Self {
        Self::ResourceCreationFailed {
            msg: msg.into(),
            context,
        }
    }

    /// Create a new shader error
    pub fn shader_error<S: Into<String>>(msg: S, context: GpuErrorContext) -> Self {
        Self::ShaderError {
            msg: msg.into(),
            context,
        }
    }

    /// Create a new buffer size mismatch error
    pub fn buffer_size_mismatch<S: Into<String>>(msg: S, context: GpuErrorContext) -> Self {
        Self::BufferSizeMismatch {
            msg: msg.into(),
            context,
        }
    }

    /// Create a new timeout error
    pub fn timeout<S: Into<String>>(msg: S, context: GpuErrorContext) -> Self {
        Self::Timeout {
            msg: msg.into(),
            context,
        }
    }

    /// Create a new device lost error
    pub fn device_lost<S: Into<String>>(msg: S, context: GpuErrorContext) -> Self {
        Self::DeviceLost {
            msg: msg.into(),
            context,
        }
    }

    /// Create a new adapter request failed error
    pub fn adapter_request_failed(context: GpuErrorContext) -> Self {
        Self::AdapterRequestFailed { context }
    }

    /// Create a new device request failed error
    pub fn device_request_failed(err: wgpu::RequestDeviceError, context: GpuErrorContext) -> Self {
        Self::DeviceRequestFailed(err, context)
    }

    /// Create a new mutex error
    pub fn mutex_error<S: Into<String>>(msg: S, context: GpuErrorContext) -> Self {
        Self::MutexError {
            msg: msg.into(),
            context,
        }
    }

    /// Create a new contradiction detected error
    pub fn contradiction_detected(context: GpuErrorContext) -> Self {
        Self::ContradictionDetected { context }
    }

    /// Create a new other error
    pub fn other<S: Into<String>>(msg: S, context: GpuErrorContext) -> Self {
        Self::Other {
            msg: msg.into(),
            context,
        }
    }

    /// Get the context information for this error
    pub fn get_context(&self) -> &GpuErrorContext {
        match self {
            Self::BufferMapFailed { context, .. } => context,
            Self::ValidationError(_, context) => context,
            Self::CommandExecutionError { context, .. } => context,
            Self::BufferOperationError { context, .. } => context,
            Self::TransferError { context, .. } => context,
            Self::ResourceCreationFailed { context, .. } => context,
            Self::ShaderError { context, .. } => context,
            Self::BufferSizeMismatch { context, .. } => context,
            Self::Timeout { context, .. } => context,
            Self::DeviceLost { context, .. } => context,
            Self::AdapterRequestFailed { context } => context,
            Self::DeviceRequestFailed(_, context) => context,
            Self::MutexError { context, .. } => context,
            Self::ContradictionDetected { context } => context,
            Self::Other { context, .. } => context,
        }
    }
}

impl ErrorWithContext for GpuError {
    fn context(&self) -> String {
        match self {
            Self::BufferMapFailed { msg, context } => format!("{}: {}", context, msg),
            Self::ValidationError(err, context) => format!("{}: {}", context, err),
            Self::CommandExecutionError { msg, context } => format!("{}: {}", context, msg),
            Self::BufferOperationError { msg, context } => format!("{}: {}", context, msg),
            Self::TransferError { msg, context } => format!("{}: {}", context, msg),
            Self::ResourceCreationFailed { msg, context } => format!("{}: {}", context, msg),
            Self::ShaderError { msg, context } => format!("{}: {}", context, msg),
            Self::BufferSizeMismatch { msg, context } => format!("{}: {}", context, msg),
            Self::Timeout { msg, context } => format!("{}: {}", context, msg),
            Self::DeviceLost { msg, context } => format!("{}: {}", context, msg),
            Self::AdapterRequestFailed { context } => context.to_string(),
            Self::DeviceRequestFailed(err, context) => format!("{}: {}", context, err),
            Self::MutexError { msg, context } => format!("{}: {}", context, msg),
            Self::ContradictionDetected { context } => context.to_string(),
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
            Self::BufferMapFailed { .. } => {
                Some("Check buffer usage flags and ensure buffer is not already mapped".to_string())
            }
            Self::ValidationError(_, _) => {
                Some("Check GPU resource creation parameters and usage flags".to_string())
            }
            Self::DeviceLost { .. } => {
                Some("GPU device was lost. Reinitialize GPU resources and retry".to_string())
            }
            Self::Timeout { .. } => Some(
                "Operation timed out. Consider increasing timeout duration or optimizing algorithm"
                    .to_string(),
            ),
            Self::ContradictionDetected { .. } => {
                Some("Algorithm reached contradiction. Check input data or constraints".to_string())
            }
            _ => None,
        }
    }

    fn severity(&self) -> ErrorSeverity {
        match self {
            Self::ValidationError(_, _) => ErrorSeverity::Fatal,
            Self::AdapterRequestFailed { .. } => ErrorSeverity::Fatal,
            Self::DeviceRequestFailed(_, _) => ErrorSeverity::Fatal,
            Self::DeviceLost { .. } => ErrorSeverity::Fatal, // Can be Recoverable with proper reinit
            Self::ShaderError { .. } => ErrorSeverity::Fatal,
            Self::ContradictionDetected { .. } => ErrorSeverity::Recoverable, // WFC contradictions are expected
            Self::BufferMapFailed { .. } => ErrorSeverity::Recoverable,
            Self::Timeout { .. } => ErrorSeverity::Recoverable,
            Self::TransferError { .. } => ErrorSeverity::Recoverable,
            Self::BufferOperationError { .. } => ErrorSeverity::Recoverable,
            Self::MutexError { .. } => ErrorSeverity::Recoverable,
            Self::CommandExecutionError { .. } => ErrorSeverity::Recoverable,
            Self::BufferSizeMismatch { .. } => ErrorSeverity::Recoverable,
            Self::ResourceCreationFailed { .. } => ErrorSeverity::Recoverable,
            Self::Other { .. } => ErrorSeverity::Fatal,
        }
    }
}
