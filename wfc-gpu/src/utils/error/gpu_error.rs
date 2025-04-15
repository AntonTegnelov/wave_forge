//! GPU-specific error types and handling logic.
//!
//! This module defines specialized error types for GPU operations,
//! with detailed context information to aid in debugging and recovery.

use crate::utils::error::{ErrorLocation, ErrorSeverity, ErrorWithContext};
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

/// Debug information about GPU state
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct GpuStateInfo {
    /// Available GPU memory at the time of error (if available)
    pub available_memory: Option<u64>,
    /// Total GPU memory (if available)
    pub total_memory: Option<u64>,
    /// Current GPU workload/utilization (if available)
    pub gpu_utilization: Option<f32>,
    /// Number of active command buffers
    pub active_commands: Option<usize>,
    /// Number of allocated buffers
    pub buffer_count: Option<usize>,
    /// Debug markers active at time of error
    pub active_debug_markers: Option<Vec<String>>,
    /// GPU adapter information
    pub adapter_info: Option<String>,
    /// GPU features enabled
    pub enabled_features: Option<Vec<String>>,
    /// Current frame number
    pub frame_number: Option<u64>,
}


impl GpuStateInfo {
    /// Create a new GPU state info
    pub fn new() -> Self {
        Self::default()
    }

    /// Add memory information to the state
    pub fn with_memory(mut self, available: u64, total: u64) -> Self {
        self.available_memory = Some(available);
        self.total_memory = Some(total);
        self
    }

    /// Add GPU utilization to the state
    pub fn with_utilization(mut self, utilization: f32) -> Self {
        self.gpu_utilization = Some(utilization);
        self
    }

    /// Add active command count to the state
    pub fn with_active_commands(mut self, count: usize) -> Self {
        self.active_commands = Some(count);
        self
    }

    /// Add buffer count to the state
    pub fn with_buffer_count(mut self, count: usize) -> Self {
        self.buffer_count = Some(count);
        self
    }

    /// Add active debug markers to the state
    pub fn with_debug_markers(mut self, markers: Vec<String>) -> Self {
        self.active_debug_markers = Some(markers);
        self
    }

    /// Add adapter info to the state
    pub fn with_adapter_info<S: Into<String>>(mut self, info: S) -> Self {
        self.adapter_info = Some(info.into());
        self
    }

    /// Add enabled features to the state
    pub fn with_enabled_features(mut self, features: Vec<String>) -> Self {
        self.enabled_features = Some(features);
        self
    }

    /// Add frame number to the state
    pub fn with_frame_number(mut self, frame: u64) -> Self {
        self.frame_number = Some(frame);
        self
    }
}

impl fmt::Display for GpuStateInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;

        if let (Some(available), Some(total)) = (self.available_memory, self.total_memory) {
            write!(f, "Memory: {}/{} bytes", available, total)?;
            first = false;
        }

        if let Some(utilization) = self.gpu_utilization {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "GPU Utilization: {:.1}%", utilization * 100.0)?;
            first = false;
        }

        if let Some(cmds) = self.active_commands {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "Active Commands: {}", cmds)?;
            first = false;
        }

        if let Some(buffers) = self.buffer_count {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "Allocated Buffers: {}", buffers)?;
            first = false;
        }

        if let Some(frame) = self.frame_number {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "Frame: {}", frame)?;
        }

        if let Some(ref markers) = self.active_debug_markers {
            write!(f, "\nActive Debug Markers: ")?;
            for (i, marker) in markers.iter().enumerate() {
                if i > 0 {
                    write!(f, " -> ")?;
                }
                write!(f, "{}", marker)?;
            }
        }

        if let Some(ref adapter) = self.adapter_info {
            write!(f, "\nAdapter: {}", adapter)?;
        }

        if let Some(ref features) = self.enabled_features {
            write!(f, "\nEnabled Features: ")?;
            for (i, feature) in features.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", feature)?;
            }
        }

        Ok(())
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
    /// GPU state information at the time of error
    pub gpu_state: Option<GpuStateInfo>,
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
            gpu_state: None,
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

    /// Add GPU state information to the context
    pub fn with_gpu_state(mut self, state: GpuStateInfo) -> Self {
        self.gpu_state = Some(state);
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

        if let Some(state) = &self.gpu_state {
            write!(f, "\nGPU State: {}", state)?;
        }

        if let Some(solution) = &self.suggested_solution {
            write!(f, "\nSuggested solution: {}", solution)?;
        }

        Ok(())
    }
}

/// Enumeration of all possible GPU errors
#[derive(Error, Debug, Clone)]
pub enum GpuError {
    #[error("Buffer mapping failed: {msg}")]
    BufferMapFailed {
        msg: String,
        context: Box<GpuErrorContext>,
    },

    #[error("Buffer mapping timed out: {0}")]
    BufferMapTimeout(String, Box<GpuErrorContext>),

    #[error("Validation error: {msg}")]
    ValidationError {
        msg: String,
        context: Box<GpuErrorContext>,
    },

    #[error("Command execution error: {msg}")]
    CommandExecutionError {
        msg: String,
        context: Box<GpuErrorContext>,
    },

    #[error("Buffer operation error: {msg}")]
    BufferOperationError {
        msg: String,
        context: Box<GpuErrorContext>,
    },

    #[error("Data transfer error: {msg}")]
    TransferError {
        msg: String,
        context: Box<GpuErrorContext>,
    },

    #[error("GPU resource creation failed: {msg}")]
    ResourceCreationFailed {
        msg: String,
        context: Box<GpuErrorContext>,
    },

    #[error("Shader compilation failed: {msg}")]
    ShaderError {
        msg: String,
        context: Box<GpuErrorContext>,
    },

    #[error("Buffer size mismatch: {msg}")]
    BufferSizeMismatch {
        msg: String,
        context: Box<GpuErrorContext>,
    },

    #[error("Timeout occurred: {msg}")]
    Timeout {
        msg: String,
        context: Box<GpuErrorContext>,
    },

    #[error("Device lost: {msg}")]
    DeviceLost {
        msg: String,
        context: Box<GpuErrorContext>,
    },

    #[error("GPU Adapter request failed")]
    AdapterRequestFailed { context: Box<GpuErrorContext> },

    #[error("GPU Device request failed: {0}")]
    DeviceRequestFailed(wgpu::RequestDeviceError, Box<GpuErrorContext>),

    #[error("Mutex lock error: {msg}")]
    MutexError {
        msg: String,
        context: Box<GpuErrorContext>,
    },

    #[error("Contradiction detected")]
    ContradictionDetected { context: Box<GpuErrorContext> },

    #[error("Other GPU error: {msg}")]
    Other {
        msg: String,
        context: Box<GpuErrorContext>,
    },
}

impl GpuError {
    /// Create a new buffer map failed error
    pub fn buffer_map_failed<S: Into<String>>(msg: S, context: GpuErrorContext) -> Self {
        Self::BufferMapFailed {
            msg: msg.into(),
            context: Box::new(context),
        }
    }

    /// Create a new buffer map timeout error
    pub fn buffer_map_timeout<S: Into<String>>(label: S, context: GpuErrorContext) -> Self {
        Self::BufferMapTimeout(label.into(), Box::new(context))
    }

    /// Create a new validation error
    pub fn validation_error(err: wgpu::Error, context: GpuErrorContext) -> Self {
        Self::ValidationError {
            msg: err.to_string(),
            context: Box::new(context),
        }
    }

    /// Create a new command execution error
    pub fn command_execution_error<S: Into<String>>(msg: S, context: GpuErrorContext) -> Self {
        Self::CommandExecutionError {
            msg: msg.into(),
            context: Box::new(context),
        }
    }

    /// Create a new buffer operation error
    pub fn buffer_operation_error<S: Into<String>>(msg: S, context: GpuErrorContext) -> Self {
        Self::BufferOperationError {
            msg: msg.into(),
            context: Box::new(context),
        }
    }

    /// Create a new transfer error
    pub fn transfer_error<S: Into<String>>(msg: S, context: GpuErrorContext) -> Self {
        Self::TransferError {
            msg: msg.into(),
            context: Box::new(context),
        }
    }

    /// Create a new resource creation failed error
    pub fn resource_creation_failed<S: Into<String>>(msg: S, context: GpuErrorContext) -> Self {
        Self::ResourceCreationFailed {
            msg: msg.into(),
            context: Box::new(context),
        }
    }

    /// Create a new shader error
    pub fn shader_error<S: Into<String>>(msg: S, context: GpuErrorContext) -> Self {
        Self::ShaderError {
            msg: msg.into(),
            context: Box::new(context),
        }
    }

    /// Create a new buffer size mismatch error
    pub fn buffer_size_mismatch<S: Into<String>>(msg: S, context: GpuErrorContext) -> Self {
        Self::BufferSizeMismatch {
            msg: msg.into(),
            context: Box::new(context),
        }
    }

    /// Create a new timeout error
    pub fn timeout<S: Into<String>>(msg: S, context: GpuErrorContext) -> Self {
        Self::Timeout {
            msg: msg.into(),
            context: Box::new(context),
        }
    }

    /// Create a new device lost error
    pub fn device_lost<S: Into<String>>(msg: S, context: GpuErrorContext) -> Self {
        Self::DeviceLost {
            msg: msg.into(),
            context: Box::new(context),
        }
    }

    /// Create a new adapter request failed error
    pub fn adapter_request_failed(context: GpuErrorContext) -> Self {
        Self::AdapterRequestFailed {
            context: Box::new(context),
        }
    }

    /// Create a new device request failed error
    pub fn device_request_failed(err: wgpu::RequestDeviceError, context: GpuErrorContext) -> Self {
        Self::DeviceRequestFailed(err, Box::new(context))
    }

    /// Create a new mutex error
    pub fn mutex_error<S: Into<String>>(msg: S, context: GpuErrorContext) -> Self {
        Self::MutexError {
            msg: msg.into(),
            context: Box::new(context),
        }
    }

    /// Create a new contradiction detected error
    pub fn contradiction_detected(context: GpuErrorContext) -> Self {
        Self::ContradictionDetected {
            context: Box::new(context),
        }
    }

    /// Create a new other error
    pub fn other<S: Into<String>>(msg: S, context: GpuErrorContext) -> Self {
        Self::Other {
            msg: msg.into(),
            context: Box::new(context),
        }
    }

    /// Get the context information for this error
    pub fn get_context(&self) -> &GpuErrorContext {
        match self {
            Self::BufferMapFailed { context, .. } => context,
            Self::BufferMapTimeout(_, context) => context,
            Self::ValidationError { context, .. } => context,
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
            Self::BufferMapTimeout(label, context) => format!("{}: {}", context, label),
            Self::ValidationError { msg, context } => format!("{}: {}", context, msg),
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
                Some(concat!(
                    "Buffer mapping failed. Try the following:\n",
                    "1. Check buffer usage flags - ensure COPY_SRC/COPY_DST are set for mapped buffers\n",
                    "2. Ensure buffer is not already mapped or in use by another operation\n",
                    "3. Check that the buffer size is sufficient for the mapping range\n",
                    "4. Verify that the GPU device hasn't been lost during the operation\n",
                    "5. Consider using a staging buffer with GPU-only buffers"
                ).to_string())
            }
            Self::BufferMapTimeout(_, _) => {
                Some(concat!(
                    "Buffer mapping timed out. Try the following:\n",
                    "1. Increase the timeout duration for buffer mapping operations\n",
                    "2. Check for GPU driver issues or high system load\n",
                    "3. Ensure the GPU is not being used by other intensive applications\n",
                    "4. Consider breaking operations into smaller chunks\n",
                    "5. Verify that the GPU is not in a sleep state or power-saving mode"
                ).to_string())
            }
            Self::ValidationError { msg, .. } => {
                let basic_msg = "Validation error in GPU operation. Try the following:\n\
                1. Check resource usage flags and ensure they match the operation\n\
                2. Verify bind group layouts match pipeline expectations\n\
                3. Ensure buffer sizes are sufficient for the operation";
                
                // Add more specific advice based on the error message
                let specific_advice = if msg.contains("out of memory") {
                    "\n4. Reduce resource usage or buffer sizes\n\
                     5. Consider batching operations into smaller chunks"
                } else if msg.contains("bind group") {
                    "\n4. Check bind group entries match the layout\n\
                     5. Ensure all required bindings are provided"
                } else if msg.contains("buffer") {
                    "\n4. Verify buffer alignment requirements\n\
                     5. Check buffer usage flags"
                } else if msg.contains("shader") {
                    "\n4. Review shader code for validation errors\n\
                     5. Ensure shader inputs match pipeline configuration"
                } else {
                    ""
                };
                
                Some(format!("{}{}", basic_msg, specific_advice))
            }
            Self::CommandExecutionError { .. } => {
                Some(concat!(
                    "Command execution failed. Try the following:\n",
                    "1. Check that resources used in the command are still valid\n",
                    "2. Ensure resources are in the correct state for the operation\n",
                    "3. Verify that buffer sizes and offsets are valid\n",
                    "4. Consider adding debug markers to identify problematic commands\n",
                    "5. Inspect device logs for further validation errors"
                ).to_string())
            }
            Self::BufferOperationError { .. } => {
                Some(concat!(
                    "Buffer operation failed. Try the following:\n",
                    "1. Check buffer usage flags match the intended operation\n",
                    "2. Verify that buffer size is sufficient for the operation\n",
                    "3. Ensure buffer alignments are correct for the operation\n",
                    "4. Check that the buffer isn't being used in conflicting operations\n",
                    "5. Consider using larger buffers or chunking data into multiple operations"
                ).to_string())
            }
            Self::TransferError { .. } => {
                Some(concat!(
                    "Data transfer error. Try the following:\n",
                    "1. Check that both source and destination buffers exist and are valid\n",
                    "2. Ensure buffers have appropriate usage flags (COPY_SRC and COPY_DST)\n",
                    "3. Verify that transfer sizes and alignments are correct\n",
                    "4. Check for potential race conditions in async operations\n",
                    "5. Consider using mapped memory or staging buffers for complex transfers"
                ).to_string())
            }
            Self::ResourceCreationFailed { .. } => {
                Some(concat!(
                    "GPU resource creation failed. Try the following:\n",
                    "1. Check resource creation parameters (size, format, usage flags)\n",
                    "2. Verify that the GPU device is still valid\n",
                    "3. Check system resources (available memory)\n",
                    "4. Consider reducing resource sizes or counts\n",
                    "5. Verify that required features are supported by the GPU"
                ).to_string())
            }
            Self::ShaderError { .. } => {
                Some(concat!(
                    "Shader compilation or execution failed. Try the following:\n",
                    "1. Check shader code for syntax errors or unsupported features\n",
                    "2. Verify that required GPU features are enabled\n",
                    "3. Check for uniform buffer or texture binding mismatches\n",
                    "4. Consider using simpler shader code or breaking into multiple passes\n",
                    "5. Check for resource conflicts or invalid bind groups"
                ).to_string())
            }
            Self::BufferSizeMismatch { .. } => {
                Some(concat!(
                    "Buffer size mismatch. Try the following:\n",
                    "1. Ensure the buffer was created with sufficient size\n",
                    "2. Check data structures for size calculations\n",
                    "3. Verify that dynamic buffer resizing is working correctly\n",
                    "4. Consider padding buffers to avoid alignment issues\n",
                    "5. Add size validation checks before buffer operations"
                ).to_string())
            }
            Self::Timeout { .. } => Some(concat!(
                "Operation timed out. Try the following:\n",
                "1. Increase timeout duration for complex operations\n",
                "2. Optimize algorithm to reduce processing time\n",
                "3. Consider breaking work into smaller chunks\n",
                "4. Check for infinite loops or deadlocks in shader code\n",
                "5. Monitor GPU utilization to detect bottlenecks"
            ).to_string()),
            Self::DeviceLost { .. } => {
                Some(concat!(
                    "GPU device was lost. Try the following:\n",
                    "1. Reinitialize the GPU device and recreate resources\n",
                    "2. Check for driver crashes or system resource issues\n",
                    "3. Reduce GPU workload if overheating may be an issue\n",
                    "4. Update GPU drivers to the latest version\n",
                    "5. Consider implementing automatic device recovery"
                ).to_string())
            }
            Self::AdapterRequestFailed { .. } => {
                Some(concat!(
                    "Failed to request GPU adapter. Try the following:\n",
                    "1. Check that a compatible GPU is available in the system\n",
                    "2. Verify GPU driver installation and update if necessary\n",
                    "3. Reduce GPU feature requirements to improve compatibility\n",
                    "4. Consider fallback to software rendering if available\n",
                    "5. Check system logs for GPU detection issues"
                ).to_string())
            }
            Self::DeviceRequestFailed(_, _) => {
                Some(concat!(
                    "Failed to request GPU device. Try the following:\n",
                    "1. Check that the requested features are supported by the GPU\n",
                    "2. Reduce limits or feature requirements\n",
                    "3. Update GPU drivers to the latest version\n",
                    "4. Verify that the GPU is not disabled or in error state\n",
                    "5. Consider fallback to a different adapter or GPU"
                ).to_string())
            }
            Self::MutexError { .. } => {
                Some(concat!(
                    "Mutex lock error. Try the following:\n",
                    "1. Check for potential deadlocks in multi-threaded code\n",
                    "2. Ensure locks are released properly in all code paths\n",
                    "3. Consider using read-write locks for better concurrency\n",
                    "4. Reduce lock contention by minimizing critical sections\n",
                    "5. Add timeout to lock acquisition to detect deadlocks"
                ).to_string())
            }
            Self::ContradictionDetected { .. } => {
                Some(concat!(
                    "Algorithm reached contradiction. Try the following:\n",
                    "1. Check input data for consistency with constraints\n",
                    "2. Verify adjacency rules for potential conflicts\n",
                    "3. Consider relaxing constraints if they're too restrictive\n",
                    "4. Use backtracking or alternative cell selection strategy\n",
                    "5. Add debug visualization to identify problematic areas"
                ).to_string())
            }
            Self::Other { .. } => {
                Some(concat!(
                    "Unknown GPU error occurred. Try the following:\n",
                    "1. Check for system resource limitations (memory, etc.)\n",
                    "2. Verify GPU driver status and update if necessary\n",
                    "3. Restart the application or reinitialize GPU resources\n",
                    "4. Check logs for additional error details\n",
                    "5. Consider simplifying GPU workload"
                ).to_string())
            }
        }
    }

    fn severity(&self) -> ErrorSeverity {
        match self {
            Self::ValidationError { .. } => ErrorSeverity::Fatal,
            Self::AdapterRequestFailed { .. } => ErrorSeverity::Fatal,
            Self::DeviceRequestFailed(_, _) => ErrorSeverity::Fatal,
            Self::DeviceLost { .. } => ErrorSeverity::Fatal, // Can be Recoverable with proper reinit
            Self::ShaderError { .. } => ErrorSeverity::Fatal,
            Self::ContradictionDetected { .. } => ErrorSeverity::Recoverable, // WFC contradictions are expected
            Self::BufferMapFailed { .. } => ErrorSeverity::Recoverable,
            Self::BufferMapTimeout(_, _) => ErrorSeverity::Recoverable,
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

impl From<crate::utils::error_recovery::GpuError> for GpuError {
    fn from(err: crate::utils::error_recovery::GpuError) -> Self {
        use crate::utils::error_recovery::GpuError as RecoveryGpuError;
        
        match err {
            RecoveryGpuError::MemoryAllocation(msg) => Self::BufferOperationError {
                msg,
                context: Box::new(GpuErrorContext::new(GpuResourceType::Buffer)
                    .with_details("Memory allocation failed")),
            },
            RecoveryGpuError::ComputationTimeout { grid_size, duration } => Self::Timeout {
                msg: format!("Computation timeout for grid {}x{} after {:?}", grid_size.0, grid_size.1, duration),
                context: Box::new(GpuErrorContext::new(GpuResourceType::Other)
                    .with_details("GPU computation timed out")),
            },
            RecoveryGpuError::KernelExecution(msg) => Self::CommandExecutionError {
                msg,
                context: Box::new(GpuErrorContext::new(GpuResourceType::Other)
                    .with_details("Kernel execution error")),
            },
            RecoveryGpuError::QueueSubmission(msg) => Self::CommandExecutionError {
                msg,
                context: Box::new(GpuErrorContext::new(GpuResourceType::Queue)
                    .with_details("Queue submission error")),
            },
            RecoveryGpuError::DeviceLost(msg) => Self::DeviceLost {
                msg,
                context: Box::new(GpuErrorContext::new(GpuResourceType::Device)
                    .with_details("Device lost")),
            },
            RecoveryGpuError::InvalidState(msg) => Self::ValidationError {
                msg,
                context: Box::new(GpuErrorContext::new(GpuResourceType::Other)
                    .with_details("Invalid state")),
            },
            RecoveryGpuError::BarrierSynchronization(msg) => Self::CommandExecutionError {
                msg,
                context: Box::new(GpuErrorContext::new(GpuResourceType::Other)
                    .with_details("Barrier synchronization error")),
            },
            RecoveryGpuError::BufferCopy(msg) => Self::TransferError {
                msg,
                context: Box::new(GpuErrorContext::new(GpuResourceType::Buffer)
                    .with_details("Buffer copy error")),
            },
            RecoveryGpuError::BufferMapping(msg) => Self::BufferMapFailed {
                msg,
                context: Box::new(GpuErrorContext::new(GpuResourceType::Buffer)
                    .with_details("Buffer mapping error")),
            },
            RecoveryGpuError::ContradictionDetected { context } => Self::ContradictionDetected {
                context: Box::new(GpuErrorContext::new(GpuResourceType::Other)
                    .with_details(format!("Contradiction detected: {}", context))),
            },
            RecoveryGpuError::Other(msg) => Self::Other {
                msg,
                context: Box::new(GpuErrorContext::new(GpuResourceType::Other)),
            },
        }
    }
}
