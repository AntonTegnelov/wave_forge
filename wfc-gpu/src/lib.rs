use thiserror::Error;

pub mod accelerator;
pub mod buffers;
pub mod pipeline;
pub mod shaders; // Module to potentially help load shaders

/// Errors related to GPU setup, buffer operations, shader compilation, and pipeline execution using WGPU.
#[derive(Error, Debug)]
pub enum GpuError {
    /// Failed to find a suitable WGPU adapter (physical GPU or backend).
    #[error("Failed to request WGPU adapter")]
    AdapterRequestFailed,
    /// Failed to get a logical WGPU device and queue from the adapter.
    #[error("Failed to request WGPU device: {0}")]
    DeviceRequestFailed(#[from] wgpu::RequestDeviceError),
    /// Failed to create a WGPU buffer (e.g., for storing grid data or rules).
    #[error("Failed to create WGPU buffer: {0}")]
    BufferCreationError(String),
    /// An error occurred during a buffer operation (e.g., reading, writing, mapping).
    #[error("GPU buffer operation error: {0}")]
    BufferOperationError(String),
    /// Failed to create a WGPU compute or render pipeline.
    #[error("Failed to create WGPU pipeline: {0}")]
    PipelineCreationError(String),
    /// Failed to submit or execute a WGPU command buffer.
    #[error("Failed to execute WGPU command: {0}")]
    CommandExecutionError(String),
    /// An error occurred during data transfer between CPU and GPU.
    #[error("Data transfer error: {0}")]
    TransferError(String),
    /// An error related to shader compilation or loading.
    #[error("Shader error: {0}")]
    ShaderError(String),
    /// A WGPU validation error occurred, often indicating incorrect API usage.
    #[error("WGPU validation error: {0}")]
    ValidationError(wgpu::Error),
    /// An unspecified or unknown GPU-related error.
    #[error("Unknown GPU error")]
    Unknown,
    /// Failed to map a GPU buffer for CPU access (e.g., reading results).
    #[error("Failed to map GPU buffer: {0}")]
    BufferMapFailed(#[from] wgpu::BufferAsyncError),
    /// A generic GPU operation error with a custom message.
    #[error("GPU operation failed: {0}")]
    Other(String),
}
