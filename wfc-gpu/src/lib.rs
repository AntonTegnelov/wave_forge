use thiserror::Error;

pub mod accelerator;
pub mod buffers;
pub mod pipeline;
pub mod shaders; // Module to potentially help load shaders

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("Failed to request WGPU adapter")]
    AdapterRequestFailed,
    #[error("Failed to request WGPU device: {0}")]
    DeviceRequestFailed(#[from] wgpu::RequestDeviceError),
    #[error("Failed to create WGPU buffer: {0}")]
    BufferCreationError(String),
    #[error("GPU buffer operation error: {0}")]
    BufferOperationError(String),
    #[error("Failed to create WGPU pipeline: {0}")]
    PipelineCreationError(String),
    #[error("Failed to execute WGPU command: {0}")]
    CommandExecutionError(String),
    #[error("Data transfer error: {0}")]
    TransferError(String),
    #[error("Shader error: {0}")]
    ShaderError(String),
    #[error("WGPU validation error: {0}")]
    ValidationError(wgpu::Error),
    #[error("Unknown GPU error")]
    Unknown,
    #[error("Failed to map GPU buffer: {0}")]
    BufferMapFailed(#[from] wgpu::BufferAsyncError),
    #[error("GPU operation failed: {0}")]
    Other(String),
}
