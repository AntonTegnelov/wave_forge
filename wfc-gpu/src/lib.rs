use thiserror::Error;

pub mod accelerator;
pub mod buffers;
pub mod pipeline;
pub mod shaders; // Module to potentially help load shaders

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("WGPU initialization failed: {0}")]
    InitializationError(String),
    #[error("Failed to create WGPU buffer: {0}")]
    BufferCreationError(String),
    #[error("Failed to create WGPU pipeline: {0}")]
    PipelineCreationError(String),
    #[error("Failed to execute WGPU command: {0}")]
    CommandExecutionError(String),
    #[error("Data transfer error: {0}")]
    TransferError(String),
    #[error("Shader error: {0}")]
    ShaderError(String),
    #[error("WGPU validation error: {0}")]
    ValidationError(#[from] wgpu::Error), // Catch wgpu validation errors
    #[error("Unknown GPU error")]
    Unknown,
}
