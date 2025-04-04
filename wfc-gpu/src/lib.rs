//! Provides GPU acceleration for the WFC algorithm using WGPU compute shaders.

use thiserror::Error;

/// Manages the WGPU context and implements the WFC traits using compute shaders.
pub mod accelerator;
/// Handles creation and management of WGPU buffers for grid state, rules, etc.
pub mod buffers;
/// GPU implementation of the EntropyCalculator trait.
pub mod entropy;
/// Error recovery mechanisms for non-fatal GPU errors.
pub mod error_recovery;
/// Logic for loading shaders and creating WGPU compute pipelines.
pub mod pipeline;
/// GPU implementation of the ConstraintPropagator trait.
pub mod propagator;
/// Contains WGSL shader code as string constants or loading utilities.
pub mod shaders; // Module to potentially help load shaders
/// Provides parallel subgrid processing for large grids.
pub mod subgrid;

// Add test_utils module, conditionally compiled for tests
pub mod test_utils;

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
    /// Generic internal error, often for logic errors or unexpected states.
    #[error("Internal GPU logic error: {0}")]
    InternalError(String),
    /// A generic GPU operation error with a custom message.
    #[error("GPU operation failed: {0}")]
    Other(String),
}

// Removed manual From<wgpu::RequestDeviceError> impl as it conflicts with derive macro
// impl From<wgpu::RequestDeviceError> for GpuError {
//     fn from(error: wgpu::RequestDeviceError) -> Self {
//         GpuError::DeviceRequestFailed(error)
//     }
// }

// Cannot easily implement From<bytemuck::PodCastError> as it's not pub
// impl From<bytemuck::PodCastError> for GpuError {
//     fn from(error: bytemuck::PodCastError) -> Self {
//         GpuError::BytemuckError(format!("Pod casting error: {}", error))
//     }
// }
