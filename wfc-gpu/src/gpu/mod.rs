// GPU Module - Handles all GPU-related functionality for Wave Function Collapse

// Internal modules
mod accelerator;
mod backend;
mod sync;

// Re-exports for use outside the gpu module
pub use accelerator::{GpuAccelerator, GridDefinition, GridStats, WfcRunResult};
pub use backend::{
    BackendError, ComputeCapable, DataTransfer, GpuBackend, GpuBackendFactory, Synchronization,
    WgpuBackend,
};
pub use sync::GpuSynchronizer;
