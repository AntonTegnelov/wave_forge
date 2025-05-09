// GPU Module - Handles all GPU-related functionality for Wave Function Collapse

// Internal modules - now public so they can be accessed from other modules
pub mod accelerator;
pub mod backend;
pub mod features;
pub mod sync;

// Re-exports for use outside the gpu module
pub use accelerator::{GpuAccelerator, GridDefinition, GridStats, WfcRunResult};
pub use backend::{
    BackendError, ComputeCapable, DataTransfer, GpuBackend, GpuBackendFactory, Synchronization,
    WgpuBackend,
};
pub use features::{AtomicsSupport, GpuCapabilities, GpuFeature, WorkgroupSupport};
pub use sync::GpuSynchronizer;
