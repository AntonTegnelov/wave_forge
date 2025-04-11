//! Utility modules for the WFC-GPU implementation.
//!
//! This module contains utility components that support the main WFC-GPU algorithm:
//! - Debug visualization tools
//! - Error recovery mechanisms
//! - Subgrid processing utilities for large grids
//! - Error handling and context enrichment

// Re-export submodules
pub mod debug_viz;
pub mod error;
pub mod error_recovery;
pub mod subgrid;

// Re-export commonly used types
pub use debug_viz::{DebugSnapshot, DebugVisualizer, VisualizationType};
pub use error::{ErrorLocation, ErrorSeverity, ErrorWithContext, WfcError};
pub use error_recovery::{GpuError, RecoverableGpuOp};
pub use subgrid::{SubgridConfig, SubgridRegion};
