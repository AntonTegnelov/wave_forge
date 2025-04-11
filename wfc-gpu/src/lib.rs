//! Provides GPU acceleration for the WFC algorithm using WGPU compute shaders.

#![allow(clippy::derive_partial_eq_without_eq)]
// Removed conflicting use statement
// use crate::backend::BackendError;
// Removed unused import
// use thiserror::Error;

// --- Private/Internal Modules ---
// These are implementation details not part of the public API unless re-exported.
pub mod gpu;
pub mod shader; // New shader module that contains all shader-related functionality // New gpu module that contains all GPU-related functionality

// --- Public Modules ---
// These form the public API surface of the crate.
pub mod buffers;
pub mod coordination;
pub mod entropy;
pub mod propagator;
pub mod utils; // New utils module that contains debug_viz, error_recovery, and subgrid

// --- Public Re-exports --- //
// Re-export key types for easier access by users of the crate.

// Core accelerator type - now from gpu module
pub use gpu::GpuAccelerator;

// Algorithm strategy types
pub use entropy::{EntropyStrategy, EntropyStrategyFactory};
pub use propagator::propagator_strategy::{PropagationStrategy, PropagationStrategyFactory};

// Buffer related types
pub use buffers::{DownloadRequest, DynamicBufferConfig, GpuBuffers, GpuDownloadResults};

// Configuration types
pub use coordination::WfcCoordinator;
pub use utils::debug_viz::{DebugVisualizationConfig, DebugVisualizer, VisualizationType};
pub use utils::subgrid::SubgridConfig; // Coordination API

// Add re-exports for propagation coordination
pub use coordination::propagation::{
    DirectPropagationCoordinator, PropagationCoordinator, SubgridPropagationCoordinator,
};

// Error type
pub use utils::error_recovery::GpuError;

// Re-export types from dependencies if they are part of the public API
// (e.g., from wfc_core if needed for function signatures)
// pub use wfc_core::{BoundaryCondition, PossibilityGrid, /* etc. */ };

// --- Conditional Compilation for Tests --- //
// Declare test modules, only compiled when running tests.
#[cfg(test)]
mod tests;

// Re-exports from shader module
pub use shader::pipeline::ComputePipelines;
pub use shader::shader_registry::ShaderRegistry;
pub use shader::ShaderType;

// Re-exports from gpu module
pub use gpu::GpuSynchronizer;
pub use gpu::{BackendError, GpuBackend, WgpuBackend};

// Re-export propagator
pub use propagator::GpuConstraintPropagator;
