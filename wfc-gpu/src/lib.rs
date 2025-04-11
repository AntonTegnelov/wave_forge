//! Provides GPU acceleration for the WFC algorithm using WGPU compute shaders.

#![allow(clippy::derive_partial_eq_without_eq)]
// Removed conflicting use statement
// use crate::backend::BackendError;
// Removed unused import
// use thiserror::Error;

// --- Private/Internal Modules ---
// These are implementation details not part of the public API unless re-exported.
mod backend;
mod error_recovery;
mod pipeline;
mod shader_compiler;
mod shader_registry;
mod shaders;

// --- Public Modules ---
// These form the public API surface of the crate.
pub mod accelerator;
pub mod algorithm;
pub mod buffers;
pub mod coordination;
pub mod debug_viz;
pub mod entropy;
pub mod propagator;
pub mod subgrid;
pub mod sync;

// --- Public Re-exports --- //
// Re-export key types for easier access by users of the crate.

// Core accelerator type
pub use accelerator::GpuAccelerator;

// Algorithm strategy types
pub use algorithm::entropy_strategy::{EntropyStrategy, EntropyStrategyFactory};
pub use algorithm::propagator_strategy::{PropagationStrategy, PropagationStrategyFactory};

// Buffer related types
pub use buffers::{DownloadRequest, DynamicBufferConfig, GpuBuffers, GpuDownloadResults};

// Configuration types
pub use coordination::WfcCoordinator;
pub use debug_viz::{DebugVisualizationConfig, DebugVisualizer, VisualizationType};
pub use subgrid::SubgridConfig; // Coordination API

// Add re-exports for propagation coordination
pub use coordination::propagation::{
    DirectPropagationCoordinator, PropagationCoordinator, SubgridPropagationCoordinator,
};

// Error type
pub use error_recovery::GpuError;

// Re-export types from dependencies if they are part of the public API
// (e.g., from wfc_core if needed for function signatures)
// pub use wfc_core::{BoundaryCondition, PossibilityGrid, /* etc. */ };

// --- Conditional Compilation for Tests --- //
// Declare test modules, only compiled when running tests.
#[cfg(test)]
mod tests;

pub use pipeline::ComputePipelines;
pub use propagator::GpuConstraintPropagator;
pub use shader_registry::ShaderRegistry;
pub use sync::GpuSynchronizer;
