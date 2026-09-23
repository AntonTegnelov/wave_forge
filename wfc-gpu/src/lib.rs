//! The GPU solver: one workgroup solves one whole region inside a single dispatch.
//!
//! A region's domains live in workgroup memory for the whole solve, so propagation, selection and
//! recovery never leave the device and a batch of regions costs one dispatch and one readback
//! (docs/architecture/solver.md, "The block kernel"). [`BlockSolver`] is the [`wfc_core::Solver`]
//! over it, and [`ComputeBackend`] is everything it needs from a device, so an engine that owns its
//! own compute API supplies a backend rather than a solver: Godot's `RenderingDevice` takes SPIR-V
//! and blocks in `sync()`, which is why nothing here is async and why polling is optional.

pub mod backend;
pub mod block_solver;
pub mod error;
pub mod kernel;
#[cfg(feature = "wgpu")]
pub mod wgpu_backend;

pub use backend::{BackendError, BackendLimits, BufferUsage, ComputeBackend};
pub use block_solver::{BlockSolver, Compilation};
pub use error::GpuError;
pub use kernel::{KernelSpec, Params, SolverConfig};
#[cfg(feature = "wgpu")]
pub use wgpu_backend::WgpuBackend;
