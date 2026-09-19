//! What can go wrong setting up or running the block solver.

use crate::backend::BackendError;
use thiserror::Error;
use wfc_core::{ModelError, SolverError};

/// A solver that could not be built or run.
#[derive(Error, Debug)]
pub enum GpuError {
    /// The compute backend refused.
    #[error(transparent)]
    Backend(#[from] BackendError),
    /// The rule set or a region is not something the solver can work on.
    #[error(transparent)]
    Model(#[from] ModelError),
    /// A region needs more workgroup memory than the device has. A smaller chunk, a smaller halo or
    /// fewer tiles all reduce it.
    #[error(
        "a region needs {needed} B of workgroup memory but the device offers {available} B; \
         use a smaller chunk, a smaller halo or fewer tiles"
    )]
    WorkgroupStorage { needed: u32, available: u32 },
    /// The configuration asks for more invocations per workgroup than the device allows.
    #[error("{wanted} invocations per workgroup, but the device allows {available}")]
    Invocations { wanted: u32, available: u32 },
}

impl GpuError {
    /// The same failure as a [`SolverError`], for the trait's methods.
    #[must_use]
    pub fn into_solver(self) -> SolverError {
        match self {
            Self::WorkgroupStorage { needed, available } => {
                SolverError::WorkgroupStorage { needed, available }
            }
            other => SolverError::Backend(other.to_string()),
        }
    }
}
