// wfc-gpu/src/coordination/mod.rs

//! Module responsible for coordinating the high-level flow of the WFC algorithm
//! on the GPU, scheduling different phases like entropy calculation, cell collapse,
//! and constraint propagation.

use crate::{GpuAccelerator, GpuError};
use async_trait::async_trait;
use wfc_core::{
    entropy::EntropyError, grid::PossibilityGrid, propagator::PropagationError, WfcError,
};

// --- Traits --- //

/// Defines the interface for a WFC coordination strategy.
/// Implementations will manage the overall algorithm loop.
#[async_trait]
pub trait WfcCoordinator {
    /// Runs the main WFC algorithm loop.
    ///
    /// # Arguments
    /// * `accelerator` - The GPU accelerator providing computational resources.
    /// * `grid` - The initial possibility grid state.
    /// * `max_iterations` - Maximum number of iterations allowed.
    ///
    /// # Returns
    /// * `Ok(final_grid)` if the algorithm completes successfully.
    /// * `Err(WfcError)` if an error (like contradiction) occurs.
    async fn run_wfc(
        &mut self,
        accelerator: &mut GpuAccelerator, // Needs mutable access?
        grid: &mut PossibilityGrid,
        max_iterations: u64,
        // TODO: Add progress callback, shutdown signal
    ) -> Result<PossibilityGrid, WfcError>;

    // TODO: Define other necessary methods for coordination,
    // e.g., step(), initialize(), finalize().
}

// --- Structs --- //

/// A basic coordinator implementation (Placeholder).
#[derive(Debug, Default)]
pub struct BasicCoordinator; // Placeholder struct

#[async_trait]
impl WfcCoordinator for BasicCoordinator {
    async fn run_wfc(
        &mut self,
        _accelerator: &mut GpuAccelerator,
        _grid: &mut PossibilityGrid,
        _max_iterations: u64,
    ) -> Result<PossibilityGrid, WfcError> {
        // Placeholder implementation - real logic will go here
        // This might call accelerator.run_with_callback() internally,
        // or reimplement the loop using delegated methods.
        unimplemented!("BasicCoordinator::run_wfc is not implemented yet.");
    }
}

// --- Submodules --- //

// pub mod propagation; // Planned submodule for propagation coordination
// pub mod entropy;     // Planned submodule for entropy coordination
