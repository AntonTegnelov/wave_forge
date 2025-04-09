// wfc-gpu/src/coordination/mod.rs

//! Module responsible for coordinating the high-level flow of the WFC algorithm
//! on the GPU, scheduling different phases like entropy calculation, cell collapse,
//! and constraint propagation.

use crate::{
    backend::GpuBackend,
    // Removed unused import GpuError
    // GpuError,
    buffers::{DownloadRequest, GpuBuffers},
    entropy::GpuEntropyCalculator,
    error_recovery::GpuError,
    propagator::GpuConstraintPropagator,
    // GpuAccelerator, GpuError, // Removed unused imports
    // Removed unused imports
    // coordination::{CoordinationError, CoordinationEvent, CoordinationStrategy},
    // entropy::GpuEntropyCalculator,
    // GpuAccelerator, GpuError, // Removed unused imports
    GpuAccelerator,
};
use async_trait::async_trait;
use std::fmt::Debug;
use std::sync::Arc;
use wfc_core::propagator::{ConstraintPropagator, PropagationError};
use wfc_core::{grid::GridCoord3D, grid::PossibilityGrid, WfcError};
use wgpu::{Device, Queue}; // Import Debug trait // Import ConstraintPropagator

// Re-export or define CoordinationError, CoordinationEvent, CoordinationStrategy here
// For now, let's assume they should be defined in this file or imported differently.
// Placeholder definitions removed as they seem to exist already
// #[derive(Debug)] pub enum CoordinationError {}
// #[derive(Debug)] pub enum CoordinationEvent {}
// pub trait CoordinationStrategy {}

// Define placeholder types until proper ones are implemented
// These should eventually be replaced by actual types from submodules or elsewhere.
#[derive(Debug, thiserror::Error)]
pub enum CoordinationError {
    #[error("Placeholder coordination error: {0}")]
    Placeholder(String),
}

#[derive(Debug)]
pub enum CoordinationEvent {
    PhaseStarted(String),
    PhaseCompleted(String),
}

pub trait CoordinationStrategy {}

// --- Traits --- //

/// Clone trait for Box<dyn WfcCoordinator + Send + Sync + Debug>.
/// Allows cloning the boxed trait object.
pub trait CloneBoxWfcCoordinator {
    fn clone_box(&self) -> Box<dyn WfcCoordinator + Send + Sync + Debug>;
}

impl<T> CloneBoxWfcCoordinator for T
where
    T: WfcCoordinator + Clone + Send + Sync + Debug + 'static,
{
    fn clone_box(&self) -> Box<dyn WfcCoordinator + Send + Sync + Debug> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn WfcCoordinator + Send + Sync + Debug> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Defines the interface for a WFC coordination strategy.
/// Implementations will manage the overall algorithm loop.
#[async_trait]
pub trait WfcCoordinator: Send + Sync + Debug + CloneBoxWfcCoordinator {
    // Add bounds
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

    /// Coordinates entropy calculation and cell selection.
    async fn coordinate_entropy_and_selection(
        &self,
        entropy_calculator: &GpuEntropyCalculator,
        buffers: &GpuBuffers,
        device: &Device,
        queue: &Queue,
    ) -> Result<Option<(usize, usize, usize)>, GpuError>;

    /// Coordinates constraint propagation.
    async fn coordinate_propagation(
        &self,
        propagator: &mut GpuConstraintPropagator,
        buffers: &GpuBuffers,
        device: &Device,
        queue: &Queue,
        updated_coords: Vec<(usize, usize, usize)>,
    ) -> Result<(), PropagationError>;

    // TODO: Define other necessary methods for coordination,
    // e.g., step(), initialize(), finalize().
}

// --- Structs --- //

/// The default coordinator implementation.
#[derive(Debug, Clone)] // Removed Default
pub struct DefaultCoordinator {
    entropy_calculator: GpuEntropyCalculator,
    propagator: GpuConstraintPropagator,
}

impl DefaultCoordinator {
    pub fn new(
        entropy_calculator: GpuEntropyCalculator,
        propagator: GpuConstraintPropagator,
    ) -> Self {
        Self {
            entropy_calculator,
            propagator,
        }
    }
}

#[async_trait]
impl WfcCoordinator for DefaultCoordinator {
    async fn run_wfc(
        &mut self,
        _accelerator: &mut GpuAccelerator,
        _grid: &mut PossibilityGrid,
        _max_iterations: u64,
    ) -> Result<PossibilityGrid, WfcError> {
        // Placeholder implementation - real logic will go here
        // This might call accelerator.run_with_callback() internally,
        // or reimplement the loop using delegated methods.
        unimplemented!("DefaultCoordinator::run_wfc is not implemented yet.");
    }

    async fn coordinate_entropy_and_selection(
        &self,
        _entropy_calculator: &GpuEntropyCalculator, // Can use self.entropy_calculator
        buffers: &GpuBuffers,
        device: &Device,
        queue: &Queue,
    ) -> Result<Option<(usize, usize, usize)>, GpuError> {
        // 1. Run entropy calculation shader
        self.entropy_calculator
            .run_entropy_pass(device, queue)
            .await?;

        // 2. Run min reduction shader (part of entropy calculation)
        self.entropy_calculator
            .run_min_reduction_pass(device, queue)
            .await?;

        // 3. Download min entropy info
        let request = DownloadRequest {
            download_min_entropy_info: true,
            ..Default::default()
        };
        // Need owned Arcs for download_results
        // This requires changing how DefaultCoordinator holds/gets device/queue/buffers
        // For now, assume we get them passed in or cloned appropriately.
        // Let's pretend download_results takes refs for now to avoid bigger refactor
        let results = buffers
            .download_results(Arc::new(device.clone()), Arc::new(queue.clone()), request)
            .await?;

        // 4. Select cell CPU-side based on downloaded info
        if let Some((min_entropy, index)) = results.min_entropy_info {
            if min_entropy < f32::INFINITY {
                // Check if any cell is not fully collapsed
                let grid_def = buffers.grid_definition(); // Need GridDefinition access
                let coords = grid_def.coords_from_index(index as usize);
                Ok(Some(coords))
            } else {
                Ok(None) // All cells collapsed or invalid state
            }
        } else {
            Err(GpuError::BufferOperationError(
                "Min entropy info not found after download".to_string(),
            ))
        }
    }

    async fn coordinate_propagation(
        &self,
        propagator: &mut GpuConstraintPropagator,
        buffers: &GpuBuffers,
        device: &Device,
        queue: &Queue,
        updated_coords: Vec<(usize, usize, usize)>,
    ) -> Result<(), PropagationError> {
        // Run propagation passes
        propagator.propagate(device, queue, updated_coords).await
    }
}

// --- Submodules --- //

pub mod propagation;
// pub mod entropy;     // Planned submodule for entropy coordination
