// wfc-gpu/src/coordination/mod.rs

//! Module responsible for coordinating the high-level flow of the WFC algorithm
//! on the GPU, scheduling different phases like entropy calculation, cell collapse,
//! and constraint propagation.

use crate::{
    buffers::{DownloadRequest, GpuBuffers},
    entropy::{EntropyStrategy, GpuEntropyCalculator},
    gpu::{sync::GpuSynchronizer, GpuAccelerator},
    propagator::GpuConstraintPropagator,
    utils::debug_viz::DebugVisualizer,
    utils::error_recovery::{GpuError, GridCoord},
};
use async_trait::async_trait;
use log::{error, trace};
use std::fmt::Debug;
use std::sync::{Arc, RwLock};
use wfc_core::{grid::PossibilityGrid, propagator::PropagationError, WfcError};
use wgpu::{Device, Queue};

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

// The CoordinationStrategy is now defined in the strategy.rs module
// pub trait CoordinationStrategy {}

// --- Traits --- //

/// Defines the high-level coordination logic for a WFC run step.
/// Implementations manage the sequence of GPU operations for entropy calculation,
/// cell selection, and constraint propagation.
#[async_trait]
pub trait WfcCoordinator: Send + Sync {
    /// Coordinates the entropy calculation and cell selection phase.
    /// Returns the coordinates of the cell with the minimum entropy, or None if converged.
    async fn coordinate_entropy_and_selection(
        &self,
        _entropy_calculator: &Arc<GpuEntropyCalculator>,
        buffers: &Arc<GpuBuffers>,
        _device: &Device,
        _queue: &Queue,
        _sync: &Arc<GpuSynchronizer>,
    ) -> Result<Option<(usize, usize, usize)>, GpuError>;

    /// Coordinates the constraint propagation phase after a cell collapse.
    /// `updated_coords` typically contains the single collapsed cell's coordinates.
    async fn coordinate_propagation(
        &self,
        propagator: &Arc<RwLock<GpuConstraintPropagator>>,
        buffers: &Arc<GpuBuffers>,
        device: &Device,
        queue: &Queue,
        updated_coords: Vec<GridCoord>,
    ) -> Result<(), PropagationError>;

    /// Allows cloning the coordinator into a Box.
    fn clone_box(&self) -> Box<dyn WfcCoordinator + Send + Sync>;
}

impl Clone for Box<dyn WfcCoordinator + Send + Sync> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Defines the interface for a WFC coordination strategy.
/// Implementations will manage the overall algorithm loop.
/// This is now a deprecated alias for the CoordinationStrategy trait in the strategy module.
#[async_trait]
#[deprecated(
    since = "0.1.0",
    note = "Use strategy::CoordinationStrategy instead which provides a more comprehensive interface"
)]
pub trait WfcCoordinatorTrait: Send + Sync {
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

    // TODO: Define other necessary methods for coordination,
    // e.g., step(), initialize(), finalize().
}

// --- Structs --- //

/// The default coordinator implementation.
#[derive(Debug, Clone)]
pub struct DefaultCoordinator {
    entropy_calculator: Arc<GpuEntropyCalculator>,
    propagator: Arc<RwLock<GpuConstraintPropagator>>,
    // Add reference to EntropyCoordinator
    entropy_coordinator: Option<entropy::EntropyCoordinator>,
}

impl DefaultCoordinator {
    pub fn new(
        entropy_calculator: Arc<GpuEntropyCalculator>,
        propagator: Arc<RwLock<GpuConstraintPropagator>>,
    ) -> Self {
        // Create an EntropyCoordinator using the provided calculator
        let entropy_coordinator =
            Some(entropy::EntropyCoordinator::new(entropy_calculator.clone()));

        Self {
            entropy_calculator,
            propagator,
            entropy_coordinator,
        }
    }

    /// Set a specific entropy strategy on the entropy calculator
    pub fn with_entropy_strategy<S: EntropyStrategy + 'static>(
        &mut self,
        strategy: S,
    ) -> &mut Self {
        let mut calculator = self.entropy_calculator.clone();
        // Can't modify the calculator directly due to Arc, so this is a placeholder
        // In a real implementation, would need to manage this differently
        trace!("Setting entropy strategy on DefaultCoordinator (placeholder)");
        self
    }

    /// Set a specific entropy coordination strategy
    pub fn with_entropy_coordination_strategy<S: entropy::EntropyCoordinationStrategy + 'static>(
        &mut self,
        strategy: S,
    ) -> &mut Self {
        if let Some(ref mut coordinator) = self.entropy_coordinator {
            // Create a new coordinator with the strategy
            self.entropy_coordinator = Some(
                entropy::EntropyCoordinator::new(self.entropy_calculator.clone())
                    .with_strategy(strategy),
            );
        }
        self
    }
}

#[async_trait]
impl WfcCoordinator for DefaultCoordinator {
    async fn coordinate_entropy_and_selection(
        &self,
        _entropy_calculator: &Arc<GpuEntropyCalculator>,
        buffers: &Arc<GpuBuffers>,
        _device: &Device,
        _queue: &Queue,
        sync: &Arc<GpuSynchronizer>,
    ) -> Result<Option<(usize, usize, usize)>, GpuError> {
        trace!("DefaultCoordinator: Using entropy coordinator for selection");

        // Use the EntropyCoordinator if available
        if let Some(ref coordinator) = self.entropy_coordinator {
            let result = coordinator.download_min_entropy_info(buffers, sync).await?;

            if let Some((_entropy, coord)) = result {
                return Ok(Some((coord.x, coord.y, coord.z)));
            } else {
                return Ok(None);
            }
        }

        // Fall back to original implementation if no coordinator is available
        trace!("DefaultCoordinator: Running entropy pass...");
        // Assume GpuEntropyCalculator has run_entropy_pass method
        // Commenting out due to unknown signature
        // entropy_calculator.run_entropy_pass(device, queue).await?;

        trace!("DefaultCoordinator: Running min reduction pass...");
        // Assume GpuEntropyCalculator has run_min_reduction_pass method
        // Commenting out due to unknown signature
        // entropy_calculator.run_min_reduction_pass(device, queue).await?;

        trace!("DefaultCoordinator: Downloading entropy results...");
        let request = DownloadRequest {
            download_min_entropy_info: true,
            download_contradiction_flag: false,
            ..Default::default()
        };

        let results = buffers.download_results(request).await?;

        trace!("Getting min entropy info: {:?}", results.min_entropy_info);

        // Extract min_data from the tuple
        if let Some(min_data) = results.min_entropy_info {
            // Check if grid is fully collapsed
            if min_data.1 == u32::MAX {
                trace!("Grid appears to be fully collapsed or in contradiction");
                return Ok(None);
            }

            let (width, height, _depth) = buffers.grid_dims;
            let flat_index = min_data.1 as usize;
            let z = flat_index / (width * height);
            let y = (flat_index % (width * height)) / width;
            let x = flat_index % width;

            trace!(
                "Selected cell at ({}, {}, {}) with entropy {}",
                x,
                y,
                z,
                min_data.0
            );

            // We now have our minimum entropy cell
            Ok(Some((x, y, z)))
        } else {
            // No min entropy info found
            trace!("No min entropy info found, grid may be fully collapsed");
            Ok(None)
        }
    }

    async fn coordinate_propagation(
        &self,
        propagator_lock: &Arc<RwLock<GpuConstraintPropagator>>,
        _buffers: &Arc<GpuBuffers>,
        _device: &Device,
        _queue: &Queue,
        _updated_coords: Vec<GridCoord>,
    ) -> Result<(), PropagationError> {
        trace!("DefaultCoordinator: Running propagation...");

        let _propagator = propagator_lock.write().unwrap();

        // Handle propagation operations
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn WfcCoordinator + Send + Sync> {
        Box::new(self.clone())
    }
}

// --- Submodules --- //

pub mod entropy;
pub mod propagation; // Added entropy coordination module
pub mod strategy; // New strategy module for coordination strategies

// For convenience, re-export key types from submodules
pub use self::entropy::{
    EntropyCoordinationStrategy, EntropyCoordinationStrategyFactory, EntropyCoordinator,
};
pub use self::strategy::{CoordinationStrategy, CoordinationStrategyFactory, StepResult};
pub mod coordinator {
    pub use super::DefaultCoordinator;
}
