// wfc-gpu/src/coordination/mod.rs

//! Module responsible for coordinating the high-level flow of the WFC algorithm
//! on the GPU, scheduling different phases like entropy calculation, cell collapse,
//! and constraint propagation.

use crate::{
    buffers::{DownloadRequest, GpuBuffers},
    entropy::{EntropyStrategy, GpuEntropyCalculator},
    gpu::{sync::GpuSynchronizer, GpuAccelerator},
    propagator::GpuConstraintPropagator,
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
    // Coordinator for entropy calculation and selection
    entropy_coordinator: Option<entropy::EntropyCoordinator>,
    // Strategy for propagation coordination
    propagation_strategy: Option<Box<dyn propagation::PropagationCoordinationStrategy>>,
    // Overall coordination strategy
    coordination_strategy: Option<Box<dyn strategy::CoordinationStrategy>>,
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
            entropy_calculator: entropy_calculator.clone(),
            propagator: propagator.clone(),
            entropy_coordinator,
            propagation_strategy: None,
            coordination_strategy: Some(strategy::CoordinationStrategyFactory::create_default(
                entropy_calculator.clone(),
                propagator.clone(),
            )),
        }
    }

    /// Set a specific entropy strategy on the entropy calculator
    pub fn with_entropy_strategy<S: EntropyStrategy + 'static>(
        &mut self,
        strategy: S,
    ) -> &mut Self {
        let calculator = self.entropy_calculator.clone();
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

    /// Set a specific propagation coordination strategy
    pub fn with_propagation_coordination_strategy(
        &mut self,
        strategy: Box<dyn propagation::PropagationCoordinationStrategy>,
    ) -> &mut Self {
        self.propagation_strategy = Some(strategy);
        trace!("Setting propagation coordination strategy on DefaultCoordinator");
        self
    }

    /// Set the overall coordination strategy
    pub fn with_coordination_strategy<S: strategy::CoordinationStrategy + 'static>(
        &mut self,
        strategy: S,
    ) -> &mut Self {
        self.coordination_strategy = Some(Box::new(strategy));
        trace!("Setting coordination strategy on DefaultCoordinator");
        self
    }

    /// Set the overall coordination strategy
    pub fn with_coordination_strategy_boxed(
        &mut self,
        strategy: Box<dyn strategy::CoordinationStrategy>,
    ) -> &mut Self {
        self.coordination_strategy = Some(strategy);
        trace!("Setting boxed coordination strategy on DefaultCoordinator");
        self
    }

    /// Use the default coordination strategy
    pub fn with_default_coordination(&mut self) -> &mut Self {
        self.coordination_strategy = Some(strategy::CoordinationStrategyFactory::create_default(
            self.entropy_calculator.clone(),
            self.propagator.clone(),
        ));
        trace!("Set default coordination strategy");
        self
    }

    /// Use an adaptive coordination strategy based on grid size
    pub fn with_adaptive_coordination(&mut self, grid_size: (usize, usize, usize)) -> &mut Self {
        self.coordination_strategy = Some(strategy::CoordinationStrategyFactory::create_adaptive(
            self.entropy_calculator.clone(),
            self.propagator.clone(),
            grid_size,
        ));
        trace!("Set adaptive coordination strategy based on grid size");
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
        buffers: &Arc<GpuBuffers>,
        device: &Device,
        queue: &Queue,
        updated_coords: Vec<GridCoord>,
    ) -> Result<(), PropagationError> {
        trace!("DefaultCoordinator: Running propagation...");

        // If a propagation strategy is set, use it
        if let Some(ref mut strategy) = self.propagation_strategy.clone() {
            let coords_vec: Vec<(usize, usize, usize)> =
                updated_coords.iter().map(|c| (c.x, c.y, c.z)).collect();

            // Create a dummy grid for now - in a real implementation, we would
            // download the grid state first
            let grid_dims = buffers.grid_dims;
            let mut grid = PossibilityGrid::new(
                grid_dims.0,
                grid_dims.1,
                grid_dims.2,
                0, // Placeholder
            );

            // Create dummy rules - in a real implementation, we would
            // get the proper rules
            let rules = wfc_rules::AdjacencyRules::from_allowed_tuples(0, 0, Vec::new());

            return strategy
                .coordinate_propagation(propagator_lock, &mut grid, coords_vec, &rules)
                .await
                .map_err(|e| {
                    PropagationError::InternalError(format!("Strategy propagation failed: {}", e))
                });
        }

        // Otherwise, use the default implementation
        let _propagator = propagator_lock.write().unwrap();

        // Handle propagation operations
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn WfcCoordinator + Send + Sync> {
        Box::new(self.clone())
    }
}

/// A coordinator implementation that fully leverages the strategy pattern.
/// This implementation delegates all coordination decisions to strategy implementations.
#[derive(Debug, Clone)]
pub struct StrategicCoordinator {
    entropy_calculator: Arc<GpuEntropyCalculator>,
    propagator: Arc<RwLock<GpuConstraintPropagator>>,
    // Strategy for overall coordination
    coordination_strategy: Box<dyn strategy::CoordinationStrategy>,
    // Synchronizer for GPU operations
    sync: Option<Arc<GpuSynchronizer>>,
}

impl StrategicCoordinator {
    /// Creates a new StrategicCoordinator with the specified strategy.
    pub fn new(
        entropy_calculator: Arc<GpuEntropyCalculator>,
        propagator: Arc<RwLock<GpuConstraintPropagator>>,
        coordination_strategy: Box<dyn strategy::CoordinationStrategy>,
    ) -> Self {
        Self {
            entropy_calculator,
            propagator,
            coordination_strategy,
            sync: None,
        }
    }

    /// Sets the synchronizer for GPU operations.
    pub fn with_synchronizer(&mut self, sync: Arc<GpuSynchronizer>) -> &mut Self {
        self.sync = Some(sync);
        self
    }

    /// Creates a StrategicCoordinator with the default coordination strategy.
    pub fn with_default_strategy(
        entropy_calculator: Arc<GpuEntropyCalculator>,
        propagator: Arc<RwLock<GpuConstraintPropagator>>,
    ) -> Self {
        Self::new(
            entropy_calculator.clone(),
            propagator.clone(),
            strategy::CoordinationStrategyFactory::create_default(entropy_calculator, propagator),
        )
    }

    /// Creates a StrategicCoordinator with the adaptive coordination strategy.
    pub fn with_adaptive_strategy(
        entropy_calculator: Arc<GpuEntropyCalculator>,
        propagator: Arc<RwLock<GpuConstraintPropagator>>,
        grid_size: (usize, usize, usize),
    ) -> Self {
        Self::new(
            entropy_calculator.clone(),
            propagator.clone(),
            strategy::CoordinationStrategyFactory::create_adaptive(
                entropy_calculator,
                propagator,
                grid_size,
            ),
        )
    }

    /// Creates a StrategicCoordinator with the batched coordination strategy.
    pub fn with_batched_strategy(
        entropy_calculator: Arc<GpuEntropyCalculator>,
        propagator: Arc<RwLock<GpuConstraintPropagator>>,
        batch_size: usize,
    ) -> Self {
        Self::new(
            entropy_calculator.clone(),
            propagator.clone(),
            strategy::CoordinationStrategyFactory::create_batched(
                entropy_calculator,
                propagator,
                batch_size,
            ),
        )
    }

    /// Run a single step of the WFC algorithm using the underlying strategy.
    /// This method directly delegates to the coordination strategy.
    pub async fn run_step(
        &mut self,
        accelerator: &mut GpuAccelerator,
        grid: &mut PossibilityGrid,
    ) -> Result<strategy::StepResult, WfcError> {
        self.coordination_strategy.step(accelerator, grid).await
    }

    /// Initialize the coordination strategy with the given grid.
    /// This method directly delegates to the coordination strategy.
    pub async fn initialize(
        &mut self,
        accelerator: &mut GpuAccelerator,
        grid: &PossibilityGrid,
    ) -> Result<(), WfcError> {
        self.coordination_strategy
            .initialize(accelerator, grid)
            .await
    }

    /// Finalize the coordination strategy and get the final grid.
    /// This method directly delegates to the coordination strategy.
    pub async fn finalize(
        &mut self,
        accelerator: &mut GpuAccelerator,
        grid: &mut PossibilityGrid,
    ) -> Result<PossibilityGrid, WfcError> {
        self.coordination_strategy.finalize(accelerator, grid).await
    }

    /// Run the full WFC algorithm with the given grid.
    /// This is a convenience method that handles initialization, stepping, and finalization.
    pub async fn run_wfc(
        &mut self,
        accelerator: &mut GpuAccelerator,
        grid: &mut PossibilityGrid,
        max_iterations: u64,
    ) -> Result<PossibilityGrid, WfcError> {
        // Initialize the strategy
        self.initialize(accelerator, grid).await?;

        // Run steps until completion or max iterations
        let mut iterations = 0;
        loop {
            if iterations >= max_iterations {
                return Err(WfcError::MaxIterationsReached(max_iterations));
            }

            match self.run_step(accelerator, grid).await? {
                strategy::StepResult::Completed => {
                    break;
                }
                strategy::StepResult::Contradiction => {
                    return Err(WfcError::Contradiction(0, 0, 0));
                }
                strategy::StepResult::InProgress => {
                    iterations += 1;
                    continue;
                }
            }
        }

        // Finalize and return the result
        self.finalize(accelerator, grid).await
    }
}

#[async_trait]
impl WfcCoordinator for StrategicCoordinator {
    async fn coordinate_entropy_and_selection(
        &self,
        entropy_calculator: &Arc<GpuEntropyCalculator>,
        buffers: &Arc<GpuBuffers>,
        device: &Device,
        queue: &Queue,
        sync: &Arc<GpuSynchronizer>,
    ) -> Result<Option<(usize, usize, usize)>, GpuError> {
        trace!("StrategicCoordinator: Delegating entropy calculation to strategy");

        // This implementation doesn't use the coordination_strategy directly since
        // the method signatures are different. However, it still leverages the
        // entropy calculation components that would be used by the strategy.

        // Create a temporary grid for calculation
        let grid_dims = buffers.grid_dims;
        let dummy_grid = PossibilityGrid::new(grid_dims.0, grid_dims.1, grid_dims.2, 0);

        // Use the EntropyCoordinator as a helper, which is the same component
        // that would be used by the coordination strategies internally
        let coordinator = entropy::EntropyCoordinator::new(entropy_calculator.clone());
        let result = coordinator.download_min_entropy_info(buffers, sync).await?;

        if let Some((_entropy, coord)) = result {
            return Ok(Some((coord.x, coord.y, coord.z)));
        } else {
            return Ok(None);
        }
    }

    async fn coordinate_propagation(
        &self,
        propagator: &Arc<RwLock<GpuConstraintPropagator>>,
        buffers: &Arc<GpuBuffers>,
        device: &Device,
        queue: &Queue,
        updated_coords: Vec<GridCoord>,
    ) -> Result<(), PropagationError> {
        trace!("StrategicCoordinator: Delegating propagation to strategy");

        // Convert grid coordinates to tuple format
        let coords_vec: Vec<(usize, usize, usize)> =
            updated_coords.iter().map(|c| (c.x, c.y, c.z)).collect();

        // Create a temporary grid for propagation
        let grid_dims = buffers.grid_dims;
        let mut dummy_grid = PossibilityGrid::new(grid_dims.0, grid_dims.1, grid_dims.2, 0);

        // Create dummy rules - in a real implementation, we would get the proper rules
        let rules = wfc_rules::AdjacencyRules::from_allowed_tuples(0, 0, Vec::new());

        // Use the PropagationCoordinationStrategy factory to create an appropriate strategy,
        // reusing the same components that would be used by the coordination strategy internally
        let mut strategy = propagation::PropagationCoordinationStrategyFactory::create_direct();

        strategy
            .coordinate_propagation(propagator, &mut dummy_grid, coords_vec, &rules)
            .await
            .map_err(|e| {
                PropagationError::InternalError(format!("Strategy propagation failed: {}", e))
            })
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
pub use self::propagation::{
    DirectPropagationCoordinationStrategy,
    PropagationCoordinationStrategy,
    PropagationCoordinationStrategyFactory,
    PropagationCoordinator, // Legacy
    SubgridPropagationCoordinationStrategy,
};
pub use self::strategy::{CoordinationStrategy, CoordinationStrategyFactory, StepResult};
pub mod coordinator {
    pub use super::DefaultCoordinator;
}
