// wfc-gpu/src/coordination/strategy.rs

//! Defines the core strategy interface for algorithm coordination in the WFC-GPU system.
//! These strategies manage how different algorithm phases (entropy calculation, cell selection,
//! and constraint propagation) work together within the Wave Function Collapse algorithm.

use crate::{
    entropy::GpuEntropyCalculator, gpu::GpuAccelerator, propagator::GpuConstraintPropagator,
    utils::RwLock,
};
use async_trait::async_trait;
use std::fmt::Debug;
use std::sync::Arc;
use wfc_core::{
    grid::EntropyGrid, grid::PossibilityGrid, propagator::ConstraintPropagator, WfcError,
};
use wfc_rules::AdjacencyRules;

/// The core strategy interface for WFC algorithm coordination.
/// Implementations provide different approaches to running the WFC algorithm
/// on the GPU, potentially optimizing for different grid sizes, hardware capabilities,
/// or application needs.
#[async_trait]
pub trait CoordinationStrategy: Debug + Send + Sync {
    /// Coordinates a single step of the WFC algorithm.
    ///
    /// # Arguments
    /// * `accelerator` - The GPU accelerator providing computational resources
    /// * `grid` - The current possibility grid state
    ///
    /// # Returns
    /// * `Ok(StepResult)` with the result of this step
    /// * `Err(WfcError)` if an error occurred
    async fn step(
        &mut self,
        _accelerator: &mut GpuAccelerator,
        grid: &mut PossibilityGrid,
    ) -> Result<StepResult, WfcError>;

    /// Initializes the coordination strategy before running.
    ///
    /// # Arguments
    /// * `accelerator` - The GPU accelerator providing computational resources
    /// * `grid` - The initial possibility grid state
    ///
    /// # Returns
    /// * `Ok(())` if initialization is successful
    /// * `Err(WfcError)` if initialization fails
    async fn initialize(
        &mut self,
        _accelerator: &mut GpuAccelerator,
        grid: &PossibilityGrid,
    ) -> Result<(), WfcError>;

    /// Finalizes the coordination strategy after running.
    ///
    /// # Arguments
    /// * `accelerator` - The GPU accelerator providing computational resources
    /// * `grid` - The final possibility grid state
    ///
    /// # Returns
    /// * `Ok(PossibilityGrid)` - The final, possibly modified grid
    /// * `Err(WfcError)` if finalization fails
    async fn finalize(
        &mut self,
        _accelerator: &mut GpuAccelerator,
        grid: &mut PossibilityGrid,
    ) -> Result<PossibilityGrid, WfcError>;

    /// Creates a clone of this strategy.
    fn clone_box(&self) -> Box<dyn CoordinationStrategy>;

    /// Coordinates the propagation of constraints through the grid.
    ///
    /// # Arguments
    /// * `worklist` - List of coordinates to propagate constraints from
    ///
    /// # Returns
    /// * `Ok(())` if propagation is successful
    /// * `Err(WfcError)` if propagation fails
    async fn coordinate_propagation(
        &mut self,
        worklist: &[(usize, usize, usize)],
    ) -> Result<(), WfcError>;
}

impl Clone for Box<dyn CoordinationStrategy> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// The result of a single WFC step.
#[derive(Debug, Clone, PartialEq)]
pub enum StepResult {
    /// The algorithm has more work to do
    InProgress,
    /// The algorithm has completed successfully
    Completed,
    /// The algorithm encountered a contradiction
    Contradiction,
}

/// Factory for creating coordination strategies.
pub struct CoordinationStrategyFactory;

impl CoordinationStrategyFactory {
    /// Creates a default coordination strategy.
    pub fn create_default(
        entropy_calculator: Arc<GpuEntropyCalculator>,
        propagator: Arc<RwLock<GpuConstraintPropagator>>,
    ) -> Box<dyn CoordinationStrategy> {
        Box::new(DefaultCoordinationStrategy::new(
            entropy_calculator,
            propagator,
        ))
    }

    /// Creates an adaptive coordination strategy that selects an appropriate
    /// strategy based on grid size and hardware capabilities.
    pub fn create_adaptive(
        entropy_calculator: Arc<GpuEntropyCalculator>,
        propagator: Arc<RwLock<GpuConstraintPropagator>>,
        grid_size: (usize, usize, usize),
    ) -> Box<dyn CoordinationStrategy> {
        // Logic to select the appropriate strategy based on grid size
        if grid_size.0 * grid_size.1 * grid_size.2 > 1_000_000 {
            // For very large grids, use a strategy optimized for large grids
            Box::new(LargeGridCoordinationStrategy::new(
                entropy_calculator,
                propagator,
            ))
        } else {
            // For smaller grids, use the default strategy
            Box::new(DefaultCoordinationStrategy::new(
                entropy_calculator,
                propagator,
            ))
        }
    }

    /// Creates a batched coordination strategy for processing cells in batches.
    pub fn create_batched(
        entropy_calculator: Arc<GpuEntropyCalculator>,
        propagator: Arc<RwLock<GpuConstraintPropagator>>,
        batch_size: usize,
    ) -> Box<dyn CoordinationStrategy> {
        Box::new(BatchedCoordinationStrategy::new(
            entropy_calculator,
            propagator,
            batch_size,
        ))
    }
}

/// The default coordination strategy implementation.
#[derive(Debug, Clone)]
struct DefaultCoordinationStrategy {
    entropy_calculator: Arc<GpuEntropyCalculator>,
    propagator: Arc<RwLock<GpuConstraintPropagator>>,
    grid: Arc<RwLock<PossibilityGrid>>,
    rules: Arc<RwLock<AdjacencyRules>>,
    // Add additional state as needed
}

impl DefaultCoordinationStrategy {
    fn new(
        entropy_calculator: Arc<GpuEntropyCalculator>,
        propagator: Arc<RwLock<GpuConstraintPropagator>>,
    ) -> Self {
        // Create placeholder grid and rules that will be initialized later
        // Use 1 tile instead of 0 to avoid assertion failure
        let grid = Arc::new(RwLock::new(PossibilityGrid::new(1, 1, 1, 1)));
        let rules = Arc::new(RwLock::new(AdjacencyRules::from_allowed_tuples(
            0,
            0,
            Vec::new(),
        )));

        Self {
            entropy_calculator,
            propagator,
            grid,
            rules,
        }
    }
}

#[async_trait]
impl CoordinationStrategy for DefaultCoordinationStrategy {
    async fn step(
        &mut self,
        accelerator: &mut GpuAccelerator,
        grid: &mut PossibilityGrid,
    ) -> Result<StepResult, WfcError> {
        // Update our internal grid state (optional, depends if needed for tile selection)
        {
            let mut internal_grid = self.grid.write().await;
            *internal_grid = grid.clone();
        }

        // 1. Select min entropy cell by reading GPU result buffer
        // Note: Assumes the entropy calculation shader has already run before this step
        let selection_result = self
            .entropy_calculator
            .select_lowest_entropy_cell_with_value_async() // No argument needed
            .await;

        // Handle the Result first
        let selection = selection_result.map_err(|e| {
            wfc_core::WfcError::InternalError(format!("GPU min entropy selection error: {}", e))
        })?;

        match selection {
            Some((coords_tuple, _entropy)) => {
                // Convert tuple to Coord3D or directly use the tuple if collapse_cell_gpu expects it
                // Assuming collapse_cell_gpu expects (usize, usize, usize)
                let coords = (coords_tuple.0, coords_tuple.1, coords_tuple.2);
                log::debug!("Coordinator selected cell {:?} for collapse", coords);

                // 2. Choose a tile to collapse to
                // TODO: Implement actual tile selection logic. Use the entropy value?
                let chosen_tile_id: u32 = 0; // Placeholder: Always pick tile 0
                log::debug!("Collapsing cell {:?} to tile {}", coords, chosen_tile_id);

                // 3. Collapse cell on GPU
                accelerator
                    .collapse_cell_gpu(coords, chosen_tile_id)
                    .map_err(|e| {
                        wfc_core::WfcError::InternalError(format!("GPU collapse error: {}", e))
                    })?; // Use wfc_core::InternalError

                // 4. Propagate constraints starting from the collapsed cell
                let worklist = vec![coords];
                self.coordinate_propagation(&worklist).await?;

                // 5. Check for contradiction (can be done after propagation)
                let contradiction_status = accelerator
                    .synchronizer() // Use new method
                    .download_contradiction_status()
                    .await
                    .map_err(|e| {
                        wfc_core::WfcError::InternalError(format!(
                            "GPU contradiction check error: {}",
                            e
                        ))
                    })?; // Use wfc_core::InternalError

                if contradiction_status.0 {
                    log::warn!("Contradiction detected after propagation!");
                    // TODO: Extract location from contradiction_status.1
                    return Ok(StepResult::Contradiction);
                }

                // Return InProgress
                Ok(StepResult::InProgress)
            }
            None => {
                // No cell found with positive entropy - Algorithm likely completed
                log::debug!("No cell with positive entropy found. Assuming completion.");
                Ok(StepResult::Completed)
            }
        }
    }

    async fn initialize(
        &mut self,
        _accelerator: &mut GpuAccelerator,
        grid: &PossibilityGrid,
    ) -> Result<(), WfcError> {
        // Initialize resources needed for coordination
        let mut internal_grid = self.grid.write().await;
        *internal_grid = grid.clone();

        // In a real implementation, we would initialize the rules here too
        // For now, we'll use placeholder empty rules
        let mut internal_rules = self.rules.write().await;
        *internal_rules = AdjacencyRules::from_allowed_tuples(0, 0, Vec::new());

        Ok(())
    }

    async fn finalize(
        &mut self,
        _accelerator: &mut GpuAccelerator,
        grid: &mut PossibilityGrid,
    ) -> Result<PossibilityGrid, WfcError> {
        // Finalize and clean up resources
        Ok(grid.clone())
    }

    fn clone_box(&self) -> Box<dyn CoordinationStrategy> {
        Box::new(self.clone())
    }

    async fn coordinate_propagation(
        &mut self,
        worklist: &[(usize, usize, usize)],
    ) -> Result<(), WfcError> {
        let propagator = self.propagator.write().await;
        let grid = &mut self.grid.write().await;
        let rules = &self.rules.read().await;
        propagator
            .propagate(grid, worklist.to_vec(), rules)
            .await
            .map_err(WfcError::PropagationError)
    }
}

/// A coordination strategy optimized for large grids.
#[derive(Debug, Clone)]
struct LargeGridCoordinationStrategy {
    _entropy_calculator: Arc<GpuEntropyCalculator>,
    propagator: Arc<RwLock<GpuConstraintPropagator>>,
    grid: Arc<RwLock<PossibilityGrid>>,
    rules: Arc<RwLock<AdjacencyRules>>,
    // Add additional state as needed for large grid optimization
}

impl LargeGridCoordinationStrategy {
    fn new(
        entropy_calculator: Arc<GpuEntropyCalculator>,
        propagator: Arc<RwLock<GpuConstraintPropagator>>,
    ) -> Self {
        // Create empty placeholder grid and rules that will be initialized later
        let grid = Arc::new(RwLock::new(PossibilityGrid::new(1, 1, 1, 0)));
        let rules = Arc::new(RwLock::new(AdjacencyRules::from_allowed_tuples(
            0,
            0,
            Vec::new(),
        )));

        Self {
            _entropy_calculator: entropy_calculator,
            propagator,
            grid,
            rules,
        }
    }
}

#[async_trait]
impl CoordinationStrategy for LargeGridCoordinationStrategy {
    async fn step(
        &mut self,
        _accelerator: &mut GpuAccelerator,
        grid: &mut PossibilityGrid,
    ) -> Result<StepResult, WfcError> {
        // Update our internal grid with the current grid state
        {
            let mut internal_grid = self.grid.write().await;
            *internal_grid = grid.clone();
        }

        // Large grid optimization of a WFC step
        // Potentially using different batching strategies or
        // more aggressive parallelization

        // This is a placeholder - the actual implementation would optimize
        // for large grids with special strategies
        Ok(StepResult::InProgress)
    }

    async fn initialize(
        &mut self,
        _accelerator: &mut GpuAccelerator,
        grid: &PossibilityGrid,
    ) -> Result<(), WfcError> {
        // Initialize resources needed for large grid coordination
        let mut internal_grid = self.grid.write().await;
        *internal_grid = grid.clone();

        // In a real implementation, we would initialize the rules here too
        // For now, we'll use placeholder empty rules
        let mut internal_rules = self.rules.write().await;
        *internal_rules = AdjacencyRules::from_allowed_tuples(0, 0, Vec::new());

        Ok(())
    }

    async fn finalize(
        &mut self,
        _accelerator: &mut GpuAccelerator,
        grid: &mut PossibilityGrid,
    ) -> Result<PossibilityGrid, WfcError> {
        // Finalize and clean up resources
        Ok(grid.clone())
    }

    fn clone_box(&self) -> Box<dyn CoordinationStrategy> {
        Box::new(self.clone())
    }

    async fn coordinate_propagation(
        &mut self,
        worklist: &[(usize, usize, usize)],
    ) -> Result<(), WfcError> {
        let propagator = self.propagator.write().await;
        let grid = &mut self.grid.write().await;
        let rules = &self.rules.read().await;
        propagator
            .propagate(grid, worklist.to_vec(), rules)
            .await
            .map_err(WfcError::PropagationError)
    }
}

/// A coordination strategy that processes cells in batches to improve performance.
///
/// This strategy selects multiple cells to collapse in a single step, which can
/// improve performance on very large grids by reducing the number of GPU synchronization
/// points and increasing parallelism.
#[derive(Debug, Clone)]
struct BatchedCoordinationStrategy {
    _entropy_calculator: Arc<GpuEntropyCalculator>,
    propagator: Arc<RwLock<GpuConstraintPropagator>>,
    grid: Arc<RwLock<PossibilityGrid>>,
    rules: Arc<RwLock<AdjacencyRules>>,
    _batch_size: usize,
    // Additional state for batch processing
    current_batch: Vec<(usize, usize, usize)>,
}

impl BatchedCoordinationStrategy {
    fn new(
        entropy_calculator: Arc<GpuEntropyCalculator>,
        propagator: Arc<RwLock<GpuConstraintPropagator>>,
        batch_size: usize,
    ) -> Self {
        // Create empty placeholder grid and rules that will be initialized later
        let grid = Arc::new(RwLock::new(PossibilityGrid::new(1, 1, 1, 0)));
        let rules = Arc::new(RwLock::new(AdjacencyRules::from_allowed_tuples(
            0,
            0,
            Vec::new(),
        )));

        Self {
            _entropy_calculator: entropy_calculator,
            propagator,
            grid,
            rules,
            _batch_size: batch_size,
            current_batch: Vec::new(),
        }
    }

    /// Select multiple low-entropy cells for simultaneous collapse
    async fn select_batch(
        &mut self,
        _accelerator: &mut GpuAccelerator,
        _grid: &PossibilityGrid,
    ) -> Result<Vec<(usize, usize, usize)>, WfcError> {
        // In a real implementation, this would use a modified entropy calculation
        // that returns multiple low-entropy cells instead of just the minimum.
        // For now, we just return a placeholder.

        let batch = vec![(0, 0, 0)]; // Placeholder
        Ok(batch)
    }
}

#[async_trait]
impl CoordinationStrategy for BatchedCoordinationStrategy {
    async fn step(
        &mut self,
        _accelerator: &mut GpuAccelerator,
        grid: &mut PossibilityGrid,
    ) -> Result<StepResult, WfcError> {
        // Check if we have a batch to process
        if self.current_batch.is_empty() {
            // No cells to process, select a new batch
            let new_batch = self.select_batch(_accelerator, grid).await?;
            if new_batch.is_empty() {
                // No more cells to select, we're done
                return Ok(StepResult::Completed);
            }
            self.current_batch = new_batch;
        }

        // Process one cell from the batch
        let _cell = self.current_batch.pop().unwrap();

        // Process the cell and propagate constraints
        // This is a placeholder - a real implementation would do actual work
        Ok(StepResult::InProgress)
    }

    async fn initialize(
        &mut self,
        _accelerator: &mut GpuAccelerator,
        grid: &PossibilityGrid,
    ) -> Result<(), WfcError> {
        // Initialize resources needed for batched coordination
        let mut internal_grid = self.grid.write().await;
        *internal_grid = grid.clone();

        // In a real implementation, we would initialize the rules here too
        // For now, we'll use placeholder empty rules
        let mut internal_rules = self.rules.write().await;
        *internal_rules = AdjacencyRules::from_allowed_tuples(0, 0, Vec::new());

        self.current_batch.clear();
        Ok(())
    }

    async fn finalize(
        &mut self,
        _accelerator: &mut GpuAccelerator,
        grid: &mut PossibilityGrid,
    ) -> Result<PossibilityGrid, WfcError> {
        // Finalize and clean up resources
        self.current_batch.clear();
        Ok(grid.clone())
    }

    fn clone_box(&self) -> Box<dyn CoordinationStrategy> {
        Box::new(self.clone())
    }

    async fn coordinate_propagation(
        &mut self,
        worklist: &[(usize, usize, usize)],
    ) -> Result<(), WfcError> {
        let propagator = self.propagator.write().await;
        let grid = &mut self.grid.write().await;
        let rules = &self.rules.read().await;
        propagator
            .propagate(grid, worklist.to_vec(), rules)
            .await
            .map_err(WfcError::PropagationError)
    }
}
