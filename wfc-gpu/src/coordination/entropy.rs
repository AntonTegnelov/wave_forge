//! Module responsible for coordinating entropy calculation and min-entropy cell selection.

use crate::{
    buffers::{DownloadRequest, GpuBuffers},
    entropy::GpuEntropyCalculator,
    gpu::sync::GpuSynchronizer,
    utils::error_recovery::GpuError,
};
use log::trace;
use std::fmt::Debug;
use std::sync::Arc;
use wfc_core::entropy::EntropyCalculator;
use wfc_core::grid::{EntropyGrid, PossibilityGrid};

/// A coordinate in 3D space (x, y, z).
/// Used since GridCoord only supports 2D coordinates (x, y)
#[derive(Debug, Clone, Copy)]
pub struct Coord3D {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

/// Strategy trait for entropy calculation coordination.
/// Implementations provide different approaches to calculating entropy
/// and selecting cells for the Wave Function Collapse algorithm.
#[async_trait::async_trait]
pub trait EntropyCoordinationStrategy: Debug + Send + Sync {
    /// Calculate entropy for the grid and select the next cell to collapse.
    ///
    /// # Arguments
    /// * `grid` - The current possibility grid
    /// * `buffers` - The GPU buffers containing grid state
    /// * `sync` - Synchronizer for GPU operations
    ///
    /// # Returns
    /// * `Ok(Some((entropy, coord)))` - The selected cell with its entropy value
    /// * `Ok(None)` - If the grid is fully collapsed or in contradiction
    /// * `Err(GpuError)` - If an error occurs during entropy calculation or selection
    async fn calculate_and_select(
        &self,
        grid: &PossibilityGrid,
        buffers: &Arc<GpuBuffers>,
        sync: &Arc<GpuSynchronizer>,
    ) -> Result<Option<(f32, Coord3D)>, GpuError>;

    /// Clone this strategy into a boxed trait object.
    fn clone_box(&self) -> Box<dyn EntropyCoordinationStrategy>;
}

impl Clone for Box<dyn EntropyCoordinationStrategy> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Factory for creating entropy coordination strategies.
pub struct EntropyCoordinationStrategyFactory;

impl EntropyCoordinationStrategyFactory {
    /// Create a default entropy coordination strategy.
    pub fn create_default(
        entropy_calculator: Arc<GpuEntropyCalculator>,
    ) -> Box<dyn EntropyCoordinationStrategy> {
        Box::new(DefaultEntropyCoordinationStrategy::new(entropy_calculator))
    }

    /// Create a strategy based on entropy heuristic.
    pub fn create_for_heuristic(
        entropy_calculator: Arc<GpuEntropyCalculator>,
        heuristic: &str,
    ) -> Box<dyn EntropyCoordinationStrategy> {
        match heuristic.to_lowercase().as_str() {
            "shannon" => Box::new(ShannonEntropyCoordinationStrategy::new(entropy_calculator)),
            "count" | "countbased" => Box::new(CountBasedEntropyCoordinationStrategy::new(
                entropy_calculator,
            )),
            _ => Box::new(DefaultEntropyCoordinationStrategy::new(entropy_calculator)),
        }
    }
}

/// Coordinator for entropy calculation and cell selection.
/// This type manages the process of calculating entropy and selecting
/// the next cell to collapse using the current entropy strategy.
#[derive(Debug, Clone)]
pub struct EntropyCoordinator {
    entropy_calculator: Arc<GpuEntropyCalculator>,
    strategy: Option<Box<dyn EntropyCoordinationStrategy>>,
}

impl EntropyCoordinator {
    /// Creates a new entropy coordinator
    pub fn new(entropy_calculator: Arc<GpuEntropyCalculator>) -> Self {
        Self {
            entropy_calculator,
            strategy: None,
        }
    }

    /// Set a specific entropy coordination strategy
    pub fn with_strategy<S: EntropyCoordinationStrategy + 'static>(mut self, strategy: S) -> Self {
        self.strategy = Some(Box::new(strategy));
        self
    }

    /// Calculate entropy grid using the current strategy
    pub async fn calculate_entropy(&self, grid: &PossibilityGrid) -> Result<EntropyGrid, GpuError> {
        // The entropy calculator delegates to the current strategy
        self.entropy_calculator
            .calculate_entropy(grid)
            .map_err(|e| GpuError::Other(format!("Entropy calculation failed: {}", e)))
    }

    /// Find the cell with minimum entropy using the current strategy
    pub async fn select_min_entropy_cell(
        &self,
        entropy_grid: &EntropyGrid,
    ) -> Result<Option<Coord3D>, GpuError> {
        // The entropy calculator delegates to the current strategy
        let result = self
            .entropy_calculator
            .select_lowest_entropy_cell(entropy_grid);

        Ok(result.map(|(x, y, z)| Coord3D { x, y, z }))
    }

    /// Combined operation to calculate entropy and select min-entropy cell
    pub async fn calculate_and_select(
        &self,
        grid: &PossibilityGrid,
    ) -> Result<(EntropyGrid, Option<Coord3D>), GpuError> {
        // Calculate entropy
        let entropy_grid = self.calculate_entropy(grid).await?;

        // Select min entropy cell
        let min_cell = self.select_min_entropy_cell(&entropy_grid).await?;

        Ok((entropy_grid, min_cell))
    }

    /// Directly download the min entropy information from GPU
    pub async fn download_min_entropy_info(
        &self,
        buffers: &Arc<GpuBuffers>,
        sync: &Arc<GpuSynchronizer>,
    ) -> Result<Option<(f32, Coord3D)>, GpuError> {
        // Use the strategy if available
        if let Some(ref strategy) = self.strategy {
            return strategy
                .calculate_and_select(
                    &PossibilityGrid::new(1, 1, 1, 1), // This is a placeholder, the strategy should use buffers directly
                    buffers,
                    sync,
                )
                .await;
        }

        // Fall back to default implementation
        trace!("EntropyCoordinator: Downloading entropy results...");

        let request = DownloadRequest {
            download_min_entropy_info: true,
            download_contradiction_flag: false,
            ..Default::default()
        };

        let results = buffers.download_results(request).await?;

        trace!(
            "EntropyCoordinator: Received min entropy info: {:?}",
            results.min_entropy_info
        );

        if let Some(min_data) = results.min_entropy_info {
            // Check if grid is fully collapsed
            if min_data.1 == u32::MAX {
                trace!(
                    "EntropyCoordinator: Grid appears to be fully collapsed or in contradiction"
                );
                return Ok(None);
            }

            let (width, height, _depth) = buffers.grid_dims;
            let flat_index = min_data.1 as usize;
            let z = flat_index / (width * height);
            let y = (flat_index % (width * height)) / width;
            let x = flat_index % width;

            trace!(
                "EntropyCoordinator: Selected cell at ({}, {}, {}) with entropy {}",
                x,
                y,
                z,
                min_data.0
            );

            Ok(Some((min_data.0, Coord3D { x, y, z })))
        } else {
            trace!("EntropyCoordinator: No min entropy info found, grid may be fully collapsed");
            Ok(None)
        }
    }

    pub async fn find_min_entropy_with_value(
        &self,
        buffers: &GpuBuffers,
        synchronizer: &GpuSynchronizer,
    ) -> Result<Option<(f32, Coord3D)>, GpuError> {
        // Create download request for the min entropy buffer
        let request = DownloadRequest {
            download_min_entropy_info: true,
            download_contradiction_flag: false,
            download_entropy: false,
            download_grid_possibilities: false,
            download_contradiction_location: false,
        };

        // Download result using the provided synchronizer
        let results = buffers.download_results(request).await?;

        // Check if min entropy info was found
        if let Some(min_data) = results.min_entropy_info {
            trace!("EntropyCoordinator: Min entropy info found: {:?}", min_data);

            // Check if the min index is valid (not u32::MAX which indicates no valid min)
            if min_data.1 != u32::MAX {
                let grid_dims = buffers.grid_dims;

                // Convert flat index to 3D coordinates
                let flat_index = min_data.1 as usize;
                let width = grid_dims.0;
                let height = grid_dims.1;
                let area = width * height;

                let z = flat_index / area;
                let y = (flat_index % area) / width;
                let x = flat_index % width;

                Ok(Some((min_data.0, Coord3D { x, y, z })))
            } else {
                trace!(
                    "EntropyCoordinator: No min entropy info found, grid may be fully collapsed"
                );
                Ok(None)
            }
        } else {
            // No min entropy data available
            let grid_dims = buffers.grid_dims;

            // Calculate grid size
            let width = grid_dims.0;
            let height = grid_dims.1;
            let depth = grid_dims.2;
            let x = 0;
            let y = 0;
            let z = 0;

            Ok(Some((0.0, Coord3D { x, y, z })))
        }
    }
}

/// The default implementation of the entropy coordination strategy.
#[derive(Debug, Clone)]
struct DefaultEntropyCoordinationStrategy {
    entropy_calculator: Arc<GpuEntropyCalculator>,
}

impl DefaultEntropyCoordinationStrategy {
    fn new(entropy_calculator: Arc<GpuEntropyCalculator>) -> Self {
        Self { entropy_calculator }
    }
}

#[async_trait::async_trait]
impl EntropyCoordinationStrategy for DefaultEntropyCoordinationStrategy {
    async fn calculate_and_select(
        &self,
        _grid: &PossibilityGrid,
        buffers: &Arc<GpuBuffers>,
        sync: &Arc<GpuSynchronizer>,
    ) -> Result<Option<(f32, Coord3D)>, GpuError> {
        trace!("DefaultEntropyCoordinationStrategy: Downloading entropy results...");

        let request = DownloadRequest {
            download_min_entropy_info: true,
            download_contradiction_flag: false,
            ..Default::default()
        };

        let results = buffers.download_results(request).await?;

        if let Some(min_data) = results.min_entropy_info {
            // Check if grid is fully collapsed
            if min_data.1 == u32::MAX {
                return Ok(None);
            }

            let (width, height, _depth) = buffers.grid_dims;
            let flat_index = min_data.1 as usize;
            let z = flat_index / (width * height);
            let y = (flat_index % (width * height)) / width;
            let x = flat_index % width;

            Ok(Some((min_data.0, Coord3D { x, y, z })))
        } else {
            Ok(None)
        }
    }

    fn clone_box(&self) -> Box<dyn EntropyCoordinationStrategy> {
        Box::new(self.clone())
    }
}

/// Shannon entropy-specific coordination strategy.
#[derive(Debug, Clone)]
struct ShannonEntropyCoordinationStrategy {
    entropy_calculator: Arc<GpuEntropyCalculator>,
}

impl ShannonEntropyCoordinationStrategy {
    fn new(entropy_calculator: Arc<GpuEntropyCalculator>) -> Self {
        Self { entropy_calculator }
    }
}

#[async_trait::async_trait]
impl EntropyCoordinationStrategy for ShannonEntropyCoordinationStrategy {
    async fn calculate_and_select(
        &self,
        _grid: &PossibilityGrid,
        buffers: &Arc<GpuBuffers>,
        sync: &Arc<GpuSynchronizer>,
    ) -> Result<Option<(f32, Coord3D)>, GpuError> {
        // Shannon entropy-specific implementation (similar to default for now)
        // In a full implementation, this would use Shannon-specific optimizations
        DefaultEntropyCoordinationStrategy::new(self.entropy_calculator.clone())
            .calculate_and_select(_grid, buffers, sync)
            .await
    }

    fn clone_box(&self) -> Box<dyn EntropyCoordinationStrategy> {
        Box::new(self.clone())
    }
}

/// Count-based entropy-specific coordination strategy.
#[derive(Debug, Clone)]
struct CountBasedEntropyCoordinationStrategy {
    entropy_calculator: Arc<GpuEntropyCalculator>,
}

impl CountBasedEntropyCoordinationStrategy {
    fn new(entropy_calculator: Arc<GpuEntropyCalculator>) -> Self {
        Self { entropy_calculator }
    }
}

#[async_trait::async_trait]
impl EntropyCoordinationStrategy for CountBasedEntropyCoordinationStrategy {
    async fn calculate_and_select(
        &self,
        _grid: &PossibilityGrid,
        buffers: &Arc<GpuBuffers>,
        sync: &Arc<GpuSynchronizer>,
    ) -> Result<Option<(f32, Coord3D)>, GpuError> {
        // Count-based entropy-specific implementation (similar to default for now)
        // In a full implementation, this would use count-based specific optimizations
        DefaultEntropyCoordinationStrategy::new(self.entropy_calculator.clone())
            .calculate_and_select(_grid, buffers, sync)
            .await
    }

    fn clone_box(&self) -> Box<dyn EntropyCoordinationStrategy> {
        Box::new(self.clone())
    }
}
