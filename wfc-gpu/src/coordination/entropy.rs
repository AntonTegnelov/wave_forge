//! Module responsible for coordinating entropy calculation and min-entropy cell selection.

use crate::{
    buffers::{DownloadRequest, GpuBuffers},
    entropy::{EntropyStrategy, GpuEntropyCalculator},
    error_recovery::{GpuError, GridCoord},
    gpu::sync::GpuSynchronizer,
};
use log::trace;
use std::sync::Arc;
use wfc_core::entropy::EntropyCalculator;
use wfc_core::grid::EntropyGrid;

/// Coordinator for entropy calculation and cell selection.
/// This type manages the process of calculating entropy and selecting
/// the next cell to collapse using the current entropy strategy.
#[derive(Debug, Clone)]
pub struct EntropyCoordinator {
    entropy_calculator: Arc<GpuEntropyCalculator>,
}

impl EntropyCoordinator {
    /// Creates a new entropy coordinator
    pub fn new(entropy_calculator: Arc<GpuEntropyCalculator>) -> Self {
        Self { entropy_calculator }
    }

    /// Calculate entropy grid using the current strategy
    pub async fn calculate_entropy(
        &self,
        grid: &wfc_core::grid::PossibilityGrid,
    ) -> Result<EntropyGrid, GpuError> {
        // The entropy calculator delegates to the current strategy
        self.entropy_calculator
            .calculate_entropy(grid)
            .map_err(|e| GpuError::Other(format!("Entropy calculation failed: {}", e)))
    }

    /// Find the cell with minimum entropy using the current strategy
    pub async fn select_min_entropy_cell(
        &self,
        entropy_grid: &EntropyGrid,
    ) -> Result<Option<GridCoord>, GpuError> {
        // The entropy calculator delegates to the current strategy
        let result = self
            .entropy_calculator
            .select_lowest_entropy_cell(entropy_grid);

        // Convert the result to GridCoord format
        Ok(result.map(|(x, y, z)| GridCoord { x, y, z }))
    }

    /// Combined operation to calculate entropy and select min-entropy cell
    pub async fn calculate_and_select(
        &self,
        grid: &wfc_core::grid::PossibilityGrid,
    ) -> Result<(EntropyGrid, Option<GridCoord>), GpuError> {
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
    ) -> Result<Option<(f32, GridCoord)>, GpuError> {
        trace!("EntropyCoordinator: Downloading entropy results...");

        let request = DownloadRequest {
            download_min_entropy_info: true,
            download_contradiction_flag: false,
            ..Default::default()
        };

        let results = buffers.download_results(request).await?;

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

            // Convert to GridCoord
            Ok(Some((min_data.0, GridCoord { x, y, z })))
        } else {
            // No min entropy info found
            trace!("No min entropy info found, grid may be fully collapsed");
            Ok(None)
        }
    }
}
