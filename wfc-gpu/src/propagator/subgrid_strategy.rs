use crate::{
    buffers::GpuBuffers, gpu::sync::GpuSynchronizer, propagator::AsyncPropagationStrategy,
    utils::error_recovery::GridCoord, utils::subgrid::SubgridRegion,
};
use async_trait;
use std::default::Default;
use std::sync::Arc;
use wfc_core::{grid::PossibilityGrid, propagator::PropagationError};

/// Subgrid propagation strategy - divides the grid into smaller subgrids
/// for more efficient parallel processing.
#[derive(Debug)]
pub struct SubgridPropagationStrategy {
    name: String,
    max_iterations: u32,
    subgrid_size: u32,
}

/// Parameters for processing a single subgrid
struct SubgridProcessContext {
    subgrid: PossibilityGrid,
    region: SubgridRegion,
    updated_coords: Vec<(usize, usize, usize)>,
    main_grid: PossibilityGrid,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl SubgridPropagationStrategy {
    /// Create a new subgrid propagation strategy
    pub fn new(max_iterations: u32, subgrid_size: u32) -> Self {
        Self {
            name: "Subgrid Propagation".to_string(),
            max_iterations,
            subgrid_size,
        }
    }

    /// Process a single subgrid
    async fn process_subgrid(
        &self,
        context: SubgridProcessContext,
    ) -> Result<PossibilityGrid, PropagationError> {
        // Create a struct for 3D coordinates
        #[derive(Debug, Clone, Copy)]
        struct LocalCoord3D {
            x: usize,
            y: usize,
            z: usize,
        }

        // Adjust the updated coordinates to be relative to the subgrid
        let adjusted_coords_3d: Vec<LocalCoord3D> = context
            .updated_coords
            .iter()
            .filter_map(|&(x, y, z)| {
                // Check if coordinate is within the region
                if x >= context.region.x_offset
                    && x < context.region.x_offset + context.region.width
                    && y >= context.region.y_offset
                    && y < context.region.y_offset + context.region.height
                    && z >= context.region.z_offset
                    && z < context.region.z_offset + context.region.depth
                {
                    // Convert to local coordinates within the subgrid
                    Some(LocalCoord3D {
                        x: x - context.region.x_offset,
                        y: y - context.region.y_offset,
                        z: z - context.region.z_offset,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Convert to GridCoord objects for propagation
        let adjusted_coords: Vec<GridCoord> = adjusted_coords_3d
            .iter()
            .map(|coord| GridCoord {
                x: coord.x,
                y: coord.y,
                z: coord.z,
            })
            .collect();

        if adjusted_coords.is_empty() {
            // No cells to update in this subgrid, return unmodified
            return Ok(context.subgrid);
        }

        // Create a mutable clone of the subgrid to work with
        let mut subgrid_mutable = context.subgrid.clone();

        // Extract the rules from the main grid
        // Since we don't have direct access to the AdjacencyRules,
        // we'll need to create a minimal one with the same dimensions
        let dummy_rules = wfc_rules::AdjacencyRules::from_allowed_tuples(
            context.main_grid.num_tiles(), // Assuming PossibilityGrid exposes this method
            1,                             // Using 1 axis for simplicity
            vec![],                        // No allowed tuples - minimal rules
        );

        // Create buffers for the subgrid
        let subgrid_buffers = Arc::new(
            GpuBuffers::new(
                &context.device,
                &context.queue,
                &subgrid_mutable,
                &dummy_rules, // Use minimal rules with same tile count
                wfc_core::BoundaryCondition::Finite, // Inside a subgrid, use finite boundaries
            )
            .map_err(|e| {
                PropagationError::InternalError(format!("Failed to create subgrid buffers: {}", e))
            })?,
        );

        // Create a synchronizer for the subgrid
        let subgrid_synchronizer = GpuSynchronizer::new(
            context.device.clone(),
            context.queue.clone(),
            subgrid_buffers.clone(),
        );

        // Create a direct propagation strategy for the subgrid
        let direct_strategy =
            crate::propagator::DirectPropagationStrategy::new(self.max_iterations);

        // Propagate constraints within the subgrid
        direct_strategy
            .propagate(
                &mut subgrid_mutable,
                &adjusted_coords,
                &subgrid_buffers,
                &subgrid_synchronizer,
            )
            .await?;

        // Return the processed subgrid
        Ok(subgrid_mutable)
    }
}

/// Implement Default trait for SubgridPropagationStrategy
impl Default for SubgridPropagationStrategy {
    fn default() -> Self {
        Self::new(1000, 32)
    }
}

impl crate::propagator::PropagationStrategy for SubgridPropagationStrategy {
    fn name(&self) -> &str {
        &self.name
    }

    fn prepare(&self, _synchronizer: &GpuSynchronizer) -> Result<(), PropagationError> {
        // Subgrid propagation doesn't need special preparation
        Ok(())
    }

    fn cleanup(&self, _synchronizer: &GpuSynchronizer) -> Result<(), PropagationError> {
        // Subgrid propagation doesn't need special cleanup
        Ok(())
    }
}

#[async_trait::async_trait]
impl crate::propagator::AsyncPropagationStrategy for SubgridPropagationStrategy {
    async fn propagate(
        &self,
        grid: &mut PossibilityGrid,
        updated_cells: &[GridCoord],
        _buffers: &Arc<GpuBuffers>,
        synchronizer: &GpuSynchronizer,
    ) -> Result<(), PropagationError> {
        use crate::utils::subgrid::{
            divide_into_subgrids, extract_subgrid, merge_subgrids, SubgridConfig,
        };
        use futures::future::join_all;

        log::debug!(
            "Using subgrid propagation strategy for grid size: {}x{}x{}",
            grid.width,
            grid.height,
            grid.depth
        );

        // Create a SubgridConfig from the strategy properties
        let config = SubgridConfig {
            max_subgrid_size: self.subgrid_size as usize,
            overlap_size: 2,                          // Default overlap size
            min_size: self.subgrid_size as usize / 2, // Minimum size threshold
        };

        // Divide the grid into subgrids
        let subgrid_regions = divide_into_subgrids(grid.width, grid.height, grid.depth, &config)
            .map_err(|e| {
                PropagationError::InternalError(format!(
                    "Failed to divide grid into subgrids: {}",
                    e
                ))
            })?;

        log::debug!("Divided grid into {} subgrids", subgrid_regions.len());

        // Convert updated cells to coordinates
        let updated_coords: Vec<(usize, usize, usize)> = updated_cells
            .iter()
            .map(|cell| (cell.x, cell.y, cell.z))
            .collect();

        // Create a mutable synchronizer for subgrid operations
        let device = synchronizer.device().clone();
        let queue = synchronizer.queue().clone();

        // Process each subgrid in parallel using futures
        let mut futures = Vec::with_capacity(subgrid_regions.len());

        for region in &subgrid_regions {
            // Extract subgrid from the main grid
            let subgrid = extract_subgrid(grid, region).map_err(|e| {
                PropagationError::InternalError(format!("Failed to extract subgrid: {}", e))
            })?;

            log::debug!(
                "Processing subgrid: x={}-{}, y={}-{}, z={}-{}",
                region.x_offset,
                region.x_offset + region.width,
                region.y_offset,
                region.y_offset + region.height,
                region.z_offset,
                region.z_offset + region.depth
            );

            // Create context for subgrid processing
            let context = SubgridProcessContext {
                subgrid,
                region: *region,
                updated_coords: updated_coords.clone(),
                main_grid: grid.clone(),
                device: device.clone(),
                queue: queue.clone(),
            };

            // Create a future for processing this subgrid
            let future = self.process_subgrid(context);
            futures.push(future);
        }

        // Await all subgrid processing futures to complete
        let results = join_all(futures).await;

        // Collect the processed subgrids
        let mut subgrid_results = Vec::with_capacity(subgrid_regions.len());

        for (i, result) in results.into_iter().enumerate() {
            match result {
                Ok(processed_subgrid) => {
                    subgrid_results.push((subgrid_regions[i], processed_subgrid));
                }
                Err(error) => {
                    return Err(error);
                }
            }
        }

        // Merge results back into the main grid
        log::debug!(
            "Merging {} subgrid results back into main grid",
            subgrid_results.len()
        );

        let updated_cells = merge_subgrids(grid, &subgrid_results, &config).map_err(|e| {
            PropagationError::InternalError(format!("Failed to merge subgrids: {}", e))
        })?;

        log::debug!(
            "Subgrid propagation complete, updated {} cells",
            updated_cells.len()
        );

        Ok(())
    }
}
