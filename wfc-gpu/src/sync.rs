// Synchronization module that handles GPU-CPU data transfer and resource management
// This separates GPU synchronization concerns from algorithm logic

use crate::{
    buffers::{EntropyParamsUniform, GpuBuffers, GpuParamsUniform},
    GpuError,
};
use log::{debug, trace};
use std::sync::Arc;
use wfc_core::grid::PossibilityGrid;
use wfc_rules::AdjacencyRules;
use wgpu::util::DeviceExt;

/// Handles synchronization between CPU and GPU for Wave Function Collapse data.
///
/// This struct manages the transfer of grid states, rules, and other data between
/// CPU memory and GPU buffers, separating these synchronization concerns from the
/// algorithm logic that operates on the data.
#[derive(Debug, Clone)]
pub struct GpuSynchronizer {
    /// WGPU device for executing GPU operations.
    device: Arc<wgpu::Device>,
    /// WGPU command queue for submitting work.
    queue: Arc<wgpu::Queue>,
    /// Shared GPU buffers for storing data.
    buffers: Arc<GpuBuffers>,
}

impl GpuSynchronizer {
    /// Creates a new GPU synchronizer.
    ///
    /// # Arguments
    ///
    /// * `device` - The WGPU device to use for GPU operations.
    /// * `queue` - The WGPU queue for submitting commands.
    /// * `buffers` - The GPU buffers to use for data storage.
    ///
    /// # Returns
    ///
    /// A new `GpuSynchronizer` instance.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        buffers: Arc<GpuBuffers>,
    ) -> Self {
        Self {
            device,
            queue,
            buffers,
        }
    }

    /// Uploads the grid possibilities from CPU to GPU.
    ///
    /// # Arguments
    ///
    /// * `grid` - The grid containing cell possibilities to upload.
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, or an error detailing what went wrong.
    pub fn upload_grid(&self, grid: &PossibilityGrid) -> Result<(), GpuError> {
        trace!("Uploading grid possibilities to GPU");
        let (width, height, depth) = (grid.width, grid.height, grid.depth);
        let num_cells = width * height * depth;
        let u32s_per_cell = self.buffers.u32s_per_cell;

        // Convert grid possibilities to u32 arrays
        let mut packed_data = Vec::with_capacity(num_cells * u32s_per_cell);

        // For each cell, get its bitvector and pack it into u32s
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    if let Some(cell) = grid.get(x, y, z) {
                        let mut cell_data = vec![0u32; u32s_per_cell];
                        for (i, bit) in cell.iter().enumerate() {
                            if *bit {
                                let u32_idx = i / 32;
                                let bit_idx = i % 32;
                                if u32_idx < cell_data.len() {
                                    cell_data[u32_idx] |= 1 << bit_idx;
                                }
                            }
                        }
                        packed_data.extend_from_slice(&cell_data);
                    } else {
                        // If cell doesn't exist, add empty data
                        packed_data.extend(std::iter::repeat(0).take(u32s_per_cell));
                    }
                }
            }
        }

        // Upload the data to the GPU buffer
        self.queue.write_buffer(
            &self.buffers.grid_possibilities_buf,
            0,
            bytemuck::cast_slice(&packed_data),
        );

        Ok(())
    }

    /// Downloads the grid possibilities from GPU to CPU.
    ///
    /// # Arguments
    ///
    /// * `grid_template` - A template grid with the correct dimensions to download into.
    ///
    /// # Returns
    ///
    /// `Ok(grid)` with the downloaded data if successful, or an error detailing what went wrong.
    pub async fn download_grid(
        &self,
        grid_template: &PossibilityGrid,
    ) -> Result<PossibilityGrid, GpuError> {
        trace!("Downloading grid possibilities from GPU");

        let (width, height, depth) = (
            grid_template.width,
            grid_template.height,
            grid_template.depth,
        );
        let num_cells = width * height * depth;
        let num_tiles = self.buffers.num_tiles;
        let u32s_per_cell = self.buffers.u32s_per_cell;

        // Create a new grid with the same dimensions
        let mut grid = PossibilityGrid::new(width, height, depth, num_tiles);

        // Download the data from the GPU buffer using download_results
        let results = self
            .buffers
            .download_results(
                self.device.clone(),
                self.queue.clone(),
                false, // Don't download entropy
                false, // Don't download min entropy info
                true,  // Download grid possibilities
                false, // Don't download worklist
                false, // Don't download worklist size
                false, // Don't download contradiction location
            )
            .await
            .map_err(|e| GpuError::TransferError(format!("Failed to download grid data: {}", e)))?;

        // Extract grid possibilities from results
        let grid_possibilities = match results.grid_possibilities {
            Some(possibilities) => possibilities,
            None => {
                return Err(GpuError::BufferOperationError(
                    "Failed to download grid possibilities from GPU".to_string(),
                ))
            }
        };

        // Check if we have enough data
        if grid_possibilities.len() < num_cells * u32s_per_cell {
            return Err(GpuError::BufferSizeMismatch(format!(
                "Grid data mismatch: expected {} u32s, got {}",
                num_cells * u32s_per_cell,
                grid_possibilities.len()
            )));
        }

        // Process the downloaded data
        let mut cell_index = 0;
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    if let Some(cell) = grid.get_mut(x, y, z) {
                        // Clear the cell's initial state (all 1s)
                        cell.fill(false);

                        // Set the bits according to the GPU data
                        let base_index = cell_index * u32s_per_cell;
                        for i in 0..u32s_per_cell {
                            if base_index + i < grid_possibilities.len() {
                                let bits = grid_possibilities[base_index + i];
                                for bit_pos in 0..32 {
                                    let tile_idx = i * 32 + bit_pos;
                                    if tile_idx < num_tiles {
                                        let bit_value = ((bits >> bit_pos) & 1) == 1;
                                        if bit_value {
                                            cell.set(tile_idx, true);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    cell_index += 1;
                }
            }
        }

        Ok(grid)
    }

    /// Updates the entropy parameters on the GPU.
    ///
    /// # Arguments
    ///
    /// * `grid_dims` - The dimensions of the grid.
    /// * `heuristic_type` - The entropy heuristic type to use.
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, or an error detailing what went wrong.
    pub fn update_entropy_params(
        &self,
        grid_dims: (usize, usize, usize),
        heuristic_type: u32,
    ) -> Result<(), GpuError> {
        let (width, height, depth) = grid_dims;

        let entropy_params = EntropyParamsUniform {
            grid_width: width as u32,
            grid_height: height as u32,
            grid_depth: depth as u32,
            _padding1: 0,
            heuristic_type,
            _padding2: 0,
            _padding3: 0,
            _padding4: 0,
        };

        // Write entropy parameters to buffer
        self.queue.write_buffer(
            &self.buffers.entropy_params_buffer,
            0,
            bytemuck::cast_slice(&[entropy_params]),
        );

        Ok(())
    }

    /// Updates the propagation parameters on the GPU.
    ///
    /// # Arguments
    ///
    /// * `params` - The propagation parameters to upload.
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, or an error detailing what went wrong.
    pub fn update_propagation_params(&self, params: &GpuParamsUniform) -> Result<(), GpuError> {
        self.queue.write_buffer(
            &self.buffers.params_uniform_buf,
            0,
            bytemuck::cast_slice(&[params.clone()]),
        );

        Ok(())
    }

    /// Uploads the updated cell indices for propagation.
    ///
    /// # Arguments
    ///
    /// * `updated_indices` - The indices of cells that have been updated.
    /// * `worklist_idx` - The worklist buffer index to use (for ping-pong buffers).
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, or an error detailing what went wrong.
    pub fn upload_updated_indices(
        &self,
        updated_indices: &[u32],
        worklist_idx: usize,
    ) -> Result<(), GpuError> {
        self.buffers
            .upload_initial_updates(&self.queue, updated_indices, worklist_idx)
    }

    /// Downloads the minimum entropy information from the GPU.
    ///
    /// # Returns
    ///
    /// `Ok(min_info)` with the minimum entropy data if successful, or an error detailing what went wrong.
    pub async fn download_min_entropy_info(&self) -> Result<Option<(f32, u32)>, GpuError> {
        // Download the min entropy information using download_results
        let results = self
            .buffers
            .download_results(
                self.device.clone(),
                self.queue.clone(),
                false, // Don't download entropy
                true,  // Download min entropy info
                false, // Don't download grid possibilities
                false, // Don't download worklist
                false, // Don't download worklist size
                false, // Don't download contradiction location
            )
            .await
            .map_err(|e| {
                GpuError::TransferError(format!("Failed to download min entropy info: {}", e))
            })?;

        Ok(results.min_entropy_info)
    }

    /// Resets the minimum entropy buffer on the GPU.
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, or an error detailing what went wrong.
    pub fn reset_min_entropy_buffer(&self) -> Result<(), GpuError> {
        self.buffers.reset_min_entropy_info(&self.queue)
    }

    /// Creates a command encoder for GPU operations.
    ///
    /// # Arguments
    ///
    /// * `label` - An optional label for the command encoder.
    ///
    /// # Returns
    ///
    /// A new wgpu command encoder.
    pub fn create_command_encoder(&self, label: Option<&str>) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label })
    }

    /// Submits a command encoder to the GPU queue.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder to submit.
    pub fn submit_commands(&self, encoder: wgpu::CommandEncoder) {
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Gets a reference to the GPU device.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Gets a reference to the GPU queue.
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// Gets a reference to the GPU buffers.
    pub fn buffers(&self) -> &Arc<GpuBuffers> {
        &self.buffers
    }
}

impl Drop for GpuSynchronizer {
    /// Performs cleanup of GPU synchronizer resources when dropped.
    ///
    /// This properly handles the cleanup of GPU-related resources to prevent leaks.
    /// The actual device and queue resources are cleaned up when their Arc references
    /// are dropped.
    fn drop(&mut self) {
        // Log the cleanup for debugging purposes
        debug!("GpuSynchronizer being dropped, releasing references to GPU resources");

        // The actual cleanup happens automatically through Arc's reference counting
        // when these fields are dropped. No explicit cleanup needed here.

        // This is here to make the cleanup process explicit in the code, following RAII principles
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wfc_core::BoundaryCondition;

    // Helper function to create a test grid
    fn create_test_grid(
        width: usize,
        height: usize,
        depth: usize,
        num_tiles: usize,
    ) -> PossibilityGrid {
        let mut grid = PossibilityGrid::new(width, height, depth, num_tiles);

        // Set some specific patterns for testing
        if let Some(cell) = grid.get_mut(0, 0, 0) {
            cell.fill(false);
            cell.set(0, true);
        }

        if let Some(cell) = grid.get_mut(1, 1, 0) {
            cell.fill(false);
            cell.set(1, true);
        }

        grid
    }

    // Test setup to create necessary GPU resources for testing
    async fn setup_test_gpu(
    ) -> Result<(Arc<wgpu::Device>, Arc<wgpu::Queue>, Arc<GpuBuffers>), GpuError> {
        // This test would require a GPU device, which may not be available in all testing environments
        // In a real test, we would create actual GPU resources, but for this example we'll just
        // return a placeholder

        // For actual implementation, this would create GPU resources similar to GpuAccelerator::new()
        unimplemented!("GPU test setup not implemented for unit tests");
    }

    #[tokio::test]
    #[ignore] // Ignore by default since it requires GPU
    async fn test_grid_upload_download() {
        // This test would verify that uploading and downloading a grid preserves its state
        // It would require an actual GPU device

        // Example test flow:
        // 1. Create a test grid with known pattern
        // 2. Set up GPU resources
        // 3. Create synchronizer
        // 4. Upload grid
        // 5. Download grid
        // 6. Verify the downloaded grid matches the original
    }
}
