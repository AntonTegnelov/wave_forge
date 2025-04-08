// Synchronization module that handles GPU-CPU data transfer and resource management
// This separates GPU synchronization concerns from algorithm logic

use crate::{
    buffers::{DownloadRequest, GpuBuffers, GpuEntropyShaderParams, GpuParamsUniform},
    GpuError,
};
use log::{debug, trace, warn};
use std::sync::Arc;
use wfc_core::grid::PossibilityGrid;
use wgpu;

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

// Basic Default impl for testing/convenience in DebugVisualizer
// This will likely panic if used without proper initialization
impl Default for GpuSynchronizer {
    fn default() -> Self {
        // This is unsafe and should only be used where the actual
        // device/queue/buffers are provided later or not used.
        // Consider using Option<Arc<...>> instead if feasible.
        panic!("Default GpuSynchronizer created without real resources");
        // // Dummy implementations - replace with proper handling or remove Default
        // let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        // let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).unwrap();
        // let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).unwrap();
        // let dummy_grid = PossibilityGrid::new(1, 1, 1, 1);
        // let dummy_rules = AdjacencyRules::new_uniform(1, 6);
        // let buffers = GpuBuffers::new(&device, &queue, &dummy_grid, &dummy_rules, BoundaryCondition::Finite).unwrap();
        // Self {
        //     device: Arc::new(device),
        //     queue: Arc::new(queue),
        //     buffers: Arc::new(buffers),
        // }
    }
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
        let u32s_per_cell = self.buffers.grid_buffers.u32s_per_cell;

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
            &self.buffers.grid_buffers.grid_possibilities_buf,
            0,
            bytemuck::cast_slice(&packed_data),
        );

        Ok(())
    }

    /// Downloads the full possibility grid state from the GPU.
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
        let u32s_per_cell = self.buffers.grid_buffers.u32s_per_cell;

        // Create a new grid with the same dimensions
        let mut grid = PossibilityGrid::new(width, height, depth, num_tiles);

        // Download the data from the GPU buffer using download_results
        let request = DownloadRequest {
            download_entropy: false,
            download_min_entropy_info: false,
            download_grid_possibilities: true,
            download_contradiction_location: false,
        };
        let results = self
            .buffers
            .download_results(self.device.clone(), self.queue.clone(), request)
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

    /// Downloads the contradiction status (flag and location) from the GPU.
    ///
    /// This is an optimized method that combines multiple status downloads into a single operation
    /// to reduce synchronization points.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * `has_contradiction` - Boolean indicating if a contradiction was detected
    /// * `contradiction_location` - Optional index of the cell where a contradiction occurred
    pub async fn download_contradiction_status(&self) -> Result<(bool, Option<u32>), GpuError> {
        // Create encoder for copying buffers
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Contradiction Status Encoder"),
            });

        // Copy the contradiction flag to staging buffer
        encoder.copy_buffer_to_buffer(
            &self.buffers.contradiction_flag_buf,
            0,
            &self.buffers.staging_contradiction_flag_buf,
            0,
            self.buffers.contradiction_flag_buf.size(),
        );

        // Copy the contradiction location to staging buffer
        encoder.copy_buffer_to_buffer(
            &self.buffers.contradiction_location_buf,
            0,
            &self.buffers.staging_contradiction_location_buf,
            0,
            self.buffers.contradiction_location_buf.size(),
        );

        self.queue.submit(Some(encoder.finish()));

        // Download and map the flag buffer
        let flag_buffer = self.buffers.staging_contradiction_flag_buf.clone();
        let flag_result = crate::buffers::map_and_process::<u32>(
            flag_buffer,
            std::time::Duration::from_secs(5),
            Some(1),
        )
        .await;

        let flag = match flag_result {
            Ok(data_box) => data_box
                .downcast::<Vec<u32>>()
                .map_err(|_| GpuError::InternalError("Failed to downcast flag".into()))?
                .get(0)
                .cloned()
                .unwrap_or(0),
            Err(e) => {
                return Err(GpuError::BufferOperationError(format!(
                    "Flag map failed: {}",
                    e
                )))
            }
        };

        let has_contradiction = flag != 0;
        let mut contradiction_location = None;

        if has_contradiction {
            // Download and map the location buffer
            let loc_buffer = self.buffers.staging_contradiction_location_buf.clone();
            let loc_result = crate::buffers::map_and_process::<u32>(
                loc_buffer,
                std::time::Duration::from_secs(5),
                Some(1),
            )
            .await;

            contradiction_location = match loc_result {
                Ok(data_box) => data_box
                    .downcast::<Vec<u32>>()
                    .map_err(|_| GpuError::InternalError("Failed to downcast loc".into()))?
                    .get(0)
                    .cloned(),
                Err(e) => {
                    warn!("Contradiction flag set, but failed to read location: {}", e);
                    None // Proceed without location if read fails
                }
            };
        }

        Ok((has_contradiction, contradiction_location))
    }

    /// Downloads the worklist count from the GPU.
    ///
    /// # Returns
    ///
    /// `Ok(count)` with the worklist count if successful, or an error detailing what went wrong.
    pub async fn download_worklist_count(&self) -> Result<u32, GpuError> {
        // Determine which count buffer to download (assuming propagator manages index)
        // We need access to the propagator's state or need to coordinate the index.
        // For now, let's assume we download from a specific buffer (e.g., A) - THIS IS LIKELY WRONG
        // TODO: Rework this logic based on how propagator state is managed.
        warn!("download_worklist_count logic needs review regarding ping-pong buffers");
        let count_buffer_gpu = &self.buffers.worklist_buffers.worklist_count_buf; // Use single count buffer
        let count_buffer_staging = &self.buffers.worklist_buffers.staging_worklist_count_buf; // Use single staging count buffer

        // Copy
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Worklist Count Copy"),
            });
        encoder.copy_buffer_to_buffer(count_buffer_gpu, 0, count_buffer_staging, 0, 4);
        self.queue.submit(Some(encoder.finish()));

        // Map and Read
        let count_result = crate::buffers::map_and_process::<u32>(
            count_buffer_staging.clone(),
            std::time::Duration::from_secs(5),
            Some(1),
        )
        .await;

        match count_result {
            Ok(data_box) => Ok(*data_box
                .downcast::<Vec<u32>>()
                .map_err(|_| GpuError::InternalError("Failed to downcast count".into()))?
                .get(0)
                .unwrap_or(&0)),
            Err(e) => Err(GpuError::BufferOperationError(format!(
                "Count map failed: {}",
                e
            ))),
        }
    }

    /// Updates the entropy parameters uniform buffer on the GPU.
    pub fn update_entropy_params(
        &self,
        grid_dims: (usize, usize, usize),
        heuristic_type: u32,
    ) -> Result<(), GpuError> {
        let num_tiles = self.buffers.num_tiles;
        let u32s_per_cell = self.buffers.grid_buffers.u32s_per_cell; // Access via grid_buffers

        let params = GpuEntropyShaderParams {
            grid_dims: [grid_dims.0 as u32, grid_dims.1 as u32, grid_dims.2 as u32],
            heuristic_type,
            num_tiles: num_tiles as u32,
            u32s_per_cell: u32s_per_cell as u32,
            _padding1: 0,
            _padding2: 0,
        };

        self.queue.write_buffer(
            &self.buffers.entropy_params_buffer, // Direct access ok
            0,
            bytemuck::cast_slice(&[params]),
        );
        Ok(())
    }

    /// Updates the propagation parameters uniform buffer on the GPU.
    pub fn update_propagation_params(&self, params: &GpuParamsUniform) -> Result<(), GpuError> {
        self.queue.write_buffer(
            &self.buffers.params_uniform_buf, // Direct access ok
            0,
            bytemuck::cast_slice(&[*params]),
        );
        Ok(())
    }

    /// Uploads updated cell indices to the specified worklist buffer.
    pub fn upload_updated_indices(
        &self,
        updated_indices: &[u32],
        worklist_idx: usize,
    ) -> Result<(), GpuError> {
        let worklist_buffer = if worklist_idx == 0 {
            &self.buffers.worklist_buffers.worklist_buf_a // Use worklist_buffers
        } else {
            &self.buffers.worklist_buffers.worklist_buf_b // Use worklist_buffers
        };
        let count_buffer = &self.buffers.worklist_buffers.worklist_count_buf; // Use single count buffer

        let data_size = (updated_indices.len() * std::mem::size_of::<u32>()) as u64;

        // Ensure buffer is large enough (use worklist_buffers method)
        // Note: This might require mutable access or coordination if called concurrently
        // self.buffers.worklist_buffers.ensure_buffer_size(&self.device, data_size, config)?;

        if worklist_buffer.size() < data_size {
            return Err(GpuError::BufferSizeMismatch(
                "Worklist buffer too small".to_string(),
            ));
        }

        self.queue
            .write_buffer(worklist_buffer, 0, bytemuck::cast_slice(updated_indices));
        self.queue.write_buffer(
            count_buffer,
            0,
            bytemuck::cast_slice(&[updated_indices.len() as u32]),
        );
        Ok(())
    }

    /// Uploads initial updated cell indices and sets the worklist size.
    pub fn upload_initial_updates(
        &self,
        updated_indices: &[u32],
        worklist_idx: usize,
    ) -> Result<(), GpuError> {
        trace!(
            "Uploading initial worklist ({} items) to GPU",
            updated_indices.len()
        );
        self.upload_updated_indices(updated_indices, worklist_idx)
    }

    /// Downloads the minimum entropy info (value and index) from the GPU.
    pub async fn download_min_entropy_info(&self) -> Result<Option<(f32, u32)>, GpuError> {
        // Use GpuBuffers::download_results
        let request = DownloadRequest {
            download_entropy: false,
            download_min_entropy_info: true,
            download_grid_possibilities: false,
            download_contradiction_location: false,
        };
        let results = self
            .buffers
            .download_results(self.device.clone(), self.queue.clone(), request)
            .await
            .map_err(|e| {
                GpuError::TransferError(format!("Failed to download min entropy: {}", e))
            })?;

        Ok(results.min_entropy_info)
    }

    /// Downloads pass statistics from the GPU.
    pub async fn download_pass_statistics(&self) -> Result<[u32; 4], GpuError> {
        let staging_buf = &self.buffers.staging_pass_statistics_buf; // Direct access ok
        let gpu_buf = &self.buffers.pass_statistics_buf; // Direct access ok

        // Copy
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Stats Copy"),
            });
        encoder.copy_buffer_to_buffer(gpu_buf, 0, staging_buf, 0, 16);
        self.queue.submit(Some(encoder.finish()));

        // Map and Read
        let stats_result = crate::buffers::map_and_process::<u32>(
            staging_buf.clone(),
            std::time::Duration::from_secs(5),
            Some(4),
        )
        .await;

        match stats_result {
            Ok(data_box) => {
                let data = data_box
                    .downcast::<Vec<u32>>()
                    .map_err(|_| GpuError::InternalError("Failed to downcast stats".into()))?;
                if data.len() >= 4 {
                    Ok([data[0], data[1], data[2], data[3]])
                } else {
                    Err(GpuError::BufferSizeMismatch(
                        "Stats data too short".to_string(),
                    ))
                }
            }
            Err(e) => Err(GpuError::BufferOperationError(format!(
                "Stats map failed: {}",
                e
            ))),
        }
    }

    /// Resets the minimum entropy buffer on the GPU.
    pub fn reset_min_entropy_buffer(&self) -> Result<(), GpuError> {
        // let mut encoder = self // Encoder not needed, write_buffer used directly
        //     .device
        //     .create_command_encoder(&wgpu::CommandEncoderDescriptor {
        //         label: Some("Reset Min Entropy"),
        //     });
        // Reset to [f32::MAX.to_bits(), u32::MAX]
        let reset_data = [f32::MAX.to_bits(), u32::MAX];
        self.queue.write_buffer(
            &self.buffers.entropy_buffers.min_entropy_info_buf, // Use entropy_buffers
            0,
            bytemuck::cast_slice(&reset_data),
        );
        // No need to submit here, can be batched
        Ok(())
    }

    /// Resets the contradiction flag buffer on the GPU.
    pub fn reset_contradiction_flag(&self) -> Result<(), GpuError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Reset Contradiction Flag"),
            });
        encoder.clear_buffer(&self.buffers.contradiction_flag_buf, 0, None); // Direct access ok
        self.submit_commands(encoder);
        Ok(())
    }

    /// Resets the contradiction location buffer on the GPU.
    pub fn reset_contradiction_location(&self) -> Result<(), GpuError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Reset Contradiction Location"),
            });
        encoder.clear_buffer(&self.buffers.contradiction_location_buf, 0, None); // Direct access ok
        self.submit_commands(encoder);
        Ok(())
    }

    /// Resets both worklist count buffers on the GPU.
    pub fn reset_worklist_count(&self) -> Result<(), GpuError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Reset Worklist Counts"),
            });
        encoder.clear_buffer(&self.buffers.worklist_buffers.worklist_count_buf, 0, None); // Use single count buffer
                                                                                          // Remove redundant clear for non-existent buffer B
        self.submit_commands(encoder);
        Ok(())
    }

    /// Resets grid possibilities, worklist, and contradiction state on the GPU.
    pub fn reset_grid_state(
        &self,
        initial_grid: &PossibilityGrid,
        updates: &[u32],
        active_worklist_idx: usize,
    ) -> Result<(), GpuError> {
        self.upload_grid(initial_grid)?;
        self.reset_contradiction_flag()?;
        self.reset_contradiction_location()?;
        self.reset_worklist_count()?; // Reset both counts
        self.upload_initial_updates(updates, active_worklist_idx)?; // Upload to active list
        Ok(())
    }

    /// Updates the worklist size parameter in the propagation params uniform buffer.
    pub fn update_params_worklist_size(&self, worklist_size: u32) -> Result<(), GpuError> {
        // Read current params
        // This requires a download which is slow. Consider passing full params instead.
        warn!("update_params_worklist_size is inefficient due to read-modify-write");

        // Placeholder: Assume we have the current params or can reconstruct them.
        // This function might need to be removed or redesigned.
        let mut current_params = GpuParamsUniform::default(); // DUMMY
        current_params.worklist_size = worklist_size;

        self.queue.write_buffer(
            &self.buffers.params_uniform_buf, // Direct access ok
            0,
            bytemuck::cast_slice(&[current_params]),
        );

        Ok(())
    }

    /// Uploads entropy shader specific parameters.
    pub fn upload_entropy_params(&self, params: &GpuEntropyShaderParams) -> Result<(), GpuError> {
        self.queue.write_buffer(
            &self.buffers.entropy_params_buffer, // Direct access ok
            0,
            bytemuck::cast_slice(&[*params]),
        );
        Ok(())
    }

    /// Internal helper to copy grid possibilities to staging buffer.
    pub(crate) fn stage_grid_possibilities_download(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), GpuError> {
        encoder.copy_buffer_to_buffer(
            &self.buffers.grid_buffers.grid_possibilities_buf, // Use grid_buffers
            0,
            &self.buffers.grid_buffers.staging_grid_possibilities_buf, // Use grid_buffers
            0,
            self.buffers.grid_buffers.grid_possibilities_buf.size(), // Use grid_buffers
        );
        Ok(())
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
    ///
    /// # Returns
    ///
    /// * `()`
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

    // Helper function to create a test grid
    #[allow(dead_code)]
    fn create_test_grid(
        width: usize,
        height: usize,
        depth: usize,
        num_transformed_tiles: usize,
    ) -> PossibilityGrid {
        let mut grid = PossibilityGrid::new(width, height, depth, num_transformed_tiles);

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

    // Helper to create a test GPU environment (async)
    #[allow(dead_code)]
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
