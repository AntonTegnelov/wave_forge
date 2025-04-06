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
        let request = DownloadRequest {
            download_entropy: false,
            download_min_entropy_info: false,
            download_grid_possibilities: true,
            download_worklist_size: false,
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

        self.queue.submit(Some(encoder.finish()));

        // Download and map the flag buffer
        let flag_buffer = self.buffers.staging_contradiction_flag_buf.clone();
        let flag_slice = flag_buffer.slice(..);
        let (flag_tx, flag_rx) = tokio::sync::oneshot::channel();

        flag_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = flag_tx.send(result);
        });

        self.device.poll(wgpu::Maintain::Wait); // Ensure copy completes

        flag_rx.await.map_err(|_| {
            GpuError::BufferOperationError("Mapping channel canceled for flag".to_string())
        })??; // Handle channel send error and map_async error

        let flag_mapped_range = flag_slice.get_mapped_range();
        let flag_data = flag_mapped_range.to_vec();
        drop(flag_mapped_range);
        flag_buffer.unmap();

        let has_contradiction = if flag_data.len() >= 4 {
            u32::from_le_bytes(flag_data[0..4].try_into().unwrap()) != 0
        } else {
            false // Default to false if buffer read failed
        };

        // If there's a contradiction, download the location
        let contradiction_location = if has_contradiction {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Contradiction Location Encoder"),
                });

            encoder.copy_buffer_to_buffer(
                &self.buffers.contradiction_location_buf,
                0,
                &self.buffers.staging_contradiction_location_buf,
                0,
                self.buffers.contradiction_location_buf.size(),
            );

            self.queue.submit(Some(encoder.finish()));

            let loc_buffer = self.buffers.staging_contradiction_location_buf.clone();
            let loc_slice = loc_buffer.slice(..);
            let (loc_tx, loc_rx) = tokio::sync::oneshot::channel();

            loc_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = loc_tx.send(result);
            });

            self.device.poll(wgpu::Maintain::Wait); // Ensure copy completes

            loc_rx.await.map_err(|_| {
                GpuError::BufferOperationError("Mapping channel canceled for loc".to_string())
            })??;

            let loc_mapped_range = loc_slice.get_mapped_range();
            let loc_data = loc_mapped_range.to_vec();
            drop(loc_mapped_range);
            loc_buffer.unmap();

            if loc_data.len() >= 4 {
                Some(u32::from_le_bytes(loc_data[0..4].try_into().unwrap()))
            } else {
                warn!("Contradiction detected, but failed to read location buffer.");
                None // Failed to read location
            }
        } else {
            None // No contradiction
        };

        Ok((has_contradiction, contradiction_location))
    }

    /// Downloads the worklist count from the GPU.
    ///
    /// # Returns
    ///
    /// `Ok(count)` with the worklist count if successful, or an error detailing what went wrong.
    pub async fn download_worklist_count(&self) -> Result<u32, GpuError> {
        let buffer_size = self.buffers.worklist_count_buf.size();
        if buffer_size < 4 {
            return Err(GpuError::BufferSizeMismatch(
                "Worklist count buffer too small".to_string(),
            ));
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Worklist Count Download Encoder"),
            });

        encoder.copy_buffer_to_buffer(
            &self.buffers.worklist_count_buf,
            0,
            &self.buffers.staging_worklist_count_buf,
            0,
            buffer_size,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = self.buffers.staging_worklist_count_buf.slice(..);
        let (tx, rx) = tokio::sync::oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        self.device.poll(wgpu::Maintain::Wait); // Ensure copy completes

        rx.await.map_err(|_| {
            GpuError::BufferOperationError("Mapping channel canceled for count".to_string())
        })??;

        let mapped_range = buffer_slice.get_mapped_range();
        let data = mapped_range.to_vec();
        drop(mapped_range);
        self.buffers.staging_worklist_count_buf.unmap();

        if data.len() >= 4 {
            let count = u32::from_le_bytes(data[0..4].try_into().unwrap());
            Ok(count)
        } else {
            Err(GpuError::BufferSizeMismatch(
                "Downloaded worklist count data too small".to_string(),
            ))
        }
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
        let num_tiles = self.buffers.num_tiles; // Assuming num_tiles is available on GpuBuffers
        let u32s_per_cell = self.buffers.u32s_per_cell;

        let params = GpuEntropyShaderParams {
            grid_dims: [grid_dims.0 as u32, grid_dims.1 as u32, grid_dims.2 as u32],
            heuristic_type,
            num_tiles: num_tiles as u32,
            u32s_per_cell: u32s_per_cell as u32,
            _padding1: 0,
            _padding2: 0,
        };

        self.queue.write_buffer(
            &self.buffers.entropy_params_buffer,
            0,
            bytemuck::cast_slice(&[params]),
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
            bytemuck::cast_slice(&[*params]),
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
        self.upload_initial_updates(updated_indices, worklist_idx)
    }

    /// Uploads the initial set of updated cell indices to the GPU worklist buffer.
    ///
    /// This is used to seed the propagation process after an initial cell collapse.
    /// Requires updating the `params_uniform_buf` with the new worklist size separately.
    ///
    /// # Arguments
    ///
    /// * `updated_indices` - A slice containing the 1D indices of the cells that were updated.
    /// * `worklist_idx` - The index (0 or 1) of the worklist buffer to use (for ping-pong).
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the upload is successful.
    /// * `Err(GpuError)` if the data size exceeds the buffer capacity.
    pub fn upload_initial_updates(
        &self,
        updated_indices: &[u32],
        worklist_idx: usize,
    ) -> Result<(), GpuError> {
        let worklist_buffer = if worklist_idx == 0 {
            &self.buffers.worklist_buf_a
        } else {
            &self.buffers.worklist_buf_b
        };

        let data_size = (updated_indices.len() * std::mem::size_of::<u32>()) as u64;
        if data_size > worklist_buffer.size() {
            return Err(GpuError::BufferSizeMismatch(format!(
                "Initial update data size ({}) exceeds worklist buffer capacity ({})",
                data_size,
                worklist_buffer.size()
            )));
        }

        debug!(
            "Uploading {} initial updates ({} bytes) to worklist buffer {}.",
            updated_indices.len(),
            data_size,
            worklist_idx
        );
        self.queue
            .write_buffer(worklist_buffer, 0, bytemuck::cast_slice(updated_indices));

        // NOTE: Updating the worklist size in the params buffer is now a separate step,
        // likely called after this method.
        // self.update_params_worklist_size(updated_indices.len() as u32)?;

        Ok(())
    }

    /// Downloads the minimum entropy information from the GPU.
    ///
    /// # Returns
    ///
    /// `Ok(min_info)` with the minimum entropy data if successful, or an error detailing what went wrong.
    pub async fn download_min_entropy_info(&self) -> Result<Option<(f32, u32)>, GpuError> {
        // Download the min entropy information using download_results
        let request = DownloadRequest {
            download_entropy: false,
            download_min_entropy_info: true,
            download_grid_possibilities: false,
            download_worklist_size: false,
            download_contradiction_location: false,
        };
        let results = self
            .buffers
            .download_results(self.device.clone(), self.queue.clone(), request)
            .await
            .map_err(|e| {
                GpuError::TransferError(format!("Failed to download min entropy info: {}", e))
            })?;

        Ok(results.min_entropy_info)
    }

    /// Downloads the pass statistics buffer from the GPU.
    ///
    /// The buffer contains [cells_added, possibilities_removed, contradictions, overflow].
    ///
    /// # Returns
    ///
    /// `Ok(stats)` with the statistics data if successful, or an error detailing what went wrong.
    pub async fn download_pass_statistics(&self) -> Result<[u32; 4], GpuError> {
        let buffer_size = self.buffers.pass_statistics_buf.size();
        if buffer_size < 16 {
            return Err(GpuError::BufferSizeMismatch(
                "Pass statistics buffer too small".to_string(),
            ));
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Pass Statistics Download Encoder"),
            });

        encoder.copy_buffer_to_buffer(
            &self.buffers.pass_statistics_buf,
            0,
            &self.buffers.staging_pass_statistics_buf,
            0,
            buffer_size, // Copy the whole buffer
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = self.buffers.staging_pass_statistics_buf.slice(..);
        let (tx, rx) = tokio::sync::oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        self.device.poll(wgpu::Maintain::Wait); // Ensure copy completes

        rx.await.map_err(|_| {
            GpuError::BufferOperationError("Mapping channel canceled for stats".to_string())
        })??; // Handle channel send error and map_async error

        let mapped_range = buffer_slice.get_mapped_range();
        let data = mapped_range.to_vec(); // Copy data out
        drop(mapped_range);
        self.buffers.staging_pass_statistics_buf.unmap();

        if data.len() >= 16 {
            let stats_slice: &[u32] = bytemuck::cast_slice(&data[0..16]);
            Ok([
                stats_slice[0],
                stats_slice[1],
                stats_slice[2],
                stats_slice[3],
            ])
        } else {
            Err(GpuError::BufferSizeMismatch(
                "Downloaded pass statistics data too small".to_string(),
            ))
        }
    }

    /// Resets the `min_entropy_info_buf` on the GPU to its initial state.
    ///
    /// Sets the minimum entropy value to `f32::MAX` (represented as bits) and the index to `u32::MAX`.
    /// This is typically done before running the entropy calculation shader.
    ///
    /// # Returns
    ///
    /// * `Ok(())` always (buffer writing is typically fire-and-forget, errors are harder to catch here).
    pub fn reset_min_entropy_buffer(&self) -> Result<(), GpuError> {
        let initial_data = [f32::MAX.to_bits(), u32::MAX]; // [min_entropy_f32_bits, min_index_u32]
        self.queue.write_buffer(
            &self.buffers.min_entropy_info_buf,
            0,
            bytemuck::cast_slice(&initial_data),
        );
        Ok(())
    }

    /// Resets the `contradiction_flag_buf` on the GPU to 0.
    ///
    /// A value of 0 indicates no contradiction has been detected.
    /// This should be called before running the propagation shader.
    pub fn reset_contradiction_flag(&self) -> Result<(), GpuError> {
        self.queue.write_buffer(
            &self.buffers.contradiction_flag_buf,
            0,
            bytemuck::cast_slice(&[0u32]),
        );
        Ok(())
    }

    /// Resets the `contradiction_location_buf` on the GPU to `u32::MAX`.
    ///
    /// `u32::MAX` is used to indicate that no specific contradiction location has been recorded yet.
    pub fn reset_contradiction_location(&self) -> Result<(), GpuError> {
        let max_u32 = [u32::MAX];
        self.queue.write_buffer(
            &self.buffers.contradiction_location_buf,
            0,
            bytemuck::cast_slice(&max_u32),
        );
        Ok(())
    }

    /// Resets the `worklist_count_buf` on the GPU to 0.
    ///
    /// Used if implementing iterative GPU propagation where the shader generates a new worklist.
    pub fn reset_worklist_count(&self) -> Result<(), GpuError> {
        self.queue.write_buffer(
            &self.buffers.worklist_count_buf,
            0,
            bytemuck::cast_slice(&[0u32]),
        );
        Ok(())
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

    /// Resets the grid state to an initial state based on a provided possibility grid.
    ///
    /// This involves:
    /// 1. Uploading the packed possibilities from the `initial_grid`.
    /// 2. Uploading the provided initial `updates` to the appropriate worklist buffer.
    ///
    /// Note: Buffer resizing logic is assumed to be handled elsewhere (e.g., accelerator).
    ///
    /// # Arguments
    /// * `initial_grid` - The `PossibilityGrid` defining the initial state.
    /// * `updates` - Initial cell indices to add to the worklist.
    /// * `active_worklist_idx` - Which worklist buffer (0 or 1) to upload updates to.
    ///
    /// # Returns
    /// * `Ok(())` on success.
    /// * `Err(GpuError)` if packing or upload fails.
    pub fn reset_grid_state(
        &self,
        initial_grid: &PossibilityGrid,
        updates: &[u32],
        active_worklist_idx: usize,
    ) -> Result<(), GpuError> {
        // 1. Upload packed possibilities
        self.upload_grid(initial_grid)?;

        // 2. Upload initial updates to the specified worklist
        self.upload_initial_updates(updates, active_worklist_idx)?;

        Ok(())
    }

    /// Updates the `worklist_size` field within the `params_uniform_buf` on the GPU.
    ///
    /// This informs the propagation shader how many updated cells are present in the `updates_buf`.
    ///
    /// # Arguments
    ///
    /// * `worklist_size` - The number of valid entries in the `updates_buf`.
    ///
    /// # Returns
    ///
    /// * `Ok(())` always.
    ///
    /// # Panics
    ///
    /// This function relies on the memory layout of `GpuParamsUniform`. Changes to that struct
    /// might require updating the offset calculation here.
    pub fn update_params_worklist_size(&self, worklist_size: u32) -> Result<(), GpuError> {
        // Calculate the byte offset of the worklist_size field within GpuParamsUniform
        // This relies on the struct layout. Field order: grid_width, grid_height, grid_depth,
        // num_tiles, num_axes, boundary_mode, heuristic_type, tie_breaking,
        // max_propagation_steps, contradiction_check_frequency, worklist_size
        // Offsets: 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40
        let offset = 40; // Byte offset of worklist_size

        // Check buffer size before writing to prevent panic
        if self.buffers.params_uniform_buf.size() < offset + 4 {
            return Err(GpuError::BufferOperationError(format!(
                "Params uniform buffer too small ({}) to write worklist_size at offset {}",
                self.buffers.params_uniform_buf.size(),
                offset
            )));
        }

        self.queue.write_buffer(
            &self.buffers.params_uniform_buf,
            offset,
            bytemuck::cast_slice(&[worklist_size]),
        );
        Ok(())
    }

    /// Uploads entropy parameters to the GPU.
    ///
    /// # Arguments
    ///
    /// * `params` - The `GpuEntropyShaderParams` struct containing the parameters.
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, or an error detailing what went wrong.
    pub fn upload_entropy_params(&self, params: &GpuEntropyShaderParams) -> Result<(), GpuError> {
        self.queue.write_buffer(
            &self.buffers.entropy_params_buffer,
            0,
            bytemuck::cast_slice(&[*params]),
        );
        Ok(())
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
