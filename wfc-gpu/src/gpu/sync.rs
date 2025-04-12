// Synchronization module that handles GPU-CPU data transfer and resource management
// This separates GPU synchronization concerns from algorithm logic

use crate::{
    buffers::{DownloadRequest, GpuBuffers, GpuEntropyShaderParams, GpuParamsUniform},
    utils::error::gpu_error::{GpuError as NewGpuError, GpuErrorContext, GpuResourceType},
    utils::error_recovery::{GpuError as OldGpuError, GridCoord},
};
use log::{debug, error, trace, warn};
use std::borrow::Cow;
use std::sync::Arc;
use wfc_core::grid::PossibilityGrid;
use wgpu;
use wgpu::util::DeviceExt;

// Type alias for backward compatibility
pub type GpuError = OldGpuError;

// Conversion function to handle error type mismatch
impl From<crate::utils::error::gpu_error::GpuError> for OldGpuError {
    fn from(err: crate::utils::error::gpu_error::GpuError) -> Self {
        match err {
            crate::utils::error::gpu_error::GpuError::BufferMapFailed { msg, .. } => {
                OldGpuError::BufferMapping(msg)
            }
            crate::utils::error::gpu_error::GpuError::BufferMapTimeout(msg, ..) => {
                OldGpuError::BufferMapping(msg)
            }
            crate::utils::error::gpu_error::GpuError::ValidationError(e, ..) => {
                OldGpuError::Other(e.to_string())
            }
            crate::utils::error::gpu_error::GpuError::CommandExecutionError { msg, .. } => {
                OldGpuError::KernelExecution(msg)
            }
            crate::utils::error::gpu_error::GpuError::BufferOperationError { msg, .. } => {
                OldGpuError::BufferCopy(msg)
            }
            crate::utils::error::gpu_error::GpuError::TransferError { msg, .. } => {
                OldGpuError::BufferCopy(msg)
            }
            crate::utils::error::gpu_error::GpuError::ResourceCreationFailed { msg, .. } => {
                OldGpuError::MemoryAllocation(msg)
            }
            crate::utils::error::gpu_error::GpuError::ShaderError { msg, .. } => {
                OldGpuError::Other(msg)
            }
            crate::utils::error::gpu_error::GpuError::BufferSizeMismatch { msg, .. } => {
                OldGpuError::BufferCopy(msg)
            }
            crate::utils::error::gpu_error::GpuError::Timeout { msg, .. } => {
                OldGpuError::ComputationTimeout {
                    grid_size: (0, 0),                           // We don't have grid size info here
                    duration: std::time::Duration::from_secs(0), // We don't have duration info here
                }
            }
            crate::utils::error::gpu_error::GpuError::DeviceLost { msg, .. } => {
                OldGpuError::DeviceLost(msg)
            }
            crate::utils::error::gpu_error::GpuError::AdapterRequestFailed { .. } => {
                OldGpuError::Other("Adapter request failed".to_string())
            }
            crate::utils::error::gpu_error::GpuError::DeviceRequestFailed(e, ..) => {
                OldGpuError::Other(format!("Device request failed: {}", e))
            }
            crate::utils::error::gpu_error::GpuError::MutexError { msg, .. } => {
                OldGpuError::Other(msg)
            }
            crate::utils::error::gpu_error::GpuError::ContradictionDetected { .. } => {
                OldGpuError::InvalidState("Contradiction detected".to_string())
            }
            crate::utils::error::gpu_error::GpuError::Other { msg, .. } => OldGpuError::Other(msg),
        }
    }
}

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

    /// Downloads the complete possibility grid state from GPU to a CPU-side grid.
    /// This involves mapping the grid buffer, copying data, and unmapping.
    ///
    /// # Arguments
    ///
    /// * `target_grid` - A mutable reference to the `PossibilityGrid` to download into.
    ///                   This grid MUST have the same dimensions and tile count as the GPU grid.
    ///
    /// # Returns
    ///
    /// Returns the `target_grid` on success, or a `GpuError` on failure.
    pub async fn download_grid(
        &self,
        target: &PossibilityGrid,
    ) -> Result<PossibilityGrid, GpuError> {
        let request = DownloadRequest {
            download_grid_possibilities: true,
            download_contradiction_flag: false,
            download_entropy: false,
            download_min_entropy_info: false,
            download_contradiction_location: false,
        };

        trace!("Downloading grid with request: {:?}", request);
        let start_time = std::time::Instant::now();

        // Access buffer via grid_buffers
        let possibility_buffer_slice = self.buffers.grid_buffers.grid_possibilities_buf.slice(..); // Get slice of the whole buffer

        // Asynchronous map operation
        let (sender, receiver) = futures::channel::oneshot::channel();
        possibility_buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).expect("Failed to send map result");
        });

        // Poll the device while waiting for the map operation to complete
        let _ = self.device.poll(wgpu::MaintainBase::Wait);

        // Wait for the mapping result
        match receiver.await {
            Ok(Ok(())) => {
                // Buffer mapped successfully
                trace!("Possibility buffer mapped successfully.");
                let mapped_range = possibility_buffer_slice.get_mapped_range();

                // --- Copy data from mapped GPU buffer to CPU grid --- ///
                // Safety: Ensure target_grid dimensions match GPU buffer layout
                // Assume data is tightly packed array of u32 bitmasks.
                let num_tiles = target.num_tiles();
                let tiles_per_u32 = 32;
                let u32_per_cell = (num_tiles + tiles_per_u32 - 1) / tiles_per_u32;
                let expected_size = target.width
                    * target.height
                    * target.depth
                    * u32_per_cell
                    * std::mem::size_of::<u32>();

                if mapped_range.len() != expected_size {
                    error!(
                        "GPU buffer size mismatch. Expected: {}, Actual: {}",
                        expected_size,
                        mapped_range.len()
                    );
                    // Unmap before returning error
                    drop(mapped_range); // Drop mapped_range to potentially trigger unmap implicitly
                    self.buffers.grid_buffers.grid_possibilities_buf.unmap();
                    return Err(GpuError::BufferCopy(
                        "GPU possibility buffer size does not match target grid dimensions"
                            .to_string(),
                    ));
                }

                // TODO: Implement the actual data copy efficiently.
                // This likely involves iterating through target_grid cells
                // and copying the corresponding u32 chunks from mapped_range.
                // Consider using unsafe code carefully for performance if needed,
                // or libraries like `bytemuck`.
                // Placeholder: Just log for now.
                trace!("TODO: Implement actual grid data copy from mapped buffer.");

                // Drop the mapped range view. This implicitly signals to WGPU
                // that we are done with the mapped memory.
                drop(mapped_range);

                // Explicitly unmap the buffer - access via grid_buffers
                self.buffers.grid_buffers.grid_possibilities_buf.unmap();
                trace!("Possibility buffer unmapped.");
            }
            Ok(Err(e)) => {
                error!("Failed to map possibility buffer: {}", e);
                return Err(GpuError::BufferMapping(e.to_string()));
            }
            Err(e) => {
                error!("Map future cancelled or panicked: {}", e);
                // Don't try to unmap if the channel was cancelled before mapping finished.
                return Err(GpuError::Other("Buffer map future cancelled".to_string()));
            }
        }

        let duration = start_time.elapsed();
        trace!("Grid download completed in {:?}", duration);
        Ok(target.clone())
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
        let flag_buffer_gpu = &self.buffers.contradiction_flag_buf;
        let flag_buffer_staging = &self.buffers.staging_contradiction_flag_buf;
        let mut contradiction_location = None;

        let flag_data = crate::buffers::download_buffer_data::<u32>(
            Some(self.device.clone()),
            Some(self.queue.clone()),
            &*flag_buffer_gpu,
            &*flag_buffer_staging,
            std::mem::size_of::<u32>() as u64,
            Some("Check Contradiction Flag".to_string()),
        )
        .await
        .map_err(|e| {
            GpuError::BufferCopy(format!("Failed to download contradiction flag: {}", e))
        })?;

        let has_contradiction = flag_data.first().map_or(false, |&flag| flag != 0);

        if has_contradiction {
            let loc_buffer_gpu = &self.buffers.contradiction_location_buf;
            let loc_buffer_staging = &self.buffers.staging_contradiction_location_buf;

            let loc_data = crate::buffers::download_buffer_data::<u32>(
                Some(self.device.clone()),
                Some(self.queue.clone()),
                &*loc_buffer_gpu,
                &*loc_buffer_staging,
                3 * std::mem::size_of::<u32>() as u64,
                Some("Download Contradiction Location".to_string()),
            )
            .await
            .map_err(|e| {
                GpuError::BufferCopy(format!("Failed to download contradiction location: {}", e))
            })?;

            contradiction_location = loc_data.first().cloned();
        }

        Ok((has_contradiction, contradiction_location))
    }

    /// Downloads the current worklist size from the GPU.
    pub async fn download_worklist_size(&self) -> Result<u32, GpuError> {
        let count_buffer_gpu = &self.buffers.worklist_buffers.worklist_count_buf;
        let staging_count_buffer = &self.buffers.worklist_buffers.staging_worklist_count_buf;

        let count_data = crate::buffers::download_buffer_data::<u32>(
            Some(self.device.clone()),
            Some(self.queue.clone()),
            &*count_buffer_gpu,
            &*staging_count_buffer,
            std::mem::size_of::<u32>() as u64,
            Some("Download Worklist Count".to_string()),
        )
        .await?;

        Ok(count_data.first().cloned().unwrap_or(0))
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
            return Err(GpuError::BufferCopy(
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
            download_contradiction_flag: false,
            download_contradiction_location: false,
        };
        let results =
            self.buffers.download_results(request).await.map_err(|e| {
                GpuError::BufferCopy(format!("Failed to download min entropy: {}", e))
            })?;

        Ok(results.min_entropy_info)
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

    /// Checks if a contradiction has occurred by downloading the contradiction flag.
    pub async fn check_for_contradiction(&self) -> Result<bool, GpuError> {
        let flag_buffer_gpu = &self.buffers.contradiction_flag_buf;
        let flag_buffer_staging = &self.buffers.staging_contradiction_flag_buf;

        let flag_data = crate::buffers::download_buffer_data::<u32>(
            Some(self.device.clone()),
            Some(self.queue.clone()),
            &*flag_buffer_gpu,
            &*flag_buffer_staging,
            std::mem::size_of::<u32>() as u64,
            Some("Check Contradiction Flag".to_string()),
        )
        .await
        .map_err(|e| {
            GpuError::BufferCopy(format!("Failed to download contradiction flag: {}", e))
        })?;

        Ok(flag_data.first().map_or(false, |&flag| flag != 0))
    }

    /// Downloads the location of the first contradiction, if one occurred.
    pub async fn download_contradiction_location(&self) -> Result<Option<u32>, GpuError> {
        let loc_buffer_gpu = &self.buffers.contradiction_location_buf;
        let loc_buffer_staging = &self.buffers.staging_contradiction_location_buf;

        // First, check the flag
        if !self.check_for_contradiction().await? {
            return Ok(None);
        }

        // If flag is set, download the location
        let loc_data = crate::buffers::download_buffer_data::<u32>(
            Some(self.device.clone()),
            Some(self.queue.clone()),
            &*loc_buffer_gpu,
            &*loc_buffer_staging,
            3 * std::mem::size_of::<u32>() as u64,
            Some("Download Contradiction Location".to_string()),
        )
        .await
        .map_err(|e| {
            GpuError::BufferCopy(format!("Failed to download contradiction location: {}", e))
        })?;

        Ok(loc_data.first().cloned())
    }

    /// Downloads the current worklist count.
    pub async fn download_worklist_count(&self) -> Result<u32, GpuError> {
        let count_buffer_gpu = &self.buffers.worklist_buffers.worklist_count_buf;
        let staging_count_buffer = &self.buffers.worklist_buffers.staging_worklist_count_buf;

        let count_data = crate::buffers::download_buffer_data::<u32>(
            Some(self.device.clone()),
            Some(self.queue.clone()),
            &*count_buffer_gpu,
            &*staging_count_buffer,
            std::mem::size_of::<u32>() as u64,
            Some("Download Worklist Count".to_string()),
        )
        .await?;
        Ok(count_data.first().cloned().unwrap_or(0))
    }

    /// Downloads the entire possibility grid state.
    pub async fn download_entire_grid_state(&self) -> Result<PossibilityGrid, GpuError> {
        // Create a request for grid data only
        let request = DownloadRequest {
            download_grid_possibilities: true,
            download_contradiction_flag: false,
            ..Default::default()
        };

        // Download grid data
        let results = self.buffers.download_results(request).await?;

        // Convert results to grid
        if let Some(grid_data) = results.grid_possibilities {
            let (width, height, depth) = self.buffers.grid_dims;
            // Map the buffer data to a PossibilityGrid
            return Ok(self.buffers.to_possibility_grid_from_data(
                &grid_data,
                (width, height, depth),
                self.buffers.num_tiles,
            )?);
        }

        Err(GpuError::Other("Failed to download grid data".to_string()))
    }

    /// Create a new error context for buffer operations
    fn create_buffer_error_context(&self, buffer_name: &str) -> GpuErrorContext {
        GpuErrorContext::new(GpuResourceType::Buffer).with_label(buffer_name.to_string())
    }

    /// Convert from old GpuError to new NewGpuError for consistent error handling
    pub fn convert_to_new_error(&self, error: &OldGpuError, buffer_name: &str) -> NewGpuError {
        let context = self.create_buffer_error_context(buffer_name);

        match error {
            OldGpuError::BufferMapFailed(msg) => NewGpuError::buffer_map_failed(msg, context),
            OldGpuError::ValidationError(e) => NewGpuError::validation_error(e.clone(), context),
            OldGpuError::CommandExecutionError(msg) => {
                NewGpuError::command_execution_error(msg, context)
            }
            OldGpuError::BufferOperationError(msg) => {
                NewGpuError::buffer_operation_error(msg, context)
            }
            OldGpuError::TransferError(msg) => NewGpuError::transfer_error(msg, context),
            OldGpuError::ResourceCreationFailed(msg) => {
                NewGpuError::resource_creation_failed(msg, context)
            }
            OldGpuError::ShaderError(msg) => NewGpuError::shader_error(msg, context),
            OldGpuError::BufferSizeMismatch(msg) => NewGpuError::buffer_size_mismatch(msg, context),
            OldGpuError::Timeout(msg) => NewGpuError::timeout(msg, context),
            OldGpuError::DeviceLost(msg) => NewGpuError::device_lost(msg, context),
            OldGpuError::AdapterRequestFailed => NewGpuError::adapter_request_failed(context),
            OldGpuError::DeviceRequestFailed(e) => {
                NewGpuError::device_request_failed(e.clone(), context)
            }
            OldGpuError::MutexError(msg) => NewGpuError::mutex_error(msg, context),
            OldGpuError::BufferError(msg) => NewGpuError::buffer_operation_error(msg, context),
            OldGpuError::BufferMapError(msg) => NewGpuError::buffer_map_failed(msg, context),
            OldGpuError::BufferMapTimeout(msg) => NewGpuError::timeout(msg, context),
            OldGpuError::ContradictionDetected { coord } => {
                let mut ctx = context;
                if let Some(c) = coord {
                    ctx = ctx.with_grid_coord(*c);
                }
                NewGpuError::contradiction_detected(ctx)
            }
            OldGpuError::Other(msg) => NewGpuError::other(msg, context),
            OldGpuError::InternalError(msg) => NewGpuError::other(msg, context),
        }
    }

    /// Download buffer data with consistent error handling
    pub fn download_buffer_consistent<T: bytemuck::Pod>(
        &self,
        buffer: &wgpu::Buffer,
        offset: u64,
        size: usize,
        buffer_name: &str,
    ) -> Result<Vec<T>, NewGpuError> {
        match self.download_buffer::<T>(buffer, offset, size) {
            Ok(data) => Ok(data),
            Err(err) => Err(self.convert_to_new_error(&err, buffer_name)),
        }
    }

    /// Generic buffer download method - reads data from a GPU buffer into a vector
    pub fn download_buffer<T: bytemuck::Pod>(
        &self,
        buffer: &wgpu::Buffer,
        offset: u64,
        size: usize,
    ) -> Result<Vec<T>, GpuError> {
        // Create a staging buffer to copy data from the device to the host
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer for Download"),
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create command encoder and copy from the source buffer to the staging buffer
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Buffer Download Encoder"),
            });
        encoder.copy_buffer_to_buffer(buffer, offset, &staging_buffer, 0, size as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map the staging buffer for reading
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        // Wait for the mapping to complete
        self.device.poll(wgpu::MaintainBase::Wait);

        // Check the mapping status
        match pollster::block_on(receiver) {
            Ok(Ok(())) => {
                // Successfully mapped, copy data
                let mapped_range = buffer_slice.get_mapped_range();
                let data = bytemuck::cast_slice(&mapped_range).to_vec();
                drop(mapped_range);
                staging_buffer.unmap();
                Ok(data)
            }
            _ => Err(GpuError::BufferMapping(
                "Failed to map staging buffer for download".to_string(),
            )),
        }
    }

    /// Upload buffer data with consistent error handling
    pub fn upload_buffer_consistent<T: bytemuck::Pod>(
        &self,
        buffer: &wgpu::Buffer,
        offset: u64,
        data: &[T],
        buffer_name: &str,
    ) -> Result<(), NewGpuError> {
        match self.upload_buffer(buffer, offset, data) {
            Ok(()) => Ok(()),
            Err(err) => Err(self.convert_to_new_error(&err, buffer_name)),
        }
    }

    /// Generic buffer upload method - writes data to a GPU buffer
    pub fn upload_buffer<T: bytemuck::Pod>(
        &self,
        buffer: &wgpu::Buffer,
        offset: u64,
        data: &[T],
    ) -> Result<(), GpuError> {
        // Calculate the byte size of the data
        let data_bytes = bytemuck::cast_slice(data);
        let data_size = data_bytes.len() as u64;

        // Ensure the buffer can accommodate the data
        if offset + data_size > buffer.size() {
            return Err(GpuError::BufferCopy(format!(
                "Data size ({}) exceeds buffer capacity ({}) at offset {}",
                data_size,
                buffer.size(),
                offset
            )));
        }

        // Write data to the buffer
        self.queue.write_buffer(buffer, offset, data_bytes);

        // Ensure the write is processed
        self.device.poll(wgpu::MaintainBase::Wait);

        Ok(())
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
