use crate::buffers::GpuBuffers;
use crate::pipeline::ComputePipelines;
use log;
use std::sync::Arc;
use wfc_core::{
    entropy::EntropyCalculator,
    grid::{EntropyGrid, Grid, PossibilityGrid},
    EntropyError,
};
use wgpu;

/// GPU implementation for entropy calculation.
///
/// This struct holds references to the necessary GPU resources (device, queue,
/// pipelines, buffers) and implements the `EntropyCalculator` trait.
#[derive(Clone)] // Derive Clone since it holds Arcs
pub struct GpuEntropyCalculator {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) pipelines: Arc<ComputePipelines>,
    pub(crate) buffers: Arc<GpuBuffers>,
    pub(crate) grid_dims: (usize, usize, usize),
    pub(crate) num_tiles: u32,
}

impl GpuEntropyCalculator {
    /// Creates a new `GpuEntropyCalculator` instance.
    /// Requires access to initialized GPU device, queue, pipelines, and buffers.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        pipelines: Arc<ComputePipelines>, // Expect Arc directly
        buffers: Arc<GpuBuffers>,         // Expect Arc directly
        grid_dims: (usize, usize, usize),
        num_tiles: u32,
    ) -> Self {
        Self {
            device,
            queue,
            pipelines,
            buffers,
            grid_dims,
            num_tiles,
        }
    }
}

impl EntropyCalculator for GpuEntropyCalculator {
    /// Calculates the entropy for each cell using a dedicated compute shader.
    ///
    /// This method orchestrates the execution of the entropy calculation shader:
    /// 1. Resets the auxiliary buffer used for finding the minimum entropy.
    /// 2. Sets up the necessary bind group containing GPU buffers (possibilities, entropy output, parameters, min_entropy_info).
    /// 3. Creates a compute pass and dispatches the entropy compute shader.
    /// 4. Submits the commands to the GPU queue.
    /// 5. Reads the calculated entropy values back from the `entropy_buf` into a CPU-side `EntropyGrid`.
    ///
    /// The actual entropy calculation logic resides within the `entropy.wgsl` compute shader.
    /// Note: The `_grid` parameter (CPU-side `PossibilityGrid`) is technically unused as this method
    /// assumes the relevant possibility data is already present in the GPU buffer (`self.buffers.grid_possibilities_buf`).
    ///
    /// # Returns
    ///
    /// An `EntropyGrid` containing the entropy values calculated by the GPU. If reading the results
    /// from the GPU fails, an empty grid with the correct dimensions is returned, and an error is logged.
    #[must_use]
    fn calculate_entropy(&self, _grid: &PossibilityGrid) -> Result<EntropyGrid, EntropyError> {
        // Assuming grid state is primarily managed on the GPU via self.buffers.grid_possibilities_buf
        // _grid parameter is technically unused as we read directly from the GPU buffer.
        // Consider changing the trait or method signature if this becomes an issue.
        log::debug!("Running GPU calculate_entropy...");

        let (width, height, depth) = self.grid_dims;
        let num_cells = width * height * depth;

        // Reset the min entropy buffer before dispatch
        if let Err(e) = self.buffers.reset_min_entropy_info(&self.queue) {
            let err_msg = format!("GPU Error: Failed to reset min entropy info buffer: {}", e);
            log::error!("{}", err_msg);
            // Use EntropyError::Other for GPU specific errors
            return Err(EntropyError::Other(err_msg));
        }

        // 1. Create Command Encoder (uses Arc<Device>)
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Entropy Compute Encoder"),
            });

        // 2. Create Bind Group (uses Arc<Device>, Arc<BindGroupLayout>, Arc<Buffer>)
        // Access pipelines and buffers through the Arc references
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Entropy Bind Group"),
            layout: &self.pipelines.entropy_bind_group_layout, // Access layout via Arc<Pipelines>
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffers.grid_possibilities_buf.as_entire_binding(), // Access buffer via Arc<Buffers>
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buffers.entropy_buf.as_entire_binding(), // Access buffer via Arc<Buffers>
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buffers.params_uniform_buf.as_entire_binding(), // Access buffer via Arc<Buffers>
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buffers.min_entropy_info_buf.as_entire_binding(), // Access buffer via Arc<Buffers>
                },
            ],
        });

        // 3. Begin Compute Pass (uses Arc<ComputePipeline>)
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Entropy Compute Pass"),
                timestamp_writes: None, // Add timestamps later if needed for profiling
            });

            compute_pass.set_pipeline(&self.pipelines.entropy_pipeline); // Access pipeline via Arc<Pipelines>
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch - Calculate workgroup counts
            // Use dynamic workgroup size from the pipeline struct
            let workgroup_size = self.pipelines.entropy_workgroup_size;
            let workgroups_needed = num_cells.div_ceil(workgroup_size as usize) as u32;

            log::debug!(
                "Dispatching entropy shader with {} workgroups of size {}",
                workgroups_needed,
                workgroup_size // Log the dynamic size
            );
            compute_pass.dispatch_workgroups(workgroups_needed, 1, 1);
        } // End compute pass scope

        // 4. Submit to Queue (uses Arc<Queue>)
        self.queue.submit(std::iter::once(encoder.finish()));
        log::debug!("Entropy compute shader submitted.");

        // Download results
        let download_results_res = pollster::block_on(self.buffers.download_results(
            self.device.clone(),
            self.queue.clone(),
            true,
            true,
            false,
            false,
            false,
            false, // Flags
        ));

        // Handle the outer Result first
        let download_results = download_results_res
            .map_err(|e| EntropyError::Other(format!("GPU download error: {}", e)))?;

        // Now process the GpuDownloadResults
        let gpu_entropy_data = download_results
            .entropy
            .ok_or_else(|| EntropyError::Other("GPU did not return entropy data".to_string()))?;

        let entropy_grid = Grid {
            width,
            height,
            depth,
            data: gpu_entropy_data,
        };

        if entropy_grid.data.len() != num_cells {
            let err_msg = format!(
                "GPU Error: Entropy result size mismatch: expected {}, got {}",
                num_cells,
                entropy_grid.data.len()
            );
            log::error!("{}", err_msg);
            Err(EntropyError::Other(err_msg))
        } else {
            Ok(entropy_grid)
        }
    }

    /// Finds the coordinates of the cell with the lowest positive entropy using results from the GPU.
    ///
    /// This method retrieves the minimum entropy value and its corresponding index, which were
    /// calculated and stored in the `min_entropy_info_buf` by the `calculate_entropy` compute shader.
    ///
    /// 1. Reads the `min_entropy_info_buf` from the GPU to the CPU.
    /// 2. Parses the buffer content to extract the minimum entropy value (`min_val`) and the flat index (`min_idx`).
    /// 3. If `min_val` is positive (indicating a valid, uncollapsed cell was found),
    ///    converts the flat `min_idx` back into 3D `(x, y, z)` coordinates.
    /// 4. Returns `Some((x, y, z))` if a minimum positive entropy cell was found, otherwise returns `None`.
    ///
    /// Note: The `_entropy_grid` parameter (CPU-side `EntropyGrid`) is unused as the minimum is determined
    /// directly from the GPU buffer populated during the `calculate_entropy` step.
    ///
    /// # Returns
    ///
    /// * `Some((x, y, z))` - Coordinates of the cell with the lowest positive entropy found by the GPU.
    /// * `None` - If no cell with positive entropy was found (e.g., grid fully collapsed or error).
    #[must_use]
    fn select_lowest_entropy_cell(
        &self,
        _entropy_grid: &EntropyGrid,
    ) -> Option<(usize, usize, usize)> {
        log::debug!("Downloading GPU minimum entropy info via download_results...");

        let download_results_res = pollster::block_on(self.buffers.download_results(
            self.device.clone(),
            self.queue.clone(),
            false,
            true,
            false,
            false,
            false,
            false, // Flags
        ));

        // Handle outer Result, map error to None for this function's signature
        match download_results_res {
            Ok(download_results) => {
                if let Some((min_entropy, flat_index)) = download_results.min_entropy_info {
                    if min_entropy > 0.0 && min_entropy < f32::MAX {
                        // Check validity
                        let width = self.grid_dims.0 as usize;
                        let height = self.grid_dims.1 as usize;
                        let z = flat_index as usize / (width * height);
                        let y = (flat_index as usize % (width * height)) / width;
                        let x = flat_index as usize % width;
                        log::info!(
                            "GPU found lowest entropy ({}) at ({}, {}, {})",
                            min_entropy,
                            x,
                            y,
                            z
                        );
                        Some((x, y, z))
                    } else {
                        log::info!(
                            "GPU reported no cell with calculable entropy (all collapsed or empty?)."
                        );
                        None
                    }
                } else {
                    log::error!(
                        "GPU Error: Min entropy info was requested but not returned in results."
                    );
                    None // Download ok, but info missing
                }
            }
            Err(e) => {
                log::error!("Failed to download min entropy info for selection: {}", e);
                None // Download failed
            }
        }
    }
}
