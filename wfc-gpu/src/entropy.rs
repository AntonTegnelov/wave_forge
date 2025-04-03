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
    ) -> Self {
        Self {
            device,
            queue,
            pipelines,
            buffers,
            grid_dims,
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

        // 5. Download results (uses Arc<Device>, Arc<Queue>)
        log::debug!("Downloading entropy results...");
        let entropy_data_result =
            pollster::block_on(self.buffers.download_entropy(&self.device, &self.queue));

        match entropy_data_result {
            Ok(entropy_data) => {
                log::debug!(
                    "Entropy results downloaded successfully ({} floats).",
                    entropy_data.len()
                );
                if entropy_data.len() != num_cells {
                    let err_msg = format!(
                        "GPU Error: Entropy result size mismatch: expected {}, got {}",
                        num_cells,
                        entropy_data.len()
                    );
                    log::error!("{}", err_msg);
                    // Use EntropyError::Other
                    Err(EntropyError::Other(err_msg))
                } else {
                    // 6. Create Grid from the downloaded data
                    Ok(Grid {
                        width,
                        height,
                        depth,
                        data: entropy_data,
                    })
                }
            }
            Err(e) => {
                let err_msg = format!("GPU Error: Failed to download entropy results: {}", e);
                log::error!("{}", err_msg);
                // Use EntropyError::Other
                Err(EntropyError::Other(err_msg))
            }
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
        // GPU reduction happens during calculate_entropy. Here we just download the result.
        // _entropy_grid parameter is unused because the min info is read directly from its GPU buffer.
        log::debug!("Downloading GPU minimum entropy info...");

        // Download the result [min_entropy_f32_bits, min_index_u32]
        // Access download method via Arc<Buffers>, pass Arc<Device> and Arc<Queue>
        let download_result = pollster::block_on(
            self.buffers
                .download_min_entropy_info(&self.device, &self.queue),
        );

        match download_result {
            Ok((min_entropy_val, min_index)) => {
                log::debug!(
                    "GPU min entropy info downloaded: value = {}, index = {}",
                    min_entropy_val,
                    min_index
                );

                // Check if a valid minimum was found (index != u32::MAX)
                if min_index != u32::MAX {
                    // Convert the 1D index back to 3D coordinates
                    let (width, height, _depth) = self.grid_dims; // Use stored dims
                    let z = min_index / (width * height) as u32;
                    let rem = min_index % (width * height) as u32;
                    let y = rem / width as u32;
                    let x = rem % width as u32;
                    log::info!(
                        "GPU found lowest entropy ({}) at ({}, {}, {})",
                        min_entropy_val,
                        x,
                        y,
                        z
                    );
                    Some((x as usize, y as usize, z as usize))
                } else {
                    // No cell with entropy > 0 found (or grid was empty/fully collapsed initially)
                    log::info!(
                        "GPU reported no cell with calculable entropy (all collapsed or empty?)."
                    );
                    None
                }
            }
            Err(e) => {
                log::error!("Failed to download minimum entropy info: {}", e);
                None // Return None on error
            }
        }
    }
}
