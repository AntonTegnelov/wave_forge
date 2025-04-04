use crate::{buffers::GpuBuffers, pipeline::ComputePipelines};
use futures::executor::block_on;
use log;
use std::sync::Arc;
use wfc_core::{
    entropy::{EntropyCalculator, EntropyError},
    grid::{EntropyGrid, Grid, PossibilityGrid},
};
use wgpu;

/// GPU implementation of the EntropyCalculator trait.
#[derive(Debug, Clone)]
pub struct GpuEntropyCalculator {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) pipelines: Arc<ComputePipelines>,
    pub(crate) buffers: Arc<GpuBuffers>,
    pub(crate) grid_dims: (usize, usize, usize),
}

impl GpuEntropyCalculator {
    /// Creates a new `GpuEntropyCalculator`.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        pipelines: Arc<ComputePipelines>,
        buffers: Arc<GpuBuffers>,
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

    /// Asynchronous version of calculate_entropy that allows proper async/await patterns
    pub async fn calculate_entropy_async(
        &self,
        grid: &PossibilityGrid,
    ) -> Result<EntropyGrid, EntropyError> {
        let (width, height, depth) = (grid.width, grid.height, grid.depth);
        let num_cells = width * height * depth;

        if num_cells == 0 {
            return Ok(EntropyGrid {
                width,
                height,
                depth,
                data: Vec::new(),
            });
        }

        // --- Create and configure bind group for the entropy pipeline ---
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Entropy Bind Group"),
            layout: &self.pipelines.entropy_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffers.grid_possibilities_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buffers.entropy_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buffers.params_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buffers.min_entropy_info_buf.as_entire_binding(),
                },
            ],
        });

        // --- Reset min entropy buffer to initial state ---
        let initial_min_info: &[u32] = &[f32::MAX.to_bits(), u32::MAX];
        self.queue.write_buffer(
            &self.buffers.min_entropy_info_buf,
            0,
            bytemuck::cast_slice(initial_min_info),
        );

        // --- Dispatch Compute Shader ---
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Entropy Encoder"),
            });
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
            let dispatch_x = num_cells.div_ceil(workgroup_size as usize) as u32;

            // Only log this for large grids or very verbose logging settings
            if num_cells > 10000 {
                log::debug!(
                    "Dispatching entropy shader with {} workgroups for {} cells",
                    dispatch_x,
                    num_cells
                );
            }

            compute_pass.dispatch_workgroups(dispatch_x, 1, 1);
        } // End compute pass scope

        // 4. Submit to Queue (uses Arc<Queue>)
        self.queue.submit(std::iter::once(encoder.finish()));

        // Download results asynchronously
        let download_results = self
            .buffers
            .download_results(
                self.device.clone(),
                self.queue.clone(),
                true,
                true,
                false,
                false,
                false,
                false, // Flags
            )
            .await
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

    /// Asynchronous version of select_lowest_entropy_cell
    pub async fn select_lowest_entropy_cell_async(
        &self,
        _entropy_grid: &EntropyGrid,
    ) -> Option<(usize, usize, usize)> {
        // Reduce verbosity - only log at trace level
        log::trace!("Downloading GPU minimum entropy info...");

        // Download min entropy info asynchronously
        let min_entropy_result = self
            .buffers
            .download_min_entropy_info(self.device.clone(), self.queue.clone())
            .await;

        // Handle outer Result, map error to None for this function's signature
        match min_entropy_result {
            Ok(opt_info) => {
                if let Some((min_entropy, flat_index)) = opt_info {
                    if min_entropy > 0.0 && min_entropy < f32::MAX {
                        // Check validity
                        let width = self.grid_dims.0;
                        let height = self.grid_dims.1;
                        let z = flat_index as usize / (width * height);
                        let y = (flat_index as usize % (width * height)) / width;
                        let x = flat_index as usize % width;
                        log::debug!(
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

impl EntropyCalculator for GpuEntropyCalculator {
    fn calculate_entropy(&self, grid: &PossibilityGrid) -> Result<EntropyGrid, EntropyError> {
        // Use block_on from futures executor instead of pollster
        // This maintains backward compatibility while using our async implementation
        block_on(self.calculate_entropy_async(grid))
    }

    fn select_lowest_entropy_cell(
        &self,
        entropy_grid: &EntropyGrid,
    ) -> Option<(usize, usize, usize)> {
        // Use block_on from futures executor instead of pollster
        block_on(self.select_lowest_entropy_cell_async(entropy_grid))
    }
}
