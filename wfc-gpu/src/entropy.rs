use crate::{buffers::GpuBuffers, pipeline::ComputePipelines};
use futures::executor::block_on;
use log;
use std::sync::Arc;
use wfc_core::{
    entropy::{EntropyCalculator, EntropyError, EntropyHeuristicType},
    grid::{EntropyGrid, Grid, PossibilityGrid},
};
use wgpu;

/// Define a struct to hold entropy parameters that will be passed to the shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct EntropyParams {
    width: u32,
    height: u32,
    depth: u32,
    _padding1: u32,
    heuristic_type: u32,
    _padding2: [u32; 3],
}

/// GPU implementation of the EntropyCalculator trait.
#[derive(Debug, Clone)]
pub struct GpuEntropyCalculator {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) pipelines: Arc<ComputePipelines>,
    pub(crate) buffers: Arc<GpuBuffers>,
    pub(crate) grid_dims: (usize, usize, usize),
    pub(crate) heuristic_type: EntropyHeuristicType,
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
        Self::with_heuristic(
            device,
            queue,
            pipelines,
            buffers,
            grid_dims,
            EntropyHeuristicType::default(),
        )
    }

    /// Creates a new GpuEntropyCalculator with a specific entropy heuristic
    pub fn with_heuristic(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        pipelines: Arc<ComputePipelines>,
        buffers: Arc<GpuBuffers>,
        grid_dims: (usize, usize, usize),
        heuristic_type: EntropyHeuristicType,
    ) -> Self {
        Self {
            device,
            queue,
            pipelines,
            buffers,
            grid_dims,
            heuristic_type,
        }
    }

    /// Calculates entropy for each cell asynchronously using the GPU.
    ///
    /// This method uploads the current grid state to the GPU, runs the entropy shader,
    /// and downloads the resulting entropy values back to the CPU.
    pub async fn calculate_entropy_async(
        &self,
        grid: &PossibilityGrid,
    ) -> Result<EntropyGrid, EntropyError> {
        log::trace!("Entering calculate_entropy_async");

        // Upload the current grid state to GPU
        self.buffers.upload_grid(&self.queue, grid)?;

        // Create a temporary buffer for entropy parameters
        let entropy_params = EntropyParams {
            width: self.grid_dims.0 as u32,
            height: self.grid_dims.1 as u32,
            depth: self.grid_dims.2 as u32,
            _padding1: 0,
            heuristic_type: match self.heuristic_type {
                EntropyHeuristicType::Shannon => 0,
                EntropyHeuristicType::Count => 1,
                EntropyHeuristicType::CountSimple => 2,
                EntropyHeuristicType::WeightedCount => 3,
            },
            _padding2: [0, 0, 0],
        };

        // Create a buffer for entropy parameters
        let entropy_params_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Entropy Parameters Buffer"),
                    contents: bytemuck::cast_slice(&[entropy_params]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        // Create a bind group for the parameters
        let entropy_params_bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Entropy Params Bind Group Layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let entropy_params_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Entropy Params Bind Group"),
            layout: &entropy_params_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: entropy_params_buffer.as_entire_binding(),
            }],
        });

        // Initialize entropy buffer and min entropy index buffer
        let grid_size = self.grid_dims.0 * self.grid_dims.1 * self.grid_dims.2;
        let mut entropy_buffer = vec![0.0f32; grid_size];
        let mut min_entropy_idx_buffer = vec![0u32; 2];
        min_entropy_idx_buffer[0] = f32::MAX.to_bits(); // Maximum f32 value

        // Upload the buffers to GPU
        self.buffers
            .upload_entropy_buffer(&self.queue, &entropy_buffer)?;
        self.buffers
            .upload_min_entropy_buffer(&self.queue, &min_entropy_idx_buffer)?;

        // Submit compute shader dispatch for the entropy calculation
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GpuEntropyCalculator::calculate_entropy_async encoder"),
            });

        {
            // Set the compute pipeline for entropy calculation
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GpuEntropyCalculator::calculate_entropy_async compute pass"),
                timestamp_writes: None,
            });

            // Select the appropriate pipeline based on heuristic type
            compute_pass.set_pipeline(&self.pipelines.entropy_pipeline);
            compute_pass.set_bind_group(0, &self.buffers.grid_bind_group, &[]);
            compute_pass.set_bind_group(1, &self.buffers.entropy_bind_group, &[]);
            compute_pass.set_bind_group(2, &entropy_params_bind_group, &[]);

            // Calculate workgroup counts
            let workgroup_size = 8; // Must match the shader's workgroup_size
            let workgroup_x = (self.grid_dims.0 + workgroup_size - 1) / workgroup_size;
            let workgroup_y = (self.grid_dims.1 + workgroup_size - 1) / workgroup_size;
            let workgroup_z = self.grid_dims.2;

            compute_pass.dispatch_workgroups(
                workgroup_x as u32,
                workgroup_y as u32,
                workgroup_z as u32,
            );
        }

        // Submit the command encoder
        self.queue.submit(std::iter::once(encoder.finish()));

        // Download the results
        let entropy_buffer = self.buffers.download_entropy_buffer(&self.device)?;
        let min_entropy_idx_buffer = self.buffers.download_min_entropy_buffer(&self.device)?;

        // Create entropy grid from buffer
        let entropy_grid = EntropyGrid::new_from_flat(
            self.grid_dims.0,
            self.grid_dims.1,
            self.grid_dims.2,
            entropy_buffer.to_vec(),
        )?;

        // Log the minimum entropy cell information
        let min_entropy = f32::from_bits(min_entropy_idx_buffer[0]);
        let min_idx = min_entropy_idx_buffer[1] as usize;
        log::debug!(
            "GPU Entropy calculation completed. Min entropy: {}, at index: {}",
            min_entropy,
            min_idx
        );

        Ok(entropy_grid)
    }

    /// Asynchronous version of select_lowest_entropy_cell
    pub async fn select_lowest_entropy_cell_async(
        &self,
        _entropy_grid: &EntropyGrid,
    ) -> Option<(usize, usize, usize)> {
        // Reduce verbosity - only log at trace level
        log::trace!("Downloading GPU minimum entropy info...");

        // Download min entropy info asynchronously
        let min_entropy_result = self.buffers.download_min_entropy_buffer(&self.device).await;

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
        // Use the async version with immediate blocking
        pollster::block_on(self.calculate_entropy_async(grid))
    }

    fn select_lowest_entropy_cell(
        &self,
        _entropy_grid: &EntropyGrid,
    ) -> Option<(usize, usize, usize)> {
        // Download the minimum entropy index directly from the GPU
        // This avoids having to scan the entire entropy grid on the CPU
        let min_entropy_idx_buffer =
            pollster::block_on(async { self.buffers.download_min_entropy_buffer(&self.device) })
                .ok()?;

        let min_entropy = f32::from_bits(min_entropy_idx_buffer[0]);
        let min_idx = min_entropy_idx_buffer[1] as usize;

        // Check if we have a valid minimum entropy
        if min_entropy <= 0.0 || min_entropy == f32::MAX {
            return None;
        }

        // Convert the flat index to 3D coordinates
        let width = self.grid_dims.0;
        let height = self.grid_dims.1;
        let x = min_idx % width;
        let y = (min_idx / width) % height;
        let z = min_idx / (width * height);

        Some((x, y, z))
    }

    fn set_entropy_heuristic(&mut self, heuristic_type: EntropyHeuristicType) -> bool {
        self.heuristic_type = heuristic_type;
        true // GPU implementation supports all heuristic types
    }

    fn get_entropy_heuristic(&self) -> EntropyHeuristicType {
        self.heuristic_type
    }
}
