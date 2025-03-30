use crate::{buffers::GpuBuffers, pipeline::ComputePipelines, GpuError};
use log::{info, warn};
use std::sync::Arc;
use wfc_core::{
    entropy::EntropyCalculator,
    grid::{EntropyGrid, Grid, PossibilityGrid},
    propagator::{ConstraintPropagator, PropagationError},
    rules::AdjacencyRules,
}; // Use Arc for shared GPU resources

// Main struct holding GPU state and implementing core traits
#[allow(dead_code)] // Allow unused fields while implementation is pending
pub struct GpuAccelerator {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipelines: ComputePipelines,
    buffers: GpuBuffers,
    grid_dims: (usize, usize, usize),
}

impl GpuAccelerator {
    pub async fn new(
        initial_grid: &PossibilityGrid,
        rules: &AdjacencyRules,
    ) -> Result<Self, GpuError> {
        info!("Initializing GPU Accelerator...");

        // 1. Initialize wgpu Instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(), // Or specify e.g., Vulkan, DX12
            ..Default::default()
        });

        // 2. Request Adapter (physical GPU)
        info!("Requesting GPU adapter...");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None, // No surface needed for compute
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::AdapterRequestFailed)?;
        info!("Adapter selected: {:?}", adapter.get_info());

        // 3. Request Device (logical device) & Queue
        info!("Requesting logical device and queue...");
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("WFC GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                    // memory_hints: wgpu::MemoryHints::Performance, // Commented out - investigate feature/version issue later
                },
                None, // Optional trace path
            )
            .await
            .map_err(GpuError::DeviceRequestFailed)?;
        info!("Device and queue obtained.");

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // 4. Create pipelines (Placeholder - needs shader loading)
        // TODO: Implement shader loading and pipeline creation
        warn!("Pipeline creation is not yet implemented.");
        let pipelines = ComputePipelines::new(&device)?;

        // 5. Create buffers (Placeholder - needs implementation)
        // TODO: Implement buffer creation and data upload
        warn!("Buffer creation is not yet implemented.");
        let buffers = GpuBuffers::new(&device, initial_grid, rules)?;

        let grid_dims = (initial_grid.width, initial_grid.height, initial_grid.depth);

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            pipelines,
            buffers,
            grid_dims,
        })
    }
}

impl EntropyCalculator for GpuAccelerator {
    #[must_use]
    fn calculate_entropy(&self, _grid: &PossibilityGrid) -> EntropyGrid {
        // Assuming grid state is primarily managed on the GPU via self.buffers.grid_possibilities_buf
        // _grid parameter is technically unused as we read directly from the GPU buffer.
        // Consider changing the trait or method signature if this becomes an issue.
        log::debug!("Running GPU calculate_entropy...");

        let (width, height, depth) = self.grid_dims;
        let num_cells = width * height * depth;

        // 1. Create Command Encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Entropy Compute Encoder"),
            });

        // 2. Create Bind Group
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
            ],
        });

        // 3. Begin Compute Pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Entropy Compute Pass"),
                timestamp_writes: None, // Add timestamps later if needed for profiling
            });

            compute_pass.set_pipeline(&self.pipelines.entropy_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch - Calculate workgroup counts
            let workgroup_size_x = 8;
            let workgroup_size_y = 8;
            let workgroup_size_z = 1;
            let workgroups_x = (width as u32).div_ceil(workgroup_size_x);
            let workgroups_y = (height as u32).div_ceil(workgroup_size_y);
            let workgroups_z = (depth as u32).div_ceil(workgroup_size_z);

            log::debug!(
                "Dispatching entropy shader with workgroups: ({}, {}, {})",
                workgroups_x,
                workgroups_y,
                workgroups_z
            );
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        } // End compute pass scope

        // 4. Submit to Queue
        self.queue.submit(std::iter::once(encoder.finish()));
        log::debug!("Entropy compute shader submitted.");

        // 5. Download results (synchronously for now to match trait)
        // We use pollster::block_on to wait for the async download_entropy to complete.
        // This is simpler for now but might block the calling thread.
        // Consider making the trait method async in the future if needed.
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
                    log::error!(
                        "GPU entropy result size mismatch: expected {}, got {}",
                        num_cells,
                        entropy_data.len()
                    );
                    // Return an empty/error grid or panic? For now, create with potentially wrong data.
                    // Construct Grid directly since data is public
                    Grid {
                        width,
                        height,
                        depth,
                        data: entropy_data,
                    }
                } else {
                    // 6. Create Grid from the downloaded data
                    // Construct Grid directly since data is public
                    Grid {
                        width,
                        height,
                        depth,
                        data: entropy_data,
                    }
                }
            }
            Err(e) => {
                log::error!("Failed to download entropy results: {}", e);
                // Return a default/error grid
                EntropyGrid::new(width, height, depth) // Creates grid with default (0.0) values
            }
        }
    }

    #[must_use]
    fn find_lowest_entropy(&self, entropy_grid: &EntropyGrid) -> Option<(usize, usize, usize)> {
        // This is typically done on the CPU after calculating entropy.
        // If entropy calculation is done on GPU and result downloaded,
        // we can reuse the CPU implementation from wfc_core.
        // Alternatively, a reduction shader could find the minimum on the GPU,
        // but that adds complexity.
        log::info!(
            "Using CPU logic to find lowest entropy from GPU-calculated grid (or placeholder)"
        );
        // Placeholder: Re-use CPU logic by creating a temporary CPU calculator
        let cpu_calc = wfc_core::entropy::CpuEntropyCalculator::new();
        cpu_calc.find_lowest_entropy(entropy_grid)
        // todo!() // Replace placeholder if GPU reduction is implemented
    }
}

impl ConstraintPropagator for GpuAccelerator {
    fn propagate(
        &mut self,
        _grid: &mut PossibilityGrid,
        _updated_coords: Vec<(usize, usize, usize)>,
        _rules: &AdjacencyRules,
    ) -> Result<(), PropagationError> {
        // Implementation still TODO, but signature matches now.
        // Use `rules` parameter if needed for setup or shader uniforms.
        let _ = _rules; // Mark as used for now to avoid warning
        let _ = _grid; // Mark as used
        let _ = _updated_coords; // Mark as used

        // This implementation needs to:
        // 1. Upload updated_coords to buffers.updates_buf.
        // 2. Reset contradiction flag buffer.
        // 3. Create command encoder.
        // 4. Create bind group for propagation shader.
        // 5. Set pipeline and bind group.
        // 6. Dispatch compute for propagation (potentially iteratively if needed).
        // 7. Copy contradiction_flag_buf to contradiction_staging_buf.
        // 8. Submit command encoder.
        // 9. Map staging buffer, read contradiction flag, unmap.
        // 10. If contradiction, return Err(PropagationError::Contradiction).
        // 11. (Optional/Complex) If PossibilityGrid needs updating on CPU side, download results.
        // This should also likely be async.
        log::warn!(
            "GPU propagate needs careful implementation regarding CPU/GPU state sync and async operations"
        );
        todo!()
    }
}
