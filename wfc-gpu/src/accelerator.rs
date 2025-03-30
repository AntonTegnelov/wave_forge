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
        info!("Entered GpuAccelerator::new");
        info!("Initializing GPU Accelerator...");

        // Check if the grid has a reasonable number of tiles (shader has hardcoded max of 4 u32s = 128 tiles)
        let num_tiles = rules.num_tiles();
        let u32s_per_cell = (num_tiles + 31) / 32; // Ceiling division
        if u32s_per_cell > 4 {
            return Err(GpuError::Other(format!(
                "GPU implementation supports a maximum of 128 tiles, but grid has {}",
                num_tiles
            )));
        }

        // 1. Initialize wgpu Instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(), // Or specify e.g., Vulkan, DX12
            ..Default::default()
        });

        // 2. Request Adapter (physical GPU)
        info!("Requesting GPU adapter...");
        let adapter = {
            info!("Awaiting adapter request...");
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None, // No surface needed for compute
                    force_fallback_adapter: false,
                })
                .await
                .ok_or(GpuError::AdapterRequestFailed)?
        };
        info!("Adapter request returned.");
        info!("Adapter selected: {:?}", adapter.get_info());

        // 3. Request Device (logical device) & Queue
        info!("Requesting logical device and queue...");
        let (device, queue) = {
            info!("Awaiting device request...");
            adapter
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
                .map_err(GpuError::DeviceRequestFailed)?
        };
        info!("Device request returned.");
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
            // Entropy shader uses workgroup_size(64)
            let workgroup_size = 64u32;
            let workgroups_needed = num_cells.div_ceil(workgroup_size as usize) as u32;

            log::debug!(
                "Dispatching entropy shader with {} workgroups of size 64",
                workgroups_needed
            );
            compute_pass.dispatch_workgroups(workgroups_needed, 1, 1);
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
        updated_coords: Vec<(usize, usize, usize)>,
        _rules: &AdjacencyRules,
    ) -> Result<(), PropagationError> {
        log::debug!("Running GPU propagate...");

        let (width, height, _depth) = self.grid_dims; // Prefix depth with underscore

        // --- 1. Prepare Data ---

        // Pack updated coordinates into 1D indices (u32)
        let worklist: Vec<u32> = updated_coords
            .iter()
            .map(|&(x, y, z)| (z * width * height + y * width + x) as u32)
            .collect();

        if worklist.is_empty() {
            log::debug!("GPU propagate: No updates to process.");
            return Ok(());
        }

        let worklist_size = worklist.len() as u32;

        // --- 2. Upload Worklist & Reset Buffers ---
        // Upload the worklist (updated cell indices) to the GPU buffer.
        self.buffers
            .upload_updates(&self.queue, &worklist)
            .map_err(|e| {
                log::error!("Failed to upload updates to GPU: {}", e);
                // Convert GpuError to a generic propagation error for now
                PropagationError::Contradiction(0, 0, 0) // TODO: Better error mapping
            })?;

        // Reset contradiction flag buffer to 0 on the GPU
        self.buffers
            .reset_contradiction_flag(&self.queue)
            .map_err(|e| {
                log::error!("Failed to reset contradiction flag on GPU: {}", e);
                PropagationError::Contradiction(0, 0, 0) // TODO: Better error mapping
            })?;

        // Reset output worklist count (if iterative propagation was implemented)
        // For single pass, this isn't strictly necessary but good practice
        self.buffers
            .reset_output_worklist_count(&self.queue)
            .map_err(|e| {
                log::error!("Failed to reset output worklist count on GPU: {}", e);
                PropagationError::Contradiction(0, 0, 0) // TODO: Better error mapping
            })?;

        // --- 3. Create Command Encoder ---
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Propagation Compute Encoder"),
            });

        // --- 4. Update Uniforms (Worklist Size) ---
        // We need to update the params uniform buffer with the current worklist_size
        self.buffers
            .update_params_worklist_size(&self.queue, worklist_size)
            .map_err(|e| {
                log::error!("Failed to update worklist size uniform on GPU: {}", e);
                PropagationError::Contradiction(0, 0, 0) // TODO: Better error mapping
            })?;

        // --- 5. Create Bind Group ---
        // Note: Bind group needs to be recreated if buffer bindings change,
        // but here the buffers themselves don't change, only their contents.
        let propagation_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Propagation Bind Group"),
            layout: &self.pipelines.propagation_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffers.grid_possibilities_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buffers.rules_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buffers.updates_buf.as_entire_binding(), // Contains the worklist
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buffers.output_worklist_buf.as_entire_binding(), // Output worklist (unused in single pass)
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buffers.params_uniform_buf.as_entire_binding(), // Contains worklist_size now
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.buffers.output_worklist_count_buf.as_entire_binding(), // Output count (unused in single pass)
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.buffers.contradiction_flag_buf.as_entire_binding(),
                },
            ],
        });

        // --- 6. Dispatch Compute ---
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Propagation Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipelines.propagation_pipeline);
            compute_pass.set_bind_group(0, &propagation_bind_group, &[]);

            // Dispatch based on the worklist size.
            // The shader uses a 3D workgroup size (8,8,4) = 256 threads per workgroup
            // We need to calculate how many 3D workgroups to dispatch to cover all items in the worklist
            let workgroup_size_total = 256u32; // 8x8x4 = 256 threads per workgroup
            let workgroups_needed = worklist_size.div_ceil(workgroup_size_total);

            // Since we need to dispatch in 3D, we'll use a simple distribution
            // Keeping z=1 and distributing across x,y dimensions
            let workgroups_x = (workgroups_needed as f32).sqrt().ceil() as u32;
            let workgroups_y = workgroups_needed.div_ceil(workgroups_x);
            let workgroups_z = 1u32;

            log::debug!(
                "Dispatching propagation shader for {} updates with {}x{}x{} workgroups (8x8x4 threads each).",
                worklist_size,
                workgroups_x,
                workgroups_y,
                workgroups_z
            );
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        } // End compute pass scope

        // --- 7. Submit and Check Contradiction ---
        self.queue.submit(std::iter::once(encoder.finish()));
        log::debug!("Propagation compute shader submitted.");

        // Download the contradiction flag (synchronously for now)
        log::debug!("Downloading contradiction flag...");
        let contradiction_detected = pollster::block_on(
            self.buffers
                .download_contradiction_flag(&self.device, &self.queue),
        )
        .map_err(|e| {
            log::error!("Failed to download contradiction flag: {}", e);
            PropagationError::Contradiction(0, 0, 0) // TODO: Better error mapping
        })?;

        if contradiction_detected {
            log::warn!("GPU propagation detected a contradiction!");
            // TODO: Can we get the *location* of the contradiction from the GPU?
            // Requires more complex shader logic and buffer reading.
            Err(PropagationError::Contradiction(0, 0, 0)) // Generic location for now
        } else {
            log::debug!("GPU propagation finished successfully.");
            Ok(())
        }

        // Note: If iterative propagation or CPU grid updates were needed,
        // this would involve reading the output worklist/count and potentially
        // downloading the entire grid_possibilities buffer.
    }
}
