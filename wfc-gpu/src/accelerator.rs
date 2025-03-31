use crate::{buffers::GpuBuffers, pipeline::ComputePipelines, GpuError};
use log::{info, warn};
use std::sync::Arc;
use wfc_core::{
    entropy::EntropyCalculator,
    grid::{EntropyGrid, Grid, PossibilityGrid},
    propagator::{ConstraintPropagator, PropagationError},
    rules::AdjacencyRules,
}; // Use Arc for shared GPU resources

/// Accelerates WFC computations using the GPU through wgpu.
///
/// This module provides GPU accelerated implementations of:
/// - Entropy calculation for finding the next cell to collapse
/// - Constraint propagation for updating neighbor cell possibilities
///
/// # Implementation Details
///
/// The GPU accelerator works by:
/// 1. Transferring grid possibilities and constraints to the GPU
/// 2. Running compute shaders to perform batch operations in parallel
/// 3. Reading back results to the CPU
///
/// # Error Prevention
///
/// To avoid GPU hangs and crashes, the implementation includes:
/// - Explicit synchronization between CPU and GPU
/// - Timeout mechanisms for GPU operations
/// - Proper device polling to ensure GPU work completes
/// - Bounds checking in shaders to prevent memory access violations
/// - Workgroup size matching between shader declaration and dispatching logic
/// - Safe buffer management with proper staging buffers
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
    /// Creates a new GPU accelerator for WFC operations.
    ///
    /// Initializes the GPU context, loads shaders, and creates buffers needed for
    /// entropy calculation and constraint propagation.
    ///
    /// # Arguments
    ///
    /// * `initial_grid` - The initial grid with possibility data for each cell
    /// * `rules` - Adjacency rules defining valid tile arrangements
    ///
    /// # Returns
    ///
    /// A Result containing either the initialized accelerator or a GPU error
    ///
    /// # Safety considerations
    ///
    /// - Enforces a maximum of 128 tiles (4 u32s per cell) to prevent shader array overflows
    /// - Uses Arc for shared GPU resources to ensure proper lifetime management
    /// - Handles GPU adapter and device initialization failures gracefully
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
    /// Calculates the entropy for each cell in the grid using the GPU.
    ///
    /// Entropy is a measure of uncertainty, with higher values indicating
    /// cells with more possibilities that are less constrained.
    ///
    /// # Arguments
    ///
    /// * `_grid` - The current grid state (not directly used, as we read from GPU buffer)
    ///
    /// # Returns
    ///
    /// An EntropyGrid containing calculated entropy values for each cell
    ///
    /// # Implementation Details
    ///
    /// 1. Creates a compute pass with appropriate bind groups
    /// 2. Dispatches the entropy calculation shader with appropriate workgroup size
    /// 3. Reads back the results from GPU memory
    /// 4. Handles potential errors like mismatched buffer sizes
    ///
    /// # Safety and Performance
    ///
    /// - Uses a 1D dispatch model for simplicity and reliability
    /// - Properly scales workgroups based on grid size
    /// - Performs bounds checking in the shader to prevent out-of-bounds access
    #[must_use]
    fn calculate_entropy(&self, _grid: &PossibilityGrid) -> EntropyGrid {
        // Assuming grid state is primarily managed on the GPU via self.buffers.grid_possibilities_buf
        // _grid parameter is technically unused as we read directly from the GPU buffer.
        // Consider changing the trait or method signature if this becomes an issue.
        log::debug!("Running GPU calculate_entropy...");

        let (width, height, depth) = self.grid_dims;
        let num_cells = width * height * depth;

        // Reset the min entropy buffer before dispatch
        if let Err(e) = self.buffers.reset_min_entropy_info(&self.queue) {
            log::error!("Failed to reset min entropy info buffer: {}", e);
            // Return an empty/error grid
            return EntropyGrid::new(width, height, depth);
        }

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
                // Add binding for min_entropy_info buffer
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buffers.min_entropy_info_buf.as_entire_binding(),
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

    /// Finds the cell with the lowest non-zero entropy.
    ///
    /// This helps identify the next cell to collapse in the WFC algorithm.
    /// Currently uses CPU implementation after downloading entropy values.
    ///
    /// # Arguments
    ///
    /// * `entropy_grid` - Grid containing entropy values for each cell
    ///
    /// # Returns
    ///
    /// Option containing the coordinates of the cell with lowest entropy,
    /// or None if no valid cell is found
    #[must_use]
    fn find_lowest_entropy(&self, _entropy_grid: &EntropyGrid) -> Option<(usize, usize, usize)> {
        // GPU reduction happens during calculate_entropy. Here we just download the result.
        // _entropy_grid parameter is unused because the min info is read directly from its GPU buffer.
        log::debug!("Downloading GPU minimum entropy info...");

        // Download the result [min_entropy_f32_bits, min_index_u32]
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
                    let (width, height, _depth) = self.grid_dims;
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

impl ConstraintPropagator for GpuAccelerator {
    /// Propagates constraints through the grid after a cell is collapsed.
    ///
    /// This is the core of the WFC algorithm, ensuring that all cells maintain
    /// consistent possibilities based on the adjacency rules and the constraints
    /// introduced by collapsing a cell.
    ///
    /// # Arguments
    ///
    /// * `_grid` - Current grid state (not directly used in GPU implementation)
    /// * `updated_coords` - Coordinates of cells that were updated/collapsed
    /// * `_rules` - Adjacency rules (not directly used, as they're already on GPU)
    ///
    /// # Returns
    ///
    /// Result indicating success or a contradiction with coordinates
    ///
    /// # Implementation Details
    ///
    /// 1. Converts updated coordinates to indices for the GPU
    /// 2. Uploads the indices to the GPU
    /// 3. Sets up and dispatches the compute shader
    /// 4. Explicitly synchronizes and waits for GPU completion
    /// 5. Checks for contradictions
    ///
    /// # Safety and Hang Prevention
    ///
    /// Several measures prevent indefinite hanging:
    /// - Emergency timeout to abort operations that take too long
    /// - Explicit GPU polling to ensure work completes
    /// - Shader bounds checking to prevent out-of-bounds memory access
    /// - Simple 1D workgroup model to avoid thread mapping errors
    /// - Output worklist size limits to prevent infinite loops
    fn propagate(
        &mut self,
        _grid: &mut PossibilityGrid,
        updated_coords: Vec<(usize, usize, usize)>,
        _rules: &AdjacencyRules,
    ) -> Result<(), PropagationError> {
        log::debug!("Running GPU propagate...");

        // Add emergency timeout to prevent indefinite hangs
        let timeout = std::time::Duration::from_secs(1); // Reduced from 5s to 1s for faster testing feedback
        let propagate_start = std::time::Instant::now();

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
            // The shader now uses a 1D workgroup size (64,1,1) = 64 threads per workgroup
            // This is much simpler and less error-prone

            let workgroup_size = 64u32;

            // Simple 1D dispatch - just calculate how many workgroups we need
            let workgroups_needed = std::cmp::max(1, worklist_size.div_ceil(workgroup_size));

            log::debug!(
                "Dispatching propagation shader for {} updates with {} workgroups (size {}).",
                worklist_size,
                workgroups_needed,
                workgroup_size
            );

            // Print these critical parameters so we can see them in test output
            println!(
                "Dispatching propagation shader: worklist_size={}, workgroups={}, threads_per_workgroup={}",
                worklist_size,
                workgroups_needed,
                workgroup_size
            );

            // Sanity check - dispatch will launch this many threads in total:
            let total_threads = workgroups_needed * workgroup_size;
            println!("Total threads launched: {}", total_threads);

            compute_pass.dispatch_workgroups(workgroups_needed, 1, 1);
        } // End compute pass scope

        // --- 7. Submit and Check Contradiction ---
        let submission_index = self.queue.submit(std::iter::once(encoder.finish()));
        log::debug!(
            "Propagation compute shader submitted with index: {:?}",
            submission_index
        );

        // Critical: Wait for GPU to complete work before proceeding
        // This helps ensure the GPU isn't still working when we try to read results
        println!("Waiting for GPU to complete work...");

        // Try polling with a timeout to avoid indefinite hangs
        let poll_start = std::time::Instant::now();
        let poll_timeout = std::time::Duration::from_millis(500);

        // Poll explicitly in a loop with timeout
        let mut poll_successful = false;
        while poll_start.elapsed() < poll_timeout {
            // Use Poll mode to check status without blocking
            let poll_result = self.device.poll(wgpu::Maintain::Poll);

            // Check if the queue is empty
            if let wgpu::MaintainResult::SubmissionQueueEmpty = poll_result {
                poll_successful = true;
                println!("GPU queue is empty - all work completed");
                break;
            }

            // Small delay to avoid busy loop
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Check if we successfully polled or timed out
        if !poll_successful {
            println!("WARNING: GPU polling timed out after {:?}", poll_timeout);
        } else {
            println!("GPU signaled completion.");
        }

        // Download the contradiction flag (synchronously for now)
        log::debug!("Downloading contradiction flag...");

        // Check if we're approaching timeout
        if propagate_start.elapsed() > timeout {
            log::error!("GPU propagation timed out after {:?}", timeout);
            return Err(PropagationError::Contradiction(0, 0, 0));
        }

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
