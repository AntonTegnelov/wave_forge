use crate::{buffers::GpuBuffers, pipeline::ComputePipelines, GpuError};
use log::info;
use std::sync::Arc;
use wfc_core::{
    entropy::EntropyCalculator,
    grid::{EntropyGrid, Grid, PossibilityGrid},
    propagator::{ConstraintPropagator, PropagationError},
    rules::AdjacencyRules,
}; // Use Arc for shared GPU resources

/// Manages the WGPU context and orchestrates GPU-accelerated WFC operations.
///
/// This struct holds the necessary WGPU resources (instance, adapter, device, queue)
/// and manages the compute pipelines (`ComputePipelines`) and GPU buffers (`GpuBuffers`)
/// required for accelerating entropy calculation and constraint propagation.
///
/// It implements the `EntropyCalculator` and `ConstraintPropagator` traits from `wfc-core`,
/// providing GPU-accelerated alternatives to the CPU implementations.
///
/// # Initialization
///
/// Use the asynchronous `GpuAccelerator::new()` function to initialize the WGPU context
/// and create the necessary resources based on the initial grid state and rules.
///
/// # Usage
///
/// Once initialized, the `GpuAccelerator` instance can be passed to the main WFC `run` function
/// (or used directly) to perform entropy calculation and constraint propagation steps on the GPU.
/// Data synchronization between CPU (`PossibilityGrid`) and GPU (`GpuBuffers`) is handled
/// internally by the respective trait method implementations.
#[allow(dead_code)] // Allow unused fields while implementation is pending
#[derive(Clone)] // Derive Clone
pub struct GpuAccelerator {
    instance: Arc<wgpu::Instance>, // Wrap in Arc
    adapter: Arc<wgpu::Adapter>,   // Wrap in Arc
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipelines: ComputePipelines, // Already derives Clone
    buffers: GpuBuffers,         // Already derives Clone
    grid_dims: (usize, usize, usize),
}

impl GpuAccelerator {
    /// Asynchronously creates and initializes a new `GpuAccelerator`.
    ///
    /// This involves:
    /// 1. Setting up the WGPU instance, adapter, logical device, and queue.
    /// 2. Loading and compiling compute shaders.
    /// 3. Creating compute pipelines (`ComputePipelines`).
    /// 4. Allocating GPU buffers (`GpuBuffers`) and uploading initial grid/rule data.
    ///
    /// # Arguments
    ///
    /// * `initial_grid` - A reference to the initial `PossibilityGrid` state.
    ///                    Used to determine buffer sizes and upload initial possibilities.
    /// * `rules` - A reference to the `AdjacencyRules` defining constraints.
    ///             Used to upload rule data to the GPU.
    ///
    /// # Returns
    ///
    /// * `Ok(Self)` containing the initialized `GpuAccelerator` if successful.
    /// * `Err(GpuError)` if any part of the WGPU setup, shader compilation, pipeline creation,
    ///   or buffer allocation fails.
    ///
    /// # Constraints
    ///
    /// * Currently supports a maximum of 128 unique tile types due to shader limitations.
    ///   An error will be returned if `rules.num_tiles()` exceeds this limit.
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

        // 1. Initialize wgpu Instance (Wrap in Arc)
        let instance = Arc::new(wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(), // Or specify e.g., Vulkan, DX12
            ..Default::default()
        }));

        // 2. Request Adapter (physical GPU) (Wrap in Arc)
        info!("Requesting GPU adapter...");
        let adapter = Arc::new({
            info!("Awaiting adapter request...");
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None, // No surface needed for compute
                    force_fallback_adapter: false,
                })
                .await
                .ok_or(GpuError::AdapterRequestFailed)?
        });
        info!("Adapter request returned.");
        info!("Adapter selected: {:?}", adapter.get_info());

        // 3. Request Device (logical device) & Queue (Already Arc)
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

        // 4. Create pipelines (uses device, returns Cloneable struct)
        let pipelines = ComputePipelines::new(&device)?;

        // 5. Create buffers (uses device & queue, returns Cloneable struct)
        let buffers = GpuBuffers::new(&device, &queue, initial_grid, rules)?;

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

        // 1. Create Command Encoder (uses Arc<Device>)
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Entropy Compute Encoder"),
            });

        // 2. Create Bind Group (uses Arc<Device>, Arc<BindGroupLayout>, Arc<Buffer>)
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Entropy Bind Group"),
            layout: &self.pipelines.entropy_bind_group_layout, // Access Arc<Layout>
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffers.grid_possibilities_buf.as_entire_binding(), // Access Arc<Buffer>
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buffers.entropy_buf.as_entire_binding(), // Access Arc<Buffer>
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buffers.params_uniform_buf.as_entire_binding(), // Access Arc<Buffer>
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buffers.min_entropy_info_buf.as_entire_binding(), // Access Arc<Buffer>
                },
            ],
        });

        // 3. Begin Compute Pass (uses Arc<ComputePipeline>)
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Entropy Compute Pass"),
                timestamp_writes: None, // Add timestamps later if needed for profiling
            });

            compute_pass.set_pipeline(&self.pipelines.entropy_pipeline); // Access Arc<Pipeline>
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
    /// Performs constraint propagation on the GPU using compute shaders.
    ///
    /// This method handles the propagation of changes across the grid based on
    /// the `updated_coords` provided. It leverages the GPU for parallel processing.
    ///
    /// # Workflow:
    ///
    /// 1.  **Upload Updates**: Writes the `updated_coords` list to the dedicated `updated_cells_buf` on the GPU.
    /// 2.  **Reset Contradiction Flag**: Clears the `contradiction_flag_buf` before running the shader.
    /// 3.  **Shader Dispatch**: Executes the `propagate.wgsl` compute shader.
    ///     -   The shader reads the updated cells.
    ///     -   For each updated cell, it examines neighbors based on `rules_buf`.
    ///     -   It calculates the intersection of allowed tiles and updates the main `grid_possibilities_buf`.
    ///     -   If a contradiction (empty possibility set) is detected, it sets the `contradiction_flag_buf`.
    ///     -   (Note: The current shader might perform a fixed number of iterations or a simplified propagation step).
    /// 4.  **Synchronization**: Submits the command buffer and waits for the GPU to finish (implicitly via buffer mapping or explicit polling if needed).
    /// 5.  **Check Contradiction**: Reads the `contradiction_flag_buf` back to the CPU.
    /// 6.  **Return Result**: Returns `Ok(())` if the contradiction flag is not set, or `Err(PropagationError::Contradiction(x, y, z))`
    ///     if a contradiction was detected (coordinates might be approximate or the first detected one).
    ///
    /// # Arguments
    ///
    /// * `_grid`: Mutable reference to the CPU-side `PossibilityGrid`. **Note:** This implementation primarily operates
    ///            on the GPU buffers (`self.buffers`). The CPU grid is *not* directly modified by this function.
    ///            The caller is responsible for potentially reading back the updated grid state from the GPU
    ///            using separate buffer reading methods if needed after the run completes.
    /// * `updated_coords`: A vector of `(x, y, z)` coordinates indicating cells whose possibilities have recently changed
    ///                     (e.g., due to a collapse) and need to be propagated from.
    /// * `_rules`: Reference to the `AdjacencyRules`. **Note:** This is not directly used, as the rules are assumed
    ///             to be already present in `self.buffers.rules_buf` on the GPU.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the propagation shader executed successfully without detecting a contradiction.
    /// * `Err(PropagationError::Contradiction(x, y, z))` if the shader detected and flagged a contradiction.
    ///   The coordinates might indicate the first cell where the contradiction was found.
    /// * `Err(PropagationError::Gpu*Error)` if a GPU-specific error occurs during buffer updates or shader execution,
    ///   wrapped within the `PropagationError` enum.
    #[must_use]
    fn propagate(
        &mut self,
        _grid: &mut PossibilityGrid,
        updated_coords: Vec<(usize, usize, usize)>,
        _rules: &AdjacencyRules,
    ) -> Result<(), PropagationError> {
        log::debug!("Running GPU propagate...");

        // Add emergency timeout to prevent indefinite hangs
        let _timeout = std::time::Duration::from_secs(1); // Reduced from 5s to 1s for faster testing feedback
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
                PropagationError::GpuCommunicationError(format!("Failed to upload updates: {}", e))
            })?;

        // Reset contradiction flag buffer to 0 on the GPU
        self.buffers
            .reset_contradiction_flag(&self.queue)
            .map_err(|e| {
                log::error!("Failed to reset contradiction flag on GPU: {}", e);
                PropagationError::GpuCommunicationError(format!(
                    "Failed to reset contradiction flag: {}",
                    e
                ))
            })?;

        // Reset contradiction location buffer to u32::MAX
        self.buffers
            .reset_contradiction_location(&self.queue)
            .map_err(|e| {
                log::error!("Failed to reset contradiction location on GPU: {}", e);
                PropagationError::GpuCommunicationError(format!(
                    "Failed to reset contradiction location: {}",
                    e
                ))
            })?;

        // Reset output worklist count (if iterative propagation was implemented)
        // For single pass, this isn't strictly necessary but good practice
        self.buffers
            .reset_output_worklist_count(&self.queue)
            .map_err(|e| {
                log::error!("Failed to reset output worklist count on GPU: {}", e);
                PropagationError::GpuCommunicationError(format!(
                    "Failed to reset output worklist count: {}",
                    e
                ))
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
                PropagationError::GpuCommunicationError(format!(
                    "Failed to update worklist size uniform: {}",
                    e
                ))
            })?;

        // --- 5. Create Bind Group ---
        // Note: Bind group needs to be recreated if buffer bindings change,
        // but here the buffers themselves don't change, only their contents.
        let propagation_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Propagation Bind Group"),
            layout: &self.pipelines.propagation_bind_group_layout, // Use Arc<Layout>
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffers.grid_possibilities_buf.as_entire_binding(), // Use Arc<Buffer>
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buffers.rules_buf.as_entire_binding(), // Use Arc<Buffer>
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
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.buffers.contradiction_location_buf.as_entire_binding(),
                },
            ],
        });

        // --- 6. Dispatch Compute ---
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Propagation Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipelines.propagation_pipeline); // Use Arc<Pipeline>
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

        // --- 8. Check for Contradiction ---
        log::debug!("Checking for GPU propagation contradictions...");

        // Poll device to ensure queue is processed before mapping buffers (crucial!)
        // Use a timeout to prevent indefinite hangs
        self.device.poll(wgpu::Maintain::Wait); // Wait for GPU to finish work submitted
        log::debug!("GPU poll completed.");

        // Download the contradiction flag buffer
        let contradiction_flag_result = pollster::block_on(
            self.buffers
                .download_contradiction_flag(&self.device, &self.queue),
        );

        match contradiction_flag_result {
            Ok(contradiction_detected) => {
                if contradiction_detected {
                    log::warn!("GPU propagation resulted in a contradiction.");

                    // If flag is set, download the location
                    let location_result = pollster::block_on(
                        self.buffers
                            .download_contradiction_location(&self.device, &self.queue),
                    );

                    match location_result {
                        Ok(location_index) if location_index != u32::MAX => {
                            // Convert 1D index back to 3D coordinates
                            let z = location_index / (width * height) as u32;
                            let rem = location_index % (width * height) as u32;
                            let y = rem / width as u32;
                            let x = rem % width as u32;
                            log::error!(
                                "Contradiction location reported by GPU: ({}, {}, {})",
                                x,
                                y,
                                z
                            );
                            return Err(PropagationError::Contradiction(
                                x as usize, y as usize, z as usize,
                            ));
                        }
                        Ok(_) => {
                            // Flag was set, but location wasn't (or was MAX), weird state
                            log::error!("GPU contradiction flag set, but location index was MAX. Returning generic contradiction.");
                            return Err(PropagationError::Contradiction(0, 0, 0));
                        }
                        Err(e) => {
                            log::error!(
                                "GPU contradiction flag set, but failed to download location: {}. Returning generic contradiction.",
                                e
                            );
                            // Map the download error
                            return Err(PropagationError::GpuCommunicationError(format!(
                                "Failed to download contradiction location after flag was set: {}",
                                e
                            )));
                        }
                    }
                }
                // If flag is Ok(false), no contradiction, proceed normally
            }
            Err(e) => {
                log::error!("Failed to download contradiction flag: {}", e);
                return Err(PropagationError::GpuCommunicationError(format!(
                    "Failed to download contradiction flag: {}",
                    e
                )));
            }
        }

        // TODO: Add logic to download the output worklist if implementing iterative propagation

        log::debug!(
            "GPU propagate finished successfully in {:?}.",
            propagate_start.elapsed()
        );
        Ok(())
    }
}
