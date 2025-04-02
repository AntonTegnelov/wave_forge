use crate::{buffers::GpuBuffers, pipeline::ComputePipelines, GpuError};
use log::info;
use std::sync::Arc;
use wfc_core::{
    entropy::EntropyCalculator,
    grid::{EntropyGrid, Grid, PossibilityGrid},
    propagator::{ConstraintPropagator, PropagationError},
}; // Use Arc for shared GPU resources
use wfc_rules::AdjacencyRules; // Added import

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
    fn propagate(
        &mut self,
        _grid: &mut PossibilityGrid,
        updated_coords: Vec<(usize, usize, usize)>,
        _rules: &AdjacencyRules,
    ) -> Result<(), PropagationError> {
        log::debug!("Starting GPU iterative propagation...");
        let propagate_start = std::time::Instant::now();

        let (width, height, _depth) = self.grid_dims;
        let _max_worklist_size = (width * height * _depth) as u64; // Max possible work items

        // --- 1. Prepare Initial Worklist ---
        let mut current_worklist: Vec<u32> = updated_coords
            .iter()
            .map(|&(x, y, z)| (z * width * height + y * width + x) as u32)
            .collect();

        if current_worklist.is_empty() {
            log::debug!("GPU propagate: No initial updates to process.");
            return Ok(());
        }

        let mut iteration = 0;
        const MAX_ITERATIONS: u32 = 100; // Safeguard against infinite loops

        // --- 2. Iterative Propagation Loop ---
        loop {
            iteration += 1;
            log::debug!("GPU Propagation Iteration: {}", iteration);
            if iteration > MAX_ITERATIONS {
                log::error!(
                    "GPU propagation exceeded max iterations ({}), assuming divergence.",
                    MAX_ITERATIONS
                );
                return Err(PropagationError::GpuCommunicationError(
                    "Propagation exceeded max iterations".to_string(),
                ));
            }

            let worklist_size = current_worklist.len() as u32;
            log::debug!("  Worklist size: {}", worklist_size);
            if worklist_size == 0 {
                log::debug!("  Worklist empty, propagation complete.");
                break; // No more work
            }

            // --- 2a. Upload Current Worklist & Reset Buffers ---
            self.buffers
                .upload_updates(&self.queue, &current_worklist)
                .map_err(|e| {
                    PropagationError::GpuCommunicationError(format!(
                        "Failed to upload updates: {}",
                        e
                    ))
                })?;

            self.buffers
                .reset_contradiction_flag(&self.queue)
                .map_err(|e| {
                    PropagationError::GpuCommunicationError(format!(
                        "Failed to reset contradiction flag: {}",
                        e
                    ))
                })?;

            self.buffers
                .reset_contradiction_location(&self.queue)
                .map_err(|e| {
                    PropagationError::GpuCommunicationError(format!(
                        "Failed to reset contradiction location: {}",
                        e
                    ))
                })?;

            self.buffers
                .reset_output_worklist_count(&self.queue)
                .map_err(|e| {
                    PropagationError::GpuCommunicationError(format!(
                        "Failed to reset output worklist count: {}",
                        e
                    ))
                })?;

            // --- 2b. Update Uniforms ---
            self.buffers
                .update_params_worklist_size(&self.queue, worklist_size)
                .map_err(|e| {
                    PropagationError::GpuCommunicationError(format!(
                        "Failed to update worklist size uniform: {}",
                        e
                    ))
                })?;

            // --- 2c. Create Command Encoder & Bind Group ---
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("Propagation Encoder Iter {}", iteration)),
                });

            let propagation_bind_group =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("Propagation Bind Group Iter {}", iteration)),
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
                            resource: self.buffers.updates_buf.as_entire_binding(),
                        }, // Input worklist
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.buffers.output_worklist_buf.as_entire_binding(),
                        }, // Output worklist
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.buffers.params_uniform_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: self.buffers.output_worklist_count_buf.as_entire_binding(),
                        }, // Output count
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

            // --- 2d. Dispatch Compute ---
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Propagation Compute Pass Iter {}", iteration)),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.pipelines.propagation_pipeline);
                compute_pass.set_bind_group(0, &propagation_bind_group, &[]);

                let workgroup_size = 64u32;
                let workgroups_needed = std::cmp::max(1, worklist_size.div_ceil(workgroup_size));
                log::trace!(
                    "Dispatching iter {} with {} workgroups.",
                    iteration,
                    workgroups_needed
                );
                compute_pass.dispatch_workgroups(workgroups_needed, 1, 1);
            } // End compute pass

            // --- 2e. Submit and Wait ---
            self.queue.submit(std::iter::once(encoder.finish()));
            // Wait for GPU to finish before checking results and preparing next iteration
            self.device.poll(wgpu::Maintain::Wait);

            // --- 2f. Check for Contradiction ---
            let contradiction_detected = pollster::block_on(
                self.buffers
                    .download_contradiction_flag(&self.device, &self.queue),
            )
            .map_err(|e| {
                PropagationError::GpuCommunicationError(format!(
                    "Failed to download contradiction flag: {}",
                    e
                ))
            })?;

            if contradiction_detected {
                log::warn!(
                    "GPU propagation contradiction detected in iteration {}.",
                    iteration
                );
                let location_index = pollster::block_on(
                    self.buffers
                        .download_contradiction_location(&self.device, &self.queue),
                )
                .unwrap_or(u32::MAX); // Default to MAX if download fails

                if location_index != u32::MAX {
                    let z = location_index / (width * height) as u32;
                    let rem = location_index % (width * height) as u32;
                    let y = rem / width as u32;
                    let x = rem % width as u32;
                    log::error!("Contradiction location: ({}, {}, {})", x, y, z);
                    return Err(PropagationError::Contradiction(
                        x as usize, y as usize, z as usize,
                    ));
                } else {
                    log::error!("Contradiction detected, but location unknown.");
                    return Err(PropagationError::Contradiction(0, 0, 0)); // Generic contradiction
                }
            }

            // --- 2g. Prepare for Next Iteration ---
            let output_count = pollster::block_on(
                self.buffers
                    .download_output_worklist_count(&self.device, &self.queue),
            )
            .map_err(|e| {
                PropagationError::GpuCommunicationError(format!(
                    "Failed to download output worklist count: {}",
                    e
                ))
            })?;

            log::debug!("  Output worklist count: {}", output_count);

            if output_count == 0 {
                log::debug!("  No new updates generated, propagation stable.");
                break; // Stable state reached
            }

            // Copy output worklist to input worklist buffer for the next iteration
            let copy_size = (output_count as u64 * std::mem::size_of::<u32>() as u64)
                .min(self.buffers.updates_buf.size()); // Don't copy more than the buffer size

            if copy_size > 0 {
                let mut copy_encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some(&format!("Worklist Copy Encoder Iter {}", iteration)),
                        });
                copy_encoder.copy_buffer_to_buffer(
                    &self.buffers.output_worklist_buf,
                    0,
                    &self.buffers.updates_buf,
                    0,
                    copy_size,
                );
                self.queue.submit(std::iter::once(copy_encoder.finish()));
                self.device.poll(wgpu::Maintain::Wait); // Ensure copy completes
            }

            // The worklist size for the *next* iteration is the output_count from *this* iteration
            // But we need to actually read the buffer contents to use as the `current_worklist` vec for the next loop
            // This is inefficient. A better approach would be to ping-pong buffers or directly use the output count.
            // For simplicity now, we'll just use the count to determine the *size* for the next iteration's uniform.
            // We don't actually need the `current_worklist` Vec<u32> anymore.
            // Let's modify the loop to directly use the count.

            // Re-thinking: We MUST update the `worklist_size` uniform *before* dispatch.
            // The `updates_buf` *is* the input buffer.
            // So, the copy MUST happen before the next loop iteration starts.
            // The size for the next iteration *is* `output_count`.

            // We don't need to re-create the `current_worklist` Vec. We just need the count.
            // The size for the next iteration is set here.
            // The loop condition `worklist_size == 0` handles termination.
            // The `updates_buf` now contains the worklist for the next iteration.
            current_worklist.resize(output_count as usize, 0); // Not actually used, just for size logic
        } // End loop

        log::info!(
            "GPU iterative propagation finished successfully after {} iterations in {:?}.",
            iteration,
            propagate_start.elapsed()
        );
        Ok(())
    }
}
