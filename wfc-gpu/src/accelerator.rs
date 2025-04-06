#![allow(clippy::redundant_field_names)]
use crate::buffers::{GpuBuffers, GpuParamsUniform};
use crate::debug_viz::{DebugVisualizationConfig, DebugVisualizer, GpuBuffersDebugExt};
use crate::sync::GpuSynchronizer;
use crate::{pipeline::ComputePipelines, subgrid::SubgridConfig, GpuError};
use async_trait::async_trait;
use bytemuck;
use futures::channel::oneshot;
use log::{debug, error, info, warn};
use std::time::Instant;
use wfc_core::{
    entropy::{EntropyCalculator, EntropyError, EntropyHeuristicType},
    grid::{EntropyGrid, PossibilityGrid},
    propagator::{self, ConstraintPropagator, PropagationError},
    BoundaryCondition, GridState, ProgressUpdate, WfcResult,
};
use wfc_rules::AdjacencyRules;
use wgpu;

/// Manages the WGPU context and orchestrates GPU-accelerated WFC operations.
///
/// This struct holds the necessary WGPU resources (instance, adapter, device, queue)
/// and manages the compute pipelines (`ComputePipelines`) and GPU buffers (`GpuBuffers`)
/// required for accelerating entropy calculation and constraint propagation.
///
/// It implements the `EntropyCalculator` and `ConstraintPropagator` traits from `wfc-core`,
/// providing GPU-acceleration.
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
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct GpuAccelerator {
    /// WGPU instance.
    instance: Arc<wgpu::Instance>,
    /// WGPU adapter (connection to physical GPU).
    adapter: Arc<wgpu::Adapter>,
    /// WGPU logical device.
    device: Arc<wgpu::Device>,
    /// WGPU command queue.
    queue: Arc<wgpu::Queue>,
    /// Collection of compute pipelines for different WFC operations.
    pipelines: Arc<ComputePipelines>,
    /// Collection of GPU buffers holding grid state, rules, etc.
    buffers: Arc<GpuBuffers>,
    /// Configuration for subgrid processing (if used).
    subgrid_config: Option<SubgridConfig>,
    /// Debug visualizer for algorithm state
    debug_visualizer: Option<DebugVisualizer>,
    grid_dims: (usize, usize, usize),
    boundary_mode: BoundaryCondition,
    num_tiles: usize,
    propagator: GpuConstraintPropagator,
    entropy_heuristic: EntropyHeuristicType,
    /// GPU synchronizer for handling data transfer between CPU and GPU
    synchronizer: GpuSynchronizer,
}

// Import the concrete GPU implementations
use crate::propagator::GpuConstraintPropagator;

impl GpuAccelerator {
    /// Creates a new GPU accelerator for Wave Function Collapse.
    ///
    /// Initializes the GPU device, compute pipelines, and buffers required for the WFC algorithm.
    /// This method performs asynchronous GPU operations and must be awaited.
    ///
    /// # Arguments
    ///
    /// * `initial_grid` - The initial grid state containing all possibilities.
    /// * `rules` - The adjacency rules for the WFC algorithm.
    /// * `boundary_mode` - Whether to use periodic or finite boundary conditions.
    /// * `subgrid_config` - Optional configuration for subgrid processing.
    ///
    /// # Returns
    ///
    /// A `Result` containing either a new `GpuAccelerator` or a `GpuError`.
    ///
    /// # Constraints
    ///
    /// * Dynamically supports arbitrary numbers of unique tile types, limited only by available GPU memory.
    pub async fn new(
        initial_grid: &PossibilityGrid,
        rules: &AdjacencyRules,
        boundary_mode: BoundaryCondition,
        entropy_heuristic: EntropyHeuristicType,
        subgrid_config: Option<SubgridConfig>,
    ) -> Result<Self, GpuError> {
        info!(
            "Entered GpuAccelerator::new with boundary mode {:?}",
            boundary_mode
        );
        info!("Initializing GPU Accelerator...");

        // Calculate num_tiles_u32 here
        let num_tiles = rules.num_tiles();
        let u32s_per_cell = (num_tiles + 31) / 32; // Ceiling division

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
                        // Request features needed for detection proxy
                        required_features: wgpu::Features::BUFFER_BINDING_ARRAY
                            | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY,
                        required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
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

        // --- Determine Supported Features ---
        // TODO: Integrate with ShaderRegistry / dedicated feature detection module
        let supports_atomics = device
            .features()
            .contains(wgpu::Features::BUFFER_BINDING_ARRAY)
            || device
                .features()
                .contains(wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY);
        log::info!(
            "GPU supports potentially relevant features (atomics proxy): {}",
            supports_atomics
        );
        let features = if supports_atomics {
            vec!["atomics"]
        } else {
            vec![]
        };

        // 4. Create pipelines (uses device, returns Cloneable struct)
        // Pass features vector to ComputePipelines::new
        let pipelines = Arc::new(ComputePipelines::new(
            &device,
            u32s_per_cell as u32,
            &features, // Pass detected features
        )?);

        // 5. Create buffers (uses device & queue, returns Cloneable struct)
        let buffers = Arc::new(GpuBuffers::new(
            &device,
            &queue,
            initial_grid,
            rules,
            boundary_mode,
        )?); // Pass boundary_mode to GpuBuffers::new

        // Create the GPU synchronizer
        let synchronizer = GpuSynchronizer::new(device.clone(), queue.clone(), buffers.clone());

        let grid_dims = (initial_grid.width, initial_grid.height, initial_grid.depth);

        // Create GpuParamsUniform
        let params = GpuParamsUniform {
            grid_width: grid_dims.0 as u32,
            grid_height: grid_dims.1 as u32,
            grid_depth: grid_dims.2 as u32,
            num_tiles: num_tiles as u32,
            num_axes: rules.num_axes() as u32,
            boundary_mode: match boundary_mode {
                BoundaryCondition::Finite => 0,
                BoundaryCondition::Periodic => 1,
            },
            heuristic_type: 0,
            tie_breaking: 0,
            max_propagation_steps: 10000,
            contradiction_check_frequency: 100,
            worklist_size: 0,
            grid_element_count: (grid_dims.0 * grid_dims.1 * grid_dims.2) as u32,
            _padding: 0,
        };

        // Create the propagator instance
        let propagator = GpuConstraintPropagator::new(
            device.clone(),
            queue.clone(),
            pipelines.clone(),
            buffers.clone(),
            grid_dims,
            boundary_mode,
            params,
        );

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            pipelines,
            buffers,
            grid_dims,
            boundary_mode,
            num_tiles,
            propagator,
            subgrid_config,
            debug_visualizer: None,
            entropy_heuristic,
            synchronizer,
        })
    }

    // --- Public Accessors for Shared Resources ---

    /// Returns a clone of the Arc-wrapped WGPU Device.
    pub fn device(&self) -> Arc<wgpu::Device> {
        self.device.clone()
    }

    /// Returns a clone of the Arc-wrapped WGPU Queue.
    pub fn queue(&self) -> Arc<wgpu::Queue> {
        self.queue.clone()
    }

    /// Returns a clone of the Arc-wrapped ComputePipelines.
    pub fn pipelines(&self) -> Arc<ComputePipelines> {
        self.pipelines.clone()
    }

    /// Returns a clone of the Arc-wrapped GpuBuffers.
    pub fn buffers(&self) -> Arc<GpuBuffers> {
        self.buffers.clone()
    }

    /// Returns the grid dimensions (width, height, depth).
    pub fn grid_dims(&self) -> (usize, usize, usize) {
        self.grid_dims
    }

    /// Returns the boundary mode used by this accelerator.
    pub fn boundary_mode(&self) -> BoundaryCondition {
        self.boundary_mode
    }

    /// Returns the number of unique transformed tiles.
    pub fn num_tiles(&self) -> usize {
        self.num_tiles
    }

    /// Returns the GPU parameters uniform.
    pub fn params(&self) -> GpuParamsUniform {
        self.propagator.params
    }

    /// Retrieves the current intermediate grid state from the GPU.
    ///
    /// This method allows accessing partial or in-progress WFC results before the algorithm completes.
    /// It downloads the current state of the grid possibilities from the GPU and converts them back
    /// to a PossibilityGrid.
    ///
    /// # Returns
    ///
    /// A `Result` containing either the current grid state or a `GpuError`.
    pub async fn get_intermediate_result(&self) -> Result<PossibilityGrid, GpuError> {
        // Create template grid with same dimensions and num_tiles
        let template = PossibilityGrid::new(
            self.grid_dims.0,
            self.grid_dims.1,
            self.grid_dims.2,
            self.num_tiles,
        );

        // Use the synchronizer to download the grid
        self.synchronizer.download_grid(&template).await
    }

    /// Enables parallel subgrid processing for large grids.
    ///
    /// When enabled, the Wave Function Collapse algorithm will divide large grids
    /// into smaller subgrids that can be processed independently, potentially
    /// improving performance for large problem sizes.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration for subgrid division and processing.
    ///              If None, a default configuration will be used.
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining.
    pub fn with_parallel_subgrid_processing(&mut self, config: Option<SubgridConfig>) -> &mut Self {
        // Create a propagator with parallel subgrid processing enabled
        let config = config.unwrap_or_default();
        self.propagator = self
            .propagator
            .clone()
            .with_parallel_subgrid_processing(config.clone());
        self.subgrid_config = Some(config);
        self
    }

    /// Disables parallel subgrid processing.
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining.
    pub fn without_parallel_subgrid_processing(&mut self) -> &mut Self {
        self.propagator = self
            .propagator
            .clone()
            .without_parallel_subgrid_processing();
        self
    }

    /// Enable debug visualization with the given configuration
    pub fn enable_debug_visualization(&mut self, config: DebugVisualizationConfig) {
        // Pass the synchronizer when creating DebugVisualizer
        let synchronizer = Arc::new(GpuSynchronizer::new(
            self.device.clone(),
            self.queue.clone(),
            self.buffers.clone(),
        ));
        self.debug_visualizer = Some(DebugVisualizer::new(config, synchronizer));
    }

    /// Enable debug visualization with default settings
    pub fn enable_default_debug_visualization(&mut self) {
        // Ensure default() also gets a synchronizer if it uses new()
        let synchronizer = Arc::new(GpuSynchronizer::new(
            self.device.clone(),
            self.queue.clone(),
            self.buffers.clone(),
        ));
        // Assuming DebugVisualizer::default() now internally calls new() with a default synchronizer,
        // or we need a specific `default_with_sync` method.
        // For now, let's use new() directly.
        self.debug_visualizer = Some(DebugVisualizer::new(
            DebugVisualizationConfig::default(),
            synchronizer,
        ));
        // self.debug_visualizer = Some(DebugVisualizer::default()); // This would panic if Default needs real resources
    }

    /// Disable debug visualization
    pub fn disable_debug_visualization(&mut self) {
        self.debug_visualizer = None;
    }

    /// Check if debug visualization is enabled
    pub fn has_debug_visualization(&self) -> bool {
        self.debug_visualizer.is_some()
    }

    /// Get a reference to the debug visualizer, if enabled
    pub fn debug_visualizer(&self) -> Option<&DebugVisualizer> {
        self.debug_visualizer.as_ref()
    }

    /// Get a mutable reference to the debug visualizer, if enabled
    pub fn debug_visualizer_mut(&mut self) -> Option<&mut DebugVisualizer> {
        self.debug_visualizer.as_mut()
    }

    /// Take a snapshot of the current state for visualization purposes
    pub async fn take_debug_snapshot(&mut self) -> Result<(), GpuError> {
        if let Some(visualizer) = &mut self.debug_visualizer {
            let buffers = Arc::clone(&self.buffers);
            // let device = Arc::clone(&self.device);
            // let queue = Arc::clone(&self.queue);

            // Call the extension method on buffers, passing only the visualizer
            buffers.take_debug_snapshot(visualizer)?;
        }

        Ok(())
    }

    /// Sets the entropy heuristic used for entropy calculation
    ///
    /// # Arguments
    ///
    /// * `heuristic_type` - The entropy heuristic type to use
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining
    pub fn with_entropy_heuristic(&mut self, heuristic_type: EntropyHeuristicType) -> &mut Self {
        self.entropy_heuristic = heuristic_type;
        self
    }

    /// Gets the current entropy heuristic type
    pub fn entropy_heuristic(&self) -> EntropyHeuristicType {
        self.entropy_heuristic
    }

    /// Runs the WFC algorithm on the GPU asynchronously.
    ///
    /// # Arguments
    /// * `grid` - The mutable `PossibilityGrid` to operate on.
    /// * `rules` - The `AdjacencyRules` for the model.
    /// * `max_iterations` - The maximum number of iterations before stopping.
    /// * `callback` - A closure called periodically with progress updates.
    ///
    /// # Returns
    /// A `Result` containing a `WfcResult` enum indicating success, contradiction, or max iterations,
    /// or a `GpuError` if a GPU-specific error occurs.
    #[allow(unused_variables)]
    pub async fn run_with_callback<'grid, F>(
        &mut self,
        grid: &'grid mut PossibilityGrid,
        rules: &AdjacencyRules,
        max_iterations: usize,
        mut callback: F,
    ) -> Result<WfcResult, GpuError>
    where
        F: FnMut(ProgressUpdate<'grid>) + Send,
    {
        debug!("Starting GPU WFC run.");
        let start_time = Instant::now();
        let mut iterations = 0;
        let max_iterations = max_iterations.unwrap_or(usize::MAX);

        // Initial grid upload
        self.synchronizer.upload_grid(grid)?;

        loop {
            if iterations >= max_iterations {
                info!("Reached max iterations ({}), stopping.", max_iterations);
                return Ok(WfcResult::MaxIterationsReached(iterations));
            }
            iterations += 1;

            // --- 1. Calculate Entropy (GPU) ---
            debug!("Iteration {}: Calculating entropy...", iterations);
            // Important: Reset the min entropy buffer BEFORE calculating entropy
            self.synchronizer
                .reset_min_entropy_buffer()
                .await
                .map_err(GpuError::SynchronizerError)?;

            // Placeholder: GPU Entropy calculation would go here.
            // For now, we might need a CPU fallback or assume GPU did its work
            // let entropy_result = self.calculate_entropy(grid).await?;
            // debug!("Entropy calculation done.");

            // --- 2. Select Lowest Entropy Cell (Download result from GPU) ---
            debug!("Iteration {}: Selecting lowest entropy cell...", iterations);
            let selection_result = self
                .select_lowest_entropy_cell(self.entropy_heuristic)
                .await;
            // debug!("Cell selection done.");

            let pos = match selection_result {
                Ok(Some(p)) => {
                    debug!("Selected cell: {:?}", p);
                    p
                }
                Ok(None) => {
                    // Grid is fully collapsed and consistent
                    debug!("Grid fully collapsed successfully.");
                    // TODO: Final download/sync?
                    // Download final grid state?
                    self.synchronizer
                        .download_grid_state(grid)
                        .await
                        .map_err(GpuError::SynchronizerError)?;

                    return Ok(WfcResult::Success(iterations));
                }
                Err(e) => {
                    error!("Error during entropy selection/download: {:?}", e);
                    // Attempt to download contradiction info
                    match self.synchronizer.download_contradiction_status().await {
                        Ok(Some(loc)) => {
                            warn!(
                                "Contradiction detected during cell selection at location: {:?}. Attempting grid download.",
                                loc
                            );
                            // Download potentially partial/contradictory grid state for inspection
                            let _ = self.synchronizer.download_grid_state(grid).await; // Ignore error here
                            return Ok(WfcResult::Contradiction(loc.map(|l| l as usize)));
                        }
                        Ok(None) => {
                            // No specific contradiction location found, return original error
                            warn!("No specific contradiction location found, returning original selection error.");
                            // It's likely a contradiction still, report it generically
                            return Ok(WfcResult::Contradiction(None));
                        }
                        Err(sync_err) => {
                            error!(
                                "Error downloading contradiction status after selection error: {:?}",
                                sync_err
                            );
                            // We know there was an error, report contradiction generically
                            return Ok(WfcResult::Contradiction(None));
                        }
                    }
                }
            };

            // Callback: Observing
            callback(ProgressUpdate {
                iterations,
                elapsed_time: start_time.elapsed(),
                current_state: GridState::Observing,
                grid: &*grid, // Borrow grid for the callback
                last_change: Some((pos.0, pos.1, pos.2)),
            });

            // --- 3. Collapse Cell (CPU Grid Update & Upload Changes) ---
            debug!(
                "Iteration {}: Collapsing cell at ({}, {}, {})...",
                iterations, pos.0, pos.1, pos.2
            );
            // Get possibilities for the selected cell from the CPU grid
            let possibilities = match grid.get_possibilities(pos.0, pos.1, pos.2) {
                Some(p) => p,
                None => {
                    error!(
                        "Attempted to collapse cell ({}, {}, {}) which is out of bounds.",
                        pos.0, pos.1, pos.2
                    );
                    // This indicates a logic error somewhere (selection gave invalid coords?)
                    return Err(GpuError::InternalError(
                        "Selected cell coordinates are out of bounds.".to_string(),
                    ));
                }
            };

            // Find the first available tile_id to collapse to
            let chosen_tile_id = possibilities
                .iter()
                .enumerate()
                .find(|(_, &enabled)| enabled)
                .map(|(id, _)| id);

            let updated_coords = if let Some(tile_id) = chosen_tile_id {
                match grid.collapse(pos.0, pos.1, pos.2, tile_id) {
                    Ok(updates) => {
                        debug!(
                            "Collapsed cell ({}, {}, {}) to tile {}. Changes: {:?}",
                            pos.0, pos.1, pos.2, tile_id, updates
                        );
                        // Upload these specific changes to the GPU
                        self.synchronizer
                            .upload_grid_changes(updates.as_slice()) // Pass slice of updates
                            .await
                            .map_err(GpuError::SynchronizerError)?;
                        debug!("Uploaded collapse changes to GPU.");
                        // The updates returned by grid.collapse are the initial worklist
                        updates
                            .iter()
                            .map(|upd| (upd.x, upd.y, upd.z))
                            .collect::<Vec<_>>() // Collect coords for propagation input
                    }
                    Err(e) => {
                        // This might happen if the cell was *already* collapsed (e.g., by init)
                        warn!(
                            "CPU grid collapse failed for cell ({:?}): {}. This might be okay if pre-collapsed.",
                            pos, e
                        );
                        // Even if CPU collapse fails, the GPU state might be different.
                        // We selected based on GPU state, so proceed with propagation from that cell.
                        // Upload the intended collapse *anyway*? Or just propagate from the selected cell?
                        // Let's assume we still need to propagate from `pos`.
                        // Create a single update for the selected cell to trigger propagation
                        let single_update = grid.create_update(pos.0, pos.1, pos.2);
                        if let Some(update_data) = single_update {
                            self.synchronizer
                                .upload_grid_changes(&[update_data]) // Pass slice of updates
                                .await
                                .map_err(GpuError::SynchronizerError)?;
                            debug!(
                                "Uploaded single cell update to GPU after CPU collapse warning."
                            );
                            vec![(pos.0, pos.1, pos.2)]
                        } else {
                            error!(
                                "Failed to create update for selected cell {:?}, cannot proceed.",
                                pos
                            );
                            return Err(GpuError::InternalError(format!(
                                "Failed to create grid update for selected cell {:?}",
                                pos
                            )));
                        }
                    }
                }
            } else {
                // This should ideally not happen if select_lowest_entropy_cell returned a valid pos
                error!(
                    "Selected cell ({:?}) has no possible tiles left according to CPU grid!",
                    pos
                );
                // Indicates a potential divergence or earlier error. Report contradiction.
                return Ok(WfcResult::Contradiction(Some(
                    grid.coords_to_index(pos.0, pos.1, pos.2),
                )));
            };

            // --- 4. Propagate Constraints (GPU) ---
            if !updated_coords.is_empty() {
                debug!(
                    "Iteration {}: Propagating constraints from {} initial updates...",
                    iterations,
                    updated_coords.len()
                );
                // Call the propagator directly using the stored instance
                match self
                    .propagator // Use the stored propagator instance
                    .propagate(
                        &self.device, // Pass required resources
                        &self.queue,
                        &mut self.buffers,
                        // `updated_coords` might need conversion if propagator expects different format
                        // Assuming propagator handles Vec<(usize, usize, usize)> for now
                    )
                    .await
                {
                    Ok(_) => {
                        debug!("Propagation successful.");
                        // Download the contradiction status *after* propagation
                        match self.synchronizer.download_contradiction_status().await {
                            Ok(Some(loc)) => {
                                warn!(
                                    "Contradiction detected by GPU after propagation at {:?}.",
                                    loc
                                );
                                // Download potentially partial/contradictory grid state
                                let _ = self.synchronizer.download_grid_state(grid).await;
                                callback(ProgressUpdate {
                                    iterations,
                                    elapsed_time: start_time.elapsed(),
                                    current_state: GridState::Contradiction,
                                    grid: &*grid,
                                    last_change: Some((pos.0, pos.1, pos.2)),
                                });
                                return Ok(WfcResult::Contradiction(loc.map(|l| l as usize)));
                            }
                            Ok(None) => {
                                // No contradiction found by GPU
                                // Report progress after successful propagation
                                // Need to download the grid state to show it in the callback
                                self.synchronizer
                                    .download_grid_state(grid)
                                    .await
                                    .map_err(GpuError::SynchronizerError)?;
                                callback(ProgressUpdate {
                                    iterations,
                                    elapsed_time: start_time.elapsed(),
                                    current_state: GridState::Propagated,
                                    grid: &*grid,
                                    last_change: Some((pos.0, pos.1, pos.2)),
                                });
                            }
                        }
                    }
                    Err(prop_err) => {
                        error!("Propagation failed with error: {:?}", prop_err);
                        // Attempt to check GPU contradiction buffer anyway, just in case
                        match self.synchronizer.download_contradiction_status().await {
                            Ok(Some(loc)) => {
                                warn!(
                                    "Contradiction location found after propagation error: {:?}",
                                    loc
                                );
                                let _ = self.synchronizer.download_grid_state(grid).await;
                                callback(ProgressUpdate {
                                    iterations,
                                    elapsed_time: start_time.elapsed(),
                                    current_state: GridState::Contradiction,
                                    grid: &*grid,
                                    last_change: Some((pos.0, pos.1, pos.2)),
                                });
                                return Ok(WfcResult::Contradiction(loc.map(|l| l as usize)));
                            }
                            Err(sync_err) => {
                                error!(
                                    "Error downloading contradiction status after propagation error: {:?}",
                                    sync_err
                                );
                                // Return the original propagation error as GpuError
                                return Err(GpuError::PropagationError(prop_err.to_string()));
                            }
                            Ok(None) => {
                                // No specific contradiction, but propagation failed.
                                warn!("Propagation failed, but no specific contradiction location found.");
                                callback(ProgressUpdate {
                                    iterations,
                                    elapsed_time: start_time.elapsed(),
                                    current_state: GridState::Error, // Indicate generic error state
                                    grid: &*grid,
                                    last_change: Some((pos.0, pos.1, pos.2)),
                                });
                                // Return the original propagation error as GpuError
                                return Err(GpuError::PropagationError(prop_err.to_string()));
                            }
                        }
                    }
                }
            } else {
                // No updates were generated by the collapse (e.g., cell was already collapsed on CPU)
                debug!(
                    "Iteration {}: No updates generated by CPU collapse, skipping propagation.",
                    iterations
                );
                // We still need to check if the grid is fully collapsed or if there's an issue.
                // Checking contradiction status is redundant here as propagation didn't run.
                // Re-running selection logic to see if grid is complete.
                match self
                    .select_lowest_entropy_cell(self.entropy_heuristic)
                    .await
                {
                    Ok(None) => {
                        debug!("Grid determined complete after no-update iteration.");
                        self.synchronizer
                            .download_grid_state(grid)
                            .await
                            .map_err(GpuError::SynchronizerError)?;
                        return Ok(WfcResult::Success(iterations));
                    }
                    Ok(Some(_)) => {
                        // Still cells to collapse, loop continues
                        debug!("Grid not yet complete, continuing loop.");
                    }
                    Err(e) => {
                        error!("Error re-selecting lowest entropy cell after no-update iteration: {:?}", e);
                        // Assume it might be a contradiction discovered during selection
                        match self.synchronizer.download_contradiction_status().await {
                            Ok(Some(loc)) => {
                                return Ok(WfcResult::Contradiction(loc.map(|l| l as usize)))
                            }
                            _ => return Ok(WfcResult::Contradiction(None)), // Report generic contradiction
                        }
                    }
                }
            }

            // Optional debug snapshot
            self.take_debug_snapshot().await?;
        }
    }
}

// --- Trait Implementations ---

#[async_trait]
impl propagator::ConstraintPropagator for GpuAccelerator {
    /// Delegates propagation to the stored `GpuConstraintPropagator` instance.
    async fn propagate(
        &mut self,
        grid: &mut PossibilityGrid,
        updated_coords: Vec<(usize, usize, usize)>,
        rules: &AdjacencyRules,
    ) -> Result<(), propagator::PropagationError> {
        // Upload the current grid state to GPU (synchronous call)
        self.synchronizer.upload_grid(grid).map_err(|e| {
            propagator::PropagationError::GpuSetupError(format!("Failed to upload grid: {}", e))
        })?;

        // Call the actual GPU propagation logic (this part IS async)
        let result = self.propagator.propagate(grid, updated_coords, rules).await;

        // If propagation succeeded, download the updated grid
        if result.is_ok() {
            // Download the results back to the CPU grid (this IS async)
            self.synchronizer.download_grid(grid).await.map_err(|e| {
                propagator::PropagationError::GpuCommunicationError(format!(
                    "Failed to download grid: {}",
                    e
                ))
            })?;
        }

        result
    }
}

impl EntropyCalculator for GpuAccelerator {
    /// Calculates entropy using the GPU.
    fn calculate_entropy(&self, grid: &PossibilityGrid) -> Result<EntropyGrid, EntropyError> {
        log::debug!("Calculating entropy (GPU path)...");

        // First upload the grid to GPU (synchronous for this trait method)
        // Note: Consider if this upload should happen elsewhere for efficiency
        pollster::block_on(self.synchronizer.upload_grid(grid))
            .map_err(|e| EntropyError::Other(format!("Failed to upload grid: {}", e)))?;

        // Update entropy parameters (synchronous)
        let heuristic_value = match self.entropy_heuristic {
            EntropyHeuristicType::Shannon => 0,
            EntropyHeuristicType::Count => 1,
            EntropyHeuristicType::CountSimple => 2,
            EntropyHeuristicType::WeightedCount => 3,
        };
        pollster::block_on(
            self.synchronizer
                .update_entropy_params(self.grid_dims, heuristic_value),
        )
        .map_err(|e| EntropyError::Other(format!("Failed to update entropy params: {}", e)))?;

        // Reset the min entropy buffer (synchronous)
        pollster::block_on(self.synchronizer.reset_min_entropy_buffer()).map_err(|e| {
            EntropyError::Other(format!("Failed to reset min entropy buffer: {}", e))
        })?;

        // --- Dispatch Compute Shader ---
        log::debug!("Dispatching entropy compute shader...");
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Entropy Calculation Encoder"),
            });

        // Create bind groups
        // Group 0: Grid, Entropy, Params, MinEntropyInfo
        let grid_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Entropy Grid Bind Group"),
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

        // Group 1: EntropyParams
        let entropy_params_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Entropy Params Bind Group"),
            // Assuming a separate layout exists on pipelines or creating it here if needed
            // layout: &self.pipelines.entropy_params_bind_group_layout,
            layout: &self.pipelines.entropy_bind_group_layout, // Reusing group 0 layout for simplicity? Check shader.
            entries: &[wgpu::BindGroupEntry {
                binding: 0, // Check binding index in shader!
                resource: self.buffers.entropy_params_buffer.as_entire_binding(),
            }],
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Entropy Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipelines.entropy_pipeline);
            compute_pass.set_bind_group(0, &grid_bind_group, &[]);
            // Check bind group index!
            // compute_pass.set_bind_group(1, &entropy_params_bind_group, &[]);

            // Dispatch - Calculate workgroup counts
            // TODO: Use pipelines.entropy_workgroup_size if available
            let workgroup_size = 8; // Must match shader's workgroup_size
            let (width, height, depth) = self.grid_dims;
            let workgroup_x = (width as u32 + workgroup_size - 1) / workgroup_size;
            let workgroup_y = (height as u32 + workgroup_size - 1) / workgroup_size;
            let workgroup_z = depth as u32; // Process full depth layers

            compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, workgroup_z);
        } // End compute pass scope

        // Submit to Queue
        self.queue.submit(std::iter::once(encoder.finish()));
        log::debug!("Entropy compute shader submitted.");

        // --- Download Entropy Results ---
        log::debug!("Downloading entropy results from GPU...");

        let buffer_size = self.buffers.entropy_buf.size();
        let num_elements = (buffer_size / std::mem::size_of::<f32>() as u64) as usize;
        if num_elements != (grid.width * grid.height * grid.depth) {
            return Err(EntropyError::Other(format!(
                "Entropy buffer size mismatch: Expected {} elements, found {}",
                grid.width * grid.height * grid.depth,
                num_elements
            )));
        }

        // 1. Create encoder for copy
        let mut download_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Entropy Download Encoder"),
                });

        // 2. Queue copy command
        download_encoder.copy_buffer_to_buffer(
            &self.buffers.entropy_buf,
            0,
            &self.buffers.staging_entropy_buf,
            0,
            buffer_size,
        );

        // 3. Submit copy command
        self.queue
            .submit(std::iter::once(download_encoder.finish()));

        // 4. Map staging buffer
        let buffer_slice = self.buffers.staging_entropy_buf.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel(); // Need futures import
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // 5. Wait for mapping (using pollster for sync context)
        self.device.poll(wgpu::Maintain::Wait); // Ensure copy command completes
        let map_result = pollster::block_on(async {
            // Add a timeout for robustness?
            rx.await
        })
        .map_err(|_| EntropyError::Other("Mapping channel canceled".to_string()))?;

        map_result.map_err(|e| EntropyError::Other(format!("Buffer mapping failed: {:?}", e)))?;

        let entropy_data: Vec<f32>;
        {
            // 6. Get mapped range and copy data
            let mapped_range = buffer_slice.get_mapped_range();
            let bytes = mapped_range.as_ref();
            entropy_data = bytemuck::cast_slice::<u8, f32>(bytes).to_vec();
            // Drop mapped_range explicitly before unmap
            drop(mapped_range);
        }

        // 7. Unmap the buffer
        self.buffers.staging_entropy_buf.unmap();

        log::debug!(
            "Entropy results downloaded successfully ({} elements).",
            entropy_data.len()
        );

        // 8. Create EntropyGrid
        if entropy_data.len() != num_elements {
            return Err(EntropyError::Other(format!(
                "Downloaded entropy data size mismatch: Expected {} elements, got {}",
                num_elements,
                entropy_data.len()
            )));
        }
        let mut result_grid = EntropyGrid::new(grid.width, grid.height, grid.depth);
        result_grid.data = entropy_data;

        Ok(result_grid)

        /* // Remove placeholder
        log::warn!("GPU Entropy calculation dispatched, but result download is not implemented yet. Returning placeholder.");
        let placeholder_entropy_grid = EntropyGrid::new(grid.width, grid.height, grid.depth);
        Ok(placeholder_entropy_grid)
        */
    }

    /// Selects the cell with the lowest entropy based on the GPU calculation.
    fn select_lowest_entropy_cell(
        &self,
        heuristic: EntropyHeuristicType,
    ) -> Option<(usize, usize, usize)> {
        // Use the synchronizer to download the min entropy info
        let min_entropy_info = pollster::block_on(self.synchronizer.download_min_entropy_info())
            .ok()
            .flatten();

        // Check if we have valid results
        if min_entropy_info.is_none() {
            log::debug!("No valid min entropy info found from GPU");
            return None;
        }

        let (min_entropy, min_idx_u32) = min_entropy_info.unwrap();

        // Check if we have a valid minimum entropy (not collapsed or MAX)
        if min_entropy <= 0.0 || min_entropy == f32::MAX || min_idx_u32 == u32::MAX {
            log::debug!(
                "Min entropy found ({}) is invalid or cell already collapsed (idx: {}).",
                min_entropy,
                min_idx_u32
            );
            return None;
        }

        // Convert grid dimensions to u32 for calculations
        let width_u32 = self.grid_dims.0 as u32;
        let height_u32 = self.grid_dims.1 as u32;

        if width_u32 == 0 || height_u32 == 0 {
            log::error!("Grid dimensions are zero, cannot calculate 3D coordinates.");
            return None; // Avoid division by zero
        }

        // Calculate 3D coordinates from 1D index (using u32)
        let x = min_idx_u32 % width_u32;
        let y = (min_idx_u32 / width_u32) % height_u32;
        let z = min_idx_u32 / (width_u32 * height_u32);

        log::debug!(
            "Selected cell ({}, {}, {}) with min entropy {}",
            x,
            y,
            z,
            min_entropy
        );
        // Return coordinates as usize
        Some((x as usize, y as usize, z as usize))
    }

    fn set_entropy_heuristic(&mut self, heuristic_type: EntropyHeuristicType) -> bool {
        self.entropy_heuristic = heuristic_type;
        true
    }

    fn get_entropy_heuristic(&self) -> EntropyHeuristicType {
        self.entropy_heuristic
    }
}

impl Drop for GpuAccelerator {
    /// Performs cleanup of GPU resources when GpuAccelerator is dropped.
    ///
    /// This ensures proper cleanup following RAII principles, releasing all GPU resources.
    fn drop(&mut self) {
        debug!("GpuAccelerator being dropped, coordinating cleanup of GPU resources");

        // The actual cleanup happens through the Drop implementations of the components
        // and the Arc reference counting system when these fields are dropped.

        // We ensure a clean reference graph by explicitly dropping components in a specific order:

        // Debug visualizer should be cleaned up first if it exists
        if self.debug_visualizer.is_some() {
            debug!("Cleaning up debug visualizer");
            self.debug_visualizer = None;
        }

        // Additional synchronization might be needed for proper GPU cleanup
        debug!("Final GPU device poll for cleanup");
        self.device.poll(wgpu::Maintain::Wait);

        // Log completion
        info!("GPU Accelerator resources cleaned up");
    }
}

#[cfg(test)]
mod tests {
    use crate::subgrid::SubgridConfig;

    // Just test that SubgridConfig API exists and works
    #[test]
    fn test_subgrid_config() {
        // Create a test subgrid config
        let config = SubgridConfig {
            max_subgrid_size: 32,
            overlap_size: 3,
            min_size: 64,
        };

        // Check values
        assert_eq!(config.max_subgrid_size, 32);
        assert_eq!(config.overlap_size, 3);
        assert_eq!(config.min_size, 64);

        // Check default values
        let default_config = SubgridConfig::default();
        assert_eq!(default_config.max_subgrid_size, 64);
        assert_eq!(default_config.overlap_size, 2);
        assert_eq!(default_config.min_size, 128);
    }
}
