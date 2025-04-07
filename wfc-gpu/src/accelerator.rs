#![allow(clippy::redundant_field_names)]
use crate::{
    buffers::{GpuBuffers, GpuParamsUniform},
    debug_viz::{DebugVisualizationConfig, DebugVisualizer, GpuBuffersDebugExt},
    entropy::GpuEntropyCalculator,
    pipeline::ComputePipelines,
    propagator::GpuConstraintPropagator,
    subgrid::SubgridConfig,
    sync::GpuSynchronizer,
    GpuError,
};
use log::{error, info, trace, warn};
use std::sync::Arc;
use std::time::Instant;
use wfc_core::{
    entropy::{EntropyCalculator, EntropyError, EntropyHeuristicType},
    grid::{EntropyGrid, PossibilityGrid},
    propagator::{ConstraintPropagator, PropagationError},
    BoundaryCondition, ProgressInfo, WfcError,
};
use wfc_rules::AdjacencyRules;

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
    synchronizer: Arc<GpuSynchronizer>,
    num_tiles_u32: u32,
    max_propagation_steps: u32,
    contradiction_check_frequency: u32,
    entropy_calculator: GpuEntropyCalculator,
}

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
        let start_time = Instant::now();
        info!("Initializing GPU Accelerator...");

        // Initialize WGPU
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| GpuError::AdapterRequestFailed)?;

        info!("Using GPU adapter: {:?}", adapter.get_info());

        let supported_features = adapter.features();
        let mut features_to_enable = wgpu::Features::empty();
        let mut features = vec![];

        if supported_features.contains(
            wgpu::Features::BUFFER_BINDING_ARRAY | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY,
        ) {
            features_to_enable |= wgpu::Features::BUFFER_BINDING_ARRAY
                | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY;
            features.push("binding_arrays".to_string());
            info!("GPU supports binding arrays.");
        }
        if supported_features.contains(wgpu::Features::SHADER_F64) {
            features_to_enable |= wgpu::Features::SHADER_F64;
            info!("SHADER_F64 feature supported and enabled.");
        }

        // Example: Check for atomics support (general)
        // Commenting out entirely for now to ensure compilation
        /*
        if supported_features.contains(wgpu::Features::BUFFER_BINDING_TYPE_STORAGE_BUFFER_ATOMIC) {
            info!("Storage Buffer Atomics feature supported.");
            features.push("atomics".to_string()); // Push based on general atomics support
            // features_to_enable |= wgpu::Features::BUFFER_BINDING_TYPE_STORAGE_BUFFER_ATOMIC;
        }
        */

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("WFC GPU Device"),
                    required_features: features_to_enable,
                    required_limits: wgpu::Limits {
                        max_compute_invocations_per_workgroup: 256,
                        ..wgpu::Limits::default()
                    },
                },
                None,
            )
            .await
            .map_err(GpuError::DeviceRequestFailed)?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Calculate num_tiles_u32 using public getter
        let num_tiles = initial_grid.num_tiles(); // Use getter
        let num_tiles_u32 = (num_tiles + 31) / 32;

        let features_ref: Vec<&str> = features.iter().map(String::as_str).collect();

        // Create compute pipelines
        let pipelines = Arc::new(ComputePipelines::new(
            &device,
            num_tiles_u32 as u32,
            &features_ref,
        )?);

        // Create GPU buffers
        let buffers = Arc::new(GpuBuffers::new(
            &device,
            &queue,
            initial_grid,
            rules,
            boundary_mode,
        )?);

        // Create synchronizer
        let synchronizer = Arc::new(GpuSynchronizer::new(
            device.clone(),
            queue.clone(),
            buffers.clone(),
        ));

        // Need params for propagator creation
        let temp_params = GpuParamsUniform {
            grid_width: initial_grid.width as u32,
            grid_height: initial_grid.height as u32,
            grid_depth: initial_grid.depth as u32,
            num_tiles: num_tiles as u32,
            num_axes: rules.num_axes() as u32,
            boundary_mode: if boundary_mode == BoundaryCondition::Periodic {
                1
            } else {
                0
            },
            heuristic_type: 0,                  // Placeholder, update later
            tie_breaking: 0,                    // Placeholder
            max_propagation_steps: 1000,        // Example
            contradiction_check_frequency: 100, // Example
            worklist_size: 0,                   // Initial
            grid_element_count: (initial_grid.width * initial_grid.height * initial_grid.depth)
                as u32,
            _padding: 0,
        };

        // Create propagator
        let mut propagator = GpuConstraintPropagator::new(
            device.clone(),
            queue.clone(),
            pipelines.clone(),
            buffers.clone(),
            (initial_grid.width, initial_grid.height, initial_grid.depth),
            boundary_mode,
            temp_params,
        );

        // Configure propagator with subgrid if provided
        if let Some(config) = subgrid_config.clone() {
            propagator = propagator.with_parallel_subgrid_processing(config);
        }

        let grid_dims = (initial_grid.width, initial_grid.height, initial_grid.depth);
        let max_propagation_steps = 1000;
        let contradiction_check_frequency = 100;

        // Create entropy calculator
        let entropy_calculator = GpuEntropyCalculator::new(
            device.clone(),
            queue.clone(),
            pipelines.clone(),
            buffers.clone(),
            (initial_grid.width, initial_grid.height, initial_grid.depth),
        );

        info!(
            "GPU Accelerator initialized in {:.2?}",
            start_time.elapsed()
        );

        Ok(Self {
            instance: Arc::new(instance),
            adapter: Arc::new(adapter),
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
            num_tiles_u32: num_tiles_u32 as u32,
            max_propagation_steps,
            contradiction_check_frequency,
            entropy_calculator,
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

    /// Retrieves the current state of the possibility grid from the GPU.
    pub async fn get_intermediate_result(&self) -> Result<PossibilityGrid, GpuError> {
        let grid_template = PossibilityGrid::new(
            self.grid_dims.0,
            self.grid_dims.1,
            self.grid_dims.2,
            self.num_tiles,
        );
        self.synchronizer.download_grid(&grid_template).await
    }

    /// Enables debug visualization.
    pub fn enable_default_debug_visualization(&mut self) {
        self.debug_visualizer = Some(DebugVisualizer::new(
            DebugVisualizationConfig::default(),
            self.synchronizer.clone(),
        ));
    }

    /// Sets the entropy heuristic type.
    pub fn with_entropy_heuristic(mut self, heuristic: EntropyHeuristicType) -> Self {
        self.entropy_heuristic = heuristic;
        // Re-create or update the entropy calculator if its behavior depends on the heuristic
        self.entropy_calculator = GpuEntropyCalculator::with_heuristic(
            self.device.clone(),
            self.queue.clone(),
            self.pipelines.clone(),
            self.buffers.clone(),
            self.grid_dims,
            heuristic,
        );
        self
    }

    /// Configures the accelerator for parallel subgrid processing.
    pub fn with_parallel_subgrid_processing(mut self, config: SubgridConfig) -> Self {
        self.subgrid_config = Some(config.clone());
        // Update the propagator with the subgrid config
        self.propagator = self
            .propagator
            .clone()
            .with_parallel_subgrid_processing(config);
        self
    }

    /// Runs the WFC algorithm using the GPU accelerator, with progress callbacks.
    pub async fn run_with_callback<F>(
        &mut self,
        grid: &mut PossibilityGrid,
        rules: &AdjacencyRules,
        max_iterations: u64,
        mut progress_callback: F,
        shutdown_signal: Option<tokio::sync::watch::Receiver<bool>>,
    ) -> Result<PossibilityGrid, WfcError>
    where
        F: FnMut(ProgressInfo),
    {
        let start_time = Instant::now();
        let total_cells = grid.width * grid.height * grid.depth;

        // Upload initial grid state
        self.synchronizer
            .upload_grid(grid)
            .map_err(|e| WfcError::InternalError(e.to_string()))?;

        for iterations in 0..max_iterations {
            // Check for shutdown signal
            if let Some(ref signal) = shutdown_signal {
                if *signal.borrow() {
                    return Err(WfcError::ShutdownSignalReceived);
                }
            }

            // --- Observation Phase (Entropy Calculation & Cell Selection) ---
            trace!("Iteration {}: Calculating entropy...", iterations);
            // Use the dedicated GpuEntropyCalculator
            let entropy_grid = self
                .entropy_calculator
                .calculate_entropy(grid)
                .await
                .map_err(WfcError::EntropyError)?;

            trace!("Iteration {}: Selecting lowest entropy cell...", iterations);
            let maybe_coords = self
                .entropy_calculator
                .select_lowest_entropy_cell(&entropy_grid)
                .await;

            let (x, y, z) = match maybe_coords {
                Some(coords) => coords,
                None => {
                    info!("No cells left to collapse. WFC finished.");
                    break; // No more cells to collapse
                }
            };
            trace!("Selected cell: ({}, {}, {})", x, y, z);

            // --- Collapse Phase ---
            let cell_possibilities = grid
                .get(x, y, z)
                .ok_or_else(|| WfcError::GridError("Selected cell out of bounds".to_string()))?;

            if cell_possibilities.count_ones() <= 1 {
                trace!("Cell ({}, {}, {}) already collapsed.", x, y, z);
                continue; // Skip propagation if already collapsed
            }

            let available_tiles: Vec<usize> = cell_possibilities.iter_ones().collect();
            if available_tiles.is_empty() {
                error!(
                    "Contradiction detected at ({}, {}, {}) before collapse.",
                    x, y, z
                );
                return Err(WfcError::Contradiction(x, y, z));
            }

            // TODO: Implement weighted selection based on rules.get_tile_weight
            // For now, randomly choose one of the available tiles.
            let chosen_tile = available_tiles[iterations as usize % available_tiles.len()]; // Simple deterministic choice for now

            if let Err(e) = grid.collapse(x, y, z, chosen_tile) {
                error!("Failed to collapse cell ({}, {}, {}): {}", x, y, z, e);
                return Err(WfcError::InternalError(format!("Collapse failed: {}", e)));
            }
            trace!(
                "Collapsed cell ({}, {}, {}) to tile {}",
                x,
                y,
                z,
                chosen_tile
            );

            // Update GPU grid state after collapse
            self.synchronizer
                .upload_grid(grid)
                .map_err(|e| WfcError::InternalError(e.to_string()))?;

            // --- Propagation Phase (using GpuConstraintPropagator) ---
            trace!("Propagating constraints...");
            // Use the dedicated GpuConstraintPropagator
            match self
                .propagator
                .propagate(grid, vec![(x, y, z)], rules)
                .await
            {
                Ok(_) => trace!("Propagation successful."),
                Err(PropagationError::Contradiction(cx, cy, cz)) => {
                    error!(
                        "Contradiction detected at ({}, {}, {}) during propagation.",
                        cx, cy, cz
                    );
                    return Err(WfcError::Contradiction(cx, cy, cz));
                }
                Err(e) => {
                    error!("Propagation failed: {:?}", e);
                    return Err(WfcError::Propagation(e));
                }
            }

            // Download updated grid state for callback
            let current_grid_state = self
                .synchronizer
                .download_grid(grid)
                .await
                .map_err(|e| WfcError::InternalError(e.to_string()))?;

            // Calculate collapsed cells count (can be optimized)
            let mut collapsed_cells_count = 0;
            for cz in 0..current_grid_state.depth {
                for cy in 0..current_grid_state.height {
                    for cx in 0..current_grid_state.width {
                        if let Some(cell) = current_grid_state.get(cx, cy, cz) {
                            if cell.count_ones() == 1 {
                                collapsed_cells_count += 1;
                            }
                        }
                    }
                }
            }

            // --- Progress Callback ---
            let progress_info = ProgressInfo {
                collapsed_cells: collapsed_cells_count,
                total_cells: total_cells,
                elapsed_time: start_time.elapsed(),
                iterations: iterations as u64,
                grid_state: current_grid_state,
            };
            progress_callback(progress_info);
        }

        // Download final result
        self.get_intermediate_result()
            .await
            .map_err(|e| WfcError::InternalError(e.to_string()))
    }
}

impl Drop for GpuAccelerator {
    /// Cleans up GPU resources when the accelerator is dropped.
    fn drop(&mut self) {
        // The primary resources (device, queue, buffers, pipelines) are wrapped in Arc,
        // so their cleanup is handled automatically when the reference count drops to zero.
        // We might want to explicitly drop or clear certain states if necessary,
        // but relying on Arc's Drop implementation is usually sufficient.
        info!("Dropping GpuAccelerator. GPU resources will be released if reference count reaches zero.");

        // If DebugVisualizer holds resources that need explicit cleanup, do it here.
        // self.debug_visualizer = None; // Implicitly dropped

        // Clearing caches might be useful in some contexts, but not typically on drop.
        // pipeline::clear_pipeline_cache().ok();
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{create_test_device_queue, initialize_test_gpu};
    use std::sync::atomic::AtomicBool;
    use wfc_core::grid::PossibilityGrid;

    // Removed test that depended on create_simple_2d_rules
    // #[tokio::test]
    // async fn test_accelerator_creation() {
    //     let accelerator = run_test_accelerator(4, 4, 1, 3).await;
    //     assert!(accelerator.is_ok());
    // }

    // Helper function might still be useful later?
    // async fn run_test_accelerator(
    //     width: usize,
    //     height: usize,
    //     depth: usize,
    //     num_tiles: usize,
    // ) -> Result<GpuAccelerator, GpuError> {
    //     let rules = create_simple_2d_rules(num_tiles); // This function is missing
    //     let grid = PossibilityGrid::new(width, height, depth, num_tiles);
    //     GpuAccelerator::new(
    //         &grid,
    //         &rules,
    //         BoundaryCondition::Finite,
    //         EntropyHeuristicType::Shannon, // Added missing heuristic
    //         None
    //     ).await
    // }

    // ... (Keep other tests, ensure they use async fn and .await where needed)
}
