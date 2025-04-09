#![allow(clippy::redundant_field_names)]
use crate::{
    backend::GpuBackend,
    buffers::{DownloadRequest, GpuBuffers},
    coordination::WfcCoordinator,
    debug_viz::{DebugVisualizationConfig, DebugVisualizer},
    entropy::{EntropyHeuristicType, GpuEntropyCalculator},
    pipeline::ComputePipelines,
    propagator::GpuConstraintPropagator,
    shader_registry::ShaderRegistry,
    sync::GpuSynchronizer,
    GpuError,
};
use log::{error, info, trace, warn};
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Instant;
use wfc_core::{
    entropy::{EntropyCalculator, EntropyError, EntropyHeuristicType},
    grid::{EntropyGrid, PossibilityGrid},
    propagator::{ConstraintPropagator, PropagationError},
    BoundaryCondition, ProgressInfo, WfcError,
};
use wfc_rules::AdjacencyRules;

/// Internal state for the GpuAccelerator, managed within an Arc<RwLock<>>.
#[derive(Debug)]
pub struct AcceleratorInstance {
    backend: Arc<dyn GpuBackend>,
    grid_definition: GridDefinition,
    rules: Arc<AdjacencyRules>,
    boundary_condition: BoundaryCondition,
    pipelines: Arc<ComputePipelines>,
    buffers: Arc<GpuBuffers>,
    entropy_calculator: GpuEntropyCalculator,
    propagator: GpuConstraintPropagator,
    coordinator: Box<dyn WfcCoordinator + Send + Sync + std::fmt::Debug>,
    subgrid_config: Option<SubgridConfig>,
    progress_callback: Option<Box<dyn FnMut(f32) + Send + Sync>>,
    debug_visualizer: Option<DebugVisualizer>,
}

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
#[derive(Clone)]
pub struct GpuAccelerator {
    instance: Arc<RwLock<AcceleratorInstance>>,
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
            &pipelines,
            &buffers,
            initial_grid.width as u32,
            initial_grid.height as u32,
            initial_grid.depth as u32,
            contradiction_check_frequency,
        )?;

        // Set subgrid config if provided
        if let Some(config) = subgrid_config {
            propagator.with_parallel_subgrid_processing(config);
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

        // Explicitly instantiate DefaultCoordinator
        let default_coord = DefaultCoordinator::new(entropy_calculator.clone(), propagator.clone());
        let instance = AcceleratorInstance {
            backend: backend.clone(),
            grid_definition,
            rules: Arc::new(rules.clone()),
            boundary_condition,
            pipelines: Arc::new(pipelines),
            buffers: Arc::new(buffers),
            entropy_calculator,
            propagator,
            coordinator: Box::new(default_coord),
            subgrid_config: None,
            progress_callback: None,
            debug_visualizer: None,
        };

        let accelerator = Self {
            instance: Arc::new(RwLock::new(instance)),
        };

        Ok(accelerator)
    }

    // Methods accessing instance fields now require locks

    pub fn backend(&self) -> Arc<dyn GpuBackend> {
        self.instance.read().unwrap().backend.clone()
    }

    pub fn pipelines(&self) -> Arc<ComputePipelines> {
        self.instance.read().unwrap().pipelines.clone()
    }

    pub fn buffers(&self) -> Arc<GpuBuffers> {
        self.instance.read().unwrap().buffers.clone()
    }

    pub fn grid_definition(&self) -> GridDefinition {
        self.instance.read().unwrap().grid_definition.clone()
    }

    pub fn boundary_condition(&self) -> BoundaryCondition {
        self.instance.read().unwrap().boundary_condition
    }

    pub fn num_tiles(&self) -> usize {
        self.instance.read().unwrap().grid_definition.num_tiles
    }

    pub async fn get_intermediate_result(&self) -> Result<PossibilityGrid, GpuError> {
        let instance = self.instance.read().unwrap();
        let device = instance.backend.device();
        let queue = instance.backend.queue();
        let buffers = instance.buffers.clone(); // Clone Arc for async boundary
        let grid_def = instance.grid_definition.clone();

        // Download grid possibilities
        let request = DownloadRequest {
            download_grid_possibilities: true,
            ..Default::default()
        };
        let results = buffers
            .download_results(device.clone(), queue.clone(), request)
            .await?;

        // Convert raw data to PossibilityGrid
        results.to_possibility_grid(grid_def.dims, grid_def.num_tiles)
    }

    pub fn enable_default_debug_visualization(&mut self) {
        // TODO: Fix DebugVisualizer setup after RwLock refactor
        /*
        let mut instance = self.instance.write().unwrap(); // Write lock
        if instance.debug_visualizer.is_none() {
            let config = DebugVisualizationConfig::default(); // Or load from somewhere
            // Need access to GpuSynchronizer which is not in AcceleratorInstance
            // let synchronizer = ???;
            match DebugVisualizer::new(&config, synchronizer) { // Updated signature guess
                Ok(visualizer) => instance.debug_visualizer = Some(visualizer),
                Err(e) => error!("Failed to create debug visualizer: {}", e),
            }
        }
        */
        warn!("Debug visualizer setup skipped due to ongoing refactoring.");
    }

    pub async fn run_with_callback<F>(
        &mut self, // Needs &mut self for write lock
        initial_grid: &PossibilityGrid,
        rules: &AdjacencyRules,
        max_iterations: u64,
        mut progress_callback: F,
        shutdown_signal: Option<tokio::sync::watch::Receiver<bool>>,
    ) -> Result<PossibilityGrid, WfcError>
    where
        F: FnMut(ProgressInfo),
    {
        let mut instance = self.instance.write().unwrap(); // Acquire write lock early
        let start_time = Instant::now();
        let mut iteration = 0;
        let total_cells = instance.grid_definition.total_cells();

        // --- Initial State Synchronization ---
        trace!("Uploading initial grid state to GPU...");
        instance
            .buffers
            .upload_grid_state(instance.backend.queue(), initial_grid)?;
        trace!("Initial grid state uploaded.");

        // --- Main WFC Loop ---
        loop {
            // Check for shutdown signal
            if let Some(ref signal) = shutdown_signal {
                if *signal.borrow() {
                    info!("Shutdown signal received, stopping WFC execution.");
                    return Err(WfcError::ExecutionInterrupted(
                        "Shutdown signal".to_string(),
                    ));
                }
            }

            if iteration >= max_iterations {
                error!(
                    "Maximum iterations ({}) reached without convergence.",
                    max_iterations
                );
                return Err(WfcError::MaxIterationsReached(max_iterations));
            }

            iteration += 1;
            trace!("--- Iteration {} ---", iteration);

            // 1. Calculate Entropy & Select Cell (using Coordinator)
            let selection_result = instance
                .coordinator
                .coordinate_entropy_and_selection(
                    &instance.entropy_calculator,
                    &instance.buffers, // Pass buffers needed by coordinator
                    instance.backend.device(),
                    instance.backend.queue(),
                )
                .await;

            let selected_coords = match selection_result {
                Ok(Some(coords)) => {
                    // Keep coords as (usize, usize, usize) for contradiction reporting
                    coords
                }
                Ok(None) => {
                    // No cell with entropy > 0 found, convergence?
                    // Check for contradictions first
                    if instance
                        .propagator
                        .check_for_contradiction(
                            instance.backend.device(),
                            instance.backend.queue(),
                        )
                        .await?
                    {
                        error!("Contradiction detected after entropy calculation.");
                        return Err(WfcError::Contradiction(None)); // Contradiction expects Option<(usize, usize, usize)>?
                    }
                    info!(
                        "Converged after {} iterations ({:.2?})",
                        iteration,
                        start_time.elapsed()
                    );
                    break; // Converged successfully
                }
                Err(EntropyError::GpuError(e)) => return Err(WfcError::Gpu(e)),
                Err(e) => return Err(WfcError::EntropyCalculation(e)), // Other entropy errors
            };

            // 2. Collapse Cell (Implicitly handled by next propagation)
            // The coordinator should provide the tile to collapse to, or handle it.
            // For now, assume coordinator handles collapse internally or provides info needed.
            // Let's assume propagation starts from the selected cell.
            let updated_coords = vec![selected_coords]; // Start propagation from the selected cell

            // 3. Propagate Constraints (using Coordinator)
            let propagation_result = instance
                .coordinator
                .coordinate_propagation(
                    &mut instance.propagator,
                    &instance.buffers, // Pass buffers needed by coordinator
                    instance.backend.device(),
                    instance.backend.queue(),
                    updated_coords,
                )
                .await;

            match propagation_result {
                Ok(_) => { /* Continue */ }
                Err(PropagationError::Contradiction(coords_option)) => {
                    // Map Option<GridCoord3D> to Option<(usize, usize, usize)>
                    let coords_tuple_option =
                        coords_option.map(|gc| (gc.x as usize, gc.y as usize, gc.z as usize));
                    error!(
                        "Contradiction detected during propagation at {:?}",
                        coords_tuple_option
                    );
                    return Err(WfcError::Contradiction(coords_tuple_option)); // Pass Option<(usize, usize, usize)>
                }
                Err(PropagationError::GpuError(e)) => return Err(WfcError::Gpu(e)),
                Err(e) => return Err(WfcError::Propagation(e)), // Other propagation errors
            }

            // --- Progress Reporting ---
            // Fetch intermediate grid for accurate progress info
            let intermediate_grid = instance
                .buffers
                .download_results(
                    instance.backend.device().clone(),
                    instance.backend.queue().clone(),
                    DownloadRequest {
                        download_grid_possibilities: true,
                        ..Default::default()
                    },
                )
                .await?
                .to_possibility_grid(
                    instance.grid_definition.dims,
                    instance.grid_definition.num_tiles,
                )?;

            let collapsed_cells_count = intermediate_grid.count_collapsed_cells();

            let progress = collapsed_cells_count as f32 / total_cells as f32;
            let progress_info = ProgressInfo {
                iteration: iteration, // Ensure field name is correct
                total_iterations: Some(max_iterations),
                collapsed_cells: collapsed_cells_count,
                total_cells: total_cells,
                progress: progress,
                start_time: start_time,
                elapsed_time: start_time.elapsed(),
                message: format!("Iteration {} complete", iteration),
                grid_state: intermediate_grid, // grid_state expects PossibilityGrid, not Option
            };
            progress_callback(progress_info);

            // --- Debug Visualization ---
            if let Some(visualizer) = &mut instance.debug_visualizer {
                trace!("Updating debug visualizer...");
                if let Err(e) = visualizer
                    .update(&instance.backend, &instance.buffers)
                    .await
                {
                    error!("Failed to update debug visualizer: {}", e);
                }
            }
        }

        // --- Final Result Download ---
        trace!("Downloading final grid state from GPU...");
        let request = DownloadRequest {
            download_grid_possibilities: true,
            ..Default::default()
        };
        let results = instance
            .buffers
            .download_results(instance.backend.device(), instance.backend.queue(), request)
            .await?;
        trace!("Final grid state downloaded.");

        results
            .to_possibility_grid(
                instance.grid_definition.dims,
                instance.grid_definition.num_tiles,
            )
            .map_err(WfcError::Gpu)
    }

    pub fn set_progress_callback<F>(&mut self, callback: F)
    where
        F: FnMut(f32) + Send + Sync + 'static,
    {
        let mut instance = self.instance.write().unwrap(); // Acquire write lock
        instance.progress_callback = Some(Box::new(callback));
    }
}

impl Drop for GpuAccelerator {
    fn drop(&mut self) {
        // Acquire write lock to ensure exclusive access during drop
        let _instance = self.instance.write().unwrap();
        // GPU resources within AcceleratorInstance (Buffers, Pipelines, Backend) should handle their own cleanup
        // via their respective Drop implementations when the Arc count reaches zero.
        info!("Dropping GpuAccelerator instance. GPU resources should be released if this is the last reference.");
    }
}

// Removed trait impls

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::create_test_device_queue;
    use futures::executor::block_on;
    use std::collections::HashSet;
    use wfc_core::grid::{GridCoord3D, PossibilityGrid};
    use wfc_rules::{AdjacencyRules, TileSet, Transformation};

    // Helper to create basic grid and rules for testing
    fn setup_basic_test_data() -> (PossibilityGrid, AdjacencyRules) {
        let width = 3;
        let height = 3;
        let depth = 1;
        let num_tiles = 2;
        let grid = PossibilityGrid::new(width, height, depth, num_tiles);

        let weights = vec![1.0; num_tiles];
        let allowed_transforms = vec![vec![Transformation::Identity]; num_tiles];
        let tileset = TileSet::new(weights, allowed_transforms).unwrap();
        let num_transformed_tiles = tileset.num_transformed_tiles();
        let allowed_tuples = (0..num_transformed_tiles).flat_map(|t1| {
            (0..num_transformed_tiles).flat_map(move |t2| (0..6).map(move |axis| (t1, t2, axis)))
        });
        let rules = AdjacencyRules::from_allowed_tuples(num_transformed_tiles, 6, allowed_tuples);

        (grid, rules)
    }

    #[tokio::test]
    async fn test_accelerator_creation_and_config() {
        let (grid, rules) = setup_basic_test_data();
        let accelerator = GpuAccelerator::new(
            &grid,
            &rules,
            BoundaryCondition::Finite,
            EntropyHeuristicType::Shannon,
            None,
        )
        .await;
        assert!(accelerator.is_ok());
        let acc = accelerator.unwrap();

        // Test basic config access
        assert_eq!(acc.boundary_condition(), BoundaryCondition::Finite);
        let grid_def = acc.grid_definition();
        assert_eq!(grid_def.width, 3);
        assert_eq!(grid_def.height, 3);
        assert_eq!(grid_def.depth, 1);
        assert_eq!(grid_def.num_tiles, 2);

        // TODO: Test setting/getting heuristic and subgrid config once refactored
    }

    // TODO: Add tests for run_with_callback
    // TODO: Add tests for get_intermediate_result
    // TODO: Add tests for error conditions (e.g., contradiction)
}
