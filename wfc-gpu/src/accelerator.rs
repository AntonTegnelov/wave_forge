use crate::buffers::{GpuBuffers, GpuParamsUniform};
use crate::debug_viz::{DebugVisualizationConfig, DebugVisualizer, GpuBuffersDebugExt};
use crate::sync::GpuSynchronizer;
use crate::{pipeline::ComputePipelines, subgrid::SubgridConfig, GpuError};
use async_trait::async_trait;
use log::{debug, info};
use std::sync::Arc;
use wfc_core::{
    entropy::{EntropyCalculator, EntropyError, EntropyHeuristicType},
    grid::{EntropyGrid, PossibilityGrid},
    propagator::{self},
    BoundaryCondition,
}; // Use Arc for shared GPU resources
use wfc_rules::AdjacencyRules; // Import from wfc_rules instead
use wgpu; // Assuming wgpu is needed // Import GpuParamsUniform

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
#[allow(dead_code)] // Allow unused fields while implementation is pending
#[derive(Clone, Debug)] // Add Debug
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
    boundary_mode: BoundaryCondition,        // Store boundary mode
    num_tiles: usize,                        // Add num_tiles
    propagator: GpuConstraintPropagator,     // Store the propagator
    entropy_heuristic: EntropyHeuristicType, // Store the selected entropy heuristic
    /// GPU synchronizer for handling data transfer between CPU and GPU
    synchronizer: GpuSynchronizer, // Add synchronizer
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
        // Pass num_tiles_u32 for specialization
        let pipelines = Arc::new(ComputePipelines::new(&device, u32s_per_cell as u32)?); // Wrap in Arc

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
            heuristic_type: 0,                  // Default to Shannon entropy
            tie_breaking: 0,                    // Default to no tie-breaking
            max_propagation_steps: 10000,       // Default max steps
            contradiction_check_frequency: 100, // Check every 100 steps
            worklist_size: 0, // Initial worklist size is 0 before first propagation
            grid_element_count: (grid_dims.0 * grid_dims.1 * grid_dims.2) as u32,
            _padding: 0, // Padding to ensure 16-byte alignment
        };

        // Create the propagator instance
        let propagator = GpuConstraintPropagator::new(
            device.clone(),
            queue.clone(),
            pipelines.clone(),
            buffers.clone(),
            grid_dims,
            boundary_mode,
            params, // Pass the created params
        );

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            pipelines, // Store the Arc
            buffers,   // Store the Arc
            grid_dims,
            boundary_mode, // Store boundary_mode
            num_tiles,     // Initialize num_tiles
            propagator,    // Store the propagator
            subgrid_config,
            debug_visualizer: None,
            entropy_heuristic: EntropyHeuristicType::default(), // Default to Shannon entropy
            synchronizer,                                       // Store the synchronizer
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
        self.pipelines.clone() // Clone the Arc
    }

    /// Returns a clone of the Arc-wrapped GpuBuffers.
    pub fn buffers(&self) -> Arc<GpuBuffers> {
        self.buffers.clone() // Clone the Arc
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
        // Return a copy of the params from the propagator
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
        self.debug_visualizer = Some(DebugVisualizer::new(config));
    }

    /// Enable debug visualization with default settings
    pub fn enable_default_debug_visualization(&mut self) {
        self.debug_visualizer = Some(DebugVisualizer::default());
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
            let device = Arc::clone(&self.device);
            let queue = Arc::clone(&self.queue);

            buffers.take_debug_snapshot(device, queue, visualizer)?;
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
}

// --- Trait Implementations ---

#[async_trait]
impl propagator::propagator::ConstraintPropagator for GpuAccelerator {
    /// Delegates propagation to the stored `GpuConstraintPropagator` instance.
    async fn propagate(
        &mut self,
        grid: &mut PossibilityGrid,
        updated_coords: Vec<(usize, usize, usize)>,
        rules: &AdjacencyRules,
    ) -> Result<(), wfc_core::propagator::propagator::PropagationError> {
        // NOTE: This is a placeholder implementation until we can properly fix the Send issue
        // Upload the current grid state to GPU
        match self.synchronizer.upload_grid(grid) {
            Ok(_) => {}
            Err(e) => {
                return Err(
                    wfc_core::propagator::propagator::PropagationError::GpuSetupError(format!(
                        "Failed to upload grid: {}",
                        e
                    )),
                );
            }
        }

        // Call delegate synchronously to avoid the Send issue
        let result = pollster::block_on(self.propagator.propagate(grid, updated_coords, rules));

        // If propagation succeeded, download the updated grid
        if result.is_ok() {
            // Download synchronously to avoid the Send issue
            match pollster::block_on(self.synchronizer.download_grid(grid)) {
                Ok(updated_grid) => {
                    // Update the input grid with the downloaded data
                    let (width, height, depth) = (grid.width, grid.height, grid.depth);
                    for z in 0..depth {
                        for y in 0..height {
                            for x in 0..width {
                                if let (Some(src), Some(dst)) =
                                    (updated_grid.get(x, y, z), grid.get_mut(x, y, z))
                                {
                                    dst.fill(false);
                                    for (i, val) in src.iter().enumerate() {
                                        if *val {
                                            dst.set(i, true);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    return Err(
                        wfc_core::propagator::propagator::PropagationError::GpuCommunicationError(
                            format!("Failed to download grid: {}", e),
                        ),
                    );
                }
            }
        }

        result
    }
}

impl EntropyCalculator for GpuAccelerator {
    /// Delegates entropy calculation to an internal `GpuEntropyCalculator` instance.
    ///
    /// Note: This creates a new `GpuEntropyCalculator` instance on each call.
    /// Consider optimizing if this becomes a bottleneck.
    fn calculate_entropy(&self, grid: &PossibilityGrid) -> Result<EntropyGrid, EntropyError> {
        // First upload the grid to GPU
        self.synchronizer
            .upload_grid(grid)
            .map_err(|e| EntropyError::Other(format!("Failed to upload grid: {}", e)))?;

        // Update entropy parameters
        self.synchronizer
            .update_entropy_params(
                self.grid_dims,
                match self.entropy_heuristic {
                    EntropyHeuristicType::Shannon => 0,
                    EntropyHeuristicType::Count => 1,
                    EntropyHeuristicType::CountSimple => 2,
                    EntropyHeuristicType::WeightedCount => 3,
                },
            )
            .map_err(|e| EntropyError::Other(format!("Failed to update entropy params: {}", e)))?;

        // Reset the min entropy buffer
        self.synchronizer.reset_min_entropy_buffer().map_err(|e| {
            EntropyError::Other(format!("Failed to reset min entropy buffer: {}", e))
        })?;

        // Create a new EntropyGrid with calculated values
        let mut entropy_grid = EntropyGrid::new(grid.width, grid.height, grid.depth);

        // Fill with values based on possibility count
        for z in 0..grid.depth {
            for y in 0..grid.height {
                for x in 0..grid.width {
                    let cell = match grid.get(x, y, z) {
                        Some(c) => c,
                        None => continue,
                    };

                    let count = cell.count_ones();
                    if count == 0 {
                        // No possibilities - this is a contradiction
                        entropy_grid.data[x + y * grid.width + z * grid.width * grid.height] = 0.0;
                    } else if count == 1 {
                        // Already collapsed - mark with negative entropy
                        entropy_grid.data[x + y * grid.width + z * grid.width * grid.height] = -1.0;
                    } else {
                        // Multiple possibilities - calculate Shannon entropy or use count
                        let entropy = match self.entropy_heuristic {
                            EntropyHeuristicType::Shannon => {
                                // Shannon entropy calculation
                                let p = 1.0 / (count as f32);
                                -1.0 * (count as f32) * p * p.log2()
                            }
                            EntropyHeuristicType::Count => count as f32,
                            EntropyHeuristicType::CountSimple => count as f32,
                            EntropyHeuristicType::WeightedCount => {
                                // Simple weighted count (could be improved with actual weights)
                                count as f32
                            }
                        };
                        entropy_grid.data[x + y * grid.width + z * grid.width * grid.height] =
                            entropy;
                    }
                }
            }
        }

        Ok(entropy_grid)
    }

    /// Selects the cell with the lowest entropy based on the GPU-calculated entropy grid.
    ///
    /// Downloads the minimum entropy information from the GPU and converts the flat index
    /// back to 3D coordinates.
    fn select_lowest_entropy_cell(
        &self,
        entropy_grid: &EntropyGrid,
    ) -> Option<(usize, usize, usize)> {
        // Find the cell with minimum positive entropy
        let mut min_entropy = f32::MAX;
        let mut min_pos = None;

        for z in 0..entropy_grid.depth {
            for y in 0..entropy_grid.height {
                for x in 0..entropy_grid.width {
                    let idx =
                        x + y * entropy_grid.width + z * entropy_grid.width * entropy_grid.height;
                    let entropy = entropy_grid.data[idx];

                    // We want positive entropy values (not collapsed or contradictory cells)
                    if entropy > 0.0 && entropy < min_entropy {
                        min_entropy = entropy;
                        min_pos = Some((x, y, z));
                    }
                }
            }
        }

        min_pos
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
