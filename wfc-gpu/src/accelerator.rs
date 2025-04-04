use crate::buffers::GpuParamsUniform;
use crate::{buffers::GpuBuffers, pipeline::ComputePipelines, subgrid::SubgridConfig, GpuError};
use async_trait::async_trait;
use log::info;
use std::sync::Arc;
use wfc_core::{
    entropy::{EntropyCalculator, EntropyError},
    grid::{EntropyGrid, PossibilityGrid},
    propagator, BoundaryCondition,
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
    instance: Arc<wgpu::Instance>, // Wrap in Arc
    adapter: Arc<wgpu::Adapter>,   // Wrap in Arc
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipelines: Arc<ComputePipelines>, // Changed to Arc
    buffers: Arc<GpuBuffers>,         // Changed to Arc
    grid_dims: (usize, usize, usize),
    boundary_mode: BoundaryCondition,    // Store boundary mode
    num_tiles: usize,                    // Add num_tiles
    propagator: GpuConstraintPropagator, // Store the propagator
}

// Import the concrete GPU implementations
use crate::entropy::GpuEntropyCalculator;
use crate::propagator::GpuConstraintPropagator;

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
    /// * `boundary_mode` - The boundary handling mode for the grid.
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
        boundary_mode: BoundaryCondition,
    ) -> Result<Self, GpuError> {
        info!(
            "Entered GpuAccelerator::new with boundary mode {:?}",
            boundary_mode
        );
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

        // Calculate num_tiles_u32 here
        let num_tiles = rules.num_tiles();
        let u32s_per_cell = (num_tiles + 31) / 32; // Ceiling division

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

        let grid_dims = (initial_grid.width, initial_grid.height, initial_grid.depth);

        // Create GpuParamsUniform
        let params = GpuParamsUniform {
            grid_width: grid_dims.0 as u32,
            grid_height: grid_dims.1 as u32,
            grid_depth: grid_dims.2 as u32,
            num_tiles: num_tiles as u32,
            num_axes: rules.num_axes() as u32,
            worklist_size: 0, // Initial worklist size is 0 before first propagation
            boundary_mode: match boundary_mode {
                BoundaryCondition::Finite => 0,
                BoundaryCondition::Periodic => 1,
            },
            _padding1: 0,
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
            .with_parallel_subgrid_processing(config);
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
        // Use the stored propagator
        self.propagator.propagate(grid, updated_coords, rules).await
    }
}

impl EntropyCalculator for GpuAccelerator {
    /// Delegates entropy calculation to an internal `GpuEntropyCalculator` instance.
    ///
    /// Note: This creates a new `GpuEntropyCalculator` instance on each call.
    /// Consider optimizing if this becomes a bottleneck.
    fn calculate_entropy(&self, grid: &PossibilityGrid) -> Result<EntropyGrid, EntropyError> {
        // Create a GpuEntropyCalculator using the accelerator's resources
        let calculator = GpuEntropyCalculator::new(
            self.device(),
            self.queue(),
            self.pipelines(),
            self.buffers(),
            self.grid_dims(),
        );
        // Delegate the actual work
        calculator.calculate_entropy(grid)
    }

    /// Selects the cell with the lowest entropy based on the GPU-calculated entropy grid.
    ///
    /// Downloads the minimum entropy information from the GPU and converts the flat index
    /// back to 3D coordinates.
    fn select_lowest_entropy_cell(
        &self,
        entropy_grid: &EntropyGrid,
    ) -> Option<(usize, usize, usize)> {
        // Create a GpuEntropyCalculator using the accelerator's resources
        let calculator = GpuEntropyCalculator::new(
            self.device(),
            self.queue(),
            self.pipelines(),
            self.buffers(),
            self.grid_dims(),
        );
        // Delegate the actual work
        // Note: GpuEntropyCalculator::select_lowest_entropy_cell might need adjustment
        // if it expects to reuse state or internal buffers.
        // For now, assuming it's relatively stateless or handles its own setup.
        calculator.select_lowest_entropy_cell(entropy_grid)
    }
}
