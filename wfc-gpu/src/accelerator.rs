use crate::{buffers::GpuBuffers, pipeline::ComputePipelines, GpuError};
use log::info;
use std::sync::Arc;
use wfc_core::{
    // Removed unused: EntropyCalculator, ConstraintPropagator, PropagationError, EntropyGrid, Grid
    grid::PossibilityGrid,
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
    pipelines: Arc<ComputePipelines>, // Changed to Arc
    buffers: Arc<GpuBuffers>,         // Changed to Arc
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

        // Calculate num_tiles_u32 here
        let num_tiles = rules.num_tiles();
        let u32s_per_cell = (num_tiles + 31) / 32; // Ceiling division

        // 4. Create pipelines (uses device, returns Cloneable struct)
        // Pass num_tiles_u32 for specialization
        let pipelines = Arc::new(ComputePipelines::new(&device, u32s_per_cell as u32)?); // Wrap in Arc

        // 5. Create buffers (uses device & queue, returns Cloneable struct)
        let buffers = Arc::new(GpuBuffers::new(&device, &queue, initial_grid, rules)?); // Wrap in Arc

        let grid_dims = (initial_grid.width, initial_grid.height, initial_grid.depth);

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            pipelines, // Store the Arc
            buffers,   // Store the Arc
            grid_dims,
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
}
